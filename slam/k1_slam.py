"""
K1 SLAM — Simultaneous Localization and Mapping for the Booster K1 robot.

Frame-to-frame RGB-D odometry (with point-cloud ICP fallback), TSDF volume
integration for 3D mesh reconstruction, and 2D occupancy-grid generation.

Coordinate convention
---------------------
Internally the camera pose is tracked in the standard pinhole-camera frame
(x=right, y=down, z=forward).  Public accessors (`get_pose`,
`get_odometry_xy_theta`, `get_occupancy_grid`) return values in a *map frame*
suitable for bird's-eye rendering where:

    map x  =  camera x   (right)
    map y  =  −camera z  (objects ahead → smaller y → higher on the image)

This matches the server's `_project_detections_to_map` which reads
``p_world[0], p_world[1]`` and draws a triangle whose tip is at (0, −10)
on screen for yaw = 0.

Requires: pip install open3d opencv-python numpy
"""

import math
import os
import tempfile
import threading
import time

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError as _e:
    raise ImportError(
        "K1 SLAM requires Open3D.  Install with:  pip install open3d"
    ) from _e

# Camera frame → map frame (static, one-time rotation).
#   map_x =  cam_x          (right)
#   map_y = -cam_z          (forward on camera → "up" on the rendered map)
#   map_z =  cam_y          (kept for completeness; unused in 2-D)
_T_CAM_TO_MAP = np.array([
    [1.,  0.,  0., 0.],
    [0.,  0., -1., 0.],
    [0.,  1.,  0., 0.],
    [0.,  0.,  0., 1.],
], dtype=np.float64)


class K1SLAM:
    """Real-time depth-based SLAM for the K1 robot's stereo camera."""

    def __init__(
        self,
        enable_tsdf=True,
        fx=320.0,
        fy=320.0,
        voxel_length=0.025,
        sdf_trunc=0.06,
        max_depth_m=5.0,
        min_depth_m=0.1,
    ):
        self._fx = fx
        self._fy = fy
        self._intrinsic = None
        self._w = self._h = 0

        # Accumulated camera-frame pose (identity at start).
        self._cam_pose = np.eye(4, dtype=np.float64)

        self._prev_rgbd = None
        self._prev_depth = None          # kept for ICP fallback
        self._frame_count = 0
        self._last_update_time = 0.0

        self._max_depth = max_depth_m
        self._min_depth = min_depth_m

        # TSDF volume (lazy-created on first depth frame).
        self._enable_tsdf = enable_tsdf
        self._tsdf = None
        self._voxel_length = voxel_length
        self._sdf_trunc = sdf_trunc

        # Occupancy: accumulated map-frame (x, y) obstacle observations.
        self._occ_pts = np.empty((0, 2), dtype=np.float32)
        self._max_occ = 500_000

        self._lock = threading.Lock()

    # ── internal helpers ──────────────────────────────────────────────────

    def _ensure_intrinsic(self, w, h, cx, cy):
        if self._intrinsic is not None and self._w == w and self._h == h:
            return
        self._w, self._h = w, h
        cx = cx if cx is not None else w / 2.0
        cy = cy if cy is not None else h / 2.0
        self._intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, self._fx, self._fy, cx, cy,
        )

    @staticmethod
    def _to_rgbd(depth_u16, rgb, max_depth, convert_to_intensity=True):
        h, w = depth_u16.shape[:2]
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_u16.astype(np.uint16)))
        if rgb is not None and rgb.dtype == np.uint8 and rgb.shape[:2] == (h, w):
            color_img = o3d.geometry.Image(np.ascontiguousarray(rgb))
        else:
            color_img = o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1000.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=convert_to_intensity,
        )

    @staticmethod
    def _is_sane_transform(T, max_translation=0.5, max_angle_deg=30):
        t_norm = float(np.linalg.norm(T[:3, 3]))
        cos_a = np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        angle = math.acos(float(cos_a))
        return t_norm < max_translation and angle < math.radians(max_angle_deg)

    def _rgbd_odometry(self, src_rgbd, tgt_rgbd):
        """RGB-D odometry (hybrid term, then color-only fallback).

        Returns 4x4 relative transform (target_T_source) or None.
        """
        option = o3d.pipelines.odometry.OdometryOption(
            depth_diff_max=0.07,
            depth_min=self._min_depth,
            depth_max=self._max_depth,
        )
        for jac_cls in (
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm,
        ):
            ok, T, _info = o3d.pipelines.odometry.compute_rgbd_odometry(
                src_rgbd, tgt_rgbd, self._intrinsic, np.eye(4),
                jac_cls(), option,
            )
            if ok and self._is_sane_transform(T):
                return T
        return None

    def _icp_odometry(self, cur_depth, prev_depth):
        """Point-to-plane ICP fallback when RGB-D odometry fails."""
        src = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(np.ascontiguousarray(cur_depth.astype(np.uint16))),
            self._intrinsic,
            depth_scale=1000.0,
            depth_trunc=self._max_depth,
        )
        tgt = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(np.ascontiguousarray(prev_depth.astype(np.uint16))),
            self._intrinsic,
            depth_scale=1000.0,
            depth_trunc=self._max_depth,
        )
        if len(src.points) < 200 or len(tgt.points) < 200:
            return None

        src = src.voxel_down_sample(0.05)
        tgt = tgt.voxel_down_sample(0.05)
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
        )

        result = o3d.pipelines.registration.registration_icp(
            src, tgt, 0.10, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
        if result.fitness > 0.3 and self._is_sane_transform(result.transformation):
            return result.transformation
        return None

    def _project_to_map_2d(self, depth_u16, step=10):
        """Depth pixels → map-frame (x, y) points for occupancy grid."""
        h, w = depth_u16.shape
        mat = self._intrinsic.intrinsic_matrix
        cx, cy = float(mat[0, 2]), float(mat[1, 2])
        fx, fy = self._fx, self._fy

        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        uu, vv = np.meshgrid(xs, ys)
        zz = depth_u16[vv, uu].astype(np.float32) / 1000.0
        mask = (zz > self._min_depth) & (zz < self._max_depth)
        z = zz[mask]
        if z.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        u = uu.astype(np.float32)[mask]
        v = vv.astype(np.float32)[mask]

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        ones = np.ones_like(z)
        pts_cam = np.stack([x_cam, y_cam, z, ones], axis=0)          # 4 × N

        world_T_cam = _T_CAM_TO_MAP @ self._cam_pose
        pts_world = world_T_cam @ pts_cam                             # 4 × N
        return np.stack([pts_world[0], pts_world[1]], axis=-1).astype(np.float32)

    # ── public API (matches server.py call sites) ─────────────────────────

    def update(self, depth, rgb=None, cx=None, cy=None, min_interval=0.3):
        """Process a new depth frame (+ optional BGR image).

        Called by the server's ``_slam_update_loop`` in an executor thread.
        """
        now = time.time()
        if now - self._last_update_time < min_interval:
            return
        self._last_update_time = now

        dh, dw = depth.shape[:2]

        if rgb is not None:
            rh, rw = rgb.shape[:2]
            if (rh, rw) != (dh, dw):
                rgb = cv2.resize(rgb, (dw, dh))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        self._ensure_intrinsic(dw, dh, cx, cy)

        # Intensity version for odometry (grayscale + float depth).
        cur_rgbd_odo = self._to_rgbd(depth, rgb, self._max_depth, convert_to_intensity=True)
        # RGB version for TSDF (keeps uint8 color so ScalableTSDFVolume/RGB8 is happy).
        cur_rgbd_tsdf = (
            self._to_rgbd(depth, rgb, self._max_depth, convert_to_intensity=False)
            if self._enable_tsdf else None
        )

        with self._lock:
            # --- odometry ---
            if self._prev_rgbd is not None:
                T = self._rgbd_odometry(cur_rgbd_odo, self._prev_rgbd)
                if T is None and self._prev_depth is not None:
                    T = self._icp_odometry(depth, self._prev_depth)
                if T is not None:
                    self._cam_pose = self._cam_pose @ T

            # --- TSDF integration ---
            if self._enable_tsdf and cur_rgbd_tsdf is not None:
                if self._tsdf is None:
                    self._tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
                        voxel_length=self._voxel_length,
                        sdf_trunc=self._sdf_trunc,
                        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                    )
                self._tsdf.integrate(
                    cur_rgbd_tsdf, self._intrinsic, np.linalg.inv(self._cam_pose),
                )

            # --- occupancy accumulation ---
            pts = self._project_to_map_2d(depth, step=12)
            if pts.shape[0] > 0:
                self._occ_pts = np.vstack([self._occ_pts, pts])
                if self._occ_pts.shape[0] > self._max_occ:
                    self._occ_pts = self._occ_pts[-self._max_occ // 2:]

            self._prev_rgbd = cur_rgbd_odo
            self._prev_depth = depth.copy()
            self._frame_count += 1

    # ── pose accessors ────────────────────────────────────────────────────

    def get_pose(self):
        """4x4 world_T_cam in the map frame.

        Used by ``_project_detections_to_map`` which reads
        ``p_world[0], p_world[1]`` as (map_x, map_y).
        """
        with self._lock:
            return (_T_CAM_TO_MAP @ self._cam_pose).copy()

    def get_odometry_xy_theta(self):
        """(x, y, yaw) in the map frame.

        *  x  — rightward displacement from start
        *  y  — displacement along the "vertical" map axis (negative = ahead)
        *  yaw — heading; 0 = initial forward, positive = turned left
        """
        with self._lock:
            pose = (_T_CAM_TO_MAP @ self._cam_pose).copy()
        x = float(pose[0, 3])
        y = float(pose[1, 3])
        R = pose[:3, :3]
        # Heading = where the camera's z-axis (forward) points in the map frame.
        # After the static rotation, that direction lands in columns 2 of R_map.
        yaw = math.atan2(float(R[0, 2]), float(-R[1, 2]))
        return x, y, yaw

    # ── occupancy grid ────────────────────────────────────────────────────

    def get_occupancy_grid(
        self,
        resolution=0.05,
        width_m=10.0,
        height_m=10.0,
        center_on_robot=True,
    ):
        """Generate a 2-D occupancy grid.

        Returns ``(grid, resolution, (origin_x, origin_y))`` or ``None``.

        *grid* is ``uint8``: 0 = free, 50 = unknown, 100 = occupied.
        """
        with self._lock:
            if self._occ_pts.shape[0] < 10:
                return None
            pts = self._occ_pts.copy()
            map_pose = (_T_CAM_TO_MAP @ self._cam_pose).copy()

        cols = int(width_m / resolution)
        rows = int(height_m / resolution)

        rx = float(map_pose[0, 3])
        ry = float(map_pose[1, 3])
        if center_on_robot:
            ox = rx - width_m / 2.0
            oy = ry - height_m / 2.0
        else:
            ox = -width_m / 2.0
            oy = -height_m / 2.0

        grid = np.full((rows, cols), 50, dtype=np.uint8)

        # Obstacle pixels
        px = ((pts[:, 0] - ox) / resolution).astype(np.int32)
        py = ((pts[:, 1] - oy) / resolution).astype(np.int32)
        valid = (px >= 0) & (px < cols) & (py >= 0) & (py < rows)
        px, py = px[valid], py[valid]

        # Robot pixel (clamped into grid)
        rcol = max(0, min(cols - 1, int((rx - ox) / resolution)))
        rrow = max(0, min(rows - 1, int((ry - oy) / resolution)))

        # Free-space rays: cv2.line is C-fast; draw from robot to each obstacle
        ray_step = max(1, len(px) // 3000)
        for i in range(0, len(px), ray_step):
            cv2.line(grid, (rcol, rrow), (int(px[i]), int(py[i])), 0, 1)

        # Stamp occupied ON TOP so rays don't erase them
        grid[py, px] = 100

        return grid, resolution, (ox, oy)

    # ── mesh export ───────────────────────────────────────────────────────

    def get_mesh_ply(self):
        """Extract the TSDF triangle mesh and return raw PLY bytes (or None)."""
        with self._lock:
            if not self._enable_tsdf or self._tsdf is None:
                return None
            try:
                mesh = self._tsdf.extract_triangle_mesh()
            except Exception:
                return None

        if len(mesh.vertices) == 0:
            return None
        mesh.compute_vertex_normals()

        fd, tmp_path = tempfile.mkstemp(suffix=".ply")
        os.close(fd)
        try:
            o3d.io.write_triangle_mesh(tmp_path, mesh, write_ascii=False)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self):
        """Clear the map, TSDF, and return pose to the origin."""
        with self._lock:
            self._cam_pose = np.eye(4, dtype=np.float64)
            self._prev_rgbd = None
            self._prev_depth = None
            self._frame_count = 0
            self._occ_pts = np.empty((0, 2), dtype=np.float32)
            if self._tsdf is not None:
                self._tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=self._voxel_length,
                    sdf_trunc=self._sdf_trunc,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                )
        print("SLAM: reset")
