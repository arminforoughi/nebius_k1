#!/usr/bin/env python3
"""
Remote Server — runs on YOUR machine (laptop/desktop with GPU).
Receives camera + depth + audio from the robot over WebSocket.
Runs YOLO, face recognition, Gemini Live API, tracking/follow logic.
Sends control commands back to the robot.

Usage:
    export GEMINI_API_KEY="your-key"
    python3 server.py
    python3 server.py --voice Charon --no-faces --port 8080
"""

import os
import sys
import asyncio
import threading
import base64
import time
import argparse
import json
import re
import math
import struct
import zlib
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2


class _NullStream:
    """No-op audio stream when no device is available (e.g. headless VM)."""
    def read(self, num_frames, exception_on_overflow=False):
        return b'\x00' * (num_frames * 2)  # int16 = 2 bytes per sample
    def write(self, data):
        pass
    def stop_stream(self):
        pass
    def close(self):
        pass


# Prefer compat on macOS to avoid "Could not import the PyAudio C module" and dylib issues.
# On headless VMs (no PortAudio), use null audio so server still runs; audio goes to robot only.
pyaudio = None
if sys.platform == 'darwin':
    try:
        import pyaudio_compat as pyaudio
    except (ImportError, OSError):
        pass
if pyaudio is None:
    try:
        import pyaudio
    except ImportError:
        try:
            import pyaudio_compat as pyaudio
        except (ImportError, OSError):
            pass
if pyaudio is None:
    # No PortAudio/PyAudio: dummy module so AUDIO_FORMAT and PyAudio() exist; streams are no-ops.
    class _NullPyAudio:
        paInt16 = 8
        def open(self, **kwargs):
            return _NullStream()  # module-level class above
        def terminate(self):
            pass
    class _NullPyaudioModule:
        PyAudio = _NullPyAudio
        paInt16 = 8
    pyaudio = _NullPyaudioModule()
    print("Warning: PyAudio/PortAudio not available; local audio I/O disabled (robot audio still works).", file=sys.stderr)

from ultralytics import YOLO
from face_engine import FaceEngine

from google import genai
from google.genai import types

# Audio config
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1024

# Binary message type prefixes (robot -> server)
MSG_VIDEO = 0x01
MSG_DEPTH = 0x02
MSG_AUDIO_IN = 0x03
# server -> robot
MSG_AUDIO_OUT = 0x10

# Detection colors (BGR)
_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255),
]

_NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)", re.IGNORECASE),
    re.compile(r"\bi'm (\w+)", re.IGNORECASE),
    re.compile(r"\bi am (\w+)", re.IGNORECASE),
    re.compile(r"\bcall me (\w+)", re.IGNORECASE),
]

_CMD_PATTERNS = [
    (re.compile(r"\b(?:i'll |i will |let me |okay,? |ok,? )?(?:follow|following)\b(?:\s+(?:you|him|her|them|that person|(\w+)))?", re.IGNORECASE), "follow"),
    (re.compile(r"\b(?:stop\s+(?:the\s+)?gesture)\b", re.IGNORECASE), "stop_gesture"),
    (re.compile(r"\b(?:i'll |i will |let me )?stop(?:ping)?\b(?:\s+(?:follow|track|mov))?", re.IGNORECASE), "stop"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:go(?:ing)?|walk(?:ing)?|head(?:ing)?|mov(?:e|ing))\s+(?:to(?:ward)?|over to)\s+(?:(\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)\s+(?:from|away from|near|of)\s+)?(?:the\s+|that\s+)?(\w+)", re.IGNORECASE), "go_to"),
    (re.compile(r"\b(?:i'll |i will |let me |here'?s? |okay,? |ok,? )?(?:do |doing |start )?(?:a |the )?(?:dance|dancing)\b(?:\s+(?:the\s+)?(\w+))?", re.IGNORECASE), "dance"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:wave|waving)\b", re.IGNORECASE), "wave"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:handshake|shake hands?|shaking hands?)\b", re.IGNORECASE), "handshake"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "look_left"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "look_right"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+up\b", re.IGNORECASE), "look_up"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+down\b", re.IGNORECASE), "look_down"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:center|straight|forward|ahead)\b", re.IGNORECASE), "look_center"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:look(?:ing)? at|track(?:ing)?|watch(?:ing)?)\b(?:\s+(?:the\s+)?(\w+))?", re.IGNORECASE), "track"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "turn_left"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "turn_right"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+around\b", re.IGNORECASE), "turn_around"),
    (re.compile(r"\b(?:i'll |let me )?(?:walk(?:ing)?|mov(?:e|ing))\s+forward\b", re.IGNORECASE), "forward"),
    (re.compile(r"\b(?:i'll |let me )?(?:walk(?:ing)?|mov(?:e|ing))\s+backward\b", re.IGNORECASE), "backward"),
    (re.compile(r"\b(?:com(?:e|ing)\s+closer|approach(?:ing)?)\b", re.IGNORECASE), "approach"),
    (re.compile(r"\b(?:back(?:ing)?\s+up|step(?:ping)?\s+back|mov(?:e|ing)\s+back)\b", re.IGNORECASE), "back_up"),
    (re.compile(r"\b(?:i'll |let me )?(?:straf(?:e|ing)|sidestep(?:ping)?|mov(?:e|ing)\s+sideways)\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "strafe_left"),
    (re.compile(r"\b(?:i'll |let me )?(?:straf(?:e|ing)|sidestep(?:ping)?|mov(?:e|ing)\s+sideways)\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "strafe_right"),
    (re.compile(r"\b(?:i'll |let me |okay,? |here'?s? (?:a )?)?dab(?:bing)?\b", re.IGNORECASE), "dab"),
    (re.compile(r"\b(?:i'll |let me |okay,? |here'?s? (?:a )?)?flex(?:ing)?\b", re.IGNORECASE), "flex"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?new\s*year(?:'?s?)?\s*dance\b", re.IGNORECASE), "dance_newyear"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?nezha\s*dance\b", re.IGNORECASE), "dance_nezha"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?future\s*dance\b", re.IGNORECASE), "dance_future"),
    (re.compile(r"\b(?:kick(?:ing)?|boxing\s*kick)\b", re.IGNORECASE), "dance_kick"),
    (re.compile(r"\b(?:moonwalk(?:ing)?)\b", re.IGNORECASE), "dance_moonwalk"),
    (re.compile(r"\b(?:michael\s*jackson)\b", re.IGNORECASE), "dance_michael jackson"),
    (re.compile(r"\b(?:roundhouse(?:\s*kick)?)\b", re.IGNORECASE), "dance_roundhouse"),
    (re.compile(r"\b(?:salsa|arabic\s*dance)\b", re.IGNORECASE), "dance_salsa"),
    (re.compile(r"\b(?:ultraman)\b", re.IGNORECASE), "dance_ultraman"),
    (re.compile(r"\b(?:respect)\b", re.IGNORECASE), "dance_respect"),
    (re.compile(r"\b(?:celebrat(?:e|ing|ion)|cheer(?:ing)?)\b", re.IGNORECASE), "dance_celebrate"),
    (re.compile(r"\b(?:lucky\s*cat)\b", re.IGNORECASE), "dance_luckycat"),
    (re.compile(r"\b(?:macarena)\b", re.IGNORECASE), "dance_macarena"),
    (re.compile(r"\b(?:twist(?:ing)?)\b", re.IGNORECASE), "dance_twist"),
    (re.compile(r"\b(?:take a |do a )?bow(?:ing)?\b", re.IGNORECASE), "dance_bow"),
    (re.compile(r"\b(?:chicken\s*dance|do(?:ing)?\s+(?:the\s+)?chicken)\b", re.IGNORECASE), "dance_chicken"),
    (re.compile(r"\b(?:disco)\b", re.IGNORECASE), "dance_disco"),
    (re.compile(r"\b(?:karate|kung\s*fu)\b", re.IGNORECASE), "dance_karate"),
    (re.compile(r"\b(?:nod(?:ding)?)\b", re.IGNORECASE), "nod"),
    (re.compile(r"\b(?:shak(?:e|ing)\s+(?:my\s+)?head)\b", re.IGNORECASE), "head_shake"),
    (re.compile(r"\b(?:get\s+up|stand\s+up|get\s+back\s+up)\b", re.IGNORECASE), "get_up"),
    (re.compile(r"\b(?:shoot|kick\s+the\s+ball|power\s+kick|score|goal!?)\b", re.IGNORECASE), "shoot"),
    (re.compile(r"\b(?:side\s*foot\s*kick|visual\s*kick|pass\s+the\s+ball|pass)\b", re.IGNORECASE), "visual_kick"),
    (re.compile(r"\b(?:stop\s+(?:the\s+)?kick|stop\s+kick)\b", re.IGNORECASE), "stop_visual_kick"),
    (re.compile(r"\b(?:soccer\s+combo|shoot\s+and\s+celebrate|score\s+and\s+celebrate)\b", re.IGNORECASE), "soccer_combo"),
    (re.compile(r"\b(?:celebrate|celebration|we\s+scored)\b", re.IGNORECASE), "dance_celebrate"),
]


def _color_for_class(cls_id):
    return _COLORS[cls_id % len(_COLORS)]


def _project_detections_to_map(detections, world_T_cam, origin_xy, resolution, grid_rows, grid_cols, frame_shape, fx=320.0, fy=320.0):
    """
    Project YOLO detections (image bbox center + depth) to 2D map pixel coords.
    world_T_cam: 4x4 camera pose. origin_xy: (ox, oy) in meters. resolution: m per pixel.
    Returns list of (col, row, label, color_bgr) for drawing on the map image.
    """
    if not detections or world_T_cam is None:
        return []
    fh, fw = frame_shape[:2] if frame_shape is not None else (480, 640)
    cx0, cy0 = fw / 2.0, fh / 2.0
    ox, oy = origin_xy
    out = []
    for i, det in enumerate(detections):
        distance_m = det.get('distance_m')
        if distance_m is None or distance_m <= 0.05 or distance_m > 15.0:
            continue
        cx, cy = det.get('center', (fw // 2, fh // 2))
        if isinstance(cx, (list, tuple)):
            cx, cy = cx[0], cx[1]
        z = float(distance_m)
        x_cam = (cx - cx0) * z / fx
        y_cam = (cy - cy0) * z / fy
        p_cam = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
        p_world = world_T_cam @ p_cam
        wx, wy = float(p_world[0]), float(p_world[1])
        col = int((wx - ox) / resolution)
        row = int((wy - oy) / resolution)
        if col < 0 or col >= grid_cols or row < 0 or row >= grid_rows:
            continue
        cls_name = det.get('class', '?')
        name = det.get('name')
        if name and cls_name == 'person':
            label = name
        else:
            label = cls_name
        cls_id = hash(cls_name) % len(_COLORS)
        color = _color_for_class(cls_id)
        out.append((col, row, label, color))
    return out


# ── Face Cache ───────────────────────────────────────────────────────────────

FACE_CACHE_DIR = os.path.expanduser('~/.face_cache')
FACE_CACHE_FILE = os.path.join(FACE_CACHE_DIR, 'known_faces.json')


class FaceCache:
    def __init__(self, tolerance=0.6, face_engine=None):
        self.tolerance = tolerance
        self.face_engine = face_engine
        self.entries = []
        self._lock = threading.Lock()
        os.makedirs(FACE_CACHE_DIR, exist_ok=True)
        self._load()

    def _load(self):
        if not os.path.exists(FACE_CACHE_FILE):
            return
        try:
            with open(FACE_CACHE_FILE) as f:
                data = json.load(f)

            if isinstance(data, dict):
                cached_backend = data.get('backend')
                entries = data.get('entries', [])
            else:
                cached_backend = 'dlib'
                entries = data

            engine_backend = self.face_engine.backend if self.face_engine else None
            if cached_backend != engine_backend:
                print(f"Face cache: backend changed ({cached_backend} -> {engine_backend}), clearing cache")
                self._persist()
                return

            for entry in entries:
                self.entries.append({
                    'name': entry['name'],
                    'encoding': np.array(entry['encoding'], dtype=np.float64),
                    'saved_at': entry.get('saved_at', ''),
                })
            print(f"Face cache: loaded {len(self.entries)} known face(s)")
        except Exception as e:
            print(f"Warning: failed to load face cache: {e}")

    def _persist(self):
        try:
            data = {
                'backend': self.face_engine.backend if self.face_engine else None,
                'entries': [
                    {'name': e['name'], 'encoding': e['encoding'].tolist(), 'saved_at': e['saved_at']}
                    for e in self.entries
                ],
            }
            with open(FACE_CACHE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: failed to save face cache: {e}")

    def recognize(self, encoding):
        if self.face_engine is None:
            return None
        with self._lock:
            if not self.entries:
                return None
            known = [e['encoding'] for e in self.entries]
            distances = self.face_engine.face_distance(known, encoding)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= self.tolerance:
                return self.entries[best_idx]['name']
            return None

    def save_face(self, name, encoding):
        with self._lock:
            for e in self.entries:
                if e['name'].lower() == name.lower():
                    e['encoding'] = encoding
                    e['saved_at'] = datetime.now().isoformat()
                    self._persist()
                    return
            self.entries.append({
                'name': name, 'encoding': encoding,
                'saved_at': datetime.now().isoformat(),
            })
            self._persist()
        print(f"Face cache: saved '{name}'")

    def delete_face(self, name):
        with self._lock:
            self.entries = [e for e in self.entries if e['name'].lower() != name.lower()]
            self._persist()

    def list_known(self):
        with self._lock:
            return [{'name': e['name'], 'saved_at': e['saved_at']} for e in self.entries]


# ── Frame Processor (replaces CameraDetectionNode — no ROS2 needed) ─────────


class FrameProcessor:
    """Receives frames from robot WebSocket, runs YOLO + face recognition."""

    def __init__(self, model_path='yolov8n.pt', confidence=0.5,
                 face_cache=None, enable_faces=True, face_engine=None):
        print(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.enable_faces = enable_faces
        self.face_cache = face_cache
        self.face_engine = face_engine

        self._unknown_faces = {}
        self._next_unknown_id = 1
        self._last_face_time = 0.0
        self._face_interval = 0.5
        self._cached_face_results = []

        self._lock = threading.Lock()
        self.latest_frame = None       # annotated frame (for web UI)
        self.latest_detections = []
        self._raw_frame = None         # original decoded frame
        self._depth_map = None         # uint16 depth
        self._frame_shape = None       # (h, w)
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_time = time.time()

        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._pending_frame = None
        self._detect_event = threading.Event()
        self._detect_thread.start()

    def on_video_frame(self, jpeg_bytes):
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self._raw_frame = frame
            self._frame_shape = frame.shape[:2]
            self._pending_frame = frame
            self._detect_event.set()

    def on_depth_frame(self, data):
        try:
            w, h = struct.unpack('<HH', data[:4])
            raw = zlib.decompress(data[4:])
            depth = np.frombuffer(raw, dtype=np.uint16).reshape((h, w))
            self._depth_map = depth
        except Exception as e:
            print(f"Depth decode error: {e}")

    def _detect_loop(self):
        while True:
            self._detect_event.wait()
            self._detect_event.clear()
            frame = self._pending_frame
            if frame is None:
                continue
            try:
                self._run_detection(frame)
            except Exception as e:
                print(f"Detection error: {e}")

    def _get_depth_at(self, x, y, window=5):
        depth_map = self._depth_map
        if depth_map is None:
            return None
        dh, dw = depth_map.shape
        fh, fw = self._frame_shape or (dh * 2, dw * 2)
        sx = x * dw / fw
        sy = y * dh / fh
        ix, iy = int(sx), int(sy)
        half = window // 2
        y1, y2 = max(0, iy - half), min(dh, iy + half + 1)
        x1, x2 = max(0, ix - half), min(dw, ix + half + 1)
        patch = depth_map[y1:y2, x1:x2].astype(np.float32)
        valid = patch[(patch > 0) & (patch < 65535)]
        if len(valid) == 0:
            return None
        return float(np.median(valid)) / 1000.0

    def _run_detection(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        annotated = frame.copy()
        detections = []
        depth_available = self._depth_map is not None
        has_persons = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                if cls_name == 'person':
                    has_persons = True
                cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                distance_m = self._get_depth_at(cx, cy) if depth_available else None
                detections.append({
                    'class': cls_name, 'confidence': round(float(conf), 2),
                    'distance_m': round(float(distance_m), 2) if distance_m else None,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int(cx), int(cy)],
                    'name': None, 'unknown_id': None,
                })

        now = time.time()
        if self.enable_faces and has_persons and now - self._last_face_time >= self._face_interval:
            self._last_face_time = now
            self._cached_face_results = self._run_face_recognition(frame)

        for fr in self._cached_face_results:
            matched_det = self._match_face_to_person(fr, detections)
            if matched_det:
                matched_det['name'] = fr['name']
                matched_det['unknown_id'] = fr.get('unknown_id')
            top, right, bottom, left = fr['face_loc']
            is_known = fr['unknown_id'] is None
            face_color = (0, 255, 255) if is_known else (0, 165, 255)
            cv2.rectangle(annotated, (left, top), (right, bottom), face_color, 2)
            name = fr['name']
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (left, bottom), (left + tw + 4, bottom + th + 8), face_color, -1)
            cv2.putText(annotated, name, (left + 2, bottom + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']
            distance_m = det['distance_m']
            cls_id = list(self.model.names.values()).index(cls_name) if cls_name in self.model.names.values() else 0
            color = _color_for_class(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.0%}"
            if det['name'] and cls_name == 'person':
                label = f"{det['name']} {conf:.0%}"
            if distance_m is not None:
                label += f" {distance_m:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        self._fps_counter += 1
        if now - self._fps_time >= 1.0:
            self._fps = self._fps_counter / (now - self._fps_time)
            self._fps_counter = 0
            self._fps_time = now

        faces_str = f"Faces: {len(self._cached_face_results)}" if self.enable_faces else "Faces: off"
        depth_str = "Depth: ON" if depth_available else "Depth: waiting..."
        status = f"FPS: {self._fps:.0f} | Objects: {len(detections)} | {faces_str} | {depth_str}"
        cv2.putText(annotated, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        with self._lock:
            self.latest_frame = annotated
            self.latest_detections = detections

    def _run_face_recognition(self, frame):
        if self.face_engine is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.face_engine.detect_and_encode(rgb)
        if not detections:
            return []
        results = []
        for top, right, bottom, left, enc in detections:
            name = self.face_cache.recognize(enc) if self.face_cache else None
            unknown_id = None
            if name is None:
                unknown_id = self._get_or_assign_unknown_id(enc)
                name = f"Unknown #{unknown_id}"
            results.append({
                'name': name, 'unknown_id': unknown_id,
                'face_loc': (top, right, bottom, left), 'encoding': enc,
            })
        return results

    def _get_or_assign_unknown_id(self, encoding):
        if self.face_engine is None:
            return 0
        best_dist, best_id = 999.0, None
        for uid, enc in self._unknown_faces.items():
            dist = float(self.face_engine.face_distance([enc], encoding)[0])
            if dist < best_dist:
                best_dist, best_id = dist, uid
        if best_id is not None and best_dist < 0.5:
            self._unknown_faces[best_id] = encoding
            return best_id
        uid = self._next_unknown_id
        self._next_unknown_id += 1
        self._unknown_faces[uid] = encoding
        return uid

    def _match_face_to_person(self, face_result, detections):
        top, right, bottom, left = face_result['face_loc']
        face_cx = (left + right) // 2
        face_cy = (top + bottom) // 2
        for det in detections:
            if det['class'] != 'person':
                continue
            bx1, by1, bx2, by2 = det['bbox']
            if bx1 <= face_cx <= bx2 and by1 <= face_cy <= by2:
                return det
        return None

    def save_unknown_face(self, unknown_id, name):
        enc = self._unknown_faces.get(unknown_id)
        if enc is None:
            return False
        self.face_cache.save_face(name, enc)
        del self._unknown_faces[unknown_id]
        self._cached_face_results = []
        self._last_face_time = 0
        return True

    def try_learn_name_from_transcript(self, text):
        if not self.enable_faces or not self._unknown_faces:
            return
        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).capitalize()
                latest_uid = max(self._unknown_faces.keys())
                if self.save_unknown_face(latest_uid, name):
                    print(f"Auto-learned face: '{name}' from speech")
                    add_transcript("System", f"Learned face: {name}")
                return

    def get_frame_b64jpeg(self, max_dim=640, quality=60):
        with self._lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def get_detection_summary(self):
        with self._lock:
            dets = list(self.latest_detections)
        if not dets:
            return ""
        lines = []
        for d in dets:
            dist = f"{d['distance_m']:.1f}m away" if d['distance_m'] else "unknown dist"
            name_str = f" ({d['name']})" if d.get('name') else ""
            lines.append(f"- {d['class']}{name_str} ({d['confidence']:.0%}, {dist})")
        return "Detected objects:\n" + "\n".join(lines)

    def get_pointcloud_binary(self, step=4, max_depth=6.0, fx=320, fy=320, cx=None, cy=None):
        """Generate a colored point cloud from depth + RGB.

        Returns bytes: N x 6 float32 array [x, y, z, r, g, b] (colors 0-1).
        Downsamples by `step` for performance.
        """
        depth_map = self._depth_map
        raw_frame = self._raw_frame
        if depth_map is None or raw_frame is None:
            return None

        dh, dw = depth_map.shape
        fh, fw = raw_frame.shape[:2]

        if cx is None:
            cx = dw / 2.0
        if cy is None:
            cy = dh / 2.0

        # Scale intrinsics if depth and frame have different resolutions
        scale_x = dw / fw
        scale_y = dh / fh
        fx_d = fx * scale_x
        fy_d = fy * scale_y

        # Resample RGB to depth resolution
        if (fh, fw) != (dh, dw):
            rgb = cv2.resize(raw_frame, (dw, dh))
        else:
            rgb = raw_frame
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Build pixel coordinate grids (downsampled)
        ys = np.arange(0, dh, step)
        xs = np.arange(0, dw, step)
        xx, yy = np.meshgrid(xs, ys)

        depths = depth_map[yy, xx].astype(np.float32) / 1000.0  # mm -> meters
        colors_r = rgb[yy, xx, 0].astype(np.float32) / 255.0
        colors_g = rgb[yy, xx, 1].astype(np.float32) / 255.0
        colors_b = rgb[yy, xx, 2].astype(np.float32) / 255.0

        # Filter valid depths
        valid = (depths > 0.05) & (depths < max_depth)
        z = depths[valid]
        u = xx.astype(np.float32)[valid]
        v = yy.astype(np.float32)[valid]
        r = colors_r[valid]
        g = colors_g[valid]
        b = colors_b[valid]

        # Deproject to 3D (camera coordinates: x=right, y=down, z=forward)
        x = (u - cx) * z / fx_d
        y = (v - cy) * z / fy_d

        points = np.stack([x, -y, -z, r, g, b], axis=-1).astype(np.float32)
        return points.tobytes()


# ── Remote Robot Controller ──────────────────────────────────────────────────


class RobotController:
    """Controls the remote robot by sending commands over WebSocket.
    Runs tracking/follow loops locally (has access to detections)."""

    def __init__(self):
        self._ws = None
        self._loop = None
        self.lock = threading.Lock()
        self.head_pitch = 0.0
        self.head_yaw = 0.0

        self.tracking_active = False
        self.tracking_target = None
        self.tracking_thread = None

        self.follow_active = False
        self.follow_target = None
        self.follow_thread = None
        self.follow_target_distance = 1.0

        self.move_active = False
        self.move_thread = None

        self.frame_processor = None

    def set_connection(self, ws, loop):
        self._ws = ws
        self._loop = loop

    def set_frame_processor(self, fp: FrameProcessor):
        self.frame_processor = fp

    def _send(self, cmd_dict):
        ws, loop = self._ws, self._loop
        if ws and loop:
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(cmd_dict)), loop)

    # ── Head control ─────────────────────────────────────────────────────

    def rotate_head(self, pitch, yaw):
        pitch = max(-0.5, min(1.0, pitch))
        yaw = max(-0.785, min(0.785, yaw))
        self.head_pitch, self.head_yaw = pitch, yaw
        self._send({'cmd': 'rotate_head', 'pitch': pitch, 'yaw': yaw})

    def nod(self):
        self._send({'cmd': 'nod'})

    def head_shake(self):
        self._send({'cmd': 'head_shake'})

    # ── Movement ─────────────────────────────────────────────────────────

    def _move(self, x, y, yaw):
        self._send({'cmd': 'move', 'x': x, 'y': y, 'yaw': yaw})

    def move_timed(self, x, y, yaw, duration):
        self.stop_movement()
        def _run():
            self.move_active = True
            start = time.time()
            while self.move_active and (time.time() - start) < duration:
                self._move(x, y, yaw)
                time.sleep(0.05)
            self._move(0, 0, 0)
            self.move_active = False
        self.move_thread = threading.Thread(target=_run, daemon=True)
        self.move_thread.start()

    def stop_movement(self):
        self.move_active = False
        if self.move_thread and self.move_thread.is_alive():
            self.move_thread.join(timeout=1.0)
        self._move(0, 0, 0)

    def turn_around(self):
        self.move_timed(0, 0, 0.5, 3.0)

    def approach(self):
        self.move_timed(0.4, 0, 0, 2.0)

    def back_up(self):
        self.move_timed(-0.2, 0, 0, 1.5)

    def turn_left(self):
        self.move_timed(0, 0, 0.5, 1.5)

    def turn_right(self):
        self.move_timed(0, 0, -0.5, 1.5)

    def forward(self):
        self.move_timed(0.5, 0, 0, 2.0)

    def backward(self):
        self.move_timed(-0.3, 0, 0, 2.0)

    def strafe_left(self):
        self.move_timed(0, 0.3, 0, 1.5)

    def strafe_right(self):
        self.move_timed(0, -0.3, 0, 1.5)

    # ── Dances / gestures (delegated to robot client) ────────────────────

    def do_dance(self, dance_name=None):
        self._send({'cmd': 'dance', 'name': dance_name or 'robot'})

    def do_wave(self):
        self._send({'cmd': 'wave'})

    def do_handshake(self):
        self._send({'cmd': 'handshake'})

    def do_dab(self):
        self._send({'cmd': 'dab'})

    def do_flex(self):
        self._send({'cmd': 'flex'})

    def do_get_up(self):
        self._send({'cmd': 'get_up'})

    def do_shoot(self):
        self._send({'cmd': 'shoot'})

    def do_visual_kick(self, start=True):
        self._send({'cmd': 'visual_kick', 'start': start})

    def do_stop_visual_kick(self):
        self._send({'cmd': 'visual_kick', 'start': False})

    def do_soccer_combo(self):
        self._send({'cmd': 'soccer_combo'})

    def do_stop_gesture(self):
        """Stop any ongoing gesture and reset arms/head on the robot."""
        self._send({'cmd': 'stop_gesture'})

    # ── Tracking ─────────────────────────────────────────────────────────

    def start_tracking(self, target=None):
        self.stop_tracking()
        self.tracking_active = True
        self.tracking_target = target
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print(f"[Robot] Head tracking started: {target or 'closest person'}")

    def stop_tracking(self):
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
            self.tracking_thread = None
        self._move(0, 0, 0)

    def _tracking_loop(self):
        YAW_BODY_TURN_THRESHOLD = 0.45
        BODY_TURN_SPEED = 0.35

        while self.tracking_active:
            if not self.frame_processor or self.frame_processor._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                if not self.follow_active:
                    self._move(0, 0, 0)
                time.sleep(0.1)
                continue

            shape = self.frame_processor._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx, cy = det['center']
            err_x = (cx - w / 2) / (w / 2)
            err_y = (cy - h / 2) / (h / 2)

            kp_yaw, kp_pitch = 0.15, 0.1
            new_yaw = self.head_yaw - err_x * kp_yaw
            new_pitch = self.head_pitch + err_y * kp_pitch

            if abs(err_x) > 0.08 or abs(err_y) > 0.08:
                self.rotate_head(new_pitch, new_yaw)

            if not self.follow_active:
                if abs(self.head_yaw) > YAW_BODY_TURN_THRESHOLD:
                    body_rot = BODY_TURN_SPEED if self.head_yaw > 0 else -BODY_TURN_SPEED
                    self._move(0, 0, body_rot)
                else:
                    self._move(0, 0, 0)

            time.sleep(0.1)

    def _find_target_detection(self):
        if not self.frame_processor:
            return None
        with self.frame_processor._lock:
            dets = list(self.frame_processor.latest_detections)
        if not dets:
            return None

        target = self.tracking_target
        if target is None or target.lower() in ('person', 'people', 'someone', 'anyone'):
            persons = [d for d in dets if d['class'] == 'person']
            if not persons:
                return None
            with_dist = [p for p in persons if p.get('distance_m')]
            if with_dist:
                return min(with_dist, key=lambda p: p['distance_m'])
            return max(persons, key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]))

        named = [d for d in dets if d.get('name') and target.lower() in d['name'].lower()]
        if named:
            return named[0]

        classed = [d for d in dets if d['class'].lower() == target.lower()]
        if classed:
            with_dist = [c for c in classed if c.get('distance_m')]
            if with_dist:
                return min(with_dist, key=lambda c: c['distance_m'])
            return max(classed, key=lambda c: (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1]))
        return None

    # ── Follow ───────────────────────────────────────────────────────────

    def start_follow(self, target=None):
        self.stop_follow()
        self.follow_active = True
        self.follow_target = target
        self.start_tracking(target or 'person')
        self.follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self.follow_thread.start()
        print(f"[Robot] Following started: {target or 'closest person'}")

    def stop_follow(self):
        self.follow_active = False
        self.stop_tracking()
        if self.follow_thread:
            self.follow_thread.join(timeout=1.0)
            self.follow_thread = None
        self._move(0, 0, 0)

    def _scan_depth_obstacles(self, depth_map, target_distance=None):
        """Fine-grained depth + edge scanning for obstacle avoidance.

        Scans the depth map in a high-resolution grid and also detects sharp
        depth discontinuities (table edges, step-offs, beams, pillars).

        Returns dict with:
            zone_min[i]   - minimum depth (meters) per column (12 cols)
            edge_score[i] - depth-edge severity per column (0-1)
            left_clear    - clearance score for left path
            right_clear   - clearance score for right path
            center_clear  - clearance score for center path
            closest       - closest obstacle distance in center corridor
            has_obstacle  - True if any obstacle < threshold in path
            emergency     - True if very close obstacle detected
            best_side     - 'left' or 'right' indicating clearer path
        """
        dh, dw = depth_map.shape
        N_COLS = 12
        N_ROWS = 4
        OBSTACLE_DIST = 0.9
        EMERGENCY_DIST = 0.45
        EDGE_GRADIENT_THRESH = 0.3  # meters of depth change = edge

        target_d = target_distance if target_distance else 5.0
        col_w = dw // N_COLS

        # Scan the path-relevant portion (middle 30%-80% of height)
        y_top = int(dh * 0.25)
        y_bot = int(dh * 0.85)
        strip = depth_map[y_top:y_bot, :]
        strip_h = y_bot - y_top
        row_h = strip_h // N_ROWS

        zone_min = []    # minimum depth per column
        edge_score = []  # depth-edge severity per column

        for c in range(N_COLS):
            x1 = c * col_w
            x2 = x1 + col_w if c < N_COLS - 1 else dw
            col_data = strip[:, x1:x2]

            # Minimum depth across rows (catches narrow obstacles at any height)
            row_mins = []
            for r in range(N_ROWS):
                ry1 = r * row_h
                ry2 = ry1 + row_h if r < N_ROWS - 1 else strip_h
                cell = col_data[ry1:ry2, :]
                valid = cell[(cell > 0) & (cell < 65535)].astype(np.float32)
                if len(valid) > 10:
                    row_mins.append(float(np.percentile(valid, 15)) / 1000.0)

            col_min = min(row_mins) if row_mins else 10.0
            zone_min.append(col_min)

            # Depth gradient: detect sharp depth changes within the column
            # (table edges, step-downs, beam edges)
            col_float = col_data.astype(np.float32)
            col_float[col_data == 0] = np.nan
            col_float[col_data >= 65535] = np.nan

            max_grad = 0.0
            # Vertical gradient (top-to-bottom depth jumps = table/step edges)
            for r in range(N_ROWS - 1):
                ry1 = r * row_h
                ry2 = ry1 + row_h
                ry3 = min(ry2 + row_h, strip_h)
                upper = col_float[ry1:ry2, :]
                lower = col_float[ry2:ry3, :]
                u_valid = upper[np.isfinite(upper)]
                l_valid = lower[np.isfinite(lower)]
                if len(u_valid) > 10 and len(l_valid) > 10:
                    diff = abs(float(np.median(l_valid)) - float(np.median(u_valid))) / 1000.0
                    max_grad = max(max_grad, diff)

            # Horizontal gradient (left-right depth jumps = pillars, beam edges)
            half = col_data.shape[1] // 2
            if half > 2:
                left_half = col_float[:, :half]
                right_half = col_float[:, half:]
                lv = left_half[np.isfinite(left_half)]
                rv = right_half[np.isfinite(right_half)]
                if len(lv) > 10 and len(rv) > 10:
                    hdiff = abs(float(np.median(rv)) - float(np.median(lv))) / 1000.0
                    max_grad = max(max_grad, hdiff)

            e_score = min(1.0, max_grad / EDGE_GRADIENT_THRESH) if max_grad > 0.1 else 0.0
            edge_score.append(e_score)

        # Combine: effective obstacle distance considers both raw depth and edges
        # An edge at 0.8m with high gradient score is dangerous even if median depth is fine
        effective = []
        for i in range(N_COLS):
            eff = zone_min[i]
            if edge_score[i] > 0.5 and zone_min[i] < 1.5:
                eff = min(eff, zone_min[i] * (1.0 - edge_score[i] * 0.4))
            effective.append(eff)

        # Score left / center / right paths
        left_cols = effective[:4]        # left third
        center_cols = effective[3:9]     # center half (overlaps slightly for safety)
        right_cols = effective[8:]       # right third

        left_clear = min(left_cols) if left_cols else 10.0
        center_clear = min(center_cols) if center_cols else 10.0
        right_clear = min(right_cols) if right_cols else 10.0

        closest = center_clear
        has_obstacle = closest < OBSTACLE_DIST and closest < target_d - 0.3
        emergency = closest < EMERGENCY_DIST

        best_side = 'left' if left_clear > right_clear else 'right'

        return {
            'zone_min': zone_min,
            'edge_score': edge_score,
            'effective': effective,
            'left_clear': left_clear,
            'center_clear': center_clear,
            'right_clear': right_clear,
            'closest': closest,
            'has_obstacle': has_obstacle,
            'emergency': emergency,
            'best_side': best_side,
        }

    def _follow_loop(self):
        STRAFE_SPEED = 0.25
        MAX_FWD = 0.5
        OBSTACLE_DIST = 0.9

        while self.follow_active:
            fp = self.frame_processor
            if not fp or fp._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                self._move(0, 0, 0)
                time.sleep(0.2)
                continue

            shape = fp._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx = det['center'][0]
            distance = det.get('distance_m')
            target_id = id(det)

            err_x = (cx - w / 2) / (w / 2)
            rot_speed = -err_x * 0.8 if abs(err_x) > 0.10 else 0.0

            fwd_speed = 0.0
            if distance is not None:
                dist_error = distance - self.follow_target_distance
                if dist_error > 0.3:
                    fwd_speed = min(MAX_FWD, dist_error * 0.4)
                elif dist_error < -0.3:
                    fwd_speed = max(-0.2, dist_error * 0.2)
            else:
                bbox_w = det['bbox'][2] - det['bbox'][0]
                bbox_ratio = bbox_w / w
                if bbox_ratio < 0.15:
                    fwd_speed = 0.3
                elif bbox_ratio > 0.4:
                    fwd_speed = -0.1

            if abs(err_x) > 0.35:
                fwd_speed *= 0.2

            strafe_speed = 0.0
            obstacle_ahead = False

            # Depth-based obstacle + edge detection
            depth_map = fp._depth_map
            if depth_map is not None and fwd_speed > 0:
                scan = self._scan_depth_obstacles(depth_map, distance)

                if scan['has_obstacle']:
                    obstacle_ahead = True

                    if scan['emergency']:
                        fwd_speed = min(fwd_speed, 0.05)
                        if scan['best_side'] == 'left':
                            strafe_speed = STRAFE_SPEED
                            rot_speed = max(rot_speed, 0.35)
                        else:
                            strafe_speed = -STRAFE_SPEED
                            rot_speed = min(rot_speed, -0.35)
                        print(f"[Follow] CLOSE obstacle {scan['closest']:.2f}m — strafe {scan['best_side']}")
                    else:
                        fwd_speed = min(fwd_speed, 0.15)
                        if scan['best_side'] == 'left':
                            rot_speed = max(rot_speed, 0.25)
                            strafe_speed = 0.15
                        else:
                            rot_speed = min(rot_speed, -0.25)
                            strafe_speed = -0.15
                        print(f"[Follow] Obstacle {scan['closest']:.2f}m — steer {scan['best_side']}")

                else:
                    # No center obstacle, but check sides for grazing
                    if scan['left_clear'] < 0.45:
                        strafe_speed = -0.12
                    elif scan['right_clear'] < 0.45:
                        strafe_speed = 0.12

                    # Check for edge hazards even when path seems clear
                    center_edges = scan['edge_score'][3:9]
                    max_edge = max(center_edges) if center_edges else 0
                    if max_edge > 0.7 and scan['center_clear'] < 1.2:
                        fwd_speed = min(fwd_speed, 0.15)
                        edge_idx = 3 + center_edges.index(max_edge)
                        if edge_idx < 6:
                            strafe_speed = max(strafe_speed, -0.1)
                        else:
                            strafe_speed = min(strafe_speed, 0.1)
                        print(f"[Follow] Edge detected (score {max_edge:.2f}) — slowing")

            # YOLO detection fallback for non-depth-detected objects
            if fwd_speed > 0 and not obstacle_ahead:
                with fp._lock:
                    all_dets = list(fp.latest_detections)
                for obj in all_dets:
                    if id(obj) == target_id:
                        continue
                    obj_dist = obj.get('distance_m')
                    if obj_dist is None or obj_dist > OBSTACLE_DIST:
                        continue
                    obj_cx = obj['center'][0]
                    obj_err = (obj_cx - w / 2) / (w / 2)
                    if abs(obj_err) < 0.4:
                        fwd_speed = min(fwd_speed, 0.1)
                        if obj_err <= 0:
                            strafe_speed = max(strafe_speed, -0.15)
                        else:
                            strafe_speed = min(strafe_speed, 0.15)
                        break

            self._move(fwd_speed, strafe_speed, rot_speed)
            time.sleep(0.05)

    # ── Go to object ─────────────────────────────────────────────────────

    def go_to_object(self, target, stop_distance=0.5):
        self.stop_follow()
        self.follow_active = True
        self.follow_target = target
        self.start_tracking(target)
        self.follow_thread = threading.Thread(
            target=self._go_to_loop, args=(target, stop_distance), daemon=True
        )
        self.follow_thread.start()
        print(f"[Robot] Going to: {target} (stop {stop_distance:.1f}m away)")

    def _go_to_loop(self, target, stop_distance=0.5):
        TIMEOUT = 30.0
        OBSTACLE_DIST = 0.8
        MIN_FWD, MAX_FWD = 0.15, 0.5
        STEER_SPEED = 0.35
        LOST_PATIENCE = 3.0

        start = time.time()
        lost_since = None

        while self.follow_active and (time.time() - start) < TIMEOUT:
            fp = self.frame_processor
            if not fp or fp._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                if lost_since is None:
                    lost_since = time.time()
                self._move(0, 0, 0)
                if time.time() - lost_since > LOST_PATIENCE:
                    print(f"[Robot] Lost target {target}, giving up")
                    break
                time.sleep(0.2)
                continue
            lost_since = None

            shape = fp._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx = det['center'][0]
            distance = det.get('distance_m')

            if distance is not None and distance <= stop_distance:
                print(f"[Robot] Arrived at {target} ({distance:.1f}m)")
                break

            bbox_w = det['bbox'][2] - det['bbox'][0]
            bbox_ratio = bbox_w / w
            bbox_threshold = max(0.30, 0.55 - stop_distance * 0.15)
            if distance is None and bbox_ratio >= bbox_threshold:
                print(f"[Robot] Arrived at {target} (bbox {bbox_ratio:.2f})")
                break

            err_x = (cx - w / 2) / (w / 2)
            rot_speed = -err_x * 0.5 if abs(err_x) > 0.10 else 0.0

            if distance is not None:
                remaining = distance - stop_distance
                if remaining <= 0:
                    break
                fwd_speed = min(MAX_FWD, max(MIN_FWD, remaining * 0.35))
            else:
                fwd_speed = 0.3

            strafe_speed = 0.0
            depth_map = fp._depth_map
            if depth_map is not None:
                scan = self._scan_depth_obstacles(depth_map, distance)

                if scan['has_obstacle']:
                    if scan['emergency']:
                        fwd_speed = min(fwd_speed, 0.05)
                        strafe_speed = 0.2 if scan['best_side'] == 'left' else -0.2
                        rot_speed = STEER_SPEED if scan['best_side'] == 'left' else -STEER_SPEED
                    else:
                        fwd_speed = min(fwd_speed, 0.2)
                        rot_speed = STEER_SPEED if scan['best_side'] == 'left' else -STEER_SPEED

                    print(f"[GoTo] Obstacle {scan['closest']:.2f}m — steer {scan['best_side']}")

                elif scan['left_clear'] < 0.45:
                    strafe_speed = -0.1
                elif scan['right_clear'] < 0.45:
                    strafe_speed = 0.1

                center_edges = scan['edge_score'][3:9]
                max_edge = max(center_edges) if center_edges else 0
                if max_edge > 0.7 and scan['center_clear'] < 1.2:
                    fwd_speed = min(fwd_speed, 0.15)
                    print(f"[GoTo] Edge hazard (score {max_edge:.2f}) — slowing")

            self._move(fwd_speed, strafe_speed, rot_speed)
            time.sleep(0.05)

        self._move(0, 0, 0)
        self.follow_active = False
        print(f"[Robot] Go-to complete: {target}")

    # ── Stop all ─────────────────────────────────────────────────────────

    def stop_all(self):
        self.stop_follow()
        self.stop_tracking()
        self.stop_movement()
        self.rotate_head(0.0, 0.0)
        self.do_stop_gesture()
        print("[Robot] All stopped")

    def shutdown(self):
        self.stop_all()


# ── Command Dispatcher ───────────────────────────────────────────────────────


class CommandDispatcher:
    def __init__(self, robot: RobotController):
        self.robot = robot
        self._last_cmd_time = 0
        self._cmd_cooldown = 2.0

    def check_transcript(self, text):
        now = time.time()
        if now - self._last_cmd_time < self._cmd_cooldown:
            return None
        text_lower = text.lower().strip()
        for pattern, cmd_name in _CMD_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                self._last_cmd_time = now
                self._execute(cmd_name, match)
                return cmd_name
        return None

    def _execute(self, cmd, match):
        if cmd == "go_to":
            groups = match.groups()
            dist_str, go_target = groups[0], groups[1]
            stop_dist = float(dist_str) if dist_str else 0.5
            target = go_target or "person"
            print(f"[CMD] go_to {target} ({stop_dist}m)")
            add_transcript("Action", f"go_to {target} ({stop_dist}m)")
            self.robot.go_to_object(target, stop_distance=stop_dist)
            return

        target = None
        for g in match.groups():
            if g:
                target = g.strip()
                break

        print(f"[CMD] {cmd}" + (f" ({target})" if target else ""))
        add_transcript("Action", f"{cmd}" + (f" ({target})" if target else ""))

        actions = {
            "follow": lambda: self.robot.start_follow(target),
            "stop_gesture": self.robot.do_stop_gesture,
            "stop": self.robot.stop_all,
            "dance": lambda: self.robot.do_dance(target),
            "wave": self.robot.do_wave,
            "handshake": self.robot.do_handshake,
            "dab": self.robot.do_dab,
            "flex": self.robot.do_flex,
            "track": lambda: self.robot.start_tracking(target),
            "look_left": lambda: self.robot.rotate_head(0.0, 0.5),
            "look_right": lambda: self.robot.rotate_head(0.0, -0.5),
            "look_up": lambda: self.robot.rotate_head(-0.3, 0.0),
            "look_down": lambda: self.robot.rotate_head(0.5, 0.0),
            "look_center": lambda: self.robot.rotate_head(0.0, 0.0),
            "turn_left": self.robot.turn_left,
            "turn_right": self.robot.turn_right,
            "turn_around": self.robot.turn_around,
            "forward": self.robot.forward,
            "backward": self.robot.backward,
            "strafe_left": self.robot.strafe_left,
            "strafe_right": self.robot.strafe_right,
            "approach": self.robot.approach,
            "back_up": self.robot.back_up,
            "nod": self.robot.nod,
            "head_shake": self.robot.head_shake,
            "get_up": self.robot.do_get_up,
            "shoot": self.robot.do_shoot,
            "visual_kick": lambda: self.robot.do_visual_kick(True),
            "stop_visual_kick": self.robot.do_stop_visual_kick,
            "soccer_combo": self.robot.do_soccer_combo,
        }

        if cmd in actions:
            actions[cmd]()
        elif cmd.startswith("dance_"):
            self.robot.do_dance(cmd.replace("dance_", ""))


# ── Transcript ───────────────────────────────────────────────────────────────

transcript = deque(maxlen=200)
transcript_lock = threading.Lock()
_frame_processor_ref = None
_cmd_dispatcher_ref = None
_slam_ref = {}  # 'slam' -> K1SLAM instance (lazy-created)
_session_ref = None
_event_loop_ref = None


def add_transcript(role, text):
    with transcript_lock:
        transcript.append({"role": role, "text": text, "ts": time.time()})
    if role == "You" and _frame_processor_ref:
        _frame_processor_ref.try_learn_name_from_transcript(text)


def get_transcript():
    with transcript_lock:
        return list(transcript)


def send_text_to_gemini(text):
    session, loop = _session_ref, _event_loop_ref
    if not session or not loop:
        print("[Chat] No active Gemini session")
        return False

    async def _send():
        try:
            await session.send_client_content(
                turns=[types.Content(role="user", parts=[types.Part(text=text)])],
                turn_complete=True,
            )
        except Exception as e:
            print(f"[Chat] Send error: {e}")

    asyncio.run_coroutine_threadsafe(_send(), loop)
    return True


# ── Web UI ───────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Gemini Robot Control (Remote)</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body { height:100%; overflow:hidden; }
  body { background:#111; color:#eee; font-family:system-ui,sans-serif; display:flex; height:100vh; min-height:0; }
  #left { flex:1; display:flex; flex-direction:column; background:#000; min-width:0; min-height:0; }
  #camera-wrap { flex:1; display:flex; align-items:center; justify-content:center; min-height:0; overflow:hidden; }
  #camera-wrap img { width:100%; height:100%; object-fit:contain; display:block; }
  #left-slam { flex-shrink:0; padding:8px; border-top:1px solid #333; background:#0a0a0a; }
  #left-slam h3 { font-size:11px; color:#888; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px; }
  #left-slam .slam-status { font-size:11px; color:#666; margin-bottom:4px; }
  #left-slam img { width:100%; max-height:260px; object-fit:contain; background:#1a1a1a; border-radius:4px; display:block; }
  #left-slam .btn-row { margin-top:6px; gap:6px; }
  #right { width:380px; flex-shrink:0; display:flex; flex-direction:column; border-left:1px solid #333; min-height:0; }
  #header { padding:12px 16px; border-bottom:1px solid #333; font-size:14px; color:#888; }
  #header span { color:#4CAF50; font-weight:bold; }
  #controls { padding:8px 16px; border-bottom:1px solid #333; }
  #controls h3 { font-size:12px; color:#888; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
  .btn-row { display:flex; gap:6px; margin-bottom:6px; flex-wrap:wrap; }
  .ctrl-btn { padding:6px 12px; border:none; border-radius:6px; cursor:pointer; font-size:12px; font-weight:bold; transition:transform 0.1s; }
  .ctrl-btn:hover { transform:scale(1.05); }
  .ctrl-btn:active { transform:scale(0.95); }
  .btn-follow { background:#4CAF50; color:#000; }
  .btn-dance { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; }
  .btn-action { background:#2196F3; color:#fff; }
  .btn-head { background:#43e97b; color:#000; }
  .btn-stop { background:#f44336; color:#fff; }
  .btn-mode { background:#FF9800; color:#000; }
  .btn-soccer { background:linear-gradient(135deg,#2e7d32,#1b5e20); color:#fff; }
  #robot-status { padding:6px 16px; border-bottom:1px solid #333; font-size:12px; }
  .connected { color:#4CAF50; } .disconnected { color:#f44336; }
  #detections { padding:8px 16px; border-bottom:1px solid #333; max-height:150px; overflow-y:auto; }
  #detections h3 { font-size:12px; color:#888; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
  .det { padding:4px 8px; margin:3px 0; background:#1a2a1a; border-radius:4px; font-size:13px; display:flex; justify-content:space-between; }
  .det .cls { color:#4CAF50; } .det .name { color:#00BCD4; font-weight:bold; }
  .det .unknown { color:#FF9800; } .det .dist { color:#2196F3; font-weight:bold; }
  .det .conf { color:#666; font-size:11px; }
  .det .name-input { width:80px; padding:2px 4px; background:#222; border:1px solid #555; color:#eee; border-radius:3px; font-size:12px; }
  .det .save-btn { padding:2px 8px; background:#4CAF50; color:#000; border:none; border-radius:3px; font-size:11px; cursor:pointer; margin-left:4px; }
  #known-faces { padding:8px 16px; border-bottom:1px solid #333; max-height:100px; overflow-y:auto; }
  #known-faces h3 { font-size:12px; color:#888; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
  .known { padding:3px 8px; margin:2px 0; background:#1a2a2a; border-radius:4px; font-size:13px; display:flex; justify-content:space-between; }
  .known .kname { color:#00BCD4; }
  .known .del-btn { padding:1px 6px; background:#c62828; color:#fff; border:none; border-radius:3px; font-size:11px; cursor:pointer; }
  #chat { flex:1; overflow-y:auto; padding:12px 16px; display:flex; flex-direction:column; gap:8px; }
  .msg { padding:8px 12px; border-radius:8px; max-width:95%; font-size:14px; line-height:1.4; word-wrap:break-word; }
  .msg.you { background:#1a3a5c; align-self:flex-end; }
  .msg.robot { background:#2d2d2d; align-self:flex-start; }
  .msg.system { background:#2a2a1a; align-self:center; font-size:12px; color:#aaa; }
  .msg.action { background:#1a2a1a; align-self:center; font-size:12px; color:#4CAF50; border:1px solid #4CAF50; }
  .msg .role { font-size:11px; color:#888; margin-bottom:2px; }
  #chat-input { display:flex; padding:8px 12px; border-top:1px solid #333; gap:6px; }
  #msg-input { flex:1; padding:8px 12px; background:#1a1a1a; border:1px solid #444; color:#eee; border-radius:8px; font-size:14px; outline:none; }
  #msg-input:focus { border-color:#4CAF50; }
  #send-btn { padding:8px 16px; background:#4CAF50; color:#000; border:none; border-radius:8px; font-weight:bold; cursor:pointer; font-size:14px; }
  #send-btn:hover { background:#66BB6A; }
  #status { padding:8px 16px; border-top:1px solid #333; font-size:12px; color:#666; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#4CAF50; margin-right:6px; }
  .slam-status { font-size:12px; color:#888; margin-bottom:4px; }
</style>
</head>
<body>
  <div id="left">
    <div id="camera-wrap"><img id="feed" src="/frame" alt="Camera"></div>
    <div id="left-slam">
      <h3>SLAM Map</h3>
      <div id="slam-status" class="slam-status">Checking...</div>
      <div id="slam-map-wrap" style="display:none;">
        <img id="slam-map-img" src="" alt="SLAM map">
        <div class="btn-row" style="display:flex; flex-wrap:wrap;">
          <button class="ctrl-btn btn-action" style="font-size:11px; padding:4px 8px;" onclick="refreshSlamMap()">Refresh</button>
          <a href="/slam/map" target="_blank" class="ctrl-btn btn-action" style="font-size:11px; padding:4px 8px; text-decoration:none; color:#fff;">Open full</a>
          <button class="ctrl-btn btn-stop" style="font-size:11px; padding:4px 8px;" onclick="resetSlam()">Reset map</button>
        </div>
      </div>
    </div>
  </div>
  <div id="right">
    <div id="header"><span>Gemini Robot Control</span> &mdash; Remote Mode</div>
    <div id="robot-status"><span id="conn-dot" class="disconnected">&#9679;</span> <span id="conn-text">Robot: connecting...</span></div>
    <div id="controls">
      <h3>Robot Controls</h3>
      <div class="btn-row">
        <button class="ctrl-btn btn-follow" onclick="cmd('follow')">Follow Me</button>
        <button class="ctrl-btn btn-follow" onclick="cmd('track')">Track Person</button>
        <button class="ctrl-btn btn-follow" onclick="goToPrompt()">Go To...</button>
        <button class="ctrl-btn btn-stop" onclick="cmd('stop')">STOP ALL</button>
        <button class="ctrl-btn btn-action" onclick="cmd('get_up')">Get Up</button>
        <button class="ctrl-btn btn-action" onclick="cmd('stop_gesture')">Stop Gesture</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-dance" onclick="cmd('dance')">Dance</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_newyear')">New Year</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_nezha')">Nezha</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_future')">Future</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_moonwalk')">Moonwalk</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_michael jackson')">MJ Dance</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_kick')">Kick</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_roundhouse')">Roundhouse</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_salsa')">Salsa</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_ultraman')">Ultraman</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_respect')">Respect</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_celebrate')">Celebrate</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_luckycat')">Lucky Cat</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_macarena')">Macarena</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_twist')">Twist</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_disco')">Disco</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_chicken')">Chicken</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_bow')">Bow</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-action" onclick="cmd('wave')">Wave</button>
        <button class="ctrl-btn btn-action" onclick="cmd('handshake')">Handshake</button>
        <button class="ctrl-btn btn-action" onclick="cmd('dab')">Dab</button>
        <button class="ctrl-btn btn-action" onclick="cmd('flex')">Flex</button>
        <button class="ctrl-btn btn-action" onclick="cmd('nod')">Nod</button>
        <button class="ctrl-btn btn-action" onclick="cmd('head_shake')">Shake Head</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-soccer" onclick="cmd('shoot')">Shoot</button>
        <button class="ctrl-btn btn-soccer" onclick="cmd('visual_kick')">Side Foot Kick</button>
        <button class="ctrl-btn btn-soccer" onclick="cmd('stop_visual_kick')">Stop Kick</button>
        <button class="ctrl-btn btn-soccer" onclick="cmd('soccer_combo')">Shoot + Celebrate</button>
        <button class="ctrl-btn btn-soccer" onclick="cmd('dance_kick')">Boxing Kick</button>
        <button class="ctrl-btn btn-soccer" onclick="cmd('dance_roundhouse')">Roundhouse</button>
        <button class="ctrl-btn btn-soccer" onclick="cmd('dance_celebrate')">Celebrate</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-head" onclick="cmd('look_up')">Look Up</button>
        <button class="ctrl-btn btn-head" onclick="cmd('look_down')">Look Down</button>
        <button class="ctrl-btn btn-head" onclick="cmd('look_left')">Look Left</button>
        <button class="ctrl-btn btn-head" onclick="cmd('look_right')">Look Right</button>
        <button class="ctrl-btn btn-head" onclick="cmd('look_center')">Center</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-mode" onclick="cmd('forward')">Forward<br><small>W</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('backward')">Backward<br><small>S</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('strafe_left')">Strafe L<br><small>A</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('strafe_right')">Strafe R<br><small>D</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('turn_left')">Turn L<br><small>Q</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('turn_right')">Turn R<br><small>E</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('turn_around')">Turn 180</button>
      </div>
    </div>
    <div id="detections"><h3>Detections</h3><div id="det-list">Waiting for robot...</div></div>
    <div id="known-faces"><h3>Known Faces</h3><div id="kf-list">None yet</div></div>
    <div id="chat"></div>
    <div id="chat-input">
      <input type="text" id="msg-input" placeholder="Type a message (or say Jimmy, then your command)..." autocomplete="off">
      <button id="send-btn" onclick="sendChat()">Send</button>
    </div>
    <div id="status"><span class="dot"></span>Listening... | <a href="/3d" target="_blank" style="color:#4CAF50;">3D View</a> | <a href="/slam/map" target="_blank" style="color:#4CAF50;">SLAM Map</a></div>
  </div>
<script>
  const img = document.getElementById('feed');
  function refreshFrame() {
    const next = new Image();
    next.onload = () => { img.src = next.src; setTimeout(refreshFrame, 100); };
    next.onerror = () => { setTimeout(refreshFrame, 500); };
    next.src = '/frame?t=' + Date.now();
  }
  refreshFrame();

  async function cmd(action, target) {
    try {
      const payload = {action};
      if (target) payload.target = target;
      const r = await fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      console.log('cmd:', await r.json());
    } catch(e) { console.error(e); }
  }

  function goToPrompt() {
    const target = prompt('Go to what? (e.g. person, chair, bottle, backpack, a name)');
    if (!target || !target.trim()) return;
    const distStr = prompt('Stop distance in meters? (default 0.5)', '0.5');
    const dist = parseFloat(distStr) || 0.5;
    fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'go_to', target:target.trim(), distance:dist})
    }).then(r=>r.json()).then(d=>console.log('go_to:',d)).catch(console.error);
  }

  async function sendChat() {
    const input = document.getElementById('msg-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    try { await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})}); } catch(e) { console.error(e); }
  }

  document.getElementById('msg-input').addEventListener('keydown', function(e) { if (e.key==='Enter') { e.preventDefault(); sendChat(); } });

  const wasdMap = {w:'forward', s:'backward', a:'strafe_left', d:'strafe_right', q:'turn_left', e:'turn_right'};
  document.addEventListener('keydown', function(e) {
    if (document.activeElement.tagName==='INPUT'||document.activeElement.tagName==='TEXTAREA') return;
    const action = wasdMap[e.key.toLowerCase()];
    if (action) { e.preventDefault(); cmd(action); }
  });

  const chat = document.getElementById('chat');
  let lastLen = 0;
  async function pollTranscript() {
    try {
      const r = await fetch('/transcript');
      const msgs = await r.json();
      if (msgs.length !== lastLen) {
        lastLen = msgs.length;
        chat.innerHTML = msgs.map(m => {
          const cls = m.role==='You'?'you':m.role==='Action'?'action':m.role==='System'?'system':'robot';
          return '<div class="msg '+cls+'"><div class="role">'+m.role+'</div>'+m.text+'</div>';
        }).join('');
        chat.scrollTop = chat.scrollHeight;
      }
    } catch(e) {}
    setTimeout(pollTranscript, 500);
  }
  pollTranscript();

  async function refreshSlamMap() {
    const img = document.getElementById('slam-map-img');
    const wrap = document.getElementById('slam-map-wrap');
    if (wrap.style.display !== 'none') img.src = '/slam/map?t=' + Date.now();
  }
  async function resetSlam() {
    try {
      await fetch('/slam/reset', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
      refreshSlamMap();
    } catch (e) { console.error(e); }
  }
  const slamStatusEl = document.getElementById('slam-status');
  const slamMapWrap = document.getElementById('slam-map-wrap');
  const slamMapImg = document.getElementById('slam-map-img');
  async function pollSlamStatus() {
    try {
      const r = await fetch('/slam/status');
      const d = await r.json();
      if (d.active) {
        slamStatusEl.textContent = 'Active (frames: ' + (d.frame_count || 0) + ')';
        slamMapWrap.style.display = 'block';
        slamMapImg.src = '/slam/map?t=' + Date.now();
      } else {
        slamStatusEl.textContent = 'Inactive (connect robot with depth for SLAM)';
        slamMapWrap.style.display = 'none';
      }
    } catch (e) {
      slamStatusEl.textContent = 'SLAM unavailable';
      slamMapWrap.style.display = 'none';
    }
    setTimeout(pollSlamStatus, 3000);
  }
  pollSlamStatus();

  const detList = document.getElementById('det-list');
  async function pollDetections() {
    try {
      const r = await fetch('/detections');
      const dets = await r.json();
      if (dets.length === 0) {
        detList.innerHTML = '<div style="color:#666;font-size:12px;">No objects detected</div>';
      } else {
        detList.innerHTML = dets.map(d => {
          const dist = d.distance_m !== null ? d.distance_m.toFixed(1)+'m' : '?';
          let nameHtml = '';
          if (d.class==='person' && d.name) {
            if (d.unknown_id !== null && d.unknown_id !== undefined) {
              nameHtml = '<span class="unknown">'+d.name+'</span> '
                +'<input class="name-input" placeholder="Name..." id="ni_'+d.unknown_id+'">'
                +'<button class="save-btn" onclick="saveFace('+d.unknown_id+')">Save</button>';
            } else {
              nameHtml = '<span class="name">'+d.name+'</span> ';
            }
          }
          return '<div class="det"><span>'+nameHtml+'<span class="cls">'+d.class+'</span> <span class="conf">'+(d.confidence*100).toFixed(0)+'%</span></span><span class="dist">'+dist+'</span></div>';
        }).join('');
      }
    } catch(e) {}
    setTimeout(pollDetections, 300);
  }
  pollDetections();

  async function saveFace(unknownId) {
    const input = document.getElementById('ni_'+unknownId);
    if (!input||!input.value.trim()) return;
    try { await fetch('/save_face', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({unknown_id:unknownId, name:input.value.trim()})}); } catch(e) { console.error(e); }
  }

  const kfList = document.getElementById('kf-list');
  async function pollKnownFaces() {
    try {
      const r = await fetch('/known_faces');
      const faces = await r.json();
      if (faces.length===0) {
        kfList.innerHTML = '<div style="color:#666;font-size:12px;">No saved faces</div>';
      } else {
        kfList.innerHTML = faces.map(f =>
          '<div class="known"><span class="kname">'+f.name+'</span><button class="del-btn" onclick="deleteFace(\\''+f.name+'\\')">x</button></div>'
        ).join('');
      }
    } catch(e) {}
    setTimeout(pollKnownFaces, 2000);
  }
  pollKnownFaces();

  async function deleteFace(name) {
    if (!confirm('Delete face: '+name+'?')) return;
    try { await fetch('/delete_face', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})}); } catch(e) { console.error(e); }
  }

  async function pollRobotStatus() {
    try {
      const r = await fetch('/robot_status');
      const s = await r.json();
      document.getElementById('conn-dot').className = s.connected ? 'connected' : 'disconnected';
      document.getElementById('conn-text').textContent = s.connected ? 'Robot: connected' : 'Robot: disconnected';
    } catch(e) {}
    setTimeout(pollRobotStatus, 2000);
  }
  pollRobotStatus();
</script>
</body>
</html>"""


HTML_3D_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<title>3D Point Cloud — Robot View</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#000; overflow:hidden; font-family:system-ui,sans-serif; }
  canvas { display:block; }
  #hud {
    position:absolute; top:12px; left:12px; color:#eee; font-size:13px;
    background:rgba(0,0,0,0.7); padding:10px 14px; border-radius:8px;
    pointer-events:none; line-height:1.6;
  }
  #hud b { color:#4CAF50; }
  #controls {
    position:absolute; top:12px; right:12px; color:#eee; font-size:12px;
    background:rgba(0,0,0,0.7); padding:10px 14px; border-radius:8px;
  }
  #controls label { display:block; margin:4px 0; }
  #controls input[type=range] { width:120px; vertical-align:middle; }
  #controls button {
    margin-top:8px; padding:6px 14px; background:#4CAF50; color:#000;
    border:none; border-radius:6px; font-weight:bold; cursor:pointer; font-size:12px;
  }
  #controls button:hover { background:#66BB6A; }
  #back-link {
    position:absolute; bottom:12px; left:12px;
    color:#4CAF50; font-size:13px; text-decoration:none;
    background:rgba(0,0,0,0.7); padding:6px 12px; border-radius:6px;
  }
</style>
</head>
<body>
<div id="hud">
  <b>3D Point Cloud</b> — Robot Depth View<br>
  Points: <span id="npts">0</span> | FPS: <span id="fps">0</span><br>
  Drag to orbit · Scroll to zoom · Right-drag to pan
</div>
<div id="controls">
  <label>Point size <input type="range" id="psize" min="1" max="8" step="0.5" value="2"></label>
  <label>Max depth (m) <input type="range" id="maxd" min="1" max="10" step="0.5" value="6"></label>
  <label>Refresh (ms) <input type="range" id="interval" min="100" max="2000" step="100" value="500"></label>
  <label><input type="checkbox" id="autoRotate"> Auto-rotate</label>
  <button onclick="resetView()">Reset View</button>
</div>
<a id="back-link" href="/">← Back to Control Panel</a>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 50);
camera.position.set(0, 1.5, 2);
camera.lookAt(0, 0, -2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, -2);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.update();

// Grid floor
const grid = new THREE.GridHelper(10, 20, 0x333333, 0x222222);
grid.rotation.x = 0;
scene.add(grid);

// Axes
scene.add(new THREE.AxesHelper(1));

// Origin marker (robot position)
const markerGeo = new THREE.SphereGeometry(0.05, 16, 16);
const markerMat = new THREE.MeshBasicMaterial({ color: 0x4CAF50 });
scene.add(new THREE.Mesh(markerGeo, markerMat));

// Point cloud
const MAX_POINTS = 200000;
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(MAX_POINTS * 3);
const colors = new Float32Array(MAX_POINTS * 3);
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geometry.setDrawRange(0, 0);

const material = new THREE.PointsMaterial({
  size: 2, vertexColors: true, sizeAttenuation: true,
  transparent: true, opacity: 0.85,
});
const points = new THREE.Points(geometry, material);
scene.add(points);

// UI controls
const psizeEl = document.getElementById('psize');
const maxdEl = document.getElementById('maxd');
const intervalEl = document.getElementById('interval');
const autoRotEl = document.getElementById('autoRotate');
const nptsEl = document.getElementById('npts');
const fpsEl = document.getElementById('fps');

psizeEl.addEventListener('input', () => { material.size = parseFloat(psizeEl.value); });
autoRotEl.addEventListener('change', () => { controls.autoRotate = autoRotEl.checked; });

window.resetView = function() {
  camera.position.set(0, 1.5, 2);
  controls.target.set(0, 0, -2);
  controls.update();
};

// Fetch point cloud
let refreshInterval = 500;
intervalEl.addEventListener('input', () => { refreshInterval = parseInt(intervalEl.value); });

async function fetchCloud() {
  try {
    const maxDepth = parseFloat(maxdEl.value);
    const resp = await fetch('/pointcloud?max_depth=' + maxDepth);
    if (!resp.ok) { setTimeout(fetchCloud, refreshInterval); return; }
    const buf = await resp.arrayBuffer();
    const floats = new Float32Array(buf);
    const nPoints = Math.min(floats.length / 6, MAX_POINTS);

    const pos = geometry.attributes.position.array;
    const col = geometry.attributes.color.array;
    for (let i = 0; i < nPoints; i++) {
      const si = i * 6;
      const di = i * 3;
      pos[di]     = floats[si];     // x
      pos[di + 1] = floats[si + 1]; // y
      pos[di + 2] = floats[si + 2]; // z
      col[di]     = floats[si + 3]; // r
      col[di + 1] = floats[si + 4]; // g
      col[di + 2] = floats[si + 5]; // b
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
    geometry.setDrawRange(0, nPoints);
    nptsEl.textContent = nPoints.toLocaleString();
  } catch(e) {}
  setTimeout(fetchCloud, refreshInterval);
}
fetchCloud();

// Render loop
let frames = 0, lastFpsTime = performance.now();
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  frames++;
  const now = performance.now();
  if (now - lastFpsTime > 1000) {
    fpsEl.textContent = Math.round(frames * 1000 / (now - lastFpsTime));
    frames = 0;
    lastFpsTime = now;
  }
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>"""


class WebHandler(BaseHTTPRequestHandler):
    slam_ref = None  # set by start_web_server
    frame_processor = None
    robot_controller = None

    def log_message(self, format, *args):
        pass

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length) if length else b''

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path.startswith('/frame'):
            fp = self.frame_processor
            if fp and fp.latest_frame is not None:
                with fp._lock:
                    frame = fp.latest_frame.copy()
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(buf.tobytes())
            else:
                self.send_error(503, 'No frame')
        elif self.path == '/transcript':
            self._json_response(get_transcript())
        elif self.path == '/detections':
            fp = self.frame_processor
            dets = []
            if fp:
                with fp._lock:
                    dets = list(fp.latest_detections)
            self._json_response(dets)
        elif self.path == '/known_faces':
            fp = self.frame_processor
            faces = fp.face_cache.list_known() if (fp and fp.face_cache) else []
            self._json_response(faces)
        elif self.path == '/robot_status':
            robot = self.robot_controller
            connected = robot is not None and robot._ws is not None
            self._json_response({'connected': connected})
        elif self.path.startswith('/pointcloud'):
            fp = self.frame_processor
            if not fp:
                self.send_error(503, 'Not ready')
                return
            pcl = fp.get_pointcloud_binary(step=4, max_depth=6.0)
            if pcl is None:
                self.send_error(503, 'No depth data')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(pcl)
        elif self.path == '/3d':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_3D_PAGE.encode())
        elif self.path.startswith('/slam/pose'):
            slam = (self.slam_ref or {}).get('slam')
            if not slam:
                self._json_response({
                    'error': 'SLAM not running',
                    'hint': 'Install Open3D (pip install open3d) and ensure the robot is connected with depth streaming. Start server without --no-slam.',
                    'pose': None,
                }, 503)
                return
            pose = slam.get_pose().tolist()
            x, y, yaw = slam.get_odometry_xy_theta()
            self._json_response({
                'pose': pose,
                'x': x, 'y': y, 'yaw_rad': yaw,
                'frame_count': getattr(slam, '_frame_count', 0),
            })
        elif self.path.startswith('/slam/status'):
            slam = (self.slam_ref or {}).get('slam')
            self._json_response({
                'active': slam is not None,
                'frame_count': getattr(slam, '_frame_count', 0) if slam else 0,
            })
        elif self.path.startswith('/slam/map'):
            slam = (self.slam_ref or {}).get('slam')
            if not slam:
                self.send_response(503)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                msg = (
                    '<!DOCTYPE html><html><body style="font-family:sans-serif; padding:2rem; background:#111; color:#eee;">'
                    '<h1>SLAM not running</h1><p>To enable SLAM:</p><ul>'
                    '<li>Install Open3D: <code>pip install open3d</code></li>'
                    '<li>Start the server without <code>--no-slam</code></li>'
                    '<li>Connect the robot so it streams depth (e.g. from /StereoNetNode/stereonet_depth)</li>'
                    '</ul><p>Then the map will appear here and in the main UI.</p></body></html>'
                )
                self.wfile.write(msg.encode('utf-8'))
                return
            # Center map on robot so robot is in the middle, surroundings around it
            result = slam.get_occupancy_grid(
                resolution=0.05, width_m=10.0, height_m=10.0, center_on_robot=True
            )
            if result is None:
                self.send_error(503, 'No map yet')
                return
            grid, res, (ox, oy) = result
            # Return as PNG image (0=free, 100=occupied); draw robot at center
            grid_vis = np.uint8(255 - grid)  # invert so occupied is dark
            grid_vis = cv2.cvtColor(grid_vis, cv2.COLOR_GRAY2BGR)
            h, w = grid_vis.shape[:2]
            cx, cy = w // 2, h // 2
            # Robot as green triangle; rotate by yaw so it points in heading direction
            _, _, yaw = slam.get_odometry_xy_theta()
            cos_a, sin_a = math.cos(-yaw), math.sin(-yaw)
            tip = np.array([0, -10])
            left = np.array([-6, 6])
            right = np.array([6, 6])
            def rot(p):
                return (cx + p[0] * cos_a - p[1] * sin_a, cy + p[0] * sin_a + p[1] * cos_a)
            pts = np.array([[rot(tip), rot(left), rot(right)]], dtype=np.int32)
            cv2.fillPoly(grid_vis, pts, (0, 255, 0))
            cv2.circle(grid_vis, (cx, cy), 3, (0, 255, 0), -1)
            # Semantic overlay: project YOLO detections onto map using depth + pose
            fp = self.frame_processor
            if fp is not None:
                with fp._lock:
                    dets = list(fp.latest_detections)
                    raw = fp._raw_frame
                if raw is not None and dets:
                    pose = slam.get_pose()
                    frame_shape = raw.shape
                    projected = _project_detections_to_map(
                        dets, pose, (ox, oy), res, h, w, frame_shape
                    )
                    for col, row, label, color in projected:
                        cv2.circle(grid_vis, (col, row), 5, color, 2)
                        cv2.putText(
                            grid_vis, label, (col + 6, row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
                        )
            _, buf = cv2.imencode('.png', grid_vis)
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(buf.tobytes())
        elif self.path.startswith('/slam/mesh'):
            slam = (self.slam_ref or {}).get('slam')
            if not slam:
                self.send_response(503)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(
                    b'SLAM not running. Install open3d (pip install open3d), start without --no-slam, and connect the robot with depth.'
                )
                return
            ply = slam.get_mesh_ply()
            if ply is None:
                self.send_error(503, 'No mesh yet')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Disposition', 'attachment; filename=k1_slam_map.ply')
            self.end_headers()
            self.wfile.write(ply)
        else:
            self.send_error(404)

    def do_POST(self):
        fp = self.frame_processor
        robot = self.robot_controller
        try:
            body = json.loads(self._read_body())
        except Exception:
            self.send_error(400, 'Invalid JSON')
            return

        if self.path == '/chat':
            text = body.get('text', '').strip()
            if not text:
                self._json_response({'error': 'need text'}, 400)
                return
            add_transcript("You", text)
            ok = send_text_to_gemini(text)
            self._json_response({'ok': ok, 'text': text})
        elif self.path == '/cmd':
            action = body.get('action', '')
            result = self._handle_cmd(action, robot, body)
            self._json_response(result)
        elif self.path == '/save_face':
            if not fp:
                self._json_response({'error': 'not ready'}, 503)
                return
            uid = body.get('unknown_id')
            name = body.get('name', '').strip()
            if not name or uid is None:
                self._json_response({'error': 'need unknown_id and name'}, 400)
                return
            ok = fp.save_unknown_face(int(uid), name)
            self._json_response({'ok': ok, 'name': name})
        elif self.path == '/delete_face':
            if not fp or not fp.face_cache:
                self._json_response({'error': 'not ready'}, 503)
                return
            name = body.get('name', '').strip()
            if not name:
                self._json_response({'error': 'need name'}, 400)
                return
            fp.face_cache.delete_face(name)
            self._json_response({'ok': True})
        elif self.path == '/slam/reset':
            slam = (self.slam_ref or {}).get('slam')
            if slam:
                slam.reset()
            self._json_response({'ok': True})
        else:
            self.send_error(404)

    def _handle_cmd(self, action, robot, body=None):
        if not robot:
            return {'status': 'error', 'message': 'Robot not connected'}
        add_transcript("Action", action)
        target = body.get('target') if isinstance(body, dict) else None
        distance = body.get('distance') if isinstance(body, dict) else None

        if action == 'follow':
            robot.start_follow(target)
        elif action == 'track':
            robot.start_tracking(target)
        elif action == 'stop':
            robot.stop_all()
        elif action.startswith('go_to'):
            obj = target or action.replace('go_to_', '').replace('go_to', 'person')
            stop_dist = float(distance) if distance else 0.5
            robot.go_to_object(obj or 'person', stop_distance=stop_dist)
            return {'status': 'ok', 'action': 'go_to', 'target': obj, 'distance': stop_dist}
        elif action == 'dance':
            robot.do_dance()
        elif action.startswith('dance_'):
            robot.do_dance(action.replace('dance_', ''))
        elif action == 'wave':
            robot.do_wave()
        elif action == 'handshake':
            robot.do_handshake()
        elif action == 'dab':
            robot.do_dab()
        elif action == 'flex':
            robot.do_flex()
        elif action == 'get_up':
            robot.do_get_up()
        elif action == 'shoot':
            robot.do_shoot()
        elif action == 'visual_kick':
            robot.do_visual_kick(start=body.get('start', True) if isinstance(body, dict) else True)
        elif action == 'stop_visual_kick':
            robot.do_stop_visual_kick()
        elif action == 'soccer_combo':
            robot.do_soccer_combo()
        elif action == 'nod':
            robot.nod()
        elif action == 'head_shake':
            robot.head_shake()
        elif action == 'stop_gesture':
            robot.do_stop_gesture()
        elif action == 'look_up':
            robot.rotate_head(-0.3, 0.0)
        elif action == 'look_down':
            robot.rotate_head(0.5, 0.0)
        elif action == 'look_left':
            robot.rotate_head(0.0, 0.5)
        elif action == 'look_right':
            robot.rotate_head(0.0, -0.5)
        elif action == 'look_center':
            robot.rotate_head(0.0, 0.0)
        elif action == 'forward':
            robot.forward()
        elif action == 'backward':
            robot.backward()
        elif action == 'strafe_left':
            robot.strafe_left()
        elif action == 'strafe_right':
            robot.strafe_right()
        elif action == 'turn_left':
            robot.turn_left()
        elif action == 'turn_right':
            robot.turn_right()
        elif action == 'turn_around':
            robot.turn_around()
        else:
            return {'status': 'error', 'message': f'Unknown: {action}'}
        return {'status': 'ok', 'action': action}


def start_web_server(frame_processor, robot_controller, host, port, slam_ref=None):
    WebHandler.frame_processor = frame_processor
    WebHandler.robot_controller = robot_controller
    WebHandler.slam_ref = slam_ref or {}
    httpd = HTTPServer((host, port), WebHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# ── Gemini Session ───────────────────────────────────────────────────────────

def _is_addressed_to_robot(text, robot_name):
    """True if the user addressed the robot by name (e.g. 'Jimmy, follow me')."""
    if not text or not robot_name:
        return False
    text = text.strip()
    name = re.escape(robot_name)
    pat = re.compile(
        r"^(?:hey\s+|hi\s+|oh\s+|um\s+)?"
        + name
        + r"[,\s:]*(.*)$",
        re.IGNORECASE | re.DOTALL,
    )
    return pat.match(text) is not None


def _get_system_instruction(robot_name):
    return f"""You are a Booster K1 humanoid robot named {robot_name}. You have stereo vision cameras, face recognition, \
and full body control. Video frames are streamed to you with real-time object detection overlays — \
each detected object has a bounding box, class label, confidence score, and distance in meters. \
People you recognize are labeled with their name; unknown people are labeled 'Unknown #N'.

WAKE WORD / NOISE FILTERING (CRITICAL):
- Your name is {robot_name}. You MUST only respond when the user addresses you by name first.
- Valid examples: "{robot_name}, follow me" / "Hey {robot_name}, walk forward" / "{robot_name} dance"
- If the user speaks without saying "{robot_name}" first (e.g. background conversation, someone talking to someone else), \
do NOT respond. Stay silent. Ignore it completely.
- Focus on the primary speaker. Ignore background conversations, TV, music, and ambient noise.
- Only process commands that are clearly directed at you by name.

You can PHYSICALLY ACT by saying certain trigger phrases in your responses. When you decide to act, \
naturally include one of these phrases — your body will respond automatically:

FOLLOWING / TRACKING:
- "I'll follow you" or "Following you now" — walks toward and follows the person
- "I'll track that" or "Tracking the [object]" — moves head AND body to keep an object centered
- "Stopping now" or "I'll stop" — stops all movement and tracking

GO TO OBJECTS:
- "Going to the [object]" or "Walking to the [object]" — walks toward a detected object
- "Going to 0.5 meters from the [object]" — walks to a specific distance from the object
- "Moving over to the [person name]" — walks toward a specific named person

LOOKING / HEAD CONTROL:
- "Looking left/right/up/down" — head movement
- "Looking forward" or "Looking straight" — centers head

MOVEMENT:
- "Walking forward/backward" — walks briefly
- "Strafing left/right" — sidesteps
- "Turning left/right" — rotates body
- "Turning around" — turns 180 degrees
- "Coming closer" or "Backing up"

DANCES & GESTURES:
- "Let me dance" — does a robot dance
- "Moonwalk!" / "Michael Jackson dance!" / "Kick!" / "Roundhouse!" / "Salsa!" etc.
- "I'll wave" / "Let me shake hands" / "Dabbing!" / "Flexing!"
- "Nodding" / "Shaking my head"

SOCCER MOVES:
- "Shoot!" / "Kick the ball!" / "Score!" / "Goal!" — powerful soccer kick
- "Pass!" / "Side foot kick!" / "Pass the ball!" — side-foot pass
- "Stop kick" — stops the side-foot kick if it's running
- "Shoot and celebrate!" / "Soccer combo!" — shoot then celebration dance
- "Celebrate!" / "We scored!" — celebration dance
- "Boxing kick!" / "Roundhouse!" — whole-body kick moves

IMPORTANT RULES:
- When someone says "follow me", respond with "I'll follow you!"
- When someone says "go to the chair", respond with "Going to the chair!"
- Keep responses short and conversational.
- Only trigger actions when explicitly asked or socially appropriate.
- Remember: only respond when addressed as {robot_name} first.
"""


async def gemini_send_video(session, frame_processor, interval):
    try:
        while True:
            b64 = frame_processor.get_frame_b64jpeg()
            if b64:
                await session.send_realtime_input(
                    video=types.Blob(data=b64, mime_type="image/jpeg")
                )
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def gemini_send_audio(session, audio_queue):
    """Forward audio from robot mic (via audio_queue) to Gemini."""
    try:
        while True:
            data = await audio_queue.get()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )
    except asyncio.CancelledError:
        pass


async def gemini_send_local_audio(session, pya, mic_device=None, mic_gain=1.0):
    """Capture audio from local mic and send to Gemini."""
    kwargs = dict(
        format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=SEND_SAMPLE_RATE,
        input=True, frames_per_buffer=AUDIO_CHUNK,
    )
    if mic_device is not None:
        kwargs['input_device_index'] = mic_device
    try:
        stream = pya.open(**kwargs)
    except OSError:
        stream = _NullStream()  # no input device; will send silence
    apply_gain = mic_gain > 1.01
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await loop.run_in_executor(
                None, lambda: stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            )
            if apply_gain:
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                samples *= mic_gain
                np.clip(samples, -32768, 32767, out=samples)
                data = samples.astype(np.int16).tobytes()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


async def _slam_update_loop(frame_processor, slam_ref, interval=0.35):
    """Background task: run SLAM updates from latest depth + RGB (throttled)."""
    slam = slam_ref.get('slam')
    if not slam:
        return
    loop = asyncio.get_event_loop()
    while True:
        try:
            await asyncio.sleep(interval)
            fp = frame_processor
            if not fp or fp._depth_map is None or fp._raw_frame is None:
                continue
            depth = fp._depth_map.copy()
            rgb = fp._raw_frame.copy()
            slam = slam_ref.get('slam')
            if not slam:
                continue
            dh, dw = depth.shape
            cx, cy = dw / 2.0, dh / 2.0
            def _run():
                slam.update(depth, rgb=rgb, cx=cx, cy=cy, min_interval=interval)
            await loop.run_in_executor(None, _run)
        except asyncio.CancelledError:
            break
        except Exception:
            pass


async def gemini_receive(session, pya, cmd_dispatcher, robot_ws_ref, robot_name):
    """Receive Gemini responses: play audio locally + send to robot, parse commands."""
    try:
        stream = pya.open(
            format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=RECV_SAMPLE_RATE,
            output=True, frames_per_buffer=AUDIO_CHUNK,
        )
    except OSError:
        stream = _NullStream()  # no output device (e.g. headless VM); audio still sent to robot
    loop = asyncio.get_event_loop()
    try:
        while True:
            async for msg in session.receive():
                if msg.data:
                    await loop.run_in_executor(None, stream.write, msg.data)
                    ws = robot_ws_ref.get('ws')
                    if ws:
                        try:
                            await ws.send(bytes([MSG_AUDIO_OUT]) + msg.data)
                        except Exception:
                            pass

                sc = msg.server_content
                if sc:
                    input_txt = (sc.input_transcription.text if sc.input_transcription else None) or ""
                    output_txt = (sc.output_transcription.text if sc.output_transcription else None) or ""
                    # Only process when user addressed robot by name (wake-word filter)
                    if input_txt and not _is_addressed_to_robot(input_txt, robot_name):
                        print(f"  [ignored - not addressed to {robot_name}]: {(input_txt[:50] + '...') if len(input_txt) > 50 else input_txt}")
                        continue
                    if input_txt:
                        print(f"  You: {input_txt}")
                        add_transcript("You", input_txt)
                    if output_txt:
                        print(f"Robot: {output_txt}")
                        add_transcript("Robot", output_txt)
                        cmd_dispatcher.check_transcript(output_txt)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


# ── WebSocket Server (robot connection) ──────────────────────────────────────


async def handle_robot_ws(websocket, frame_processor: FrameProcessor,
                          robot: RobotController, audio_queue: asyncio.Queue,
                          robot_ws_ref: dict):
    """Handle a single robot WebSocket connection."""
    print(f"[WS] Robot connected from {websocket.remote_address}")
    robot.set_connection(websocket, asyncio.get_event_loop())
    robot_ws_ref['ws'] = websocket

    try:
        async for message in websocket:
            if isinstance(message, bytes) and len(message) > 1:
                msg_type = message[0]
                payload = message[1:]
                if msg_type == MSG_VIDEO:
                    frame_processor.on_video_frame(payload)
                elif msg_type == MSG_DEPTH:
                    frame_processor.on_depth_frame(payload)
                elif msg_type == MSG_AUDIO_IN:
                    await audio_queue.put(payload)
            # text messages from robot (status, etc.) — currently unused
    except Exception as e:
        print(f"[WS] Robot disconnected: {e}")
    finally:
        robot.set_connection(None, None)
        robot_ws_ref['ws'] = None
        print("[WS] Robot connection closed")


# ── Main ─────────────────────────────────────────────────────────────────────


async def run_server(args):
    global _frame_processor_ref, _cmd_dispatcher_ref, _session_ref, _event_loop_ref

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Error: provide --api-key or set GEMINI_API_KEY env variable")
        sys.exit(1)

    os.environ.pop('GOOGLE_API_KEY', None)
    os.environ.pop('GEMINI_API_KEY', None)

    face_engine = None
    if not args.no_faces:
        try:
            face_engine = FaceEngine()
        except RuntimeError as e:
            print(f"Note: {e}\nRunning without face recognition.")

    face_cache = None
    if face_engine is not None:
        face_cache = FaceCache(tolerance=args.face_tolerance, face_engine=face_engine)

    enable_faces = face_engine is not None
    frame_processor = FrameProcessor(
        model_path=args.model, confidence=args.confidence,
        face_cache=face_cache, enable_faces=enable_faces,
        face_engine=face_engine,
    )
    _frame_processor_ref = frame_processor

    robot = RobotController()
    robot.set_frame_processor(frame_processor)
    robot.follow_target_distance = args.follow_distance

    cmd_dispatcher = CommandDispatcher(robot)
    _cmd_dispatcher_ref = cmd_dispatcher

    # Eager-init SLAM so /slam/status and /slam/pose work immediately; map fills when robot sends depth
    if not getattr(args, 'no_slam', False):
        try:
            from slam.k1_slam import K1SLAM
            _slam_ref['slam'] = K1SLAM(enable_tsdf=True)
            print("SLAM: enabled (map will appear when robot streams depth)")
        except ImportError as e:
            print("SLAM: disabled (install with: pip install open3d)")
        except Exception as e:
            print(f"SLAM: disabled ({e})")

    httpd = start_web_server(frame_processor, robot, '0.0.0.0', args.port, slam_ref=_slam_ref)
    print(f"Web UI: http://0.0.0.0:{args.port}")

    audio_queue = asyncio.Queue(maxsize=100)
    robot_ws_ref = {'ws': None}

    # WebSocket server for robot connection
    import websockets
    print(f"websockets version: {websockets.__version__}")

    async def _ws_handler(websocket, path=None):
        await handle_robot_ws(websocket, frame_processor, robot, audio_queue, robot_ws_ref)

    ws_server = await websockets.serve(
        _ws_handler,
        '0.0.0.0', args.ws_port,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    )
    print(f"Robot WebSocket: ws://0.0.0.0:{args.ws_port}")
    print(f"  Tell robot_client.py to connect to: ws://<THIS_IP>:{args.ws_port}")

    # Gemini session
    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice)
            ),
        ),
        system_instruction=_get_system_instruction(args.robot_name),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    pya = pyaudio.PyAudio()
    _event_loop_ref = asyncio.get_event_loop()

    print("Connecting to Gemini Live...")
    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config,
        ) as session:
            _session_ref = session
            print("Connected to Gemini!")
            print("Waiting for robot to connect...")
            print("Press Ctrl+C to stop.\n")

            tasks = [
                asyncio.create_task(gemini_send_video(session, frame_processor, args.frame_interval)),
                asyncio.create_task(gemini_receive(session, pya, cmd_dispatcher, robot_ws_ref, args.robot_name)),
            ]
            if not getattr(args, 'no_slam', False):
                tasks.append(asyncio.create_task(_slam_update_loop(frame_processor, _slam_ref)))

            if args.audio_source == 'local':
                tasks.append(asyncio.create_task(
                    gemini_send_local_audio(session, pya, args.mic_device, args.mic_gain)
                ))
            else:
                tasks.append(asyncio.create_task(
                    gemini_send_audio(session, audio_queue)
                ))

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            _session_ref = None
    finally:
        pya.terminate()
        ws_server.close()
        robot.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Remote Server — YOLO + Face + Gemini + Robot Control'
    )
    parser.add_argument('--api-key', type=str, default=None,
                        help='Gemini API key (or set GEMINI_API_KEY)')
    parser.add_argument('--voice', type=str, default='Puck',
                        choices=['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'])
    parser.add_argument('--frame-interval', type=float, default=1.0,
                        help='Seconds between frames sent to Gemini')
    parser.add_argument('--port', type=int, default=8080, help='Web UI port')
    parser.add_argument('--ws-port', type=int, default=9090,
                        help='WebSocket port for robot connection')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--face-tolerance', type=float, default=0.6)
    parser.add_argument('--no-faces', action='store_true')
    parser.add_argument('--follow-distance', type=float, default=1.0)
    parser.add_argument('--audio-source', choices=['local', 'robot'], default='robot',
                        help="'local' = use this machine's mic; 'robot' = stream from robot mic")
    parser.add_argument('--mic-gain', type=float, default=3.0,
                        help='Mic gain (only for --audio-source local)')
    parser.add_argument('--mic-device', type=int, default=None,
                        help='PyAudio mic device (only for --audio-source local)')
    parser.add_argument('--no-slam', action='store_true',
                        help='Disable SLAM (localization + map from depth stream)')
    parser.add_argument('--robot-name', type=str, default='Jimmy',
                        help='Robot wake word: only respond when addressed by this name (default: Jimmy)')
    args = parser.parse_args()

    print("=" * 60)
    print("Remote Robot Server")
    print("  YOLO + Face Recognition + Gemini Live + Robot Control")
    print(f"  Audio: {args.audio_source} mic | Wake word: {args.robot_name}")
    print("=" * 60)

    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
