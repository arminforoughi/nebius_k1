"""
Portable face detection & recognition engine.

Primary backend: face_recognition (dlib) — best on NVIDIA GPUs with CUDA-enabled dlib.
Fallback backend: facenet-pytorch (MTCNN + InceptionResnetV1) — works everywhere PyTorch runs.
"""

import threading
import numpy as np


class FaceEngine:
    """Unified face detection + embedding interface with automatic backend selection."""

    def __init__(self, device=None):
        self.backend = None
        self._lock = threading.Lock()

        self._try_dlib()
        if self.backend is None:
            self._try_facenet(device)
        if self.backend is None:
            raise RuntimeError(
                "No face recognition backend available.\n"
                "Install one of:\n"
                "  pip install face_recognition  (recommended for NVIDIA GPU)\n"
                "  pip install facenet-pytorch   (fallback)\n"
            )

    # -- Backend initialisation ------------------------------------------------

    def _try_dlib(self):
        try:
            import face_recognition
            self._fr = face_recognition
            self.backend = 'dlib'
            print("Face engine: dlib/face_recognition (CNN model)")
        except (ImportError, OSError):
            pass

    def _try_facenet(self, device):
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1

            if device is None:
                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'

            self._mtcnn = MTCNN(
                keep_all=True,
                device=device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False,
            )
            self._resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            self._torch = torch
            self._device = device
            self.backend = 'facenet'
            print(f"Face engine: facenet-pytorch (device={device})")
        except ImportError:
            pass
        except Exception as e:
            print(f"Face engine: facenet-pytorch init failed: {e}")

    # -- Public API ------------------------------------------------------------

    @property
    def default_tolerance(self):
        """Recommended match-distance threshold for each backend."""
        if self.backend == 'facenet':
            return 0.65
        return 0.6

    def detect_and_encode(self, rgb_frame):
        """Detect faces and compute embeddings in a single pass.

        Returns list of (top, right, bottom, left, encoding_ndarray) tuples.
        The bounding-box order matches face_recognition's convention.
        """
        with self._lock:
            if self.backend == 'dlib':
                return self._dlib_detect(rgb_frame)
            if self.backend == 'facenet':
                return self._facenet_detect(rgb_frame)
        return []

    def face_distance(self, known_encodings, face_encoding):
        """Compute distance from *face_encoding* to each entry in *known_encodings*.

        Returns an ndarray of floats (lower = more similar).
        """
        if len(known_encodings) == 0:
            return np.array([])

        if self.backend == 'dlib':
            return self._fr.face_distance(known_encodings, face_encoding)

        if self.backend == 'facenet':
            known = np.asarray(known_encodings, dtype=np.float32)
            enc = np.asarray(face_encoding, dtype=np.float32)
            known_norm = known / (np.linalg.norm(known, axis=1, keepdims=True) + 1e-10)
            enc_norm = enc / (np.linalg.norm(enc) + 1e-10)
            return 1.0 - (known_norm @ enc_norm)

        return np.array([])

    # -- Dlib backend ----------------------------------------------------------

    def _dlib_detect(self, rgb_frame):
        locs = self._fr.face_locations(rgb_frame, model='cnn')
        if not locs:
            return []
        encs = self._fr.face_encodings(rgb_frame, locs, model='small')
        return [
            (top, right, bottom, left, enc)
            for (top, right, bottom, left), enc in zip(locs, encs)
        ]

    # -- Facenet backend -------------------------------------------------------

    def _facenet_detect(self, rgb_frame):
        from PIL import Image

        pil_img = Image.fromarray(rgb_frame)
        boxes, probs = self._mtcnn.detect(pil_img)

        if boxes is None or len(boxes) == 0:
            return []

        faces = []
        valid_indices = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None or prob < 0.9:
                continue
            face_tensor = self._crop_face(pil_img, box)
            if face_tensor is not None:
                faces.append(face_tensor)
                valid_indices.append(i)

        if not faces:
            return []

        batch = self._torch.stack(faces).to(self._device)
        with self._torch.no_grad():
            embeddings = self._resnet(batch).cpu().numpy()

        results = []
        for j, idx in enumerate(valid_indices):
            x1, y1, x2, y2 = (int(b) for b in boxes[idx])
            top, right, bottom, left = y1, x2, y2, x1
            results.append((top, right, bottom, left, embeddings[j]))

        return results

    def _crop_face(self, pil_img, box, size=160, margin=32):
        """Crop, resize and normalise a face patch for InceptionResnetV1."""
        from PIL import Image as _Image

        w, h = pil_img.size
        x1 = max(0, int(box[0]) - margin)
        y1 = max(0, int(box[1]) - margin)
        x2 = min(w, int(box[2]) + margin)
        y2 = min(h, int(box[3]) + margin)

        if x2 <= x1 or y2 <= y1:
            return None

        face = pil_img.crop((x1, y1, x2, y2)).resize((size, size), _Image.BILINEAR)
        arr = np.array(face, dtype=np.float32)
        tensor = self._torch.FloatTensor(arr).permute(2, 0, 1)  # HWC -> CHW
        tensor = (tensor - 127.5) / 128.0  # fixed_image_standardization
        return tensor
