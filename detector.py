"""
src/detector.py — MediaPipe Face Detection Module.

Replaces the legacy Haar Cascade pipeline entirely.  Uses MediaPipe's
Face Detection task API (BlazeFace) — a lightweight, mobile-friendly
detector that provides sub-millisecond inference on CPU.

Compatible with mediapipe >= 0.10.14 (both legacy solutions and the
new tasks API are supported; the module auto-detects which is available).

Model
─────
On first run the BlazeFace `.tflite` model is automatically downloaded
from Google's MediaPipe model server.  Supports short-range (default)
and full-range models.

This module wraps MediaPipe Face Detection into a simple API:

    detector = FaceDetector(config)
    detector.initialise()
    results = detector.detect(frame)
    # results = [{"bbox": (x1,y1,x2,y2), "confidence": 0.98, "landmarks": [...]}, ...]
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import MODELS_DIR, Config, log

# ──────────────────────── Model URLs ────────────────────────

# Google MediaPipe model server (official, reliable, no auth needed)
_MODEL_BASE = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
)
_SHORT_RANGE_MODEL_URL = _MODEL_BASE + "blaze_face_short_range.tflite"
_SHORT_RANGE_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"

_FULL_RANGE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_full_range/float16/latest/"
    "blaze_face_full_range.tflite"
)
_FULL_RANGE_MODEL_PATH = MODELS_DIR / "blaze_face_full_range.tflite"

# Keypoint names returned by the BlazeFace model
_KP_NAMES = [
    "right_eye", "left_eye", "nose_tip",
    "mouth_center", "right_ear_tragion", "left_ear_tragion",
]


def _download_model(url: str, dest: Path) -> bool:
    """Download a TFLite model file if not already present."""
    if dest.exists():
        return True

    log.info("Downloading face detection model...")
    log.info("  URL: %s", url)

    try:
        def _report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(
                    f"\r  Downloading: {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(str(url), str(dest), _report)
        print()  # newline after progress
        log.info("Model saved: %s", dest)
        return True
    except Exception as exc:
        log.error("Failed to download model: %s", exc)
        return False


# ──────────────────────── Detector Class ────────────────────────


class FaceDetector:
    """
    MediaPipe-based face detector.

    Supports both the legacy ``mediapipe.solutions`` API and the newer
    ``mediapipe.tasks`` API.  Auto-detects at import time.

    Attributes
    ----------
    _api : str
        "tasks" (new API) or "legacy" (solutions API).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._detector = None
        self._api = "tasks"  # default; overridden in initialise()

    # ─────────── Initialisation ───────────

    def initialise(self) -> bool:
        """
        Create the MediaPipe Face Detection instance.

        Tries the new tasks API first, then falls back to legacy solutions.
        Returns True if initialised successfully.
        """
        if self._init_tasks_api():
            return True
        if self._init_legacy_api():
            return True
        return False

    def _init_tasks_api(self) -> bool:
        """Try initialising with the new mediapipe.tasks API."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_selection = self._config.detection_model  # 0 = short-range, 1 = full-range
            min_confidence = self._config.detection_confidence

            # Select and download model
            if model_selection == 1:
                model_path = _FULL_RANGE_MODEL_PATH
                model_url = _FULL_RANGE_MODEL_URL
                model_label = "full-range"
            else:
                model_path = _SHORT_RANGE_MODEL_PATH
                model_url = _SHORT_RANGE_MODEL_URL
                model_label = "short-range"

            if not _download_model(model_url, model_path):
                log.warning(
                    "Could not download face detection model. "
                    "Falling back to legacy API."
                )
                return False

            # Create detector options
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_confidence,
            )

            self._detector = vision.FaceDetector.create_from_options(options)
            self._mp_image = __import__("mediapipe").Image
            self._mp_image_format = __import__("mediapipe").ImageFormat
            self._api = "tasks"

            log.info(
                "MediaPipe Face Detection initialised [tasks API] "
                "(model=%s, min_conf=%.2f).",
                model_label,
                min_confidence,
            )
            return True

        except Exception as exc:
            log.debug("tasks API not available: %s", exc)
            return False

    def _init_legacy_api(self) -> bool:
        """Try initialising with the legacy mediapipe.solutions API."""
        try:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                log.warning(
                    "mediapipe.solutions not found in this version. "
                    "Install mediapipe < 0.10.15 or use the tasks API."
                )
                return False

            model_selection = self._config.detection_model
            min_confidence = self._config.detection_confidence

            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_confidence,
            )
            self._api = "legacy"

            log.info(
                "MediaPipe Face Detection initialised [legacy API] "
                "(model=%s, min_conf=%.2f).",
                "short-range" if model_selection == 0 else "full-range",
                min_confidence,
            )
            return True

        except Exception as exc:
            log.error("Failed to initialise MediaPipe (legacy): %s", exc)
            return False

    # ─────────── Detection ───────────

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a BGR frame.

        Parameters
        ----------
        frame : ndarray — BGR image from OpenCV VideoCapture.

        Returns
        -------
        List of dicts, each containing:
            bbox        : (x1, y1, x2, y2) in pixel coordinates
            confidence  : float 0-1
            landmarks   : list of (x, y) tuples (6 MediaPipe landmarks)
            keypoint    : dict with keys: right_eye, left_eye, nose_tip,
                          mouth_center, right_ear_tragion, left_ear_tragion
        """
        if self._detector is None:
            log.error("Detector not initialised — call initialise() first.")
            return []

        if self._api == "tasks":
            return self._detect_tasks(frame)
        else:
            return self._detect_legacy(frame)

    def _detect_tasks(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using the new mediapipe.tasks API."""
        h, w = frame.shape[:2]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = self._mp_image(
            image_format=self._mp_image_format.SRGB, data=rgb,
        )

        # Run detection
        result = self._detector.detect(mp_image)

        faces: List[Dict[str, Any]] = []

        if result.detections is None:
            return faces

        for detection in result.detections:
            # Bounding box — the tasks API returns pixel coordinates directly
            bbox = detection.bounding_box
            if bbox is None:
                continue

            x1 = max(0, int(bbox.origin_x))
            y1 = max(0, int(bbox.origin_y))
            x2 = min(w, int(bbox.origin_x + bbox.width))
            y2 = min(h, int(bbox.origin_y + bbox.height))

            if x2 <= x1 or y2 <= y1:
                continue

            # Confidence from categories
            confidence = 0.0
            if detection.categories and len(detection.categories) > 0:
                confidence = float(detection.categories[0].score)

            # Keypoints (6 per face)
            landmarks: List[Tuple[int, int]] = []
            keypoint_dict: Dict[str, Tuple[int, int]] = {}
            if detection.keypoints:
                for idx, kp in enumerate(detection.keypoints):
                    px = max(0, min(w, int(kp.x)))
                    py = max(0, min(h, int(kp.y)))
                    landmarks.append((px, py))
                    if idx < len(_KP_NAMES):
                        keypoint_dict[_KP_NAMES[idx]] = (px, py)

            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "landmarks": landmarks,
                "keypoint": keypoint_dict,
            })

        # Sort by confidence descending, keep top N
        faces.sort(key=lambda f: f["confidence"], reverse=True)
        if self._config.max_faces > 0:
            faces = faces[: self._config.max_faces]

        return faces

    def _detect_legacy(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using the legacy mediapipe.solutions API."""
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        faces: List[Dict[str, Any]] = []

        if results.detections is None:
            return faces

        for detection in results.detections:
            bbox_rel = detection.location_data.relative_bounding_box
            x_min = bbox_rel.xmin
            y_min = bbox_rel.ymin
            bw = bbox_rel.width
            bh = bbox_rel.height

            x1 = max(0, int(x_min * w))
            y1 = max(0, int(y_min * h))
            x2 = min(w, int((x_min + bw) * w))
            y2 = min(h, int((y_min + bh) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            confidence = float(
                detection.score[0] if detection.score else 0.0
            )

            landmarks: List[Tuple[float, float]] = []
            keypoint_dict: Dict[str, Tuple[float, float]] = {}
            for idx, kp in enumerate(detection.location_data.relative_keypoints):
                px = max(0, min(w, int(kp.x * w)))
                py = max(0, min(h, int(kp.y * h)))
                landmarks.append((px, py))
                if idx < len(_KP_NAMES):
                    keypoint_dict[_KP_NAMES[idx]] = (px, py)

            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "landmarks": landmarks,
                "keypoint": keypoint_dict,
            })

        faces.sort(key=lambda f: f["confidence"], reverse=True)
        if self._config.max_faces > 0:
            faces = faces[: self._config.max_faces]

        return faces

    # ─────────── Cleanup ───────────

    def close(self) -> None:
        """Release the MediaPipe detector."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
            log.info("FaceDetector closed.")

    def __del__(self) -> None:
        self.close()
