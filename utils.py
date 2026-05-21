"""
src/utils.py — Shared configuration, logging, FPS tracker, drawing helpers.

Provides the central Config dataclass consumed by every module, a
thread-safe FPS counter, and all the OpenCV drawing routines used by
the main display loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ────────────────────────── Paths ──────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

for d in (MODELS_DIR, DATA_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ────────────────────────── Logging ──────────────────────────
def setup_logger(name: str = "faceguard", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to both console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File — one log per session
    session_file = LOGS_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(session_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = setup_logger()

# ────────────────────────── Config ──────────────────────────

# Known virtual-camera substrings (case-insensitive)
KNOWN_VIRTUAL_CAM_NAMES: List[str] = [
    "obs virtual camera",
    "obs studio",
    "virtual camera",
    "manyCam virtual webcam",
    "snap camera",
    "xsplit broadcaster",
    "xsplit vcam",
    "unity webcam",
    "unloaded virtual camera",
    "fake camera",
    "vcam",
    "droidcam virtual camera",
    "irdroidcam",
    "epoccam",
    "ndi video",
    "screen capture",
    "wirecast",
    "vMix video",
    "streamlabs virtual",
    "elgato virtual",
    "logitech capture virtual",
    "ristretto virtual camera",
    "ak virtual camera",
    "high resolution camera",       # generic virtual cam on some Linux
    "dummy video",
    "video4linux loopback",
    "v4l2loopback",
    "gstreamer",
]


@dataclass
class Config:
    """Central configuration for the entire pipeline."""

    # ── Camera ──
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps_target: int = 30

    # ── Face Detection (MediaPipe) ──
    detection_model: int = 0            # 0 = short-range, 1 = full-range
    detection_confidence: float = 0.50  # min detection confidence
    max_faces: int = 5                  # max faces to process per frame

    # ── Face Recognition (InsightFace / ArcFace) ──
    recognition_model: str = "buffalo_l"  # InsightFace model pack
    recognition_threshold: float = 0.45   # cosine similarity threshold
    recognition_input_size: Tuple[int, int] = (112, 112)

    # ── Anti-Spoof (Silent-Face ONNX) ──
    antispoof_model_path: str = ""   # path to .onnx; auto-downloaded if empty
    antispoof_threshold: float = 0.50
    antispoof_input_size: Tuple[int, int] = (80, 80)

    # ── Attendance ──
    cooldown_seconds: int = 300  # 5 min between duplicate marks
    data_file: str = str(DATA_DIR / "registered_faces.json")

    # ── Virtual Camera ──
    virtual_cam_names: List[str] = field(
        default_factory=lambda: list(KNOWN_VIRTUAL_CAM_NAMES)
    )

    # ── UI ──
    show_fps: bool = True
    show_confidence: bool = True
    box_thickness: int = 2

    # ── Performance ──
    use_gpu: bool = True       # prefer CUDA if available
    target_fps: int = 20
    skip_frames: int = 0       # process every (skip_frames+1)th frame

    # ── Advanced heuristics ──
    enable_texture_analysis: bool = True
    enable_moire_detection: bool = True
    enable_screen_reflection: bool = True
    enable_temporal_consistency: bool = True
    temporal_window_size: int = 15    # frames for consistency check

    def save(self, path: Optional[str] = None) -> None:
        path = path or str(DATA_DIR / "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, default=str)
        log.info("Config saved to %s", path)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        path = path or str(DATA_DIR / "config.json")
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ──────────────────────── FPS Tracker ────────────────────────

class FPSTracker:
    """Thread-safe rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: List[float] = []
        self._window = window
        self._fps: float = 0.0

    def tick(self) -> None:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            self._fps = (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def reset(self) -> None:
        self._times.clear()
        self._fps = 0.0


# ──────────────────────── Drawing Helpers ────────────────────────

# Color palette (BGR for OpenCV)
COLORS = {
    "real":      (0, 220, 120),    # green
    "spoof":     (0, 0, 220),      # red
    "unknown":   (0, 165, 255),    # orange
    "no_face":   (100, 100, 100),  # grey
    "bg":        (18, 18, 30),     # dark background
    "text":      (220, 220, 235),  # light text
    "accent":    (0, 220, 180),    # teal accent
}


def draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    score: float,
    status: str,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a labelled bounding box on *frame* in place.

    Parameters
    ----------
    frame : ndarray   — BGR image (mutated).
    x1,y1,x2,y2 : int — box corners.
    label : str       — name text.
    score : float     — confidence 0-1.
    status : str      — "real", "spoof", or "unknown".
    thickness : int   — line width.
    """
    color = COLORS.get(status, COLORS["unknown"])

    # Outer box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Corner accents
    cl = min(18, int((x2 - x1) * 0.22), int((y2 - y1) * 0.22))
    cv2.line(frame, (x1, y1 + cl), (x1, y1), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x2 - cl, y1), (x2, y1), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y2 - cl), (x1, y2), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x2 - cl, y2), (x2, y2), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x2, y2 - cl), (x2, y2), color, thickness + 1, cv2.LINE_AA)

    # Label background
    status_upper = status.upper()
    display = f"{label}  {score * 100:.0f}%  [{status_upper}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    (tw, th), _ = cv2.getTextSize(display, font, scale, 1)
    label_y = max(y1 - 8, th + 10)
    cv2.rectangle(
        frame,
        (x1, label_y - th - 6),
        (x1 + tw + 10, label_y + 4),
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, display,
        (x1 + 5, label_y - 3),
        font, scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    n_faces: int,
    is_virtual: bool,
    antispoof_on: bool,
) -> np.ndarray:
    """Draw heads-up-display overlay: FPS, face count, warnings."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # FPS — top-left
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (12, 28), font, 0.65, COLORS["accent"], 2, cv2.LINE_AA,
    )

    # Face count
    cv2.putText(
        frame, f"Faces: {n_faces}",
        (12, 56), font, 0.55, COLORS["text"], 1, cv2.LINE_AA,
    )

    # Warnings
    y_off = 84
    if is_virtual:
        cv2.putText(
            frame, "[!] VIRTUAL CAMERA DETECTED",
            (12, y_off), font, 0.6, COLORS["spoof"], 2, cv2.LINE_AA,
        )
        y_off += 28

    if antispoof_on:
        cv2.putText(
            frame, "[i] Anti-Spoof: ACTIVE",
            (12, y_off), font, 0.5, COLORS["real"], 1, cv2.LINE_AA,
        )

    return frame


def draw_instruction_bar(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    """Draw a translucent info bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    bar_h = 26 + len(lines) * 22
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (14, h - bar_h + 20 + i * 22),
            font, 0.48,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )
    return frame
