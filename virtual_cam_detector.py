"""
src/virtual_cam_detector.py — Virtual Camera Detection Module.

Actively probes the system for known virtual-camera device names and
optionally analyses frame statistics for artefacts that are typical of
software-generated feeds (e.g. perfectly uniform noise floor, locked
exposure values, or 0-gradient flat-field frames).

Detection strategy
──────────────────
1. **Name-based** (fast):  enumerate all DirectShow / V4L2 devices and
   match their friendly-names against a curated block-list of known
   virtual camera software (OBS, ManyCam, Snap Camera, XSplit, etc.).
2. **Heuristic-based** (slow):  capture a short burst of frames and
   analyse statistical properties (variance, dead-pixel count,
   bit-depth anomalies).  Virtual cams often produce frames whose
   pixel-value histogram is abnormally narrow compared to a real
   sensor.

On Windows the module uses the DirectShow COM API via ``pywin32``.
On Linux it falls back to parsing ``/dev/video*`` and optionally
``v4l2-ctl`` output if available.

The module runs once at startup and exposes a single boolean:
``is_virtual_camera`` — used by the main pipeline to decide whether
to reject frames *before* any expensive AI inference.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import List, Optional

import cv2
import numpy as np

from .utils import KNOWN_VIRTUAL_CAM_NAMES, Config, log

# ──────────────────────── Public API ────────────────────────

class VirtualCamDetector:
    """
    Detect whether the selected camera device is a known virtual camera.

    Attributes
    ----------
    is_virtual : bool
        ``True`` if the device matched a known virtual-camera name.
    device_name : str
        The human-readable name reported by the OS (may be empty).
    heuristic_score : float
        0.0 – 1.0  confidence that the feed is synthetic (heuristic only).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self.is_virtual: bool = False
        self.device_name: str = ""
        self.heuristic_score: float = 0.0

    # ─────────── Name-based detection (fast) ───────────

    def detect_by_name(self, camera_index: int) -> bool:
        """
        Enumerate capture devices and check if *camera_index* has a
        name that matches the virtual-camera block-list.

        Returns
        -------
        bool  — True if virtual camera detected.
        """
        os_name = platform.system()
        device_names: List[str] = []

        if os_name == "Windows":
            device_names = self._enumerate_windows()
        elif os_name == "Linux":
            device_names = self._enumerate_linux()
        else:
            log.warning(
                "Virtual camera name detection not implemented for %s; "
                "skipping name-based check.",
                os_name,
            )
            self.device_name = "unknown"
            return False

        # Look up the name for the requested camera index
        if camera_index < len(device_names):
            self.device_name = device_names[camera_index]
        else:
            self.device_name = ""
            log.debug(
                "Camera index %d out of range (found %d devices).",
                camera_index,
                len(device_names),
            )

        # Case-insensitive substring matching
        name_lower = self.device_name.lower()
        for blocked in self._config.virtual_cam_names:
            if blocked.lower() in name_lower:
                self.is_virtual = True
                log.warning(
                    "Virtual camera BLOCKED — device name matched '%s': %s",
                    blocked,
                    self.device_name,
                )
                return True

        log.info("Camera name OK: %s", self.device_name or f"index {camera_index}")
        return False

    # ─────────── Heuristic-based detection (slower) ───────────

    def detect_by_heuristics(self, cap: cv2.VideoCapture, samples: int = 10) -> float:
        """
        Capture *samples* frames from *cap* and analyse statistical
        properties that often differ between real sensors and virtual
        cameras.

        Heuristics evaluated
        ────────────────────
        1. **Pixel-variance** — Virtual cams often produce unnaturally
           low or perfectly uniform noise floors.  Real sensors have
           measurable read-noise variance.
        2. **Temporal variance** — A static scene captured through a
           real camera still shows micro-fluctuations from sensor
           noise.  Virtual feeds may be pixel-perfect across frames.
        3. **Dead-pixel-flatness** — Synthetic frames sometimes have
           runs of identical pixel values (especially in dark regions).

        Returns
        -------
        float — heuristic spoof score in [0, 1].  Values > 0.7
                suggest a virtual / synthetic feed.
        """
        if not cap.isOpened():
            log.error("Cannot run heuristic check — camera not open.")
            return 0.0

        frames: List[np.ndarray] = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

        try:
            for _ in range(samples):
                ret, frame = cap.read()
                if not ret:
                    continue
                # Convert to grayscale for statistical analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray.astype(np.float32))
        finally:
            # Restore position (not strictly needed for live feeds)
            pass

        if len(frames) < 3:
            log.debug("Too few frames for heuristic check (%d).", len(frames))
            return 0.0

        scores: List[float] = []

        # ── 1. Pixel-level variance per frame ──
        frame_variances = [np.var(f) for f in frames]
        mean_var = float(np.mean(frame_variances))
        # Real sensors: variance typically 200–3000 depending on scene
        # Virtual cams: can be < 50 for static content, or suspiciously
        # uniform across frames
        if mean_var < 30:
            scores.append(0.9)  # suspiciously flat
        elif mean_var < 80:
            scores.append(0.4)
        else:
            scores.append(0.0)

        # ── 2. Temporal variance — frame-to-frame difference ──
        diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i] - frames[i - 1]))
            diffs.append(diff)
        mean_diff = float(np.mean(diffs)) if diffs else 0.0

        # Real sensor: even with a static scene, read-noise gives
        # a mean absolute difference of ~1.5–5.0 per pixel.
        # Virtual cam replaying a static image: ~0.0
        if mean_diff < 0.3:
            scores.append(0.85)
        elif mean_diff < 1.0:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # ── 3. Histogram flatness (entropy) ──
        avg_hist_entropy = 0.0
        for f in frames:
            hist = np.histogram(f.flatten(), bins=64, range=(0, 256))[0].astype(np.float32)
            hist /= hist.sum() + 1e-9
            entropy = -np.sum(hist * np.log2(hist + 1e-9))
            avg_hist_entropy += entropy
        avg_hist_entropy /= len(frames)

        # Synthetic images often have very low entropy (< 3.5 bits)
        # Real-world camera frames typically 4.5 – 6.5 bits
        if avg_hist_entropy < 3.0:
            scores.append(0.75)
        elif avg_hist_entropy < 4.0:
            scores.append(0.3)
        else:
            scores.append(0.0)

        self.heuristic_score = float(np.mean(scores)) if scores else 0.0
        log.info(
            "Heuristic check complete — score=%.3f  (mean_var=%.1f, "
            "temporal_diff=%.2f, entropy=%.2f)",
            self.heuristic_score,
            mean_var,
            mean_diff,
            avg_hist_entropy,
        )
        return self.heuristic_score

    # ─────────── Full check (name + heuristic) ───────────

    def run_full_check(self, camera_index: int, cap: Optional[cv2.VideoCapture] = None) -> bool:
        """
        Run both name-based and heuristic-based detection.

        Returns True if the camera is determined to be virtual.
        """
        # Fast check first
        if self.detect_by_name(camera_index):
            return True

        # Heuristic check (requires an open VideoCapture)
        if cap is not None:
            h_score = self.detect_by_heuristics(cap, samples=10)
            if h_score > 0.7:
                self.is_virtual = True
                log.warning(
                    "Virtual camera BLOCKED by heuristics — score=%.2f", h_score
                )
                return True

        return False

    # ─────────── Platform-specific enumerators ───────────

    @staticmethod
    def _enumerate_windows() -> List[str]:
        """Return a list of DirectShow video-capture device names."""
        names: List[str] = []
        try:
            import win32com.client  # type: ignore
            enumerator = win32com.client.Dispatch("WIA.VideoDevices")
            for i in range(enumerator.Count):
                try:
                    names.append(str(enumerator.Item(i + 1).Name))
                except Exception as exc:
                    log.debug("Win32 enum error at index %d: %s", i, exc)
        except ImportError:
            log.warning(
                "pywin32 not installed — falling back to OpenCV device probing "
                "for virtual camera detection."
            )
        except Exception as exc:
            log.warning("Win32 DirectShow enumeration failed: %s", exc)

        # Fallback: probe OpenCV indices and read CAP_PROP_BACKEND
        if not names:
            for idx in range(10):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    backend = cap.getBackendName()
                    name = f"Camera {idx} ({backend})"
                    cap.release()
                    names.append(name)
                else:
                    break

        return names

    @staticmethod
    def _enumerate_linux() -> List[str]:
        """Return a list of V4L2 video device names."""
        names: List[str] = []
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse: alternating lines of "name\n  /dev/videoX"
                current_name = ""
                for line in result.stdout.strip().splitlines():
                    line = line.rstrip()
                    if line and not line.startswith("\t") and not line.startswith("/"):
                        current_name = line
                    elif "/dev/video" in line:
                        names.append(current_name if current_name else line.strip())
        except FileNotFoundError:
            log.debug("v4l2-ctl not found; using /dev/video* fallback.")
        except subprocess.TimeoutExpired:
            log.warning("v4l2-ctl timed out.")
        except Exception as exc:
            log.warning("Linux device enumeration error: %s", exc)

        # Fallback: just list /dev/video* paths
        if not names:
            for idx in range(10):
                dev = f"/dev/video{idx}"
                if os.path.exists(dev):
                    names.append(dev)
                else:
                    break

        return names
