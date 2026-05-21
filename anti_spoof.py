"""
src/anti_spoof.py — Passive Anti-Spoofing Engine.

This module implements a multi-layered, *passive* (no challenge-response)
anti-spoofing system that runs *before* face recognition, rejecting spoof
inputs early to save compute and prevent fraudulent attendance marks.

Pipeline
────────
  Frame → Crop to face bbox
       → [ONNX Model]  Silent-Face classification (real / spoof)
       → [Heuristics]  texture, moiré, reflection, temporal, depth
       → Combined score → REAL / SPOOF / UNKNOWN decision

The ONNX model used is the **MiniFASNet** from the Silent-Face-Anti-Spoofing
project (https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).  If
the model file is not present it is automatically downloaded from the
project's GitHub releases.

Heuristic layer
───────────────
Even when the ONNX model is unavailable the heuristic layer provides a
reasonable defence by analysing:

* **Texture (LBP variance)**  — printed photos and screen captures have
  different local binary pattern variance compared to live skin texture.

* **Moiré pattern detection**  — replaying a screen with a camera
  produces characteristic interference fringes that can be detected via
  frequency-domain analysis of the face crop.

* **Screen reflection**  — if a face is being displayed on a monitor
  and re-captured, the specular highlight pattern differs from natural
  skin reflections.  We look for abnormally bright, sharp-edged
  highlight regions.

* **Low-depth estimation**  — a 2-D photo has zero depth variation
  within the face region.  We approximate this using defocus-aware
  gradient analysis (sharp edges everywhere = likely flat photo).

* **Temporal consistency**  — live faces exhibit subtle micro-movements
  (breathing, blinking).  Over a sliding window of frames we check
  whether the face crop changes by at least a small amount each frame.
  A completely static face across many frames is suspicious.

All heuristic scores are normalised to [0, 1] and averaged.  The final
decision uses a weighted combination of the ONNX score and the heuristic
score (configurable via Config).
"""

from __future__ import annotations

import logging
import time
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import MODELS_DIR, Config, log

# ──────────────────────── ONNX Model Setup ────────────────────────

# Multiple mirror URLs for the Silent-Face MiniFASNet ONNX model.
# The model is ~2.7 MB.  If all mirrors fail, the engine runs in
# heuristic-only mode (5 heuristic layers still provide good protection).
_ANTI_SPOOF_MODEL_MIRRORS = [
    (
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/"
        "releases/download/v1.0/anti_spoof_model.onnx"
    ),
    (
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/"
        "raw/master/resources/anti_spoof_models/anti_spoof_model.onnx"
    ),
]
_ANTI_SPOOF_MODEL_PATH = MODELS_DIR / "anti_spoof_model.onnx"


def _download_antispoof_model() -> Path:
    """Try multiple mirrors to download the anti-spoof ONNX model."""
    if _ANTI_SPOOF_MODEL_PATH.exists():
        return _ANTI_SPOOF_MODEL_PATH

    for i, url in enumerate(_ANTI_SPOOF_MODEL_MIRRORS):
        log.info("Downloading anti-spoof model (mirror %d/%d)...",
                 i + 1, len(_ANTI_SPOOF_MODEL_MIRRORS))
        try:
            urllib.request.urlretrieve(
                url, _ANTI_SPOOF_MODEL_PATH,
            )
            log.info("Anti-spoof model saved to %s", _ANTI_SPOOF_MODEL_PATH)
            return _ANTI_SPOOF_MODEL_PATH
        except Exception as exc:
            log.warning("Mirror %d failed: %s", i + 1, exc)
            # Clean up partial download
            if _ANTI_SPOOF_MODEL_PATH.exists():
                _ANTI_SPOOF_MODEL_PATH.unlink()

    log.error(
        "All download mirrors failed.  The system will use "
        "heuristic-only anti-spoofing (5 layers still active)."
    )
    log.warning(
        "To use the ONNX anti-spoof model, manually download it and "
        "place it at: %s", _ANTI_SPOOF_MODEL_PATH,
    )
    return _ANTI_SPOOF_MODEL_PATH


# ──────────────────────── Heuristic Functions ────────────────────────

def _lbp_variance(gray_face: np.ndarray) -> float:
    """
    Compute Local Binary Pattern variance over the face crop.

    Real skin has organic texture with moderate LBP variance.
    Printed photos / screen captures produce either very high variance
    (halftone dots, pixel grid) or abnormally low variance (smooth
    interpolation).
    """
    h, w = gray_face.shape
    if h < 3 or w < 3:
        return 0.5  # indeterminate

    gray = gray_face.astype(np.int16)
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    # 8 neighbours at radius 1
    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1),
    ]
    power = 0
    for dy, dx in neighbours:
        shifted = gray[1 + dy: h - 1 + dy, 1 + dx: w - 1 + dx]
        center = gray[1: h - 1, 1: w - 1]
        lbp |= ((shifted >= center).astype(np.uint8) << power)
        power += 1

    var = float(np.var(lbp))
    # Normalise: real skin typically var 500–2000
    # Printed: often < 200 or > 5000
    if var < 200:
        return 0.8
    elif var > 5000:
        return 0.7
    elif var < 400:
        return 0.3
    return 0.0


def _moire_detection(gray_face: np.ndarray) -> float:
    """
    Detect moiré interference patterns using frequency-domain analysis.

    Camera-to-screen replay produces periodic stripe patterns that
    manifest as strong peaks in the 2-D DFT magnitude spectrum.
    We check for energy concentration at high spatial frequencies.
    """
    h, w = gray_face.shape
    if h < 64 or w < 64:
        return 0.0

    # Resize to fixed size for consistent frequency analysis
    resized = cv2.resize(gray_face, (128, 128))
    f = np.fft.fft2(resized.astype(np.float32))
    magnitude = np.abs(np.fft.fftshift(f))

    # Remove DC component
    center = magnitude.shape[0] // 2
    mask = np.ones_like(magnitude, dtype=bool)
    r = 10  # radius to suppress DC
    mask[center - r: center + r, center - r: center + r] = False
    magnitude_masked = magnitude[mask]

    if magnitude_masked.size == 0:
        return 0.0

    total_energy = float(np.sum(magnitude_masked ** 2))
    if total_energy == 0:
        return 0.0

    # Compute ratio of energy in high-frequency bands (outer 40% of spectrum)
    edge_r = int(128 * 0.3)
    outer_mask = np.zeros_like(magnitude, dtype=bool)
    outer_mask[: edge_r, :] = True
    outer_mask[-edge_r:, :] = True
    outer_mask[:, : edge_r] = True
    outer_mask[:, -edge_r:] = True
    # Intersect with DC mask
    outer_mask &= mask

    high_freq_energy = float(np.sum(magnitude[outer_mask] ** 2))
    ratio = high_freq_energy / total_energy if total_energy > 0 else 0

    # Moiré patterns increase high-frequency energy ratio
    if ratio > 0.55:
        return min(1.0, (ratio - 0.55) * 5)
    return 0.0


def _screen_reflection(bgr_face: np.ndarray) -> float:
    """
    Detect screen-reflection artefacts.

    When a photo on a monitor is re-captured, the specular highlights
    from the screen's surface appear as abnormally bright, sharply
    delineated regions that do not follow natural skin-lighting models.

    We detect these by:
    1. Thresholding for very bright pixels (> 240)
    2. Measuring edge density around those bright regions
       (screen reflections have sharper edges than diffuse skin highlights)
    """
    if bgr_face.shape[0] < 32 or bgr_face.shape[1] < 32:
        return 0.0

    gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Bright pixel mask
    bright = (gray > 240).astype(np.uint8) * 255

    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

    bright_ratio = cv2.countNonZero(bright) / (h * w)
    if bright_ratio < 0.002:
        return 0.0  # negligible bright area

    # Edge density of the bright regions
    edges = cv2.Canny(bright, 50, 150)
    edge_pixels = cv2.countNonZero(edges)
    edge_density = edge_pixels / (h * w)

    # Natural skin highlights: broad, low edge-density
    # Screen reflections: compact, high edge-density
    if bright_ratio > 0.05 and edge_density > 0.02:
        return min(1.0, bright_ratio * 8 + edge_density * 15)
    return 0.0


def _low_depth_heuristic(bgr_face: np.ndarray) -> float:
    """
    Estimate depth variation within the face region.

    A flat 2-D photo has uniform focus across the entire face — every
    region is equally sharp.  A real 3-D face shows natural depth-dependent
    defocus: the nose tip is sharper than the ears, the eyes differ from
    the cheeks, etc.

    We approximate depth by measuring local sharpness (Laplacian variance)
    across a grid of patches.  If the variance *between patches* is too
    low (uniform sharpness everywhere), the face is likely flat.
    """
    h, w = bgr_face.shape[:2]
    if h < 64 or w < 64:
        return 0.0

    gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)

    # Divide into a 4x4 grid of patches
    grid_n = 4
    patch_h = h // grid_n
    patch_w = w // grid_n
    sharpness_values: List[float] = []

    for gy in range(grid_n):
        for gx in range(grid_n):
            py = gy * patch_h
            px = gx * patch_w
            patch = gray[py: py + patch_h, px: px + patch_w]
            # Laplacian variance as sharpness metric
            lap_var = cv2.Laplacian(patch, cv2.CV_64F).var()
            sharpness_values.append(lap_var)

    if len(sharpness_values) < 4:
        return 0.0

    mean_sharp = float(np.mean(sharpness_values))
    std_sharp = float(np.std(sharpness_values))

    if mean_sharp < 1.0:
        return 0.3  # too blurry to tell, slightly suspicious

    # Coefficient of variation of sharpness across patches
    cv_sharp = std_sharp / (mean_sharp + 1e-9)

    # Real 3-D face: CV typically 0.3 – 1.0 (some patches sharper)
    # Flat photo:    CV typically < 0.15 (uniform sharpness)
    if cv_sharp < 0.10:
        return 0.7
    elif cv_sharp < 0.18:
        return 0.3
    return 0.0


def _temporal_consistency(frame_buffer: Deque[np.ndarray]) -> float:
    """
    Analyse temporal consistency of face crops over a sliding window.

    A real face always has micro-movements (breathing, tiny twitches)
    that cause pixel-level differences between consecutive frames even
    when the subject is trying to stay still.

    A replayed video or a held-up photo produces either:
    - Zero change between frames (static photo), or
    - Too-perfect motion patterns (video replay — but this is harder to
      detect without a reference database).

    Returns a spoof score [0, 1].  High = suspicious.
    """
    if len(frame_buffer) < 3:
        return 0.0  # not enough data

    diffs: List[float] = []
    frames = list(frame_buffer)
    for i in range(1, len(frames)):
        a = frames[i].astype(np.float32)
        b = frames[i - 1].astype(np.float32)
        # Frames may differ in size — resize to match before comparing
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = cv2.resize(a, (w, h))
            b = cv2.resize(b, (w, h))
        diff = float(np.mean(np.abs(a - b))) / 255.0
        diffs.append(diff)

    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    std_diff = float(np.std(diffs)) if diffs else 0.0

    # Real face: mean_diff typically 0.005 – 0.04 (breathing, noise)
    # Static photo: mean_diff ~ 0.0
    # Video replay: similar to real (harder to distinguish)

    if mean_diff < 0.002:
        return 0.85  # effectively static — likely a photo
    elif mean_diff < 0.006:
        return 0.35  # very little motion — suspicious
    return 0.0


# ──────────────────────── Main Anti-Spoof Class ────────────────────────

class AntiSpoofEngine:
    """
    Multi-layer passive anti-spoofing engine.

    Combines an ONNX-based classifier (Silent-Face MiniFASNet) with
    five heuristic analysers.  The ONNX model provides the primary
    signal; heuristics provide a fallback and additional signal.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._session = None  # ONNX InferenceSession
        self._model_loaded = False
        self._input_name = ""
        self._frame_buffer: Deque[np.ndarray] = deque(
            maxlen=config.temporal_window_size
        )
        self._temporal_enabled = config.enable_temporal_consistency

    # ─────────── Initialisation ───────────

    def initialise(self) -> bool:
        """
        Load the ONNX anti-spoof model.  Returns True if successful.
        If the model cannot be loaded, the engine falls back to
        heuristic-only mode (still useful).
        """
        import onnxruntime as ort

        model_path = self._config.antispoof_model_path or str(_ANTI_SPOOF_MODEL_PATH)

        if not Path(model_path).exists():
            model_path = str(_download_antispoof_model())

        if not Path(model_path).exists():
            log.warning(
                "ONNX anti-spoof model not available — "
                "running in heuristic-only mode."
            )
            return False

        # Select execution provider
        providers = ["CPUExecutionProvider"]
        if self._config.use_gpu:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
                log.info("ONNX anti-spoof: using CUDA GPU acceleration.")
            else:
                log.info("CUDA not available for ONNX — using CPU.")

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                model_path, sess_options=opts, providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self._model_loaded = True
            log.info("ONNX anti-spoof model loaded: %s", model_path)
            return True
        except Exception as exc:
            log.error("Failed to load ONNX anti-spoof model: %s", exc)
            log.warning("Falling back to heuristic-only anti-spoofing.")
            return False

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    # ─────────── Main Classification ───────────

    def classify(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> Dict:
        """
        Run anti-spoof analysis on a single face within *frame*.

        Parameters
        ----------
        frame : ndarray — BGR image (full camera frame).
        face_bbox : tuple — (x1, y1, x2, y2) bounding box.

        Returns
        -------
        dict with keys:
            is_real    : bool
            score      : float  (0 = spoof, 1 = real)
            label      : str    "REAL", "SPOOF", or "UNKNOWN"
            onnx_score : float or None
            heuristic  : float  average heuristic spoof score
            details    : dict   per-heuristic breakdown
        """
        x1, y1, x2, y2 = face_bbox
        # Clamp coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._unknown_result("empty crop")

        # ── Step 1: ONNX model inference ──
        onnx_score: Optional[float] = None
        if self._model_loaded and self._session is not None:
            onnx_score = self._run_onnx(crop)

        # ── Step 2: Heuristic analyses ──
        heuristics: Dict[str, float] = {}

        # Convert to grayscale once
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if self._config.enable_texture_analysis:
            heuristics["texture_lbp"] = _lbp_variance(gray_crop)

        if self._config.enable_moire_detection:
            heuristics["moire"] = _moire_detection(gray_crop)

        if self._config.enable_screen_reflection:
            heuristics["screen_reflection"] = _screen_reflection(crop)

        # Depth heuristic uses colour
        heuristics["low_depth"] = _low_depth_heuristic(crop)

        # Temporal consistency
        self._frame_buffer.append(gray_crop.copy())
        if self._temporal_enabled and len(self._frame_buffer) >= 3:
            heuristics["temporal"] = _temporal_consistency(self._frame_buffer)
        else:
            heuristics["temporal"] = 0.0

        # Average heuristic spoof score
        h_scores = list(heuristics.values())
        heuristic_avg = float(np.mean(h_scores)) if h_scores else 0.0

        # ── Step 3: Combine scores ──
        if onnx_score is not None:
            # ONNX score: 0 = spoof, 1 = real
            # Heuristic score: 0 = real-like, 1 = spoof-like
            # Combine: final_real = 0.6 * onnx_real + 0.4 * (1 - heuristic)
            final_real = 0.6 * onnx_score + 0.4 * (1.0 - heuristic_avg)
        else:
            # Heuristic-only mode
            final_real = 1.0 - heuristic_avg

        threshold = self._config.antispoof_threshold

        if final_real >= threshold:
            label = "REAL"
            is_real = True
        elif final_real < threshold * 0.5:
            label = "SPOOF"
            is_real = False
        else:
            label = "UNKNOWN"
            is_real = False

        return {
            "is_real": is_real,
            "score": final_real,
            "label": label,
            "onnx_score": onnx_score,
            "heuristic": heuristic_avg,
            "details": heuristics,
        }

    # ─────────── ONNX Inference ───────────

    def _run_onnx(self, bgr_crop: np.ndarray) -> float:
        """
        Pre-process the face crop and run ONNX inference.

        Returns a real-score in [0, 1].
        """
        if self._session is None:
            return 0.5

        ih, iw = self._config.antispoof_input_size

        # Resize, convert RGB, normalise to [0, 1]
        resized = cv2.resize(bgr_crop, (iw, ih))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0

        # HWC → NCHW
        input_tensor = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        try:
            outputs = self._session.run(None, {self._input_name: input_tensor})
            # Model output shape: (1, 3) — [spoof_prob, real_prob, unknown_prob]
            probs = outputs[0][0]  # assuming softmax output

            if len(probs) >= 2:
                # prob[0] = spoof, prob[1] = real
                return float(np.clip(probs[1], 0, 1))
            elif len(probs) == 1:
                # Single sigmoid output
                return float(np.clip(probs[0], 0, 1))
            else:
                return 0.5
        except Exception as exc:
            log.debug("ONNX inference error: %s", exc)
            return 0.5

    # ─────────── Helper ───────────

    @staticmethod
    def _unknown_result(reason: str) -> Dict:
        return {
            "is_real": False,
            "score": 0.5,
            "label": "UNKNOWN",
            "onnx_score": None,
            "heuristic": 0.0,
            "details": {"note": reason},
        }

    def reset_temporal(self) -> None:
        """Clear the temporal frame buffer (call when scene changes)."""
        self._frame_buffer.clear()
