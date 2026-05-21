"""
src/recognition.py — ArcFace Face Recognition via ONNX Runtime.

Extracts 512-D ArcFace embeddings directly using ONNX Runtime — NO
insightface package needed.  This avoids the C++ / Cython compilation
requirement that blocks installation on Windows without Visual Studio.

How it works
────────────
  Face crop + MediaPipe landmarks
       → 5-point affine alignment to standard 112×112 template
       → BGR → RGB → normalise → NCHW tensor
       → ONNX Runtime inference (w600k_r50.onnx)
       → L2-normalised 512-D embedding
       → Cosine similarity against registered database

Model download
──────────────
On first run the ArcFace ONNX model (w600k_r50.onnx, ~166 MB) is
automatically downloaded from the InsightFace GitHub releases inside
the buffalo_l model pack.  Only the recognition file is kept; the rest
of the pack (detection, landmarks) is discarded since we use MediaPipe
for those tasks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import MODELS_DIR, Config, log

# ──────────────────────── Model paths ────────────────────────

_ARCFACE_ONNX = MODELS_DIR / "w600k_r50.onnx"

# InsightFace buffalo_l pack (contains w600k_r50.onnx among other models)
_ARCFACE_PACK_URL = (
    "https://github.com/deepinsight/insightface/releases/"
    "download/v0.7/buffalo_l.zip"
)

# ArcFace standard alignment template (112×112, 5 landmarks)
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


# ──────────────────────── Model download ────────────────────────

def _download_arcface_model() -> Path:
    """
    Download the ArcFace ONNX model if not present.

    Downloads the buffalo_l pack from InsightFace GitHub releases,
    extracts w600k_r50.onnx, and deletes the rest.
    """
    if _ARCFACE_ONNX.exists():
        return _ARCFACE_ONNX

    log.info(
        "ArcFace model not found. Downloading from InsightFace releases..."
    )
    log.info("  URL: %s", _ARCFACE_PACK_URL)
    log.info("  This is a ~300 MB download (one-time only).")

    zip_path = MODELS_DIR / "buffalo_l.zip"

    try:
        # Download with progress
        def _report(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(
                    f"\r  Downloading: {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)"
                )
                sys.stdout.flush()

        import sys
        urllib.request.urlretrieve(_ARCFACE_PACK_URL, zip_path, _report)
        print()  # newline after progress bar

        # Extract only w600k_r50.onnx
        log.info("  Extracting w600k_r50.onnx from pack...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Look for the recognition model inside the zip
            for member in zf.namelist():
                if member.endswith("w600k_r50.onnx"):
                    # Extract to models/ dir (flatten path)
                    source = zf.open(member)
                    with open(_ARCFACE_ONNX, "wb") as target:
                        shutil.copyfileobj(source, target)
                    log.info("  Extracted: %s", _ARCFACE_ONNX)
                    break
            else:
                log.error(
                    "w600k_r50.onnx not found inside buffalo_l.zip. "
                    "The pack structure may have changed."
                )
                return _ARCFACE_ONNX  # return path anyway; init will fail gracefully

        # Clean up the zip (best-effort — may be locked on Windows)
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            log.debug("Could not delete zip (may be locked). Not critical.")
        log.info("ArcFace model ready: %s", _ARCFACE_ONNX)
        return _ARCFACE_ONNX

    except Exception as exc:
        log.error("Failed to download ArcFace model: %s", exc)
        log.error(
            "Manual fix: download %s , extract w600k_r50.onnx, "
            "and place it at: %s",
            _ARCFACE_PACK_URL,
            _ARCFACE_ONNX,
        )
        return _ARCFACE_ONNX


# ──────────────────────── Face alignment ────────────────────────

def _align_face(
    img: np.ndarray,
    landmarks: Optional[List[Tuple[int, int]]] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    out_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """
    Align a face to the standard ArcFace template using MediaPipe
    landmarks.

    Uses a similarity transform (rotation + scale + translation)
    computed from the 5 facial landmarks mapped to the ArcFace
    reference positions.

    Parameters
    ----------
    img : ndarray          — BGR input image.
    landmarks : list       — List of (x, y) tuples for facial keypoints.
                             Expects at least: left_eye, right_eye, nose.
    bbox : tuple           — (x1, y1, x2, y2) fallback if no landmarks.
    out_size : tuple       — Output size (width, height).

    Returns
    -------
    Aligned BGR face image, or None on failure.
    """
    if landmarks is not None and len(landmarks) >= 3:
        # The caller (gui.py / main.py) passes landmarks as:
        #   [0] left_eye, [1] right_eye, [2] nose  (crop-relative)
        #
        # ArcFace reference template:
        #   [0] left_eye  (38.29, 51.70)
        #   [1] right_eye (73.53, 51.50)
        #   [2] nose_tip  (56.03, 71.74)

        src_pts = np.array(
            [
                landmarks[0],  # left eye
                landmarks[1],  # right eye
                landmarks[2],  # nose
            ],
            dtype=np.float32,
        )

        dst_pts = np.array(
            [
                _ARCFACE_DST[0],  # left eye ref
                _ARCFACE_DST[1],  # right eye ref
                _ARCFACE_DST[2],  # nose ref
            ],
            dtype=np.float32,
        )

        # Compute similarity transform (3-point alignment)
        transform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if transform is not None:
            aligned = cv2.warpAffine(
                img, transform, out_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            if aligned is not None and aligned.size > 0:
                return aligned
            log.warning("warpAffine produced empty result, falling back to resize.")

    # Fallback: just resize the whole image to out_size.
    # This works even when bbox is not provided (we use the full crop).
    h, w = img.shape[:2]
    if w > 0 and h > 0:
        try:
            return cv2.resize(img, out_size)
        except Exception as exc:
            log.warning("Fallback resize failed: %s", exc)
    return None


def _crop_and_resize(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """Simple crop + resize fallback when no landmarks are available."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, out_size)


# ──────────────────────── Face Database ────────────────────────


class FaceDatabase:
    """
    Persistent store of registered face embeddings.

    Saved as JSON:
        {
            "persons": [
                {
                    "name": "John Doe",
                    "embedding": [0.012, -0.034, ...],
                    "samples": 3,
                    "registered_at": "2024-01-15T10:30:00"
                }
            ]
        }
    """

    def __init__(self, data_file: str) -> None:
        self._data_file = data_file
        self._persons: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._data_file):
            self._persons = []
            return
        try:
            with open(self._data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._persons = data.get("persons", [])
            log.info(
                "Face database loaded: %d person(s) from %s",
                len(self._persons),
                self._data_file,
            )
        except Exception as exc:
            log.error("Failed to load face database: %s", exc)
            self._persons = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        try:
            with open(self._data_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"persons": self._persons}, f, indent=2, ensure_ascii=False,
                )
        except Exception as exc:
            log.error("Failed to save face database: %s", exc)

    def add_person(self, name: str, embedding: List[float], samples: int = 1) -> None:
        for person in self._persons:
            if person["name"].lower() == name.lower():
                person["embedding"] = embedding
                person["samples"] = samples
                person["registered_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                self._save()
                log.info("Updated registration for '%s'.", name)
                return
        self._persons.append({
            "name": name,
            "embedding": embedding,
            "samples": samples,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save()
        log.info("Registered new person '%s' (%d samples).", name, samples)

    def remove_person(self, name: str) -> bool:
        original_len = len(self._persons)
        self._persons = [
            p for p in self._persons if p["name"].lower() != name.lower()
        ]
        if len(self._persons) < original_len:
            self._save()
            log.info("Removed person '%s'.", name)
            return True
        return False

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        result = []
        for p in self._persons:
            emb = np.array(p["embedding"], dtype=np.float32)
            result.append((p["name"], emb))
        return result

    @property
    def count(self) -> int:
        return len(self._persons)

    @property
    def names(self) -> List[str]:
        return [p["name"] for p in self._persons]

    def clear(self) -> None:
        self._persons = []
        self._save()
        log.info("Face database cleared.")


# ──────────────────────── Recognition Engine ────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(dot / denom)


class FaceRecognizer:
    """
    ArcFace-based face recognition using ONNX Runtime directly.

    No insightface package required — just onnxruntime.

    Usage
    -----
        rec = FaceRecognizer(config)
        rec.initialise()
        rec.register("John", [(crop1, kps1), (crop2, kps2)])
        result = rec.recognize(crop, landmarks=kps)
        # result = {"name": "John", "confidence": 0.92} or None
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._session = None   # ONNX InferenceSession
        self._db: Optional[FaceDatabase] = None
        self._ready = False

    # ─────────── Initialisation ───────────

    def initialise(self) -> bool:
        """Load the ArcFace ONNX model and face database."""
        import onnxruntime as ort

        # Download model if needed
        model_path = str(_download_arcface_model())

        if not Path(model_path).exists():
            log.error("ArcFace ONNX model not found at: %s", model_path)
            return False

        # Select execution provider
        providers = ["CPUExecutionProvider"]
        if self._config.use_gpu:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
                log.info("ArcFace: using CUDA GPU acceleration.")
            else:
                log.info("CUDA not available — using CPU for recognition.")

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                model_path, sess_options=opts, providers=providers,
            )
            # Log full model I/O metadata for diagnostics
            inputs_info = []
            for inp in self._session.get_inputs():
                inputs_info.append(f"{inp.name}={inp.shape}")
            outputs_info = []
            for out in self._session.get_outputs():
                outputs_info.append(f"{out.name}={out.shape}")
            log.info(
                "ArcFace ONNX model loaded: %s", model_path,
            )
            log.info("  inputs:  %s", inputs_info)
            log.info("  outputs: %s", outputs_info)
        except Exception as exc:
            log.error("Failed to load ArcFace ONNX model: %s", exc)
            return False

        # Load face database
        self._db = FaceDatabase(self._config.data_file)
        self._ready = True

        log.info(
            "Face Recognizer ready.  Database: %d person(s).",
            self._db.count,
        )
        return True

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def database(self) -> Optional[FaceDatabase]:
        return self._db

    # ─────────── Registration ───────────

    def register(
        self,
        name: str,
        face_samples: List[Tuple[np.ndarray, List[Tuple[int, int]]]],
    ) -> bool:
        """
        Register a face by averaging embeddings from multiple samples.

        Parameters
        ----------
        name : str     — Person's name.
        face_samples : list of (crop_bgr, landmarks) tuples.
                       landmarks is a list of (x, y) ints from MediaPipe.

        Returns True on success.
        """
        if not self._ready or self._session is None or self._db is None:
            log.error("Recognizer not initialised.")
            return False
        if not face_samples:
            log.error("No samples provided for registration.")
            return False

        embeddings: List[np.ndarray] = []
        iw, ih = self._config.recognition_input_size

        for i, (crop, kps) in enumerate(face_samples):
            aligned = _align_face(crop, landmarks=kps, out_size=(iw, ih))
            if aligned is None:
                log.warning("Alignment failed for sample %d.", i + 1)
                continue
            emb = self._run_onnx(aligned)
            if emb is not None:
                embeddings.append(emb)
            else:
                log.warning("ONNX inference failed for sample %d.", i + 1)

        if not embeddings:
            log.error("No valid embeddings extracted — registration failed.")
            return False

        # Average and normalise
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-10:
            avg = avg / norm

        self._db.add_person(name=name, embedding=avg.tolist(), samples=len(embeddings))
        return True

    # ─────────── Recognition ───────────

    def recognize(
        self,
        face_crop: np.ndarray,
        keypoints: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Identify a face crop against the registered database.

        Parameters
        ----------
        face_crop : ndarray — BGR face image.
        keypoints : list    — MediaPipe landmarks [(x,y), ...].

        Returns dict {"name": ..., "confidence": ...} or None.
        """
        if not self._ready or self._session is None or self._db is None:
            return None
        if self._db.count == 0:
            return None

        iw, ih = self._config.recognition_input_size
        aligned = _align_face(face_crop, landmarks=keypoints, out_size=(iw, ih))
        if aligned is None:
            return None

        query_emb = self._run_onnx(aligned)
        if query_emb is None:
            return None

        best_name: Optional[str] = None
        best_score: float = -1.0

        for name, db_emb in self._db.get_all_embeddings():
            sim = _cosine_similarity(query_emb, db_emb)
            log.debug("  %s → similarity=%.4f", name, sim)
            if sim > best_score:
                best_score = sim
                best_name = name

        threshold = self._config.recognition_threshold
        if best_name is not None:
            log.info(
                "Recognition: best='%s' score=%.4f threshold=%.2f → %s",
                best_name, best_score, threshold,
                "MATCH" if best_score >= threshold else "NO MATCH",
            )
        else:
            log.debug("Recognition: no registered faces to compare.")

        if best_score >= threshold and best_name is not None:
            return {"name": best_name, "confidence": best_score}
        return None

    # ─────────── ONNX Inference ───────────

    def _run_onnx(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Run ArcFace inference on an aligned 112x112 face image.

        Accepts either BGR or RGB input (auto-detects by channel count).
        Returns L2-normalised 512-D embedding, or None on failure.
        """
        if self._session is None:
            return None

        try:
            # Auto-detect color: if 3 channels, ensure RGB
            if len(aligned_face.shape) == 3 and aligned_face.shape[2] == 3:
                # cv2.cvtColor is a no-op if already BGR→BGR, but we
                # normalise to RGB since the model expects it.
                rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            else:
                rgb = aligned_face

            # Normalise to [-1, 1]
            blob = rgb.astype(np.float32) / 255.0
            blob = (blob - 0.5) / 0.5

            # HWC → NCHW
            input_tensor = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            # Validate tensor shape
            inp = self._session.get_inputs()[0]
            expected_shape = inp.shape
            input_name = inp.name

            if len(input_tensor.shape) != len(expected_shape):
                log.error(
                    "ONNX shape mismatch: got %s, expected %s (input='%s')",
                    input_tensor.shape, expected_shape, input_name,
                )
                return None

            output = self._session.run(None, {input_name: input_tensor})

            embedding = np.array(output[0][0], dtype=np.float32).flatten()

            if embedding.size == 0 or np.any(np.isnan(embedding)):
                log.warning(
                    "ONNX produced invalid embedding (size=%d, has_nan=%s)",
                    embedding.size, np.any(np.isnan(embedding)),
                )
                return None

            # L2 normalise
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm

            return embedding

        except Exception as exc:
            log.error("ONNX inference error: %s", exc)
            return None

    # ─────────── Cleanup ───────────

    def close(self) -> None:
        self._session = None
        self._ready = False
        log.info("FaceRecognizer closed.")

    def __del__(self) -> None:
        self.close()
