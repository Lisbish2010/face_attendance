"""
src/main.py — FaceGuard AI: Production Face Attendance System.

Main entry point that orchestrates the full pipeline:

    Camera → Virtual Cam Detection → Anti-Spoof → Face Detection
          → Face Recognition → Attendance Logging → UI Display

Keyboard Controls
─────────────────
    q / ESC        Quit
    a              Toggle attendance session on/off
    r              Register a new face (interactive)
    c              Clear attendance records
    e              Export attendance to CSV
    s              Show current summary / statistics
    v              Toggle virtual-camera block on/off
    + / -          Adjust anti-spoof threshold

Architecture
────────────
* **Capture thread** — dedicated thread reads frames from the camera via
  OpenCV VideoCapture and pushes them to a thread-safe queue.  This
  prevents I/O latency from blocking the inference pipeline.
* **Main thread** — pops frames from the queue, runs the inference
  pipeline, and renders the UI overlay.
* **Graceful shutdown** — SIGINT / SIGTERM handlers ensure all resources
  (camera, ONNX sessions, threads) are released cleanly.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .anti_spoof import AntiSpoofEngine
from .attendance import AttendanceManager
from .detector import FaceDetector
from .recognition import FaceRecognizer
from .utils import (
    COLORS,
    FPSTracker,
    Config,
    draw_box,
    draw_hud,
    draw_instruction_bar,
    log,
)
from .virtual_cam_detector import VirtualCamDetector

# ──────────────────────── Constants ────────────────────────

FRAME_QUEUE_MAXSIZE = 4   # max buffered frames; prevents lag buildup
CAPTURE_TARGET_FPS = 30
REGISTRATION_SAMPLES = 8   # number of face captures for registration


# ═══════════════════════════════════════════════════════════════
#  Camera Capture Thread
# ═══════════════════════════════════════════════════════════════

class CameraThread:
    """
    Dedicated thread for reading frames from the camera.

    Pushes BGR frames into a thread-safe queue at the target FPS.
    Includes automatic reconnection on camera disconnect.
    """

    def __init__(
        self,
        camera_index: int,
        target_fps: int = CAPTURE_TARGET_FPS,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._index = camera_index
        self._target_fps = target_fps
        self._width = width
        self._height = height
        self._queue: Queue = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._reconnect_delay = 2.0  # seconds between reconnection attempts
        self.camera_opened = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraThread",
            daemon=True,
        )
        self._thread.start()
        log.info("Camera thread started (index=%d, target_fps=%d).", self._index, self._target_fps)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._release_cap()
        log.info("Camera thread stopped.")

    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """Get the latest frame from the queue.  Returns None if empty."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def get_capture(self) -> Optional[cv2.VideoCapture]:
        """Return the underlying VideoCapture for heuristic analysis."""
        return self._cap

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / self._target_fps

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._open_camera()
                if self._cap is None:
                    time.sleep(self._reconnect_delay)
                    continue

            ret, frame = self._cap.read()
            if not ret:
                log.warning("Camera read failed — attempting reconnect...")
                self._release_cap()
                time.sleep(self._reconnect_delay)
                continue

            self.camera_opened = True

            # Drain old frames to prevent lag
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass

            self._queue.put(frame)
            time.sleep(frame_interval)

    def _open_camera(self) -> None:
        self._release_cap()
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback to default backend
            self._cap = cv2.VideoCapture(self._index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info("Camera opened: %dx%d", actual_w, actual_h)

    def _release_cap(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.camera_opened = False


# ═══════════════════════════════════════════════════════════════
#  Face Registration Helper
# ═══════════════════════════════════════════════════════════════

def interactive_register(
    recognizer: FaceRecognizer,
    detector: FaceDetector,
    cam: CameraThread,
    samples: int = REGISTRATION_SAMPLES,
    callback=None,
) -> bool:
    """
    Capture multiple face samples and register them interactively.

    Shows a live preview and captures when a face is detected.
    Returns True if registration succeeded.
    """
    if not recognizer.ready:
        log.error("Recognizer not ready for registration.")
        return False

    name = input("  Enter name to register: ").strip()
    if not name:
        print("  [!] Name cannot be empty.")
        return False

    print(f"  Registering '{name}' — please face the camera directly.")
    print(f"  Capturing {samples} samples with slight head variations...\n")

    face_samples: List[Tuple[np.ndarray, List[Tuple[int, int]]]] = []
    captured = 0
    pause_frames = 30  # frames to wait between captures

    while captured < samples:
        frame = cam.get_frame(timeout=1.0)
        frame = cv2.flip(frame, 1)
        if frame is None:
            continue

        faces = detector.detect(frame)

        if not faces:
            # Show instruction
            display = frame.copy()
            cv2.putText(
                display, "No face detected — looking for face...",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                display, f"Sample {captured + 1}/{samples}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 210), 1, cv2.LINE_AA,
            )
            if callback:
                callback(display, captured, samples, name)
            cv2.imshow("FaceGuard — Registration", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
            continue

        # Use the largest face
        best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        x1, y1, x2, y2 = best["bbox"]

        # Add margin
        h, w = frame.shape[:2]
        margin_x = int((x2 - x1) * 0.20)
        margin_y = int((y2 - y1) * 0.25)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        crop = frame[y1:y2, x1:x2]  # Keep BGR — _run_onnx converts to RGB
        if crop.size == 0:
            continue

        raw_kps = best.get("landmarks", [])

        kps = []

        if len(raw_kps) >= 3:
            left_eye = raw_kps[1]
            right_eye = raw_kps[0]
            nose = raw_kps[2]

            kps = [
                (int(left_eye[0] - x1), int(left_eye[1] - y1)),
                (int(right_eye[0] - x1), int(right_eye[1] - y1)),
                (int(nose[0] - x1), int(nose[1] - y1)),
            ]

        # Show countdown between captures
        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(
            display, f"CAPTURED sample {captured + 1}/{samples}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2, cv2.LINE_AA,
        )
        if callback:
            callback(display, captured + 1, samples, name)
        cv2.imshow("FaceGuard — Registration", display)
        cv2.waitKey(500)

        face_samples.append((crop.copy(), kps))
        captured += 1

        if captured < samples:
            print(f"  [OK] Sample {captured}/{samples} captured. "
                  f"Please move your head slightly...")
            # Wait for pause_frames
            for _ in range(pause_frames):
                frame_tmp = cam.get_frame(timeout=0.05)
                if frame_tmp is not None:
                    if callback:
                        callback(frame_tmp, captured, samples, name)
                    cv2.imshow("FaceGuard — Registration", frame_tmp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return False

    # Register with the recognizer
    success = recognizer.register(name, face_samples)
    if success:
        print(f"\n  [OK] '{name}' registered successfully with {len(face_samples)} samples.\n")
    else:
        print(f"\n  [FAIL] Registration failed for '{name}'.\n")

    cv2.destroyWindow("FaceGuard — Registration")
    return success


# ═══════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════

class FaceGuardPipeline:
    """
    Orchestrates the full detection → anti-spoof → recognition →
    attendance → UI pipeline.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._fps_tracker = FPSTracker(window=30)

        # Components (initialised in start())
        self._cam: Optional[CameraThread] = None
        self._vcd: Optional[VirtualCamDetector] = None
        self._antispoof: Optional[AntiSpoofEngine] = None
        self._detector: Optional[FaceDetector] = None
        self._recognizer: Optional[FaceRecognizer] = None
        self._attendance: Optional[AttendanceManager] = None

        # State
        self._is_virtual = False
        self._skip_count = 0

        # Recognition stabilization
        self._face_memory = {}

    # ─────────── Initialisation ───────────

    def start(self) -> None:
        """Initialise all pipeline components."""
        log.info("=" * 60)
        log.info("FaceGuard AI — Starting...")
        log.info("=" * 60)

        # 1. Virtual Camera Detector
        log.info("[1/5] Virtual Camera Detector...")
        self._vcd = VirtualCamDetector(self._config)

        # 2. Camera Thread
        log.info("[2/5] Camera...")
        self._cam = CameraThread(
            camera_index=self._config.camera_index,
            target_fps=self._config.camera_fps_target,
            width=self._config.camera_width,
            height=self._config.camera_height,
        )
        self._cam.start()

        # Wait for camera to open
        for _ in range(50):
            if self._cam.camera_opened:
                break
            time.sleep(0.1)
        else:
            log.error("Camera failed to open after 5 seconds.")
            sys.exit(1)

        # Run virtual camera detection
        cap = self._cam.get_capture()
        self._is_virtual = self._vcd.run_full_check(self._config.camera_index, cap)
        if self._is_virtual:
            log.warning("╔══════════════════════════════════════════════╗")
            log.warning("║  VIRTUAL CAMERA DETECTED                  ║")
            log.warning("║  All frames will be BLOCKED.              ║")
            log.warning("║  Switch to a physical webcam to continue. ║")
            log.warning("╚══════════════════════════════════════════════╝")

        # 3. Anti-Spoof Engine
        log.info("[3/5] Anti-Spoof Engine...")
        self._antispoof = AntiSpoofEngine(self._config)
        self._antispoof.initialise()

        # 4. Face Detector (MediaPipe)
        log.info("[4/5] Face Detector (MediaPipe)...")
        self._detector = FaceDetector(self._config)
        if not self._detector.initialise():
            log.error("Face detector failed to initialise.")
            sys.exit(1)

        # 5. Face Recognizer (ArcFace ONNX)
        log.info("[5/5] Face Recognizer (ArcFace ONNX)...")
        self._recognizer = FaceRecognizer(self._config)
        if not self._recognizer.initialise():
            log.warning(
                "Face recognizer failed to initialise.  "
                "You can still register and use detection + anti-spoof."
            )

        # Attendance Manager
        self._attendance = AttendanceManager(self._config)

        log.info("=" * 60)
        log.info("FaceGuard AI — All systems ready.")
        log.info("=" * 60)

    # ─────────── Main Loop ───────────

    def run(self) -> None:
        """Run the main processing loop until quit."""
        if self._cam is None:
            return

        window_name = "FaceGuard AI — Attendance System"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                # ── Get frame ──
                frame = self._cam.get_frame(timeout=0.1)
                if frame is None:
                    continue

                self._fps_tracker.tick()
                self._skip_count += 1

                # Frame-skip for performance
                if self._config.skip_frames > 0:
                    if (self._skip_count - 1) % (self._config.skip_frames + 1) != 0:
                        # Still draw FPS and show the frame
                        display = self._render_no_detection(frame)
                        cv2.imshow(window_name, display)
                        if self._handle_keys() == "quit":
                            break
                        continue

                # ── Pipeline ──
                display = self._process_frame(frame)
                cv2.imshow(window_name, display)

                if self._handle_keys() == "quit":
                    break

        finally:
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full inference pipeline on one frame and render the
        result with bounding boxes, labels, and HUD.
        """

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # ── Virtual camera block ──
        if self._is_virtual:
            display = self._render_virtual_blocked(display)
            draw_hud(display, self._fps_tracker.fps, 0, True, False)
            return display

        # ── Face Detection ──
        if self._detector is None:
            return display

        faces = self._detector.detect(frame)

        # ── No faces ──
        if not faces:
            draw_hud(
                display,
                self._fps_tracker.fps,
                0,
                False,
                self._antispoof is not None and self._antispoof.model_loaded,
            )
            return display

        # ── Cleanup stale tracked faces ──
        now = time.time()

        expired = [
            fid for fid, data in self._face_memory.items()
            if now - data["last_seen"] > 2.0
        ]

        for fid in expired:
            del self._face_memory[fid]

        # ── Per-face pipeline ──
        n_attended = 0

        for face_info in faces:

            bbox = face_info["bbox"]
            x1, y1, x2, y2 = bbox
            conf = face_info["confidence"]

            # ─────────────────────────────────────────
            # Create simple face ID from bbox center
            # ─────────────────────────────────────────

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            face_id = None

            for fid, data in self._face_memory.items():

                old_x, old_y = data["center"]

                dist = ((cx - old_x) ** 2 + (cy - old_y) ** 2) ** 0.5

                if dist < 80:
                    face_id = fid
                    break

            if face_id is None:
                face_id = f"{cx}_{cy}_{time.time()}"

                self._face_memory[face_id] = {
                    "center": (cx, cy),
                    "stable_count": 0,
                    "last_name": None,
                    "last_seen": now,
                }

            memory = self._face_memory[face_id]

            memory["center"] = (cx, cy)
            memory["last_seen"] = now

            # ── Anti-Spoof ──

            spoof_result = None

            if self._antispoof is not None:
                spoof_result = self._antispoof.classify(frame, bbox)

            is_real = spoof_result["is_real"] if spoof_result else True
            spoof_label = spoof_result["label"] if spoof_result else "UNKNOWN"
            spoof_score = spoof_result["score"] if spoof_result else 1.0

            # ── Recognition ──

            name = "Unknown"
            rec_conf = 0.0

            if (
                is_real
                and self._attendance is not None
                and self._attendance.is_active
            ):

                if self._recognizer is not None and self._recognizer.ready:

                    face_h = y2 - y1
                    face_w = x2 - x1

                    # Add margin to match registration crop proportions
                    h, w = frame.shape[:2]
                    margin_x = int((x2 - x1) * 0.15)
                    margin_y = int((y2 - y1) * 0.20)
                    rx1 = max(0, x1 - margin_x)
                    ry1 = max(0, y1 - margin_y)
                    rx2 = min(w, x2 + margin_x)
                    ry2 = min(h, y2 + margin_y)

                    crop = frame[ry1:ry2, rx1:rx2]  # Keep BGR
                    if crop.size > 0 and face_h > 40 and face_w > 40:

                        raw_kps = face_info.get("landmarks", [])

                        kps = []

                        if len(raw_kps) >= 3:
                            left_eye = raw_kps[1]
                            right_eye = raw_kps[0]
                            nose = raw_kps[2]

                            kps = [
                                (
                                    int(left_eye[0] - rx1),
                                    int(left_eye[1] - ry1),
                                ),
                                (
                                    int(right_eye[0] - rx1),
                                    int(right_eye[1] - ry1),
                                ),
                                (
                                    int(nose[0] - rx1),
                                    int(nose[1] - ry1),
                                ),
                            ]

                        rec_result = self._recognizer.recognize(
                            crop,
                            keypoints=kps,
                        )

                        if rec_result is not None:

                            candidate_name = rec_result["name"]
                            candidate_conf = rec_result["confidence"]

                            if (
                                candidate_conf
                                < self._config.recognition_threshold
                            ):

                                memory["stable_count"] = 0
                                name = "Unknown"

                            else:

                                # Same identity stabilized
                                if candidate_name == memory["last_name"]:
                                    memory["stable_count"] += 1
                                else:
                                    memory["stable_count"] = 1

                                memory["last_name"] = candidate_name

                                # Require stable frames
                                if memory["stable_count"] >= 3:

                                    name = candidate_name
                                    rec_conf = candidate_conf

                                    result = self._attendance.mark_attendance(
                                        name,
                                        rec_conf,
                                        spoof_score,
                                    )

                                    if result["marked"]:
                                        n_attended += 1

                                else:
                                    name = "Stabilizing..."

                else:

                    if is_real:
                        name = "Detected"

            elif (
                not is_real
                and self._attendance is not None
                and self._attendance.is_active
            ):

                self._attendance.mark_rejected(
                    None,
                    "spoof",
                    conf,
                    spoof_score,
                )

            # ── Draw Box ──

            status = (
                "real"
                if is_real
                else (
                    "spoof"
                    if spoof_label == "SPOOF"
                    else "unknown"
                )
            )

            draw_box(
                display,
                x1,
                y1,
                x2,
                y2,
                label=name,
                score=rec_conf if rec_conf > 0 else conf,
                status=status,
                thickness=self._config.box_thickness,
            )

        # ── HUD ──

        draw_hud(
            display,
            self._fps_tracker.fps,
            len(faces),
            False,
            self._antispoof is not None
            and self._antispoof.model_loaded,
        )

        # ── Session indicator ──

        if (
            self._attendance is not None
            and self._attendance.is_active
        ):

            cv2.putText(
                display,
                "SESSION ACTIVE",
                (display.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLORS["real"],
                2,
                cv2.LINE_AA,
            )

        # ── Bottom instructions ──

        lines = [
            "[A]ttendance  [R]egister  [C]lear  [E]xport  [S]ummary  [Q]uit"
        ]

        draw_instruction_bar(display, lines)

        return display

    # ─────────── Renderers ───────────

    def _render_no_detection(self, frame: np.ndarray) -> np.ndarray:
        """Render frame when no detection is performed (frame-skipped)."""
        display = frame.copy()
        draw_hud(
            display,
            self._fps_tracker.fps,
            0,
            False,
            False,
        )
        lines = ["[A]ttendance  [R]egister  [C]lear  [E]xport  [S]ummary  [Q]uit"]
        draw_instruction_bar(display, lines)
        return display

    def _render_virtual_blocked(self, frame: np.ndarray) -> np.ndarray:
        """Render a full-screen warning when a virtual camera is detected."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Semi-transparent red overlay
        overlay = display.copy()
        overlay[:] = (0, 0, 50)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

        # Warning text
        cv2.putText(
            display, "VIRTUAL CAMERA BLOCKED",
            (w // 2 - 220, h // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display, "Switch to a physical webcam",
            (w // 2 - 170, h // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )

        return display

    # ─────────── Keyboard Handler ───────────

    def _handle_keys(self) -> Optional[str]:
        """Process keyboard input.  Returns 'quit' if user wants to exit."""
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            return "quit"

        elif key == ord("a"):
            self._toggle_attendance()

        elif key == ord("r"):
            if self._detector is not None and self._recognizer is not None:
                interactive_register(
                    self._recognizer, self._detector, self._cam,
                )

        elif key == ord("c"):
            if self._attendance is not None:
                self._attendance.clear_records()
                log.info("Attendance records cleared.")

        elif key == ord("e"):
            if self._attendance is not None:
                path = self._attendance.export_csv()
                if path:
                    log.info("Exported to: %s", path)
                else:
                    log.warning("No records to export.")

        elif key == ord("s"):
            self._print_summary()

        elif key == ord("v"):
            self._toggle_virtual_block()

        elif key in (ord("+"), ord("=")):
            if self._antispoof is not None:
                self._config.antispoof_threshold = min(
                    0.95, self._config.antispoof_threshold + 0.05
                )
                log.info("Anti-spoof threshold: %.2f", self._config.antispoof_threshold)

        elif key == ord("-"):
            if self._antispoof is not None:
                self._config.antispoof_threshold = max(
                    0.10, self._config.antispoof_threshold - 0.05
                )
                log.info("Anti-spoof threshold: %.2f", self._config.antispoof_threshold)

        return None

    def _toggle_attendance(self) -> None:
        if self._attendance is None:
            return
        if self._attendance.is_active:
            self._attendance.stop_session()
            if self._antispoof is not None:
                self._antispoof.reset_temporal()
            log.info("Attendance session STOPPED.")
        else:
            if self._recognizer is not None and self._recognizer.database is not None:
                if self._recognizer.database.count == 0:
                    log.warning("No registered faces.  Register someone first (press R).")
                    return
            self._attendance.start_session()
            log.info("Attendance session STARTED.")

    def _toggle_virtual_block(self) -> None:
        self._is_virtual = not self._is_virtual
        state = "BLOCKED" if self._is_virtual else "ALLOWED"
        log.info("Virtual camera block: %s", state)

    def _print_summary(self) -> None:
        if self._attendance is None:
            return
        registered = (
            self._recognizer.database.names
            if self._recognizer is not None and self._recognizer.database is not None
            else []
        )
        summary = self._attendance.get_summary(registered)
        log.info(
            "Summary: %d/%d present (%.1f%%), %d absent, %d rejected",
            summary["unique_present"],
            len(registered),
            summary["attendance_rate"],
            summary["total_absent"],
            summary["total_rejected"],
        )
        if summary["present_names"]:
            log.info("Present: %s", ", ".join(summary["present_names"]))
        if summary["absent_names"]:
            log.info("Absent: %s", ", ".join(summary["absent_names"]))

    # ─────────── Cleanup ───────────

    def shutdown(self) -> None:
        """Gracefully shut down all components."""
        log.info("Shutting down FaceGuard AI...")

        if self._attendance is not None and self._attendance.is_active:
            self._attendance.stop_session()

        if self._cam is not None:
            self._cam.stop()

        if self._detector is not None:
            self._detector.close()

        if self._recognizer is not None:
            self._recognizer.close()

        log.info("FaceGuard AI — Shutdown complete.")


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaceGuard AI — Production Face Attendance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard Controls:
  q / ESC     Quit
  a           Toggle attendance session
  r           Register a new face
  c           Clear attendance records
  e           Export attendance to CSV
  s           Show attendance summary
  v           Toggle virtual-camera block
  + / -       Adjust anti-spoof threshold
        """,
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Camera width (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Camera height (default: 480)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Face recognition cosine similarity threshold (default: 0.55)",
    )
    parser.add_argument(
        "--spoof-threshold", type=float, default=0.50,
        help="Anti-spoof real-score threshold (default: 0.50)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--cooldown", type=int, default=300,
        help="Attendance cooldown in seconds (default: 300)",
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Process every N+1 frames (default: 0 = every frame)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.json file",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch with graphical interface (tkinter)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config (CLI args override file config)
    config = Config.load(args.config)

    # Apply CLI overrides
    config.camera_index = args.camera
    config.camera_width = args.width
    config.camera_height = args.height
    config.recognition_threshold = args.threshold
    config.antispoof_threshold = args.spoof_threshold
    config.use_gpu = not args.no_gpu
    config.cooldown_seconds = args.cooldown
    config.skip_frames = args.skip_frames

    if args.gui:
        from .gui import FaceGuardGUI
        app = FaceGuardGUI(config)
        app.run()
    else:
        # Create and start the pipeline
        pipeline = FaceGuardPipeline(config)

        # Graceful shutdown on signals
        def _signal_handler(sig, frame):
            pipeline.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            pipeline.start()
            pipeline.run()
        except KeyboardInterrupt:
            pass
        finally:
            pipeline.shutdown()


if __name__ == "__main__":
    main()
