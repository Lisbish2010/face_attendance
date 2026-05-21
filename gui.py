"""
src/gui.py — FaceGuard AI: Graphical User Interface.

A professional dark-themed tkinter GUI that wraps the FaceGuard pipeline
components.  Replaces the OpenCV cv2.imshow() loop with a native desktop
window containing:

    * Live camera feed with detection overlays
    * Registration dialog with live preview
    * Attendance session controls
    * Real-time status indicators
    * Attendance log table
    * Registered faces management

Usage
─────
    python run.py --gui
    python run.py --gui --camera 0 --width 1280 --height 720
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .anti_spoof import AntiSpoofEngine
from .attendance import AttendanceManager
from .detector import FaceDetector
from .recognition import FaceRecognizer
from .utils import COLORS, FPSTracker, Config, draw_box, draw_hud, log
from .virtual_cam_detector import VirtualCamDetector

# ──────────────────────── tkinter imports ────────────────────────

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageTk


# ═══════════════════════════════════════════════════════════════
#  Dark Theme Configuration
# ═══════════════════════════════════════════════════════════════

# Colour palette — designed to match FaceGuard branding
DARK_BG = "#12121e"
DARK_BG2 = "#1a1a2e"
DARK_BG3 = "#222240"
ACCENT = "#00dcc8"
ACCENT2 = "#00aaff"
TEXT_PRIMARY = "#e0e0f0"
TEXT_SECONDARY = "#8888aa"
TEXT_DIM = "#555570"
GREEN = "#00dc78"
RED = "#dc3030"
ORANGE = "#ffa500"
BORDER = "#333355"
BTN_BG = "#282848"
BTN_HOVER = "#383868"
BTN_ACTIVE = "#00dcc8"
ENTRY_BG = "#1e1e38"
ENTRY_FG = "#e0e0f0"
SCROLLBAR_BG = "#2a2a4a"
SCROLLBAR_FG = "#555570"
TABLE_HEADER = "#282848"
TABLE_ROW1 = "#16162a"
TABLE_ROW2 = "#1c1c36"

REGISTRATION_SAMPLES = 8


# ═══════════════════════════════════════════════════════════════
#  Main GUI Application
# ═══════════════════════════════════════════════════════════════


class FaceGuardGUI:
    """
    Complete GUI wrapper for FaceGuard AI.

    Manages the full lifecycle: initialisation → live preview → attendance
    → registration → export, all through clickable buttons and sliders.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._fps_tracker = FPSTracker(window=30)

        # Pipeline components
        self._cam = None
        self._vcd: Optional[VirtualCamDetector] = None
        self._antispoof: Optional[AntiSpoofEngine] = None
        self._detector: Optional[FaceDetector] = None
        self._recognizer: Optional[FaceRecognizer] = None
        self._attendance: Optional[AttendanceManager] = None

        # State
        self._is_virtual = False
        self._running = False
        self._pipeline_running = False
        self._skip_count = 0
        self._face_memory: Dict[str, Dict] = {}
        self._registering = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_display: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._last_attendance_count = 0
        self._log_update_pending = False

        # GUI references
        self._root: Optional[tk.Tk] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._camera_label: Optional[tk.Label] = None

    # ═══════════════════════════════════════════════════════════
    #  Initialisation
    # ═══════════════════════════════════════════════════════════

    def _init_pipeline(self) -> bool:
        """Initialise all pipeline components. Returns True on success."""
        log.info("=" * 60)
        log.info("FaceGuard AI — Starting (GUI Mode)...")
        log.info("=" * 60)

        # 1. Virtual Camera Detector
        self._status("Initialising Virtual Camera Detector...")
        self._vcd = VirtualCamDetector(self._config)

        # 2. Camera Thread
        self._status("Initialising Camera...")
        from .main import CameraThread
        self._cam = CameraThread(
            camera_index=self._config.camera_index,
            target_fps=self._config.camera_fps_target,
            width=self._config.camera_width,
            height=self._config.camera_height,
        )
        self._cam.start()

        # Wait for camera
        for _ in range(50):
            if self._cam.camera_opened:
                break
            time.sleep(0.1)
        else:
            log.error("Camera failed to open.")
            self._status("ERROR: Camera failed to open!", RED)
            return False

        # Virtual cam check
        cap = self._cam.get_capture()
        self._is_virtual = self._vcd.run_full_check(self._config.camera_index, cap)
        if self._is_virtual:
            self._update_status_indicator("virtual_cam", "BLOCKED", RED)

        # 3. Anti-Spoof
        self._status("Initialising Anti-Spoof Engine...")
        self._antispoof = AntiSpoofEngine(self._config)
        self._antispoof.initialise()
        if self._antispoof.model_loaded:
            self._update_status_indicator("antispoof", "ONNX Model", GREEN)
        else:
            self._update_status_indicator("antispoof", "Heuristic Only", ORANGE)

        # 4. Face Detector
        self._status("Initialising Face Detector (MediaPipe)...")
        self._detector = FaceDetector(self._config)
        if not self._detector.initialise():
            self._status("ERROR: Face detector failed!", RED)
            return False
        self._update_status_indicator("detector", "Active", GREEN)

        # 5. Face Recognizer
        self._status("Initialising Face Recognizer (ArcFace ONNX)...")
        self._recognizer = FaceRecognizer(self._config)
        if not self._recognizer.initialise():
            self._status("WARNING: Recognizer failed to init.", ORANGE)
            self._update_status_indicator("recognizer", "Not Ready", RED)
        else:
            self._update_status_indicator("recognizer", "Active", GREEN)

        # Attendance Manager
        self._attendance = AttendanceManager(self._config)

        # Update registered faces list
        self._update_registered_list()

        log.info("All systems ready.")
        self._status("All systems ready.", GREEN)
        return True

    # ═══════════════════════════════════════════════════════════
    #  GUI Building
    # ═══════════════════════════════════════════════════════════

    def _build_gui(self) -> None:
        """Create the main tkinter window and all widgets."""
        self._root = tk.Tk()
        self._root.title("FaceGuard AI — Attendance System")
        self._root.configure(bg=DARK_BG)
        self._root.resizable(True, True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Try to set DPI awareness on Windows
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        # Configure ttk styles
        self._setup_styles()

        # ─── Main container ───
        main_frame = tk.Frame(self._root, bg=DARK_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ─── Top row: Camera + Controls ───
        top_frame = tk.Frame(main_frame, bg=DARK_BG)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # Camera feed (left)
        camera_frame = tk.LabelFrame(
            top_frame, text="  LIVE CAMERA FEED  ",
            bg=DARK_BG2, fg=ACCENT, font=("Segoe UI", 11, "bold"),
            bd=1, relief=tk.GROOVE, highlightbackground=BORDER,
        )
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self._camera_label = tk.Label(
            camera_frame, bg=DARK_BG3,
            text="Initialising camera...",
            fg=TEXT_SECONDARY, font=("Segoe UI", 14),
        )
        self._camera_label.pack(padx=4, pady=4, fill=tk.BOTH, expand=True)

        # Right panel (controls + status)
        right_frame = tk.Frame(top_frame, bg=DARK_BG, width=310)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        right_frame.pack_propagate(False)

        self._build_controls(right_frame)
        self._build_status_panel(right_frame)
        self._build_registered_panel(right_frame)

        # ─── Bottom: Attendance log ───
        self._build_attendance_log(main_frame)

        # ─── Status bar ───
        self._status_bar_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            self._root, textvariable=self._status_bar_var,
            bg=DARK_BG3, fg=TEXT_SECONDARY, font=("Consolas", 9),
            anchor=tk.W, padx=10, pady=3,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_styles(self) -> None:
        """Configure ttk styles for dark theme."""
        style = ttk.Style()

        # Try to use 'clam' theme as base for customisation
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")

        # Treeview (attendance table)
        style.configure(
            "Dark.Treeview",
            background=TABLE_ROW1,
            foreground=TEXT_PRIMARY,
            fieldbackground=TABLE_ROW1,
            borderwidth=0,
            font=("Consolas", 9),
            rowheight=24,
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=TABLE_HEADER,
            foreground=ACCENT,
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
            relief=tk.FLAT,
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", ACCENT)],
            foreground=[("selected", DARK_BG)],
        )

        # Scrollbar
        style.configure(
            "Dark.Vertical.TScrollbar",
            background=SCROLLBAR_BG,
            troughcolor=DARK_BG2,
            borderwidth=0,
            arrowcolor=TEXT_SECONDARY,
        )
        style.map(
            "Dark.Vertical.TScrollbar",
            background=[("active", SCROLLBAR_FG)],
        )

        # Scale (sliders)
        style.configure(
            "Dark.Horizontal.TScale",
            background=DARK_BG,
            troughcolor=DARK_BG3,
            borderwidth=0,
        )

        # Separator
        style.configure(
            "Dark.TSeparator",
            background=BORDER,
        )

    def _make_button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        color: str = ACCENT,
        width: int = 28,
        font_size: int = 10,
    ) -> tk.Button:
        """Create a styled dark button."""
        btn = tk.Button(
            parent, text=text, command=command,
            bg=BTN_BG, fg=TEXT_PRIMARY,
            activebackground=BTN_HOVER, activeforeground=TEXT_PRIMARY,
            font=("Segoe UI", font_size, "bold"),
            relief=tk.FLAT, bd=0, padx=8, pady=5,
            cursor="hand2", width=width,
        )

        def _on_enter(e):
            btn.configure(bg=BTN_HOVER)

        def _on_leave(e):
            btn.configure(bg=BTN_BG)

        btn.bind("<Enter>", _on_enter)
        btn.bind("<Leave>", _on_leave)
        return btn

    # ─────────── Control Panel ───────────

    def _build_controls(self, parent: tk.Frame) -> None:
        """Build the control buttons section."""
        ctrl_frame = tk.LabelFrame(
            parent, text="  CONTROLS  ",
            bg=DARK_BG2, fg=ACCENT, font=("Segoe UI", 10, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        ctrl_frame.pack(fill=tk.X, pady=(0, 4))

        inner = tk.Frame(ctrl_frame, bg=DARK_BG2)
        inner.pack(fill=tk.X, padx=8, pady=6)

        # Attendance session toggle
        self._session_btn_text = tk.StringVar(value="▶  Start Attendance")
        self._session_btn = self._make_button(
            inner, "", self._toggle_attendance_session,
            color=GREEN, width=30,
        )
        self._session_btn.configure(textvariable=self._session_btn_text)
        self._session_btn.pack(fill=tk.X, pady=2)

        # Register face
        reg_btn = self._make_button(
            inner, "+  Register Face", self._gui_register_face,
            color=ACCENT2, width=30,
        )
        reg_btn.pack(fill=tk.X, pady=2)

        # Separator
        ttk.Separator(inner, orient=tk.HORIZONTAL, style="Dark.TSeparator").pack(
            fill=tk.X, pady=6
        )

        # Threshold sliders
        # Recognition threshold
        tk.Label(
            inner, text="Recognition Threshold",
            bg=DARK_BG2, fg=TEXT_SECONDARY, font=("Segoe UI", 9),
        ).pack(anchor=tk.W)

        threshold_frame = tk.Frame(inner, bg=DARK_BG2)
        threshold_frame.pack(fill=tk.X, pady=(0, 4))

        self._rec_threshold_var = tk.DoubleVar(value=self._config.recognition_threshold)
        self._rec_threshold_label = tk.Label(
            threshold_frame, text=f"{self._config.recognition_threshold:.2f}",
            bg=DARK_BG2, fg=ACCENT, font=("Consolas", 10, "bold"), width=5,
        )
        self._rec_threshold_label.pack(side=tk.RIGHT, padx=(4, 0))

        rec_scale = ttk.Scale(
            threshold_frame, from_=0.1, to=0.95,
            variable=self._rec_threshold_var, orient=tk.HORIZONTAL,
            style="Dark.Horizontal.TScale",
            command=self._on_rec_threshold_change,
        )
        rec_scale.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # Anti-spoof threshold
        tk.Label(
            inner, text="Anti-Spoof Threshold",
            bg=DARK_BG2, fg=TEXT_SECONDARY, font=("Segoe UI", 9),
        ).pack(anchor=tk.W)

        spoof_frame = tk.Frame(inner, bg=DARK_BG2)
        spoof_frame.pack(fill=tk.X, pady=(0, 4))

        self._spoof_threshold_var = tk.DoubleVar(value=self._config.antispoof_threshold)
        self._spoof_threshold_label = tk.Label(
            spoof_frame, text=f"{self._config.antispoof_threshold:.2f}",
            bg=DARK_BG2, fg=ACCENT, font=("Consolas", 10, "bold"), width=5,
        )
        self._spoof_threshold_label.pack(side=tk.RIGHT, padx=(4, 0))

        spoof_scale = ttk.Scale(
            spoof_frame, from_=0.1, to=0.95,
            variable=self._spoof_threshold_var, orient=tk.HORIZONTAL,
            style="Dark.Horizontal.TScale",
            command=self._on_spoof_threshold_change,
        )
        spoof_scale.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # Separator
        ttk.Separator(inner, orient=tk.HORIZONTAL, style="Dark.TSeparator").pack(
            fill=tk.X, pady=6
        )

        # Action buttons row
        btn_row = tk.Frame(inner, bg=DARK_BG2)
        btn_row.pack(fill=tk.X)

        self._make_button(
            btn_row, "Clear", self._clear_records,
            color=RED, width=12, font_size=9,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        self._make_button(
            btn_row, "Export CSV", self._export_csv,
            color=ORANGE, width=12, font_size=9,
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))

        self._make_button(
            btn_row, "Summary", self._show_summary,
            color=ACCENT2, width=12, font_size=9,
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))

    # ─────────── Status Panel ───────────

    def _build_status_panel(self, parent: tk.Frame) -> None:
        """Build the system status indicators section."""
        status_frame = tk.LabelFrame(
            parent, text="  SYSTEM STATUS  ",
            bg=DARK_BG2, fg=ACCENT, font=("Segoe UI", 10, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        status_frame.pack(fill=tk.X, pady=4)

        inner = tk.Frame(status_frame, bg=DARK_BG2)
        inner.pack(fill=tk.X, padx=8, pady=6)

        # Status indicators stored in dict for easy updates
        self._status_labels: Dict[str, tk.Label] = {}

        indicators = [
            ("fps", "FPS", "0.0"),
            ("faces", "Faces Detected", "0"),
            ("camera", "Camera", "Initialising..."),
            ("virtual_cam", "Virtual Cam", "Checking..."),
            ("antispoof", "Anti-Spoof", "Loading..."),
            ("detector", "Face Detector", "Loading..."),
            ("recognizer", "Recognizer", "Loading..."),
            ("session", "Attendance Session", "Stopped"),
            ("registered", "Registered Faces", "0"),
        ]

        for i, (key, label_text, default_val) in enumerate(indicators):
            row = tk.Frame(inner, bg=DARK_BG2)
            row.pack(fill=tk.X, pady=1)

            # Key label
            tk.Label(
                row, text=f"{label_text}:",
                bg=DARK_BG2, fg=TEXT_SECONDARY, font=("Segoe UI", 9),
                anchor=tk.W,
            ).pack(side=tk.LEFT)

            # Value label
            val_label = tk.Label(
                row, text=default_val,
                bg=DARK_BG2, fg=TEXT_DIM, font=("Consolas", 9, "bold"),
                anchor=tk.E,
            )
            val_label.pack(side=tk.RIGHT)
            self._status_labels[key] = val_label

    # ─────────── Registered Faces Panel ───────────

    def _build_registered_panel(self, parent: tk.Frame) -> None:
        """Build the registered faces list with remove button."""
        reg_frame = tk.LabelFrame(
            parent, text="  REGISTERED FACES  ",
            bg=DARK_BG2, fg=ACCENT, font=("Segoe UI", 10, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        reg_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        inner = tk.Frame(reg_frame, bg=DARK_BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Listbox for registered names
        list_frame = tk.Frame(inner, bg=DARK_BG2)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._registered_listbox = tk.Listbox(
            list_frame,
            bg=ENTRY_BG, fg=TEXT_PRIMARY,
            selectbackground=ACCENT, selectforeground=DARK_BG,
            font=("Consolas", 10),
            borderwidth=0, highlightthickness=1,
            highlightcolor=BORDER, highlightbackground=BORDER,
            height=4,
        )
        self._registered_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL,
            command=self._registered_listbox.yview,
            style="Dark.Vertical.TScrollbar",
        )
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self._registered_listbox.configure(yscrollcommand=scrollbar.set)

        # Remove button
        self._make_button(
            inner, "Remove Selected", self._remove_selected_face,
            color=RED, width=28, font_size=9,
        ).pack(fill=tk.X, pady=(6, 0))

    # ─────────── Attendance Log ───────────

    def _build_attendance_log(self, parent: tk.Frame) -> None:
        """Build the attendance records table."""
        log_frame = tk.LabelFrame(
            parent, text="  ATTENDANCE LOG  ",
            bg=DARK_BG2, fg=ACCENT, font=("Segoe UI", 10, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        # Treeview
        columns = ("name", "time", "confidence", "spoof", "status")
        self._tree = ttk.Treeview(
            log_frame, columns=columns, show="headings",
            height=6, style="Dark.Treeview",
        )

        self._tree.heading("name", text="Name")
        self._tree.heading("time", text="Time")
        self._tree.heading("confidence", text="Confidence")
        self._tree.heading("spoof", text="Spoof Score")
        self._tree.heading("status", text="Status")

        self._tree.column("name", width=160, anchor=tk.W)
        self._tree.column("time", width=170, anchor=tk.W)
        self._tree.column("confidence", width=90, anchor=tk.CENTER)
        self._tree.column("spoof", width=90, anchor=tk.CENTER)
        self._tree.column("status", width=140, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL,
            command=self._tree.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self._tree.configure(yscrollcommand=tree_scroll.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)

        # Tag colours for status
        self._tree.tag_configure("PRESENT", foreground=GREEN)
        self._tree.tag_configure("REJECTED", foreground=RED)
        self._tree.tag_configure("default", foreground=TEXT_PRIMARY)

    # ═══════════════════════════════════════════════════════════
    #  Pipeline Thread
    # ═══════════════════════════════════════════════════════════

    def _pipeline_loop(self) -> None:
        """Run the inference pipeline in a background thread."""
        self._pipeline_running = True

        while self._pipeline_running:
            if self._cam is None:
                time.sleep(0.1)
                continue

            frame = self._cam.get_frame(timeout=0.1)
            if frame is None:
                continue

            self._fps_tracker.tick()
            self._skip_count += 1

            # Frame skip
            if self._config.skip_frames > 0:
                if (self._skip_count - 1) % (self._config.skip_frames + 1) != 0:
                    with self._frame_lock:
                        self._latest_frame = frame
                    continue

            # Process frame
            display = self._process_frame(frame)

            with self._frame_lock:
                self._latest_display = display
                self._latest_frame = frame

        log.info("Pipeline thread exited.")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run the full inference pipeline on one frame."""
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Virtual camera block
        if self._is_virtual:
            display = self._render_virtual_blocked(display)
            draw_hud(display, self._fps_tracker.fps, 0, True, False)
            return display

        # Face Detection
        if self._detector is None:
            return display

        faces = self._detector.detect(frame)

        # No faces
        if not faces:
            draw_hud(
                display, self._fps_tracker.fps, 0, False,
                self._antispoof is not None and self._antispoof.model_loaded,
            )
            # Session indicator
            if self._attendance is not None and self._attendance.is_active:
                cv2.putText(
                    display, "SESSION ACTIVE",
                    (display.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["real"], 2, cv2.LINE_AA,
                )
            return display

        # Cleanup stale faces
        now = time.time()
        expired = [
            fid for fid, data in self._face_memory.items()
            if now - data["last_seen"] > 2.0
        ]
        for fid in expired:
            del self._face_memory[fid]

        # Per-face pipeline
        for face_info in faces:
            bbox = face_info["bbox"]
            x1, y1, x2, y2 = bbox
            conf = face_info["confidence"]

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

            # Anti-Spoof
            spoof_result = None
            if self._antispoof is not None:
                spoof_result = self._antispoof.classify(frame, bbox)

            is_real = spoof_result["is_real"] if spoof_result else True
            spoof_label = spoof_result["label"] if spoof_result else "UNKNOWN"
            spoof_score = spoof_result["score"] if spoof_result else 1.0

            # Recognition
            name = "Unknown"
            rec_conf = 0.0

            if is_real and self._attendance is not None and self._attendance.is_active:
                if self._recognizer is not None and self._recognizer.ready:
                    face_h = y2 - y1
                    face_w = x2 - x1
                    # Add margin to match registration crop proportions
                    fh, fw = frame.shape[:2]
                    margin_x = int((x2 - x1) * 0.15)
                    margin_y = int((y2 - y1) * 0.20)
                    rx1 = max(0, x1 - margin_x)
                    ry1 = max(0, y1 - margin_y)
                    rx2 = min(fw, x2 + margin_x)
                    ry2 = min(fh, y2 + margin_y)
                    crop = frame[ry1:ry2, rx1:rx2]  # Keep BGR
                    if crop.size > 0 and face_h > 40 and face_w > 40:
                        raw_kps = face_info.get("landmarks", [])
                        kps = []
                        if len(raw_kps) >= 3:
                            left_eye = raw_kps[1]
                            right_eye = raw_kps[0]
                            nose = raw_kps[2]
                            kps = [
                                (int(left_eye[0] - rx1), int(left_eye[1] - ry1)),
                                (int(right_eye[0] - rx1), int(right_eye[1] - ry1)),
                                (int(nose[0] - rx1), int(nose[1] - ry1)),
                            ]

                        rec_result = self._recognizer.recognize(crop, keypoints=kps)
                        if rec_result is not None:
                            candidate_name = rec_result["name"]
                            candidate_conf = rec_result["confidence"]
                            if candidate_conf < self._config.recognition_threshold:
                                memory["stable_count"] = 0
                                name = "Unknown"
                            else:
                                if candidate_name == memory["last_name"]:
                                    memory["stable_count"] += 1
                                else:
                                    memory["stable_count"] = 1
                                memory["last_name"] = candidate_name
                                if memory["stable_count"] >= 3:
                                    name = candidate_name
                                    rec_conf = candidate_conf
                                    self._attendance.mark_attendance(
                                        name, rec_conf, spoof_score,
                                    )
                                else:
                                    name = "Stabilizing..."
                else:
                    if is_real:
                        name = "Detected"

            elif (
                not is_real and self._attendance is not None
                and self._attendance.is_active
            ):
                self._attendance.mark_rejected(None, "spoof", conf, spoof_score)

            # Draw box
            status = (
                "real" if is_real
                else ("spoof" if spoof_label == "SPOOF" else "unknown")
            )
            draw_box(
                display, x1, y1, x2, y2,
                label=name, score=rec_conf if rec_conf > 0 else conf,
                status=status, thickness=self._config.box_thickness,
            )

        # HUD
        draw_hud(
            display, self._fps_tracker.fps, len(faces), False,
            self._antispoof is not None and self._antispoof.model_loaded,
        )

        # Session indicator
        if self._attendance is not None and self._attendance.is_active:
            cv2.putText(
                display, "SESSION ACTIVE",
                (display.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["real"], 2, cv2.LINE_AA,
            )

        return display

    def _render_virtual_blocked(self, frame: np.ndarray) -> np.ndarray:
        """Render a virtual camera warning."""
        display = frame.copy()
        h, w = display.shape[:2]
        overlay = display.copy()
        overlay[:] = (0, 0, 50)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.putText(
            display, "VIRTUAL CAMERA BLOCKED",
            (w // 2 - 220, h // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA,
        )
        cv2.putText(
            display, "Switch to a physical webcam",
            (w // 2 - 170, h // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 210), 1, cv2.LINE_AA,
        )
        return display

    # ═══════════════════════════════════════════════════════════
    #  GUI Update Loop (runs on main thread via after())
    # ═══════════════════════════════════════════════════════════

    def _update_display(self) -> None:
        """Periodically update the camera feed and status labels."""
        if not self._running:
            return

        # Update camera feed
        with self._frame_lock:
            display = self._latest_display

        if display is not None and self._camera_label is not None:
            try:
                # Convert BGR to RGB, then to PIL, then to ImageTk
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Scale to fit label while preserving aspect ratio
                if self._camera_label.winfo_width() > 1 and self._camera_label.winfo_height() > 1:
                    label_w = self._camera_label.winfo_width() - 8
                    label_h = self._camera_label.winfo_height() - 8
                    img_w, img_h = pil_img.size
                    scale = min(label_w / img_w, label_h / img_h)
                    if scale < 1.0:
                        new_w = int(img_w * scale)
                        new_h = int(img_h * scale)
                        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

                self._photo = ImageTk.PhotoImage(pil_img)
                self._camera_label.configure(image=self._photo, text="")
            except Exception:
                pass

        # Update status indicators
        fps = self._fps_tracker.fps
        self._update_status_indicator("fps", f"{fps:.1f}", GREEN if fps > 10 else ORANGE)

        # Update attendance log if new records
        if self._attendance is not None:
            n_records = len(self._attendance.records)
            if n_records != self._last_attendance_count:
                self._last_attendance_count = n_records
                self._refresh_attendance_table()

        # Update registered count
        if self._recognizer is not None and self._recognizer.database is not None:
            count = self._recognizer.database.count
            self._update_status_indicator("registered", str(count), TEXT_PRIMARY)

        # Schedule next update (~30 FPS)
        if self._running:
            self._root.after(33, self._update_display)

    # ═══════════════════════════════════════════════════════════
    #  Event Handlers
    # ═══════════════════════════════════════════════════════════

    def _toggle_attendance_session(self) -> None:
        """Start or stop the attendance session."""
        if self._attendance is None:
            return

        if self._attendance.is_active:
            self._attendance.stop_session()
            if self._antispoof is not None:
                self._antispoof.reset_temporal()
            self._session_btn_text.set("▶  Start Attendance")
            self._update_status_indicator("session", "Stopped", TEXT_DIM)
            self._status("Attendance session STOPPED.", ORANGE)
        else:
            # Check if faces are registered
            if self._recognizer is not None and self._recognizer.database is not None:
                if self._recognizer.database.count == 0:
                    messagebox.showwarning(
                        "No Registered Faces",
                        "Please register at least one face before "
                        "starting an attendance session.\n\n"
                        "Click 'Register Face' to add a person."
                    )
                    return
            self._attendance.start_session()
            self._session_btn_text.set("■  Stop Attendance")
            self._update_status_indicator("session", "ACTIVE", GREEN)
            self._status("Attendance session STARTED.", GREEN)

    def _gui_register_face(self) -> None:
        """Open a registration dialog with live camera preview."""
        if self._registering:
            return
        if self._detector is None or self._recognizer is None:
            messagebox.showerror("Error", "Detector or Recognizer not ready.")
            return
        if not self._recognizer.ready:
            messagebox.showerror("Error", "Recognizer not ready. Cannot register faces.")
            return

        self._registering = True
        self._status("Opening registration dialog...", ACCENT2)

        # Create registration dialog
        self._reg_window = tk.Toplevel(self._root)
        self._reg_window.title("FaceGuard — Register New Face")
        self._reg_window.configure(bg=DARK_BG)
        self._reg_window.resizable(False, True)
        self._reg_window.transient(self._root)
        self._reg_window.grab_set()
        self._reg_window.protocol("WM_DELETE_WINDOW", self._cancel_registration)

        # Center on parent
        self._reg_window.update_idletasks()
        pw = self._root.winfo_width()
        ph = self._root.winfo_height()
        px = self._root.winfo_x()
        py = self._root.winfo_y()
        ww = 520
        wh = 560
        x = px + (pw - ww) // 2
        y = py + (ph - wh) // 2
        self._reg_window.geometry(f"{ww}x{wh}+{x}+{y}")

        # Title
        tk.Label(
            self._reg_window, text="REGISTER NEW FACE",
            bg=DARK_BG, fg=ACCENT, font=("Segoe UI", 14, "bold"),
        ).pack(pady=(12, 4))

        tk.Label(
            self._reg_window,
            text="Enter a name and face the camera directly.",
            bg=DARK_BG, fg=TEXT_SECONDARY, font=("Segoe UI", 9),
        ).pack(pady=(0, 8))

        # Name entry
        name_frame = tk.Frame(self._reg_window, bg=DARK_BG)
        name_frame.pack(fill=tk.X, padx=20)

        tk.Label(
            name_frame, text="Name:",
            bg=DARK_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 11),
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._reg_name_var = tk.StringVar()
        name_entry = tk.Entry(
            name_frame, textvariable=self._reg_name_var,
            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=TEXT_PRIMARY,
            font=("Segoe UI", 11), relief=tk.FLAT, bd=4,
        )
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        name_entry.focus_set()
        name_entry.bind("<Return>", lambda e: self._start_registration())

        # Camera preview
        self._reg_camera_label = tk.Label(
            self._reg_window, bg=DARK_BG3,
            text="Preparing camera preview...",
            fg=TEXT_SECONDARY, font=("Segoe UI", 10),
        )
        self._reg_camera_label.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Progress label
        self._reg_progress_var = tk.StringVar(value="")
        tk.Label(
            self._reg_window, textvariable=self._reg_progress_var,
            bg=DARK_BG, fg=ACCENT, font=("Segoe UI", 10, "bold"),
        ).pack(pady=2)

        # Progress bar
        self._reg_progress_bar = ttk.Progressbar(
            self._reg_window, maximum=REGISTRATION_SAMPLES,
            length=400, mode="determinate",
        )
        self._reg_progress_bar.pack(pady=4)

        # Buttons
        btn_frame = tk.Frame(self._reg_window, bg=DARK_BG)
        btn_frame.pack(fill=tk.X, padx=20, pady=(4, 12))

        self._make_button(
            btn_frame, "Start Capture", self._start_registration,
            color=GREEN, width=16,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self._make_button(
            btn_frame, "Cancel", self._cancel_registration,
            color=RED, width=16,
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(4, 0))

        # Start preview update loop
        self._reg_photo = None
        self._reg_capturing = False
        self._update_reg_preview()

    def _update_reg_preview(self) -> None:
        """Update the registration dialog's camera preview."""
        if not self._registering or not hasattr(self, "_reg_window"):
            return
        try:
            if not self._reg_window.winfo_exists():
                self._registering = False
                return
        except tk.TclError:
            self._registering = False
            return

        if not self._reg_capturing:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                display = cv2.flip(frame, 1)
                # Detect and draw face box in preview
                try:
                    faces = self._detector.detect(display)
                    for f in faces:
                        x1, y1, x2, y2 = f["bbox"]
                        cv2.rectangle(display, (x1, y1), (x2, y2), ACCENT, 2, cv2.LINE_AA)
                except Exception:
                    pass

                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Scale to fit label
                if self._reg_camera_label.winfo_width() > 1:
                    lw = self._reg_camera_label.winfo_width() - 8
                    lh = self._reg_camera_label.winfo_height() - 8
                    iw, ih = pil_img.size
                    scale = min(lw / iw, lh / ih)
                    if scale < 1.0:
                        pil_img = pil_img.resize(
                            (int(iw * scale), int(ih * scale)), Image.LANCZOS
                        )

                self._reg_photo = ImageTk.PhotoImage(pil_img)
                self._reg_camera_label.configure(image=self._reg_photo, text="")

        # Schedule next preview update
        if self._registering:
            try:
                self._reg_window.after(50, self._update_reg_preview)
            except tk.TclError:
                self._registering = False

    def _start_registration(self) -> None:
        """Begin capturing registration samples."""
        name = self._reg_name_var.get().strip()
        if not name:
            messagebox.showwarning(
                "Invalid Name",
                "Please enter a valid name for the person.",
                parent=self._reg_window,
            )
            return

        self._reg_capturing = True
        self._reg_progress_bar["value"] = 0
        self._reg_progress_var.set("Capturing samples... face the camera directly")

        # Run capture in a thread to avoid blocking the GUI
        threading.Thread(
            target=self._capture_registration_samples,
            args=(name,),
            daemon=True,
        ).start()

    def _capture_registration_samples(self, name: str) -> None:
        """Capture face samples in a background thread."""
        if not self._recognizer or not self._recognizer.ready:
            self._root.after(0, lambda: messagebox.showerror(
                "Error", "Recognizer not ready.", parent=self._reg_window,
            ))
            self._reg_capturing = False
            return

        face_samples = []
        captured = 0
        pause_frames = 30

        while captured < REGISTRATION_SAMPLES and self._registering:
            frame = self._cam.get_frame(timeout=1.0) if self._cam else None
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            faces = self._detector.detect(frame)

            if not faces:
                self._root.after(0, lambda: self._reg_progress_var.set(
                    "No face detected — looking for face..."
                ))
                time.sleep(0.1)
                continue

            # Use the largest face
            best = max(
                faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            )
            x1, y1, x2, y2 = best["bbox"]

            h, w = frame.shape[:2]
            margin_x = int((x2 - x1) * 0.20)
            margin_y = int((y2 - y1) * 0.25)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            crop = frame[y1:y2, x1:x2]  # Keep BGR
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

            face_samples.append((crop.copy(), kps))
            captured += 1

            self._root.after(0, lambda c=captured: (
                self._reg_progress_bar.configure(value=c),
                self._reg_progress_var.set(f"Captured {c}/{REGISTRATION_SAMPLES} samples"),
            ))

            # Pause between captures
            if captured < REGISTRATION_SAMPLES:
                time.sleep(0.5)

        # Register
        if captured >= REGISTRATION_SAMPLES:
            success = self._recognizer.register(name, face_samples)
            if success:
                self._root.after(0, lambda: (
                    self._reg_progress_var.set(
                        f"'{name}' registered with {len(face_samples)} samples!"
                    ),
                    self._update_registered_list(),
                    self._status(f"'{name}' registered successfully.", GREEN),
                ))
            else:
                self._root.after(0, lambda: (
                    self._reg_progress_var.set("Registration FAILED."),
                    self._status("Registration failed.", RED),
                ))
        else:
            self._root.after(0, lambda: self._reg_progress_var.set("Registration cancelled."))

        self._reg_capturing = False

    def _cancel_registration(self) -> None:
        """Cancel the registration process."""
        self._registering = False
        self._reg_capturing = False
        try:
            if hasattr(self, "_reg_window") and self._reg_window.winfo_exists():
                self._reg_window.destroy()
        except tk.TclError:
            pass
        self._status("Registration cancelled.", TEXT_SECONDARY)

    def _remove_selected_face(self) -> None:
        """Remove the selected face from the database."""
        selection = self._registered_listbox.curselection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a face to remove.")
            return

        idx = selection[0]
        if self._recognizer is None or self._recognizer.database is None:
            return

        names = self._recognizer.database.names
        if idx < len(names):
            name = names[idx]
            if messagebox.askyesno("Confirm Removal", f"Remove '{name}' from the database?"):
                self._recognizer.database.remove_person(name)
                self._update_registered_list()
                self._status(f"Removed '{name}'.", ORANGE)

    def _clear_records(self) -> None:
        """Clear all attendance records."""
        if self._attendance is None:
            return
        if messagebox.askyesno("Clear Records", "Clear all attendance records?"):
            self._attendance.clear_records()
            self._last_attendance_count = 0
            self._refresh_attendance_table()
            self._status("Attendance records cleared.", ORANGE)

    def _export_csv(self) -> None:
        """Export attendance records to CSV."""
        if self._attendance is None:
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            title="Export Attendance CSV",
        )

        if not path:
            return

        result = self._attendance.export_csv(path)
        if result:
            self._status(f"Exported {len(self._attendance.records)} records to CSV.", GREEN)
            messagebox.showinfo("Export Complete", f"Records exported to:\n{path}")
        else:
            self._status("No records to export.", ORANGE)
            messagebox.showwarning("No Records", "There are no attendance records to export.")

    def _show_summary(self) -> None:
        """Show attendance summary in a dialog."""
        if self._attendance is None:
            return

        registered = (
            self._recognizer.database.names
            if self._recognizer is not None and self._recognizer.database is not None
            else []
        )
        summary = self._attendance.get_summary(registered)

        msg = (
            f"Attendance Summary\n"
            f"{'=' * 40}\n\n"
            f"Total Records:       {summary['total_records']}\n"
            f"Unique Present:      {summary['unique_present']}\n"
            f"Total Present:       {summary['total_present']}\n"
            f"Total Absent:        {summary['total_absent']}\n"
            f"Total Rejected:      {summary['total_rejected']}\n"
            f"Attendance Rate:     {summary['attendance_rate']}%\n"
        )

        if summary["present_names"]:
            msg += f"\nPresent:\n  {', '.join(summary['present_names'])}\n"

        if summary["absent_names"]:
            msg += f"\nAbsent:\n  {', '.join(summary['absent_names'])}\n"

        # Create summary dialog
        summary_win = tk.Toplevel(self._root)
        summary_win.title("Attendance Summary")
        summary_win.configure(bg=DARK_BG)
        summary_win.resizable(False, False)
        summary_win.transient(self._root)
        summary_win.grab_set()

        # Center
        summary_win.update_idletasks()
        pw = self._root.winfo_width()
        ph = self._root.winfo_height()
        px = self._root.winfo_x()
        py = self._root.winfo_y()
        ww = 460
        wh = 380
        summary_win.geometry(f"{ww}x{wh}+{px + (pw - ww) // 2}+{py + (ph - wh) // 2}")

        text = tk.Text(
            summary_win, bg=ENTRY_BG, fg=TEXT_PRIMARY,
            font=("Consolas", 10), relief=tk.FLAT, bd=8,
            wrap=tk.WORD,
        )
        text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 4))
        text.insert(tk.END, msg)
        text.configure(state=tk.DISABLED)

        self._make_button(
            summary_win, "Close", summary_win.destroy,
            color=ACCENT, width=20,
        ).pack(pady=(4, 12))

    # ═══════════════════════════════════════════════════════════
    #  Helper Methods
    # ═══════════════════════════════════════════════════════════

    def _update_status_indicator(
        self, key: str, value: str, color: str = TEXT_PRIMARY
    ) -> None:
        """Update a status label."""
        if key in self._status_labels:
            try:
                self._status_labels[key].configure(text=value, fg=color)
            except tk.TclError:
                pass

    def _status(self, text: str, color: str = TEXT_SECONDARY) -> None:
        """Update the status bar text."""
        if hasattr(self, "_status_bar_var"):
            try:
                self._status_bar_var.set(f"  {text}")
            except tk.TclError:
                pass

    def _update_registered_list(self) -> None:
        """Refresh the registered faces listbox."""
        try:
            self._registered_listbox.delete(0, tk.END)
            if self._recognizer is not None and self._recognizer.database is not None:
                for name in self._recognizer.database.names:
                    self._registered_listbox.insert(tk.END, name)
        except tk.TclError:
            pass

    def _refresh_attendance_table(self) -> None:
        """Refresh the attendance log treeview."""
        try:
            # Clear existing items
            for item in self._tree.get_children():
                self._tree.delete(item)

            if self._attendance is None:
                return

            # Add records (newest first)
            for rec in reversed(self._attendance.records):
                status = rec.get("status", "UNKNOWN")
                tag = "PRESENT" if status == "PRESENT" else "REJECTED"

                # Format timestamp
                ts = rec.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(ts)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    time_str = ts

                self._tree.insert("", tk.END, values=(
                    rec.get("name", "Unknown"),
                    time_str,
                    f"{rec.get('confidence', 0) * 100:.1f}%",
                    f"{rec.get('spoof_score', 0):.3f}",
                    status,
                ), tags=(tag,))

        except tk.TclError:
            pass

    def _on_rec_threshold_change(self, val) -> None:
        """Handle recognition threshold slider change."""
        v = float(val)
        self._config.recognition_threshold = v
        self._rec_threshold_label.configure(text=f"{v:.2f}")

    def _on_spoof_threshold_change(self, val) -> None:
        """Handle anti-spoof threshold slider change."""
        v = float(val)
        self._config.antispoof_threshold = v
        self._spoof_threshold_label.configure(text=f"{v:.2f}")

    # ═══════════════════════════════════════════════════════════
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════

    def _on_close(self) -> None:
        """Handle window close — graceful shutdown."""
        self._running = False
        self._pipeline_running = False
        self._registering = False

        # Shutdown pipeline components
        if self._attendance is not None and self._attendance.is_active:
            self._attendance.stop_session()
        if self._cam is not None:
            self._cam.stop()
        if self._detector is not None:
            self._detector.close()
        if self._recognizer is not None:
            self._recognizer.close()

        log.info("FaceGuard AI GUI — Shutdown complete.")

        try:
            if self._root is not None:
                self._root.destroy()
        except tk.TclError:
            pass

    def run(self) -> None:
        """Build the GUI, initialise pipeline, and start the main loop."""
        self._build_gui()

        # Initialise pipeline after GUI is visible
        self._root.update()
        self._root.after(100, self._deferred_start)

        self._running = True
        self._root.mainloop()

    def _deferred_start(self) -> None:
        """Initialise pipeline and start inference after GUI is ready."""
        if not self._init_pipeline():
            self._status("Failed to initialise. Check logs.", RED)
            messagebox.showerror(
                "Initialisation Failed",
                "Could not initialise the FaceGuard pipeline.\n"
                "Check the console for error details.",
            )
            return

        # Update camera status
        self._update_status_indicator("camera", "Active", GREEN)
        self._update_status_indicator("virtual_cam",
                                      "BLOCKED" if self._is_virtual else "OK",
                                      RED if self._is_virtual else GREEN)

        # Start pipeline thread
        pipeline_thread = threading.Thread(
            target=self._pipeline_loop, name="PipelineThread", daemon=True,
        )
        pipeline_thread.start()

        # Start GUI update loop
        self._update_display()
