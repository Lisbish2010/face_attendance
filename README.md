# FaceGuard AI — Production Face Attendance System

A production-grade, passive anti-spoofing face attendance system built in Python.
No Haar Cascades. No challenge-response. Fully automated.
**No C++ compiler required** — all packages are pre-built Python wheels.

---

## Architecture

```
Camera Frame
      |
      v
 Virtual Camera Detection  ---- BLOCK if OBS / ManyCam / Snap / XSplit detected
      |
      v
 MediaPipe Face Detection   ---- BlazeFace (not Haar Cascade)
      |
      v
 Anti-Spoof Analysis         ---- ONNX model + 5 heuristic layers
      |                            - Texture (LBP variance)
      |                            - Moire pattern (FFT)
      |                            - Screen reflection
      |                            - Low-depth estimation
      |                            - Temporal consistency
      |
      v
 ArcFace Recognition          ---- ONNX Runtime + 512-D embedding (no insightface pkg)
      |
      v
 Attendance Logging          ---- JSONL persistence + CSV export
```

---

## Project Structure

```
face_attendance/
|
+-- models/                  Downloaded ONNX models stored here (auto-download)
+-- data/                    Registered face database + config
+-- logs/                    Attendance logs + session logs
+-- src/
|   +-- __init__.py
|   +-- anti_spoof.py             Anti-spoof engine (ONNX + 5 heuristics)
|   +-- recognition.py            ArcFace 512-D recognition via ONNX Runtime
|   +-- detector.py               MediaPipe face detection (BlazeFace)
|   +-- virtual_cam_detector.py   Virtual camera detection (name + heuristic)
|   +-- attendance.py             Attendance logging & management
|   +-- utils.py                  Config, logging, drawing, FPS tracker
|   +-- main.py                   Entry point & pipeline orchestrator
+-- run.py                    Quick launcher (python run.py)
+-- requirements.txt          Python dependencies (all pre-built wheels)
+-- README.md                 (this file)
```

---

## Windows Setup (PowerShell)

### Step 1: Extract the zip

```powershell
# Right-click the zip -> Extract All
# Or:
Expand-Archive -Path "face_attendance.zip" -DestinationPath "face_attendance"
cd face_attendance
```

### Step 2: Create virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If you get a PowerShell execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then try the activate command again.

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

> **IMPORTANT:** This will NOT ask for a C++ compiler. All packages are
> pre-built wheels. If pip tries to compile anything, you have the wrong
> requirements.txt — make sure it matches the one included in this zip.

### Step 4: Run

```powershell
python run.py
```

Or with options:

```powershell
python run.py --camera 0 --width 1280 --height 720 --threshold 0.45
python run.py --no-gpu
python run.py --skip-frames 1
```

---

## First-Time Run: Model Downloads

On the very first run, the system automatically downloads two ONNX models:

| Model | Size | Source | Saved to |
|-------|------|--------|----------|
| **ArcFace** (w600k_r50.onnx) | ~166 MB | InsightFace GitHub releases | `models/w600k_r50.onnx` |
| **Anti-Spoof** (anti_spoof_model.onnx) | ~2.7 MB | Silent-Face GitHub releases | `models/anti_spoof_model.onnx` |

You need an internet connection for the first run. After that, the models
are cached locally and no download is needed.

**Manual download (if auto-download fails):**
1. ArcFace: Download `buffalo_l.zip` from https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
   - Extract `w600k_r50.onnx` and place it in `models/`
2. Anti-Spoof: Download from https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases
   - Place `anti_spoof_model.onnx` in `models/`

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `a` | Toggle attendance session on/off |
| `r` | Register a new face (interactive) |
| `c` | Clear attendance records |
| `e` | Export attendance to CSV |
| `s` | Show attendance summary in console |
| `v` | Toggle virtual-camera block on/off |
| `+` / `-` | Adjust anti-spoof threshold |

---

## How to Use

### 1. Register a Face

Press `R` in the running window. You will be prompted in the terminal to
enter a name. Face the camera directly and the system will capture 3 samples
automatically with slight head movement between each.

### 2. Start Attendance

Press `A` to start an attendance session. The system will now begin
recognizing registered faces and logging attendance.

### 3. View Results

Press `S` for a summary in the console, or press `E` to export all
records to a CSV file.

---

## Anti-Spoofing Details

### ONNX Model Layer

Uses the **Silent-Face Anti-Spoofing** MiniFASNet model via ONNX Runtime:
- 80x80 input
- Outputs spoof probability vs real probability
- ~2 ms inference on CPU

### Heuristic Layers (5 analysers)

1. **Texture Analysis (LBP Variance)** — Real skin has moderate LBP variance;
   printed photos show abnormal distributions.
2. **Moire Pattern Detection** — 2D FFT detects screen replay interference.
3. **Screen Reflection Detection** — Identifies unnatural bright highlight regions.
4. **Low-Depth Estimation** — Flat photos show uniform sharpness; real 3D faces vary.
5. **Temporal Consistency** — Static photos show zero frame-to-frame change.

### Score Fusion

```
final_real_score = 0.6 * onnx_real_score + 0.4 * (1.0 - heuristic_avg)
```

---

## Virtual Camera Detection

Blocks 20+ known virtual cameras by name (OBS, ManyCam, Snap, XSplit,
Streamlabs, Elgato, DroidCam, etc.) plus heuristic analysis of frame
statistics to catch unknown virtual feeds.

---

## Troubleshooting

### "No module named 'src'" error
Make sure you are running from INSIDE the `face_attendance` folder:
```powershell
cd face_attendance
python run.py
```

### Camera not opening
- Close any other app using the webcam (Zoom, Teams, etc.)
- Try `python run.py --camera 1` (different camera index)
- Check Windows Settings > Privacy > Camera permissions

### Model download fails
- Check your internet connection
- See "Manual download" section above
- Models only need to download once

### Low FPS
- Use `python run.py --skip-frames 1` to process every other frame
- Reduce resolution: `python run.py --width 640 --height 480`
- On CPU, 15-20 FPS is normal; GPU gives 25-40+ FPS

### pip install fails with C++ compiler error
- Make sure you are using the `requirements.txt` from THIS zip
- The correct file has NO `insightface` or `dlib` or `face-recognition`
- If venv is corrupted, delete it and recreate:
  ```powershell
  Remove-Item -Recurse -Force venv
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
