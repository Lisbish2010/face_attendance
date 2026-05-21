#!/usr/bin/env python3
"""
FaceGuard AI — Quick launcher.

Simply run:
    python run.py                    # CLI mode (OpenCV window)
    python run.py --gui              # GUI mode (tkinter desktop app)

Or with options:
    python run.py --camera 0 --width 1280 --height 720
    python run.py --no-gpu
    python run.py --skip-frames 1
"""

from src.main import main

if __name__ == "__main__":
    main()
