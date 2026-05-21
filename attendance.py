"""
src/attendance.py — Attendance Logging & Management.

Manages the attendance lifecycle: marking present, enforcing cooldowns,
persisting logs to CSV/JSON, and exporting attendance records.

Key features
────────────
* Configurable per-person cooldown to prevent duplicate marks
* JSON-backed persistent log (survives restarts)
* CSV export for downstream analysis
* Summary statistics (total, present, absent)
* Timestamped entries with recognition confidence
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import Config, DATA_DIR, LOGS_DIR, log


class AttendanceManager:
    """
    Attendance session manager.

    Records are stored as:
        {
            "name": "John Doe",
            "timestamp": "2024-01-15T10:30:00",
            "confidence": 0.92,
            "spoof_score": 0.05,
            "status": "PRESENT"   // PRESENT, REJECTED_SPOOF, REJECTED_LOW_CONF
        }
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._log_file = str(LOGS_DIR / f"attendance_{time.strftime('%Y%m%d')}.jsonl")
        self._cooldowns: Dict[str, float] = {}  # name → last_mark_timestamp
        self._records: List[Dict[str, Any]] = []
        self._session_active: bool = False

    # ─────────── Session Control ───────────

    def start_session(self) -> None:
        """Begin a new attendance session."""
        self._session_active = True
        self._cooldowns.clear()
        log.info("Attendance session STARTED.")

    def stop_session(self) -> None:
        """End the current attendance session."""
        self._session_active = False
        log.info(
            "Attendance session STOPPED.  Total records: %d",
            len(self._records),
        )

    @property
    def is_active(self) -> bool:
        return self._session_active

    # ─────────── Marking Attendance ───────────

    def mark_attendance(
        self,
        name: str,
        confidence: float,
        spoof_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Attempt to mark a person as present.

        Checks the cooldown timer first.  If within the cooldown window
        the mark is silently skipped (not an error).

        Parameters
        ----------
        name : str         — Recognised person's name.
        confidence : float — Recognition cosine similarity (0–1).
        spoof_score : float — Anti-spoof real-score (0–1).

        Returns
        -------
        dict with keys:
            marked   : bool
            reason   : str  ("ok", "cooldown", "not_active", "low_conf")
            record   : dict or None
        """
        if not self._session_active:
            return {"marked": False, "reason": "not_active", "record": None}

        # ── Cooldown check ──
        now = time.time()
        last_mark = self._cooldowns.get(name, 0)
        elapsed = now - last_mark
        cd_seconds = self._config.cooldown_seconds

        if elapsed < cd_seconds and last_mark > 0:
            remaining = cd_seconds - elapsed
            log.debug(
                "'%s' on cooldown (%.0fs remaining). Skipping.",
                name, remaining,
            )
            return {"marked": False, "reason": "cooldown", "record": None}

        # ── Confidence check ──
        if confidence < self._config.recognition_threshold:
            return {"marked": False, "reason": "low_conf", "record": None}

        # ── Create record ──
        record = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "confidence": round(confidence, 4),
            "spoof_score": round(spoof_score, 4),
            "status": "PRESENT",
        }

        self._records.append(record)
        self._cooldowns[name] = now
        self._persist_record(record)

        log.info(
            "ATTENDANCE: %s marked PRESENT  (conf=%.2f, spoof=%.2f)",
            name, confidence, spoof_score,
        )

        return {"marked": True, "reason": "ok", "record": record}

    def mark_rejected(
        self,
        name: Optional[str],
        reason: str,
        confidence: float = 0.0,
        spoof_score: float = 0.0,
    ) -> None:
        """Log a rejected attempt (spoof or low-confidence)."""
        if not self._session_active:
            return

        record = {
            "name": name or "UNKNOWN",
            "timestamp": datetime.now().isoformat(),
            "confidence": round(confidence, 4),
            "spoof_score": round(spoof_score, 4),
            "status": f"REJECTED_{reason.upper()}",
        }

        self._records.append(record)
        self._persist_record(record)

        log.info(
            "REJECTED: %s — reason=%s  (conf=%.2f, spoof=%.2f)",
            record["name"], reason, confidence, spoof_score,
        )

    # ─────────── Persistence ───────────

    def _persist_record(self, record: Dict[str, Any]) -> None:
        """Append a JSON record to the log file (JSONL format)."""
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.error("Failed to persist attendance record: %s", exc)

    # ─────────── Statistics ───────────

    def get_summary(self, registered_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute attendance summary.

        Returns dict with: total_records, unique_present, total_present,
        total_absent, total_rejected, records breakdown.
        """
        present_names = set()
        total_present = 0
        total_rejected = 0

        for rec in self._records:
            if rec["status"] == "PRESENT":
                present_names.add(rec["name"])
                total_present += 1
            else:
                total_rejected += 1

        total_registered = len(registered_names) if registered_names else 0
        absent_names = (
            set(registered_names) - present_names
            if registered_names
            else set()
        )

        return {
            "total_records": len(self._records),
            "unique_present": len(present_names),
            "present_names": sorted(present_names),
            "absent_names": sorted(absent_names),
            "total_present": total_present,
            "total_absent": len(absent_names),
            "total_rejected": total_rejected,
            "attendance_rate": (
                round(len(present_names) / total_registered * 100, 1)
                if total_registered > 0
                else 0
            ),
        }

    # ─────────── Export ───────────

    def export_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export all records to CSV.

        Returns the path of the generated CSV file.
        """
        if not self._records:
            log.warning("No records to export.")
            return ""

        output_path = output_path or str(
            LOGS_DIR / f"attendance_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fieldnames = ["name", "timestamp", "confidence", "spoof_score", "status"]

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._records)

            log.info("Exported %d records to %s", len(self._records), output_path)
            return output_path
        except Exception as exc:
            log.error("CSV export failed: %s", exc)
            return ""

    # ─────────── Reset ───────────

    def clear_records(self) -> None:
        """Clear all in-memory records and cooldowns."""
        self._records.clear()
        self._cooldowns.clear()
        log.info("Attendance records cleared.")

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)
