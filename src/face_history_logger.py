# src/face_history_logger.py
"""
face_history_logger.py
Logs face actions to timestamped history files.

File format:
<face>_history_<timestamp>.txt

Each line format:
[HH:MM:SS.mmm] ACTION_TYPE | description | confidence | value

Example:
[12:30:45.123] blink | Eye blink detected (open -> closed -> open) | 0.85 | 0.45
[12:30:46.456] move_right | Head movement right (12.5px) | 0.92 | 12.5
[12:30:47.789] smile | Smile/laugh detected (mouth height ratio: 1.15) | 0.88 | 1.15
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import time

from .action_detector import Action


class FaceHistoryLogger:
    """
    Logs face actions to persistent history files.
    One file per face locking session.
    """

    def __init__(
        self,
        face_name: str,
        output_dir: Path = Path("data/face_histories"),
        session_start_time: Optional[float] = None,
    ):
        """
        Args:
            face_name: name of the locked face (e.g., "Gabi", "Fani")
            output_dir: directory to store history files
            session_start_time: session start time (defaults to now)
        """
        self.face_name = str(face_name).lower()
        self.output_dir = Path(output_dir)
        self.session_start_time = session_start_time or time.time()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        ts = int(self.session_start_time * 1000)  # milliseconds
        self.filename = f"{self.face_name}_history_{ts}.txt"
        self.filepath = self.output_dir / self.filename

        # Write header
        self._write_header()

        self._action_count = 0

    def _write_header(self) -> None:
        """Write file header with session metadata."""
        header = f"""================================================================================
Face Locking Session History
================================================================================
Face Name: {self.face_name.upper()}
Session Start: {datetime.fromtimestamp(self.session_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}
File: {self.filename}
================================================================================
Format: [HH:MM:SS.mmm] ACTION_TYPE | description | confidence | value
================================================================================

"""
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(header)

    def log_action(self, action: Action) -> None:
        """
        Log a single action to the history file.

        Args:
            action: Action object to log
        """
        self._action_count += 1

        # Convert timestamp to HH:MM:SS.mmm format relative to session start
        elapsed_seconds = action.timestamp - self.session_start_time
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = elapsed_seconds % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        # Format line
        line = (
            f"[{time_str}] {action.action_type.upper():<15} | "
            f"{action.description:<50} | "
            f"conf={action.confidence:.2f} | val={action.value:.4f}\n"
        )

        # Append to file
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(line)

    def log_actions(self, actions: List[Action]) -> None:
        """
        Log multiple actions.

        Args:
            actions: List of Action objects
        """
        for action in actions:
            self.log_action(action)

    def log_status(self, message: str) -> None:
        """
        Log a status message (e.g., "lock acquired", "lock lost").

        Args:
            message: status message
        """
        elapsed_seconds = time.time() - self.session_start_time
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = elapsed_seconds % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        line = f"[{time_str}] STATUS                | {message}\n"

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(line)

    def get_summary(self) -> str:
        """
        Get summary statistics.

        Returns:
            formatted summary string
        """
        if not self.filepath.exists():
            return f"History file not found: {self.filepath}"

        content = self.filepath.read_text(encoding="utf-8")
        action_lines = [line for line in content.split("\n") if line and not line.startswith("=")]

        action_types = {}
        for line in action_lines:
            if "|" in line:
                parts = line.split("|")
                if len(parts) > 0:
                    action_part = parts[0].strip().split()
                    if action_part:
                        action_type = action_part[1].lower()
                        action_types[action_type] = action_types.get(action_type, 0) + 1

        summary = f"\n{'='*80}\nFace History Summary\n{'='*80}\n"
        summary += f"Face: {self.face_name}\n"
        summary += f"Total actions recorded: {self._action_count}\n"
        if action_types:
            summary += f"\nAction Breakdown:\n"
            for action_type, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
                summary += f"  - {action_type}: {count}\n"
        summary += f"\nHistory file: {self.filepath}\n"
        summary += f"{'='*80}\n"

        return summary

    def finalize(self) -> str:
        """
        Finalize the session and write footer.

        Returns:
            path to the history file
        """
        footer = f"\n{'='*80}\n"
        footer += f"Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
        footer += f"Total actions recorded: {self._action_count}\n"
        footer += f"{'='*80}\n"

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(footer)

        return str(self.filepath)