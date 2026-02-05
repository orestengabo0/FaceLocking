# src/action_detector.py
"""
action_detector.py
Detects simple face actions from 5-point landmarks and face position.

Detectable actions:
- eye blink: rapid closure and reopening
- head movement: left/right based on nose position
- smile/laugh: mouth height increase
- face scale change: distance between eye landmarks

Actions are detected frame-by-frame and logged with timestamps.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import time


@dataclass
class Action:
    """Represents a detected action."""
    action_type: str  # "blink", "move_left", "move_right", "smile", "face_scale_change"
    timestamp: float  # time.time()
    confidence: float  # 0.0 to 1.0
    value: float  # action-specific metric
    description: str  # human-readable description


class ActionDetector:
    """
    Detects face actions from 5-point landmarks and spatial history.

    Landmarks order (from align.py):
    0: left_eye
    1: right_eye
    2: nose_tip
    3: left_mouth
    4: right_mouth
    """

    def __init__(
        self,
        blink_threshold: float = 0.15,
        smile_threshold: float = 0.08,
        movement_threshold_px: float = 8.0,
        scale_change_threshold: float = 0.12,
    ):
        """
        Args:
            blink_threshold: vertical eye opening ratio (0-1) below which blink is detected
            smile_threshold: mouth height increase ratio to trigger smile
            movement_threshold_px: pixel distance to trigger left/right movement
            scale_change_threshold: ratio change in eye distance to detect scale change
        """
        self.blink_threshold = float(blink_threshold)
        self.smile_threshold = float(smile_threshold)
        self.movement_threshold_px = float(movement_threshold_px)
        self.scale_change_threshold = float(scale_change_threshold)

        # History for smoothing and action detection
        self._prev_kps: Optional[np.ndarray] = None
        self._prev_eye_opening: Optional[float] = None
        self._prev_mouth_height: Optional[float] = None
        self._prev_nose_x: Optional[float] = None
        self._prev_eye_distance: Optional[float] = None

        # Blink detection state machine
        self._blink_state = "open"  # open, closing, closed, opening
        self._blink_frames = 0
        self._blink_threshold_frames = 2  # minimum frames closed to count as blink

    def detect(self, kps_5x2: np.ndarray) -> List[Action]:
        """
        Detect actions from 5-point landmarks.

        Args:
            kps_5x2: (5, 2) array of landmarks
                [left_eye, right_eye, nose_tip, left_mouth, right_mouth]

        Returns:
            List of Action objects detected in this frame
        """
        actions: List[Action] = []
        now = time.time()

        if kps_5x2 is None or kps_5x2.size == 0:
            return actions

        kps = kps_5x2.astype(np.float32)

        # Extract metrics
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        left_mouth = kps[3]
        right_mouth = kps[4]

        # ------ Eye Blink Detection ------
        eye_opening = self._compute_eye_opening(left_eye, right_eye)
        blink_action = self._detect_blink(eye_opening, now)
        if blink_action is not None:
            actions.append(blink_action)

        # ------ Smile/Laugh Detection ------
        mouth_height = self._compute_mouth_height(left_mouth, right_mouth)
        if self._prev_mouth_height is not None:
            smile_action = self._detect_smile(mouth_height, self._prev_mouth_height, now)
            if smile_action is not None:
                actions.append(smile_action)

        # ------ Head Movement Detection (left/right) ------
        movement_actions = self._detect_movement(nose, now)
        actions.extend(movement_actions)

        # ------ Face Scale Change Detection ------
        eye_distance = self._compute_eye_distance(left_eye, right_eye)
        scale_action = self._detect_scale_change(eye_distance, now)
        if scale_action is not None:
            actions.append(scale_action)

        # Update history
        self._prev_kps = kps.copy()
        self._prev_eye_opening = eye_opening
        self._prev_mouth_height = mouth_height
        self._prev_nose_x = float(nose[0])
        self._prev_eye_distance = eye_distance

        return actions

    # ========================
    # Metric Computation
    # ========================

    @staticmethod
    def _compute_eye_opening(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """
        Compute eye opening ratio (0.0 = closed, 1.0 = fully open).
        Uses vertical distance between eye keypoints as proxy.
        """
        # Simple heuristic: eye opening ≈ 1.0 at rest
        # When blinking, vertical span decreases dramatically
        # We use distance between eyes as reference
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        vert_span = float(abs(right_eye[1] - left_eye[1]))

        # Ratio of vertical to horizontal distance
        # Blinking reduces vertical span
        if eye_dist < 1.0:
            return 1.0
        ratio = vert_span / eye_dist
        # Clamp to [0, 1]
        return float(np.clip(ratio, 0.0, 1.0))

    @staticmethod
    def _compute_mouth_height(left_mouth: np.ndarray, right_mouth: np.ndarray) -> float:
        """
        Compute mouth height (vertical span).
        Increase indicates smile or laugh.
        """
        return float(abs(right_mouth[1] - left_mouth[1]))

    @staticmethod
    def _compute_eye_distance(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """
        Compute distance between eyes (proxy for face scale/distance from camera).
        """
        return float(np.linalg.norm(right_eye - left_eye))

    # ========================
    # Action Detection Logic
    # ========================

    def _detect_blink(self, eye_opening: float, now: float) -> Optional[Action]:
        """
        Detect blink using eye opening state machine.
        Blink = closed for at least N frames, then reopened.
        """
        # State machine: open -> closing -> closed -> opening -> open
        if eye_opening > 0.6:  # Eyes open
            if self._blink_state in ("closing", "closed", "opening"):
                # Transition back to open
                if self._blink_state == "opening" and self._blink_frames > self._blink_threshold_frames:
                    # Completed blink
                    self._blink_state = "open"
                    self._blink_frames = 0
                    return Action(
                        action_type="blink",
                        timestamp=now,
                        confidence=0.85,
                        value=eye_opening,
                        description="Eye blink detected (open -> closed -> open)",
                    )
                elif self._blink_state != "opening":
                    self._blink_state = "opening"
                    self._blink_frames = 1
            else:
                self._blink_state = "open"
                self._blink_frames = 0
        else:  # Eyes closed
            if self._blink_state == "open":
                self._blink_state = "closing"
                self._blink_frames = 1
            elif self._blink_state == "closing":
                self._blink_frames += 1
                if self._blink_frames >= self._blink_threshold_frames:
                    self._blink_state = "closed"
                    self._blink_frames = 0
            elif self._blink_state == "closed":
                self._blink_frames += 1

        return None

    def _detect_smile(
        self, mouth_height: float, prev_mouth_height: float, now: float
    ) -> Optional[Action]:
        """
        Detect smile/laugh by detecting increase in mouth height.
        """
        if prev_mouth_height <= 0:
            return None

        ratio = mouth_height / prev_mouth_height
        if ratio > (1.0 + self.smile_threshold):
            return Action(
                action_type="smile",
                timestamp=now,
                confidence=min(0.9, (ratio - 1.0) / 0.15),
                value=ratio,
                description=f"Smile/laugh detected (mouth height ratio: {ratio:.2f})",
            )
        return None

    def _detect_movement(self, nose: np.ndarray, now: float) -> List[Action]:
        """
        Detect head left/right movement by tracking nose position.
        """
        actions: List[Action] = []

        if self._prev_nose_x is None:
            return actions

        current_x = float(nose[0])
        delta_x = current_x - self._prev_nose_x

        if abs(delta_x) > self.movement_threshold_px:
            if delta_x > 0:
                actions.append(
                    Action(
                        action_type="move_right",
                        timestamp=now,
                        confidence=min(0.95, abs(delta_x) / 20.0),
                        value=delta_x,
                        description=f"Head movement right ({delta_x:.1f}px)",
                    )
                )
            else:
                actions.append(
                    Action(
                        action_type="move_left",
                        timestamp=now,
                        confidence=min(0.95, abs(delta_x) / 20.0),
                        value=abs(delta_x),
                        description=f"Head movement left ({abs(delta_x):.1f}px)",
                    )
                )

        return actions

    def _detect_scale_change(self, eye_distance: float, now: float) -> Optional[Action]:
        """
        Detect face scale change (move closer/farther from camera).
        """
        if self._prev_eye_distance is None or self._prev_eye_distance <= 0:
            return None

        ratio = eye_distance / self._prev_eye_distance

        if ratio > (1.0 + self.scale_change_threshold):
            return Action(
                action_type="face_closer",
                timestamp=now,
                confidence=min(0.85, (ratio - 1.0) / 0.15),
                value=ratio,
                description=f"Face moved closer to camera (scale ratio: {ratio:.2f})",
            )
        elif ratio < (1.0 - self.scale_change_threshold):
            return Action(
                action_type="face_farther",
                timestamp=now,
                confidence=min(0.85, (1.0 - ratio) / 0.15),
                value=1.0 / ratio,
                description=f"Face moved farther from camera (scale ratio: {ratio:.2f})",
            )

        return None