# servo_controller.py
"""
Optimized Servo Motor Controller for Face Tracking System.
Smooth, proportional, non-blocking movement via MQTT.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ServoController:
    """Controls servo motor using EMA smoothing and proportional steps."""

    def __init__(
        self,
        mqtt_manager,
        center_angle: int = 90,
        min_angle: int = 0,
        max_angle: int = 180,
        smoothing_alpha: float = 0.2,
        max_step: int = 5,
        update_interval_ms: int = 15,
    ):
        """
        Args:
            mqtt_manager: MQTTManager instance
            center_angle: Servo center angle
            min_angle: Minimum angle
            max_angle: Maximum angle
            smoothing_alpha: EMA smoothing factor
            max_step: Maximum degrees per update
            update_interval_ms: Minimum interval between MQTT updates
        """
        self.mqtt = mqtt_manager
        self.center_angle = center_angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.smoothing_alpha = smoothing_alpha
        self.max_step = max_step
        self.update_interval = update_interval_ms / 1000.0

        self.current_angle = float(center_angle)
        self.smoothed_angle = float(center_angle)
        self.target_angle = center_angle
        self.last_publish_time = 0.0
        self.last_gesture_time = 0.0
        self.gesture_queue = []

    def track_face_position(self, face_center_x: float, frame_width: int) -> int:
        """Update target angle based on face horizontal position (0-left, 1-right)."""
        normalized_pos = max(0.0, min(face_center_x / frame_width, 1.0))
        self.target_angle = int(normalized_pos * (self.max_angle - self.min_angle) + self.min_angle)
        return self._update_servo()

    def set_angle(self, angle: int) -> int:
        """Set target servo angle directly."""
        self.target_angle = max(self.min_angle, min(angle, self.max_angle))
        return self._update_servo()

    def reset_to_center(self) -> int:
        """Reset servo to center."""
        return self.set_angle(self.center_angle)

    def handle_action(self, action_type: str, action_data: Optional[dict] = None) -> None:
        """Queue gestures for non-blocking execution."""
        action_data = action_data or {}
        now = time.time()
        if now - self.last_gesture_time < 1.0:  # simple cooldown
            return

        if action_type == "BLINK":
            self.gesture_queue.append(("pulse",))
        elif action_type in ("SMILE", "EXPRESSION"):
            self.gesture_queue.append(("acknowledge",))
        elif action_type == "LEFT":
            self.set_angle(45)
        elif action_type == "RIGHT":
            self.set_angle(135)
        self.last_gesture_time = now

    def update(self) -> None:
        """
        Call in main loop to update servo angle smoothly and execute gestures.
        Non-blocking.
        """
        self._update_servo()
        self._execute_gestures()

    def _update_servo(self) -> int:
        """Smooth servo movement toward target angle and publish via MQTT."""
        now = time.time()
        # EMA smoothing
        self.smoothed_angle = (
            self.smoothing_alpha * self.target_angle +
            (1.0 - self.smoothing_alpha) * self.smoothed_angle
        )
        # Proportional step
        diff = self.smoothed_angle - self.current_angle
        if abs(diff) > self.max_step:
            diff = self.max_step if diff > 0 else -self.max_step
        self.current_angle += diff
        self.current_angle = max(self.min_angle, min(self.current_angle, self.max_angle))

        # Rate-limited MQTT publish
        if self.mqtt.is_connected and (now - self.last_publish_time) >= self.update_interval:
            self.mqtt.publish_servo_angle(int(self.current_angle))
            self.last_publish_time = now

        return int(self.current_angle)

    def _execute_gestures(self) -> None:
        """Non-blocking gesture executor."""
        if not self.gesture_queue:
            return
        gesture = self.gesture_queue.pop(0)
        if gesture[0] == "pulse":
            # quick side-to-side pulse (non-blocking)
            offsets = [-10, 10, 0]
            for offset in offsets:
                self.target_angle = self.center_angle + offset
        elif gesture[0] == "acknowledge":
            # gentle sweep
            sweep = [self.center_angle, 70, 110, self.center_angle]
            for angle in sweep:
                self.target_angle = angle

    def get_current_angle(self) -> int:
        return int(self.current_angle)

    def get_target_angle(self) -> int:
        return self.target_angle
