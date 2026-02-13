# src/integrated_system.py
"""
Complete Integrated Face Recognition & Locked Face Tracking System.

This system combines:
- Face detection and recognition using ArcFace embeddings
- Face locking with temporal stability tracking
- Action detection (blink, smile, movement)
- MQTT messaging for remote control and telemetry
- Servo motor control for automated face tracking

Run:
    python -m src.integrated_system

Keys:
    n/p : Next/Previous identity to lock
    +/- : Adjust recognition threshold
    s   : Toggle servo tracking (on/off)
    c   : Manually call servo gesture
    q   : Quit
"""

import logging
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from .face_lock import FaceLockingSystem
from .recognize import HaarFaceMesh5pt, ArcFaceEmbedderONNX, load_db_npz
from .align import align_face_5pt

# Setup MQTT and Servo modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mqtt_handler import MQTTManager
from servo_controller import ServoController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class IntegratedFaceTrackingSystem:
    """Complete integrated system combining face recognition, locking, and servo control."""
    
    def __init__(
        self,
        db_path: Path = Path("data/db/face_db.npz"),
        model_path: str = "models/embedder_arcface.onnx",
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        camera_index: int = 0,
    ):
        """
        Initialize the integrated system.
        
        Args:
            db_path: Path to face database
            model_path: Path to ArcFace model
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            camera_index: Camera device index
        """
        self.db_path = db_path
        self.model_path = model_path
        self.camera_index = camera_index
        
        # Initialize components
        logger.info("Initializing Face Locking System...")
        self.face_lock_system = FaceLockingSystem(db_path, model_path)
        
        logger.info("Initializing MQTT Manager...")
        self.mqtt_manager = MQTTManager(broker=mqtt_broker, port=mqtt_port)
        
        logger.info("Initializing Servo Controller...")
        self.servo_controller = ServoController(self.mqtt_manager)
        
        # System state
        self.is_tracking = True
        self.servo_enabled = True
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Last published state (to avoid spam)
        self.last_published_state: Dict[str, Any] = {}
        self.state_publish_cooldown = 1.0  # seconds
        self.last_state_publish = 0
        
        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width = 0
        self.frame_height = 0
    
    def connect_mqtt(self) -> bool:
        """Connect to MQTT broker.
        
        Returns:
            True if connection successful
        """
        if self.mqtt_manager.connect():
            logger.info("✓ MQTT broker connected")
            return True
        else:
            logger.warning("✗ Failed to connect to MQTT broker (running in offline mode)")
            return False
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture.
        
        Returns:
            True if camera initialized successfully
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return False
        
        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"✓ Camera initialized: {self.frame_width}x{self.frame_height}")
        return True
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary with processing results
        """
        # Get locked face and all detected faces
        target_face, all_faces = self.face_lock_system.process_frame(frame)
        
        # Extract position for servo tracking
        face_position = None
        if target_face:
            face_center_x = (target_face.x1 + target_face.x2) / 2.0
            face_position = face_center_x / self.frame_width
        
        # Publish state periodically
        self._maybe_publish_state(target_face, all_faces, face_position)
        
        # Track servo position if face is locked and servo is enabled
        if target_face and self.face_lock_system.is_locked and self.servo_enabled:
            self.servo_controller.track_face_position(
                face_center_x,
                self.frame_width,
                auto_publish=True
            )
        
        return {
            "target_face": target_face,
            "all_faces": all_faces,
            "face_position": face_position,
            "is_locked": self.face_lock_system.is_locked,
            "selected_name": self.face_lock_system.selected_name,
            "detected_blink": self.face_lock_system.is_blinking,
            "detected_smile": self.face_lock_system.is_smiling,
        }
    
    def _maybe_publish_state(
        self,
        target_face,
        all_faces,
        face_position: Optional[float]
    ):
        """Publish face state to MQTT with cooldown."""
        current_time = time.time()
        if (current_time - self.last_state_publish) < self.state_publish_cooldown:
            return
        
        if not self.mqtt_manager.is_connected:
            return
        
        if target_face and self.face_lock_system.is_locked:
            # Calculate confidence
            if self.face_lock_system.matcher._mat is not None:
                aligned, _ = align_face_5pt(
                    self.cap.read()[1],
                    target_face.kps,
                    out_size=(112, 112)
                )
                emb = self.face_lock_system.embedder.embed(aligned)
                sims = self.face_lock_system.matcher._mat @ emb.reshape(-1, 1)
                confidence = float(np.max(sims))
            else:
                confidence = None
            
            self.mqtt_manager.publish_face_state(
                face_name=self.face_lock_system.selected_name,
                is_locked=True,
                face_position=face_position,
                confidence=confidence
            )
        
        self.last_state_publish = current_time
    
    def draw_visualization(
        self,
        frame: np.ndarray,
        result: Dict[str, Any]
    ) -> np.ndarray:
        """Draw visualization on frame.
        
        Args:
            frame: Input frame
            result: Processing results
            
        Returns:
            Visualized frame
        """
        vis = frame.copy()
        H, W = vis.shape[:2]
        
        target_face = result["target_face"]
        all_faces = result["all_faces"]
        is_locked = result["is_locked"]
        selected_name = result["selected_name"]
        
        # Draw all detected faces
        for face in all_faces:
            is_target = target_face and (
                face.x1 == target_face.x1 and face.y1 == target_face.y1
            )
            
            if is_target:
                # Target face (locked or searching)
                color = (0, 255, 0) if is_locked else (0, 255, 255)
                status = "LOCKED" if is_locked else "SEARCHING"
                thickness = 3
                
                # Draw smoothed box if available
                if self.face_lock_system.smooth_box is not None:
                    box = self.face_lock_system.smooth_box.astype(int)
                    cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, thickness)
                    cv2.putText(
                        vis,
                        f"[{status}] {selected_name}",
                        (box[0], max(0, box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                else:
                    cv2.rectangle(
                        vis,
                        (face.x1, face.y1),
                        (face.x2, face.y2),
                        color,
                        thickness
                    )
            else:
                # Other detected faces
                color = (100, 100, 100)
                cv2.rectangle(
                    vis,
                    (face.x1, face.y1),
                    (face.x2, face.y2),
                    color,
                    1
                )
        
        # Draw keypoints for target face
        if target_face:
            for (x, y) in target_face.kps.astype(int):
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        
        # Draw status information
        status_y = 30
        
        # Header
        header = f"Faces: {len(all_faces)} | "
        header += f"FPS: {self.fps:.1f} | "
        header += f"Servo: {'ON' if self.servo_enabled else 'OFF'}"
        cv2.putText(
            vis,
            header,
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        status_y += 30
        
        # Target info
        if selected_name:
            target_info = f"Target: {selected_name} | "
            target_info += f"Thresh: {self.face_lock_system.matcher.dist_thresh:.2f}"
            cv2.putText(
                vis,
                target_info,
                (10, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            status_y += 30
        
        # Action indicators
        if is_locked:
            actions = []
            if result["detected_blink"]:
                actions.append("BLINKING")
            if result["detected_smile"]:
                actions.append("SMILING")
            
            if actions:
                action_text = " | ".join(actions)
                cv2.putText(
                    vis,
                    action_text,
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
        
        # Servo tracking indicator
        if self.servo_enabled and target_face:
            servo_angle = self.servo_controller.get_current_angle()
            servo_text = f"Servo: {servo_angle}°"
            cv2.putText(
                vis,
                servo_text,
                (W - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )
        
        return vis
    
    def run(self):
        """Run the integrated system."""
        logger.info("=" * 60)
        logger.info("Face Recognition & Locked Face Tracking System")
        logger.info("=" * 60)
        
        # Initialize components
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return
        
        self.connect_mqtt()
        
        # Get available identities
        names = sorted(self.face_lock_system.matcher.db.keys())
        if not names:
            logger.error("Database is empty! Run enrollment first.")
            return
        
        selected_idx = 0
        self.face_lock_system.set_selected_name(names[selected_idx])
        
        logger.info("\nControls:")
        logger.info("  n/p     : Next/Previous identity")
        logger.info("  +/-     : Adjust threshold")
        logger.info("  s       : Toggle servo tracking")
        logger.info("  c       : Call servo gesture")
        logger.info("  q       : Quit")
        logger.info("")
        
        # Main loop
        self.start_time = time.time()
        self.frame_count = 0
        
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Draw visualization
                vis = self.draw_visualization(frame, result)
                
                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                
                # Display
                cv2.imshow("Integrated Face Tracking System", vis)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Exiting...")
                    break
                
                elif key == ord("n"):
                    if len(names) > 1:
                        selected_idx = (selected_idx + 1) % len(names)
                        self.face_lock_system.set_selected_name(names[selected_idx])
                        logger.info(f"Selected: {names[selected_idx]}")
                
                elif key == ord("p"):
                    if len(names) > 1:
                        selected_idx = (selected_idx - 1) % len(names)
                        self.face_lock_system.set_selected_name(names[selected_idx])
                        logger.info(f"Selected: {names[selected_idx]}")
                
                elif key in (ord("+"), ord("=")):
                    self.face_lock_system.matcher.dist_thresh = min(
                        0.8,
                        self.face_lock_system.matcher.dist_thresh + 0.02
                    )
                    logger.info(f"Threshold: {self.face_lock_system.matcher.dist_thresh:.2f}")
                
                elif key == ord("-"):
                    self.face_lock_system.matcher.dist_thresh = max(
                        0.1,
                        self.face_lock_system.matcher.dist_thresh - 0.02
                    )
                    logger.info(f"Threshold: {self.face_lock_system.matcher.dist_thresh:.2f}")
                
                elif key == ord("s"):
                    self.servo_enabled = not self.servo_enabled
                    logger.info(f"Servo tracking: {'ENABLED' if self.servo_enabled else 'DISABLED'}")
                    if not self.servo_enabled:
                        self.servo_controller.reset_to_center()
                
                elif key == ord("c"):
                    logger.info("Calling servo gesture...")
                    self.servo_controller._gesture_acknowledge()
        
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        
        finally:
            logger.info("Cleaning up...")
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.mqtt_manager.disconnect()
            logger.info("✓ System shutdown complete")


def main():
    """Entry point for the integrated system."""
    system = IntegratedFaceTrackingSystem(
        db_path=Path("data/db/face_db.npz"),
        model_path="models/embedder_arcface.onnx",
        mqtt_broker="broker.hivemq.com",
        mqtt_port=1883,
        camera_index=0,
    )
    system.run()


if __name__ == "__main__":
    main()
