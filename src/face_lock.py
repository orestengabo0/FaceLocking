# src/face_lock.py
"""
face_lock.py
Main face locking module.

Implements stable face tracking with action detection and history recording.

Features:
- Manual identity selection (choose which face to lock)
- Robust face locking with timeout
- Stable tracking across frames
- Action detection while locked
- Persistent action history to file

Run:
python -m src.face_lock
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .action_detector import ActionDetector, Action
from .face_history_logger import FaceHistoryLogger
from .camera_display import CameraDisplay


# =====================================================================
# Face Database & Matcher (from recognize.py, simplified)
# =====================================================================

def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    """Load face database from NPZ file."""
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity (both vectors must be L2-normalized)."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance = 1 - cosine_similarity."""
    return 1.0 - cosine_similarity(a, b)


# =====================================================================
# Face Lock State Machine
# =====================================================================

class FaceLockState:
    """Represents the state of the face lock."""

    SEARCHING = "searching"  # looking for target face
    LOCKED = "locked"  # target face found and locked
    LOST = "lost"  # target face temporarily lost but not released

    def __init__(self):
        self.state = self.SEARCHING
        self.locked_identity: Optional[str] = None
        self.locked_bbox: Optional[Tuple[int, int, int, int]] = None
        self.locked_kps: Optional[np.ndarray] = None
        self.lock_confidence: float = 0.0
        self.frames_since_detection = 0
        self.lock_acquired_time: Optional[float] = None


# =====================================================================
# Main Face Locking System
# =====================================================================

class FaceLockSystem:
    """
    Face locking and action detection system.

    Workflow:
    1. User selects target identity
    2. System searches for this identity
    3. When found with high confidence, acquires lock
    4. While locked, system tracks face and detects actions
    5. If face disappears, lock is held for N frames
    6. If face reappears, lock is re-acquired
    7. User can release lock manually
    """

    def __init__(
        self,
        db_path: Path = Path("data/db/face_db.npz"),
        enroll_dir: Path = Path("data/enroll"),
        model_path: Path = Path("models/embedder_arcface.onnx"),
        distance_threshold: float = 0.54,
        lock_timeout_frames: int = 30,
        min_lock_confidence: float = 0.65,
    ):
        """
        Args:
            db_path: path to face database NPZ
            model_path: path to ArcFace ONNX model
            distance_threshold: cosine distance threshold for recognition
            lock_timeout_frames: frames to wait before releasing lock if face lost
            min_lock_confidence: minimum confidence to acquire lock
        """
        # Load database
        self.db = load_db_npz(db_path)
        self.db_names = sorted(self.db.keys())

        # Initialize components
        self.detector = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)
        self.embedder = ArcFaceEmbedderONNX(model_path=model_path, debug=False)
        self.action_detector = ActionDetector()

        # Configuration
        self.distance_threshold = float(distance_threshold)
        self.lock_timeout_frames = int(lock_timeout_frames)
        self.min_lock_confidence = float(min_lock_confidence)

        # State
        self.state = FaceLockState()
        self.history_logger: Optional[FaceHistoryLogger] = None

    def select_target(self, face_name: str) -> bool:
        """
        Select target identity to lock.

        Args:
            face_name: name of enrolled identity

        Returns:
            True if name exists in database, False otherwise
        """
        if face_name.lower() not in [n.lower() for n in self.db_names]:
            return False

        # Match case to database
        for db_name in self.db_names:
            if db_name.lower() == face_name.lower():
                self.state.locked_identity = db_name
                break

        # Initialize history logger
        self.history_logger = FaceHistoryLogger(
            face_name=self.state.locked_identity,
            output_dir=Path("data/face_histories"),
        )
        self.history_logger.log_status(f"Target face selected: {self.state.locked_identity}")

        return True

    def _recognize_face(self, aligned_face: np.ndarray) -> Tuple[Optional[str], float, float]:
        """
        Recognize a single aligned face.

        Returns:
            (identity_name, distance, confidence) or (None, 1.0, 0.0) if unknown
        """
        if not self.db_names:
            return None, 1.0, 0.0

        # Get embedding
        emb_result = self.embedder.embed(aligned_face)
        emb = emb_result.embedding  # Extract numpy array from EmbeddingResult

        # Find best match
        best_name = None
        best_distance = float("inf")
        best_confidence = 0.0

        for db_name in self.db_names:
            db_emb = self.db[db_name]
            dist = cosine_distance(emb, db_emb)

            if dist < best_distance:
                best_distance = dist
                best_name = db_name if dist <= self.distance_threshold else None
                best_confidence = max(0.0, 1.0 - dist)

        return best_name, best_distance, best_confidence

    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        Process a single frame for face locking and action detection.

        Args:
            frame_bgr: BGR frame from camera

        Returns:
            Dictionary with state information:
            {
                "state": "searching" | "locked" | "lost",
                "locked_identity": name or None,
                "face_box": (x1, y1, x2, y2) or None,
                "face_kps": (5, 2) array or None,
                "recognition_distance": float,
                "lock_confidence": float,
                "actions": [Action, ...],
                "time_locked_seconds": float or None,
            }
        """
        H, W = frame_bgr.shape[:2]
        now = time.time()

        # Detect faces
        detected_faces = self.detector.detect(frame_bgr, max_faces=1)

        result = {
            "state": self.state.state,
            "locked_identity": self.state.locked_identity,
            "face_box": self.state.locked_bbox,
            "face_kps": self.state.locked_kps,
            "recognition_distance": None,
            "lock_confidence": self.state.lock_confidence,
            "actions": [],
            "time_locked_seconds": None,
        }

        # No faces detected
        if not detected_faces:
            self.state.frames_since_detection += 1

            if self.state.state == self.state.LOCKED:
                if self.state.frames_since_detection > self.lock_timeout_frames:
                    # Timeout, release lock
                    self.state.state = self.state.SEARCHING
                    self.state.locked_bbox = None
                    self.state.locked_kps = None
                    if self.history_logger:
                        self.history_logger.log_status("Lock LOST (face disappeared)")
                else:
                    # Still in timeout window, hold lock
                    self.state.state = self.state.LOST
                    result["state"] = self.state.LOST

            return result

        # Face detected
        face = detected_faces[0]
        self.state.frames_since_detection = 0

        # Align and recognize
        aligned, _ = align_face_5pt(frame_bgr, face.kps, out_size=(112, 112))
        identity, distance, confidence = self._recognize_face(aligned)

        result["recognition_distance"] = float(distance)

        # State machine: SEARCHING -> LOCKED or stay SEARCHING
        if self.state.state == self.state.SEARCHING:
            if identity == self.state.locked_identity and confidence >= self.min_lock_confidence:
                # Lock acquired
                self.state.state = self.state.LOCKED
                self.state.locked_bbox = (face.x1, face.y1, face.x2, face.y2)
                self.state.locked_kps = face.kps.copy()
                self.state.lock_confidence = confidence
                self.state.lock_acquired_time = now

                result["state"] = self.state.LOCKED
                result["face_box"] = self.state.locked_bbox
                result["face_kps"] = self.state.locked_kps
                result["lock_confidence"] = confidence

                if self.history_logger:
                    self.history_logger.log_status(
                        f"Lock ACQUIRED for {self.state.locked_identity} (confidence={confidence:.3f})"
                    )

        # State machine: LOCKED -> LOCKED or LOST
        elif self.state.state == self.state.LOCKED:
            # Update tracked position
            self.state.locked_bbox = (face.x1, face.y1, face.x2, face.y2)
            self.state.locked_kps = face.kps.copy()
            self.state.lock_confidence = confidence

            result["state"] = self.state.LOCKED
            result["face_box"] = self.state.locked_bbox
            result["face_kps"] = self.state.locked_kps
            result["lock_confidence"] = confidence

            # Detect actions only while locked
            actions = self.action_detector.detect(face.kps)
            result["actions"] = actions

            if self.history_logger:
                self.history_logger.log_actions(actions)

            # Verify lock still valid (optional: re-check identity)
            if identity != self.state.locked_identity:
                # Wrong identity detected, stay locked but log
                pass

        # State machine: LOST -> LOCKED (face reappeared) or LOST (still lost)
        elif self.state.state == self.state.LOST:
            if identity == self.state.locked_identity and confidence >= self.min_lock_confidence:
                # Lock re-acquired
                self.state.state = self.state.LOCKED
                self.state.locked_bbox = (face.x1, face.y1, face.x2, face.y2)
                self.state.locked_kps = face.kps.copy()
                self.state.lock_confidence = confidence

                result["state"] = self.state.LOCKED
                result["face_box"] = self.state.locked_bbox
                result["face_kps"] = self.state.locked_kps

                if self.history_logger:
                    self.history_logger.log_status(
                        f"Lock RE-ACQUIRED for {self.state.locked_identity}"
                    )

        # Time locked
        if self.state.lock_acquired_time is not None and self.state.state in (
            self.state.LOCKED,
            self.state.LOST,
        ):
            result["time_locked_seconds"] = now - self.state.lock_acquired_time

        return result

    def release_lock(self) -> None:
        """Manually release the lock."""
        if self.state.state in (self.state.LOCKED, self.state.LOST):
            self.state.state = self.state.SEARCHING
            self.state.locked_bbox = None
            self.state.locked_kps = None
            self.state.lock_acquired_time = None
            if self.history_logger:
                self.history_logger.log_status("Lock RELEASED by user")

    def finalize_session(self) -> str:
        """
        Finalize the locking session and save history.

        Returns:
            path to history file
        """
        if self.history_logger:
            return self.history_logger.finalize()
        return ""


# =====================================================================
# UI & Demo
# =====================================================================

def _put_text(img, text: str, xy=(10, 30), scale=0.8, thickness=2):
    """Draw text with white color and black outline."""
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)


def main():
    """Interactive face locking demo."""
    # Load system WITHOUT opening camera yet
    system = FaceLockSystem(
        enroll_dir=Path("data/enroll"),
        distance_threshold=0.54,
    )
    
    # Get available faces
    if not system.db_names:
        print("No enrolled faces found. Run enrollment first.")
        return

    print("\n" + "=" * 80)
    print("FACE LOCKING SYSTEM")
    print("=" * 80)
    print(f"\nAvailable faces: {', '.join(system.db_names)}\n")

    # User selects target BEFORE opening camera
    while True:
        target = input("Select face to lock (or 'q' to quit): ").strip()
        if target.lower() == "q":
            return
        if system.select_target(target):
            print(f"✓ Target selected: {target}")
            break
        print(f"✗ Face '{target}' not found. Try again.")

    # NOW open camera after selection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")
    
    # Create large display manager
    display = CameraDisplay(mode=CameraDisplay.LARGE)
    display.create_window("Face Locking", resizable=True)

    print("\nStarting face locking...")
    print("Controls:")
    print("  r  : release lock")
    print("  q  : quit")
    print("=" * 80 + "\n")

    t0 = time.time()
    frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Process frame
            result = system.process_frame(frame)

            # Visualize
            vis = frame.copy()
            H, W = vis.shape[:2]

            # Draw state indicator
            state_text = result["state"].upper()
            if result["state"] == "locked":
                state_color = (0, 255, 0)
                state_symbol = "🔒"
            elif result["state"] == "lost":
                state_color = (0, 165, 255)
                state_symbol = "⏱"
            else:
                state_color = (0, 0, 255)
                state_symbol = "🔍"

            cv2.rectangle(vis, (5, 5), (W - 5, 50), (0, 0, 0), -1)
            cv2.putText(
                vis,
                f"Lock: {state_text} | Target: {result['locked_identity']}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                state_color,
                2,
            )

            # Draw detected face if locked
            if result["state"] in ("locked", "lost") and result["face_box"]:
                x1, y1, x2, y2 = result["face_box"]
                color = state_color
                thickness = 3 if result["state"] == "locked" else 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

                # Draw landmarks
                if result["face_kps"] is not None:
                    for (x, y) in result["face_kps"].astype(int):
                        cv2.circle(vis, (int(x), int(y)), 3, color, -1)

                # Draw info
                info_y = y1 - 10 if y1 > 40 else y2 + 20
                conf_text = f"Conf: {result['lock_confidence']:.2f}"
                if result["time_locked_seconds"] is not None:
                    time_text = f" | Time: {result['time_locked_seconds']:.1f}s"
                else:
                    time_text = ""

                cv2.putText(
                    vis,
                    conf_text + time_text,
                    (x1, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # Show actions detected
                if result["actions"]:
                    action_text = " | ".join([a.action_type for a in result["actions"]])
                    cv2.putText(
                        vis,
                        f"Actions: {action_text}",
                        (x1, info_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            # FPS
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (W - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Face Locking", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                system.release_lock()
                print("✗ Lock released by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Finalize
        history_file = system.finalize_session()
        print(f"\n✓ Session ended")
        print(f"✓ History saved to: {history_file}")
        if system.history_logger:
            print(system.history_logger.get_summary())


if __name__ == "__main__":
    main()