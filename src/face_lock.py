import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

try:
    import mediapipe as mp
except ImportError:
    mp = None

from .recognize import FaceDBMatcher, ArcFaceEmbedderONNX, HaarFaceMesh5pt, load_db_npz, align_face_5pt

@dataclass
class ActionRecord:
    timestamp: float
    action_type: str
    description: str

class FaceLockingSystem:
    def __init__(self, db_path: Path, model_path: str = "models/embedder_arcface.onnx"):
        self.db_path = db_path
        self.matcher = FaceDBMatcher(load_db_npz(db_path), dist_thresh=0.35)
        self.embedder = ArcFaceEmbedderONNX(model_path=model_path)
        self.detector = HaarFaceMesh5pt()
        
        # Locking state
        self.selected_name: Optional[str] = None
        self.is_locked = False
        self.last_seen_time = 0
        self.lock_timeout = 3.0 # seconds
        
        # Tracking Stability
        self.smooth_box: Optional[np.ndarray] = None # [x1, y1, x2, y2]
        self.prev_centroid: Optional[np.ndarray] = None
        self.ema_alpha = 0.5 # Smoothing for box
        
        # Action Detection State (v2 with temporal filtering)
        self.blink_frames = 0
        self.blink_req_frames = 2 # Minimum frames closed to count as blink
        self.is_blinking = False
        self.blink_threshold = 0.20 # Adjusted EAR
        
        self.smile_mar_buffer: List[float] = []
        self.smile_buffer_size = 5
        self.is_smiling = False
        self.smile_threshold = 0.45 
        
        self.history: List[ActionRecord] = []
        self.history_file: Optional[Path] = None

        # Landmarks indices for EAR/MAR
        self.L_EYE = [33, 160, 158, 133, 153, 144]
        self.R_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 0, 17]

    def set_selected_name(self, name: str):
        if name != self.selected_name:
            self.selected_name = name
            self.is_locked = False
            self.history = []
            self.history_file = None
            self.smooth_box = None
            self.prev_centroid = None
            print(f"Selected identity: {name}")

    def start_history_logging(self):
        if not self.selected_name: return
        ts = time.strftime("%Y%m%d%H%M%S")
        filename = f"{self.selected_name.lower()}_history_{ts}.txt"
        self.history_file = Path("data/history") / filename
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.record_action("LOCK_START", f"System locked onto {self.selected_name}")

    def record_action(self, action_type: str, description: str):
        now = time.time()
        record = ActionRecord(now, action_type, description)
        self.history.append(record)
        
        if self.history_file:
            time_str = time.strftime("%H:%M:%S", time.localtime(now))
            milli = int((now % 1) * 1000)
            with open(self.history_file, "a") as f:
                f.write(f"[{time_str}.{milli:03d}] {action_type}: {description}\n")

    def calculate_ear(self, landmarks, indices):
        p2, p3, p5, p6 = [landmarks[indices[i]] for i in [1, 2, 4, 5]]
        p1, p4 = [landmarks[indices[i]] for i in [0, 3]]
        d_v1 = np.linalg.norm(np.array([p2.x, p2.y]) - np.array([p6.x, p6.y]))
        d_v2 = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p5.x, p5.y]))
        d_h = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p4.x, p4.y]))
        return (d_v1 + d_v2) / (2.0 * d_h)

    def calculate_mar(self, landmarks, indices):
        p1, p2, p3, p4 = [landmarks[indices[i]] for i in range(4)]
        d_v = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p4.x, p4.y]))
        d_h = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
        return d_v / d_h

    def process_frame(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        faces = self.detector.detect(frame)
        
        locked_face_found = False
        target_f = None
        
        # 1. Identity Verification (with Grace Multiplier if already locked)
        curr_thresh = self.matcher.dist_thresh
        if self.is_locked:
            curr_thresh *= 1.4 # Sticker logic: harder to lose lock than to gain it
            
        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = self.embedder.embed(aligned)
            
            # Manual match to use our curr_thresh
            if self.matcher._mat is None: continue
            sims = self.matcher._mat @ emb.reshape(-1, 1)
            if sims.ndim > 1:
                sims = sims.flatten()
            i = int(np.argmax(sims))
            dist = 1.0 - float(sims[i])
            name = self.matcher._names[i]
            
            if dist <= curr_thresh and name == self.selected_name:
                locked_face_found = True
                target_f = f
                self.is_locked = True
                self.last_seen_time = time.time()
                if not self.history_file:
                    self.start_history_logging()
                break
        
        # 2. Stable Tracking (if recognition missed but identity was recently locked)
        if not locked_face_found and self.is_locked:
            now = time.time()
            if now - self.last_seen_time < self.lock_timeout:
                if self.prev_centroid is not None and faces:
                    best_dist = float('inf')
                    for f in faces:
                        curr_centroid = np.array([(f.x1 + f.x2)/2, (f.y1 + f.y2)/2])
                        dist = np.linalg.norm(curr_centroid - self.prev_centroid)
                        if dist < 150 and dist < best_dist: # Increased to 150
                            best_dist = dist
                            target_f = f
                            locked_face_found = True
                            # Update timeout so tracking keeps the search alive
                            self.last_seen_time = time.time() 
            else:
                self.is_locked = False
                self.record_action("LOCK_RELEASE", "Face lost for too long")
                self.history_file = None
                self.smooth_box = None

        # 3. Post-Process Locked Face (Smoothing & Action Detection)
        if target_f:
            # Box Smoothing (EMA)
            curr_box = np.array([target_f.x1, target_f.y1, target_f.x2, target_f.y2], dtype=np.float32)
            if self.smooth_box is None:
                self.smooth_box = curr_box
            else:
                self.smooth_box = self.ema_alpha * self.smooth_box + (1.0 - self.ema_alpha) * curr_box
            
            # Centroid & Movement
            curr_centroid = np.array([(self.smooth_box[0] + self.smooth_box[2])/2, 
                                      (self.smooth_box[1] + self.smooth_box[3])/2])
            if self.prev_centroid is not None:
                dx = curr_centroid[0] - self.prev_centroid[0]
                if dx > 20: self.record_action("MOVE", "Moved Right")
                elif dx < -20: self.record_action("MOVE", "Moved Left")
            self.prev_centroid = curr_centroid
            
            # Action Detection
            roi_x1, roi_y1, roi_x2, roi_y2 = self.smooth_box.astype(int)
            # Ensure ROI is within frame
            roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
            roi_x2, roi_y2 = min(W, roi_x2), min(H, roi_y2)
            
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size > 0:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                # Use the detector's FaceLandmarker (MediaPipe Tasks API)
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
                    res = self.detector.landmarker.detect(mp_image)
                except Exception:
                    res = None

                if res and getattr(res, 'face_landmarks', None):
                    lms = res.face_landmarks[0]
                    
                    # EAR (Blink with temporal filtering)
                    ear = self.calculate_ear(lms, self.L_EYE + self.R_EYE) # simple approach
                    ear_l = self.calculate_ear(lms, self.L_EYE)
                    ear_r = self.calculate_ear(lms, self.R_EYE)
                    ear = (ear_l + ear_r) / 2.0
                    
                    if ear < self.blink_threshold:
                        self.blink_frames += 1
                    else:
                        if self.blink_frames >= self.blink_req_frames:
                            self.record_action("BLINK", "Eye blink detected")
                        self.blink_frames = 0
                    self.is_blinking = (self.blink_frames > 0)
                    
                    # MAR (Smile with Smoothing)
                    mar = self.calculate_mar(lms, self.MOUTH)
                    self.smile_mar_buffer.append(mar)
                    if len(self.smile_mar_buffer) > self.smile_buffer_size:
                        self.smile_mar_buffer.pop(0)
                    
                    avg_mar = sum(self.smile_mar_buffer) / len(self.smile_mar_buffer)
                    if avg_mar > self.smile_threshold:
                        if not self.is_smiling:
                            self.is_smiling = True
                            self.record_action("EXPRESSION", "Smile/Laugh detected")
                    else:
                        self.is_smiling = False

        return target_f, faces

def main():
    db_path = Path("data/db/face_db.npz")
    system = FaceLockingSystem(db_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not available")
        return

    names = sorted(system.matcher.db.keys())
    if not names:
        print("\n[ERROR] Database is empty! Enrollment required.\n")
    
    selected_idx = 0
    if names:
        system.set_selected_name(names[selected_idx])

    print("--- Face Locking System v2 ---")
    print("Keys:")
    print("  n/p : Next/Prev identity")
    print("  +/- : Adjust distance threshold")
    print("  q   : Quit")
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        
        target_f, all_faces = system.process_frame(frame)
        vis = frame.copy()
        
        if not names:
            cv2.putText(vis, "DATABASE EMPTY - ENROLL FIRST", (50, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for f in all_faces:
                label = "Unknown"
                color = (100, 100, 100)
                
                # Check recognition for everyone for display
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = system.embedder.embed(aligned)
                mr = system.matcher.match(emb)
                if mr.accepted: label = mr.name

                is_target = target_f and f.x1 == target_f.x1 and f.y1 == target_f.y1
                
                if is_target:
                    color = (0, 255, 0) if system.is_locked else (0, 255, 255)
                    status = "LOCKED" if system.is_locked else "SEARCHING"
                    
                    # Also calculate current distance for target for UI
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = system.embedder.embed(aligned)
                    sims = system.matcher._mat @ emb.reshape(-1, 1)
                    dist = 1.0 - float(np.max(sims))
                    
                    # Use smoothed box for target display
                    bx = system.smooth_box.astype(int) if system.smooth_box is not None else [f.x1, f.y1, f.x2, f.y2]
                    cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), color, 3)
                    cv2.putText(vis, f"[{status}] {system.selected_name} ({dist:.2f})", (bx[0], bx[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 1)
                    cv2.putText(vis, label, (f.x1, f.y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Overlay status
        total = len(names)
        curr_idx_display = selected_idx + 1 if total > 0 else 0
        target_name = system.selected_name if system.selected_name else "N/A"
        
        cv2.putText(vis, f"Target: {target_name} ({curr_idx_display}/{total})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Thresh: {system.matcher.dist_thresh:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if system.is_locked:
            y_off = 90
            if system.is_blinking:
                cv2.putText(vis, "BLINKING", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                y_off += 25
            if system.is_smiling:
                cv2.putText(vis, "SMILING", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Face Locking", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            if len(names) > 1:
                selected_idx = (selected_idx + 1) % len(names)
                system.set_selected_name(names[selected_idx])
        elif key == ord('p'):
            if len(names) > 1:
                selected_idx = (selected_idx - 1) % len(names)
                system.set_selected_name(names[selected_idx])
        elif key in (ord('+'), ord('=')):
            system.matcher.dist_thresh = min(0.8, system.matcher.dist_thresh + 0.02)
        elif key == ord('-'):
            system.matcher.dist_thresh = max(0.1, system.matcher.dist_thresh - 0.02)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()