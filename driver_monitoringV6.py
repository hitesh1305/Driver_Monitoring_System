import cv2
import numpy as np
import time
import threading
from collections import deque
from datetime import datetime
import json
import logging
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.spatial import distance as dist
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import subprocess
import psutil  # Added for process management
warnings.filterwarnings('ignore')

@dataclass
class DriverState:
    consciousness_level: str
    confidence: float
    eye_aspect_ratio: float
    mouth_aspect_ratio: float
    head_pose: Tuple[float, float, float]
    blink_rate: float
    yawn_detected: bool
    microsleep_detected: bool
    timestamp: datetime

class DriverMonitoringSystem:
    def __init__(self, camera_id=0):
        self.last_alert_time = 0
        self.ALERT_COOLDOWN = 30  # 30 seconds between alerts to prevent spam
        self.alert_process = None  # Track the alert GUI process
        self.alert_launched = False  # Flag to track if alert has been launched
        self.alert_pid = None  # Track the process ID
        
        """
        Comprehensive Driver Monitoring System using MediaPipe (NO DLIB REQUIRED!)
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize MediaPipe Face Mesh (replaces dlib)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe Face Detection for backup
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        )
        
        # Eye landmark indices for MediaPipe (468 face landmarks)
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH_LANDMARKS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 91]
        
        # Simplified eye and mouth points for easier calculation
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]  # Key eye landmarks
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]  # Key eye landmarks
        self.MOUTH_POINTS = [78, 81, 13, 311, 308, 415, 310, 317, 14, 87]  # Key mouth landmarks
        
        # FIXED THRESHOLDS - More reasonable values
        self.EYE_AR_THRESH = 0.2  # Lowered threshold
        self.EYE_AR_CONSEC_FRAMES = 15  # Increased frames needed
        self.MOUTH_AR_THRESH = 0.7  # Increased threshold
        self.YAWN_CONSEC_FRAMES = 20  # Increased frames needed
        self.MICROSLEEP_THRESH = 3.0  # Increased time threshold
        self.DROWSINESS_THRESH = 0.4  # More lenient threshold
        
        # State tracking variables
        self.eye_counter = 0
        self.yawn_counter = 0
        self.blink_counter = 0
        self.frame_counter = 0
        self.start_time = time.time()
        self.eyes_closed_start = None
        
        # Historical data for pattern analysis
        self.ear_history = deque(maxlen=100)
        self.mar_history = deque(maxlen=100)
        self.head_pose_history = deque(maxlen=50)
        self.consciousness_history = deque(maxlen=20)
        
        # Machine learning model
        self.scaler = StandardScaler()
        self.classifier = self._initialize_classifier()
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('driver_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.current_state = DriverState(
            consciousness_level="CONSCIOUS",
            confidence=1.0,
            eye_aspect_ratio=0.3,
            mouth_aspect_ratio=0.3,
            head_pose=(0, 0, 0),
            blink_rate=15.0,
            yawn_detected=False,
            microsleep_detected=False,
            timestamp=datetime.now()
        )
        
        print("🚗 Driver Monitoring System initialized successfully!")
        print("✅ Using MediaPipe - NO external model files needed!")
        print("🎥 Camera ready for monitoring...")
    
    def _initialize_classifier(self):
        """Initialize and return a machine learning classifier"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
    
    def calculate_eye_aspect_ratio_mediapipe(self, landmarks, eye_points, img_width, img_height):
        """Calculate Eye Aspect Ratio using MediaPipe landmarks"""
        # Convert normalized coordinates to pixel coordinates
        eye_coords = []
        for point_idx in eye_points:
            if point_idx < len(landmarks):
                x = int(landmarks[point_idx].x * img_width)
                y = int(landmarks[point_idx].y * img_height)
                eye_coords.append([x, y])
        
        if len(eye_coords) < 6:
            return 0.3  # Default value
        
        # Calculate EAR using the 6 key points
        # Vertical eye landmarks
        A = dist.euclidean(eye_coords[1], eye_coords[5])
        B = dist.euclidean(eye_coords[2], eye_coords[4])
        # Horizontal eye landmark
        C = dist.euclidean(eye_coords[0], eye_coords[3])
        
        # Calculate EAR
        if C == 0:
            return 0.3
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mouth_aspect_ratio_mediapipe(self, landmarks, mouth_points, img_width, img_height):
        """Calculate Mouth Aspect Ratio using MediaPipe landmarks"""
        # Convert normalized coordinates to pixel coordinates
        mouth_coords = []
        for point_idx in mouth_points:
            if point_idx < len(landmarks):
                x = int(landmarks[point_idx].x * img_width)
                y = int(landmarks[point_idx].y * img_height)
                mouth_coords.append([x, y])
        
        if len(mouth_coords) < 6:
            return 0.3  # Default value
        
        # Calculate MAR using key mouth points
        # Vertical mouth landmarks
        A = dist.euclidean(mouth_coords[1], mouth_coords[7])
        B = dist.euclidean(mouth_coords[2], mouth_coords[6])
        # Horizontal mouth landmark
        C = dist.euclidean(mouth_coords[0], mouth_coords[4])
        
        # Calculate MAR
        if C == 0:
            return 0.3
        mar = (A + B) / (2.0 * C)
        return mar
    
    def estimate_head_pose_mediapipe(self, landmarks, img_width, img_height):
        """Estimate head pose using MediaPipe face landmarks"""
        # Key facial landmarks for pose estimation
        nose_tip = [landmarks[1].x * img_width, landmarks[1].y * img_height, landmarks[1].z * img_width]
        chin = [landmarks[175].x * img_width, landmarks[175].y * img_height, landmarks[175].z * img_width]
        left_eye = [landmarks[33].x * img_width, landmarks[33].y * img_height, landmarks[33].z * img_width]
        right_eye = [landmarks[362].x * img_width, landmarks[362].y * img_height, landmarks[362].z * img_width]
        left_mouth = [landmarks[61].x * img_width, landmarks[61].y * img_height, landmarks[61].z * img_width]
        right_mouth = [landmarks[291].x * img_width, landmarks[291].y * img_height, landmarks[291].z * img_width]
        
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye
            (225.0, 170.0, -135.0),      # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points
        image_points = np.array([
            nose_tip[:2],
            chin[:2],
            left_eye[:2],
            right_eye[:2],
            left_mouth[:2],
            right_mouth[:2]
        ], dtype="double")
        
        # Camera parameters
        center = (img_width/2, img_height/2)
        focal_length = center[0] / np.tan(60/2 * np.pi / 180)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1))
        
        # Solve PnP
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Calculate Euler angles
                sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
                singular = sy < 1e-6
                
                if not singular:
                    x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                    y = np.arctan2(-rotation_matrix[2,0], sy)
                    z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
                else:
                    x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                    y = np.arctan2(-rotation_matrix[2,0], sy)
                    z = 0
                
                # Convert to degrees
                pitch = np.degrees(x)
                yaw = np.degrees(y)
                roll = np.degrees(z)
                
                return pitch, yaw, roll
        except:
            pass
        
        return 0.0, 0.0, 0.0  # Default values if estimation fails
    
    def extract_features(self, ear_left, ear_right, mar, head_pose):
        """Extract comprehensive features for ML classification"""
        features = []
        
        # Eye features
        features.extend([ear_left, ear_right, (ear_left + ear_right) / 2])
        
        # Mouth features
        features.append(mar)
        
        # Head pose features
        features.extend(head_pose)
        
        # Blink rate (blinks per minute)
        current_time = time.time()
        time_diff = current_time - self.start_time
        blink_rate = (self.blink_counter / max(time_diff, 1)) * 60
        features.append(blink_rate)
        
        # Historical features
        if len(self.ear_history) > 10:
            features.extend([
                np.mean(list(self.ear_history)[-10:]),
                np.std(list(self.ear_history)[-10:]),
                np.mean(list(self.mar_history)[-10:]),
                np.std(list(self.mar_history)[-10:])
            ])
        else:
            features.extend([0.3, 0.1, 0.3, 0.1])  # Default values
        
        # Eye closure duration
        if self.eye_counter > 0:
            closure_duration = self.eye_counter / 30.0  # Assuming 30 FPS
            features.append(closure_duration)
        else:
            features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def classify_consciousness_state(self, features):
        """FIXED: More responsive consciousness state classification"""
        ear_avg = features[0][2]  # Average EAR
        mar = features[0][3]      # MAR
        head_pose = features[0][4:7]  # Pitch, Yaw, Roll
        blink_rate = features[0][7]
        closure_duration = features[0][-1]
        
        # Start with high consciousness score
        consciousness_score = 1.0
        state = "CONSCIOUS"
        
        # FIXED: More lenient eye-based indicators
        if ear_avg < self.EYE_AR_THRESH:
            # Only penalize if eyes are closed for extended period
            if closure_duration > 1.0:  # More than 1 second
                consciousness_score -= 0.2
            if closure_duration > self.MICROSLEEP_THRESH:
                consciousness_score -= 0.3
                state = "MICROSLEEP"
        else:
            # Eyes are open - reset penalties and boost score
            consciousness_score = min(1.0, consciousness_score + 0.1)
        
        # FIXED: More reasonable blink rate analysis
        if blink_rate < 3:  # Very low blink rate (more lenient)
            consciousness_score -= 0.1
        elif blink_rate > 40:  # Very high blink rate (more lenient)
            consciousness_score -= 0.15
        
        # FIXED: Yawn detection with higher threshold
        if mar > self.MOUTH_AR_THRESH and self.yawn_counter > self.YAWN_CONSEC_FRAMES:
            consciousness_score -= 0.1  # Reduced penalty
        
        # FIXED: More lenient head pose analysis
        pitch, yaw, roll = head_pose
        if abs(pitch) > 30 or abs(yaw) > 35 or abs(roll) > 25:  # Increased thresholds
            consciousness_score -= 0.15  # Reduced penalty
        
        # FIXED: Extreme head movements (more lenient)
        if abs(pitch) > 45:  # Increased threshold
            consciousness_score -= 0.2  # Reduced penalty
        
        # FIXED: More reasonable state determination
        if consciousness_score > 0.85:
            state = "CONSCIOUS"
        elif consciousness_score > 0.7:
            state = "ALERT_FATIGUE"
        elif consciousness_score > 0.5:  # Increased threshold
            state = "DROWSY"
        elif consciousness_score > 0.3:  # Increased threshold
            state = "SEVERELY_DROWSY"
        else:
            state = "UNCONSCIOUS"
        
        # FIXED: Special case for microsleep - more strict conditions
        if closure_duration > self.MICROSLEEP_THRESH and ear_avg < 0.1:  # Lower EAR threshold
            state = "MICROSLEEP"
            consciousness_score = 0.2  # Less severe penalty
        
        # BOOST: If eyes are clearly open, ensure conscious state
        if ear_avg > 0.25:  # Eyes clearly open
            if state not in ["CONSCIOUS", "ALERT_FATIGUE"]:
                state = "CONSCIOUS"
                consciousness_score = max(0.8, consciousness_score)
        
        return state, max(0.0, min(1.0, consciousness_score))
    
    def detect_anomalies(self):
        """Detect anomalous patterns that might indicate unconsciousness"""
        if len(self.consciousness_history) < 5:
            return False
        
        recent_states = list(self.consciousness_history)[-5:]
        unconscious_count = sum(1 for state in recent_states if state in ["UNCONSCIOUS", "SEVERELY_DROWSY", "MICROSLEEP"])
        
        return unconscious_count >= 4  # Increased threshold
    
    def is_alert_process_running(self):
        """Enhanced check if alert GUI process is still running"""
        if self.alert_process is None and self.alert_pid is None:
            return False
        
        # Check using subprocess object
        if self.alert_process is not None:
            try:
                return self.alert_process.poll() is None
            except:
                self.alert_process = None
        
        # Check using PID with psutil for more reliable detection
        if self.alert_pid is not None:
            try:
                process = psutil.Process(self.alert_pid)
                return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.alert_pid = None
                return False
        
        return False
    
    def kill_existing_alert_processes(self):
        """Kill any existing alert_gui.py processes to ensure single instance"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and any('alert_gui.py' in arg for arg in proc.info['cmdline']):
                        print(f"🔄 Terminating existing alert process (PID: {proc.info['pid']})")
                        proc.terminate()
                        proc.wait(timeout=3)  # Wait up to 3 seconds for clean termination
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
        except Exception as e:
            print(f"⚠️  Warning: Could not check for existing processes: {e}")
    
    def process_frame(self, frame):
        """Process a single frame and return driver state"""
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # No face detected - potential emergency
            self.current_state.consciousness_level = "NO_FACE_DETECTED"
            self.current_state.confidence = 0.0
            return self.current_state, frame
        
        # Process the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        # Calculate eye aspect ratios
        ear_left = self.calculate_eye_aspect_ratio_mediapipe(landmarks, self.LEFT_EYE_POINTS, width, height)
        ear_right = self.calculate_eye_aspect_ratio_mediapipe(landmarks, self.RIGHT_EYE_POINTS, width, height)
        ear_avg = (ear_left + ear_right) / 2.0
        
        # Calculate mouth aspect ratio
        mar = self.calculate_mouth_aspect_ratio_mediapipe(landmarks, self.MOUTH_POINTS, width, height)
        
        # Estimate head pose
        head_pose = self.estimate_head_pose_mediapipe(landmarks, width, height)
        
        # Update histories
        self.ear_history.append(ear_avg)
        self.mar_history.append(mar)
        self.head_pose_history.append(head_pose)
        
        # FIXED: Better blink and yawn detection
        if ear_avg < self.EYE_AR_THRESH:
            self.eye_counter += 1
            if self.eyes_closed_start is None:
                self.eyes_closed_start = time.time()
        else:
            if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                self.blink_counter += 1
            self.eye_counter = 0
            self.eyes_closed_start = None
        
        yawn_detected = False
        if mar > self.MOUTH_AR_THRESH:
            self.yawn_counter += 1
            if self.yawn_counter >= self.YAWN_CONSEC_FRAMES:
                yawn_detected = True
        else:
            self.yawn_counter = 0
        
        # Extract features and classify
        features = self.extract_features(ear_left, ear_right, mar, head_pose)
        consciousness_state, confidence = self.classify_consciousness_state(features)
        
        # Update consciousness history
        self.consciousness_history.append(consciousness_state)
        
        # FIXED: Better microsleep detection
        current_time = time.time()
        if self.eyes_closed_start:
            eyes_closed_duration = current_time - self.eyes_closed_start
            microsleep_detected = eyes_closed_duration > self.MICROSLEEP_THRESH
        else:
            microsleep_detected = False
        
        # Calculate blink rate
        time_diff = current_time - self.start_time
        blink_rate = (self.blink_counter / max(time_diff, 1)) * 60
        
        # Update current state
        self.current_state = DriverState(
            consciousness_level=consciousness_state,
            confidence=confidence,
            eye_aspect_ratio=ear_avg,
            mouth_aspect_ratio=mar,
            head_pose=head_pose,
            blink_rate=blink_rate,
            yawn_detected=yawn_detected,
            microsleep_detected=microsleep_detected,
            timestamp=datetime.now()
        )
        
        # Draw annotations on frame
        annotated_frame = self.draw_annotations(frame, face_landmarks, self.current_state, width, height)
        
        return self.current_state, annotated_frame
    
    def draw_annotations(self, frame, face_landmarks, state, width, height):
        """Draw visual annotations on the frame"""
        # Draw face mesh landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            None,
            self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        
        # Draw eye landmarks
        for point_idx in self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS:
            if point_idx < len(face_landmarks.landmark):
                x = int(face_landmarks.landmark[point_idx].x * width)
                y = int(face_landmarks.landmark[point_idx].y * height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw mouth landmarks
        for point_idx in self.MOUTH_POINTS:
            if point_idx < len(face_landmarks.landmark):
                x = int(face_landmarks.landmark[point_idx].x * width)
                y = int(face_landmarks.landmark[point_idx].y * height)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        
        # Status display
        status_color = self.get_status_color(state.consciousness_level)
        
        # Display information
        info_text = [
            f"State: {state.consciousness_level}",
            f"Confidence: {state.confidence:.2f}",
            f"EAR: {state.eye_aspect_ratio:.3f}",
            f"MAR: {state.mouth_aspect_ratio:.3f}",
            f"Blink Rate: {state.blink_rate:.1f}/min",
            f"Head Pose: P:{state.head_pose[0]:.1f} Y:{state.head_pose[1]:.1f} R:{state.head_pose[2]:.1f}",
            f"Alert Status: {'ACTIVE' if self.is_alert_process_running() else 'STANDBY'}"
        ]
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (450, 225), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 225), status_color, 3)
        
        # Draw text
        for i, text in enumerate(info_text):
            y_pos = 35 + i * 25
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Warning indicators
        if state.yawn_detected:
            cv2.putText(frame, "YAWN DETECTED!", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if state.microsleep_detected:
            cv2.putText(frame, "MICROSLEEP ALERT!", (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add MediaPipe attribution
        cv2.putText(frame, "Powered by MediaPipe", (width-200, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        return frame
    
    def get_status_color(self, state):
        """Get color based on consciousness state"""
        color_map = {
            "CONSCIOUS": (0, 255, 0),           # Green
            "ALERT_FATIGUE": (0, 255, 255),    # Yellow
            "DROWSY": (0, 165, 255),           # Orange
            "SEVERELY_DROWSY": (0, 0, 255),    # Red
            "UNCONSCIOUS": (0, 0, 139),        # Dark Red
            "MICROSLEEP": (128, 0, 128),       # Purple
            "NO_FACE_DETECTED": (255, 255, 255) # White
        }
        return color_map.get(state, (255, 255, 255))
    
    def trigger_emergency_alert(self, state):
        """Enhanced emergency alert system - ENSURES ONLY ONE INSTANCE"""
        current_time = time.time()
        
        # Check if enough time has passed since last alert
        if (current_time - self.last_alert_time) < self.ALERT_COOLDOWN:
            return  # Still in cooldown period, skip alert
        
        # Check if alert GUI is already running
        if self.is_alert_process_running():
            print("🔒 Alert GUI already running - Skipping duplicate alert")
            return  # Alert GUI is still running, don't open another
        
        # Check if we've already launched an alert for this session
        if self.alert_launched:
            print("🔒 Alert already launched this session - Skipping duplicate")
            return  # Already launched alert, don't spam
        
        # Kill any existing alert processes first (safety measure)
        self.kill_existing_alert_processes()
        time.sleep(0.5)  # Brief pause to allow cleanup
        
        # Update last alert time and set flag
        self.last_alert_time = current_time
        self.alert_launched = True
        
        alert_data = {
            'timestamp': state.timestamp.isoformat(),
            'state': state.consciousness_level,
            'confidence': state.confidence,
            'eye_aspect_ratio': state.eye_aspect_ratio,
            'head_pose': state.head_pose,
            'location': 'GPS_COORDINATES_HERE',  # Integrate with GPS
            'vehicle_id': 'VEHICLE_ID_HERE'
        }
        
        # Save alert data to file for the GUI to read
        try:
            with open('alert_data.json', 'w') as f:
                json.dump(alert_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save alert data: {e}")
        
        # Launch alert GUI with enhanced process management
        try:
            # Create the alert GUI process
            self.alert_process = subprocess.Popen(
                ["python", "alert_gui.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Store the process ID for tracking
            self.alert_pid = self.alert_process.pid
            
            print(f"\n🚨 EMERGENCY ALERT LAUNCHED 🚨")
            print(f"Driver State: {state.consciousness_level}")
            print(f"Confidence: {state.confidence:.2f}")
            print(f"Timestamp: {state.timestamp}")
            print(f"🤖 Alert GUI opened successfully (PID: {self.alert_pid})")
            print("📱 Single alert instance active")
            print("🔒 Further alerts blocked until cooldown")
            
            # Log critical event
            self.logger.critical(f"🚨 EMERGENCY ALERT: Driver {state.consciousness_level} - Confidence: {state.confidence:.2f} - PID: {self.alert_pid}")
            
            # Start a thread to monitor the alert process
            threading.Thread(target=self._monitor_alert_process, daemon=True).start()
            
        except FileNotFoundError:
            print("❌ Error: alert_gui.py not found. Please ensure it exists in the same directory.")
            self.logger.error("alert_gui.py not found")
        except Exception as e:
            print(f"❌ Error launching alert GUI: {e}")
            self.logger.error(f"Failed to launch alert GUI: {e}")
            # Reset flags if launch failed
            self.alert_process = None
            self.alert_pid = None
            self.alert_launched = False
    
    def _monitor_alert_process(self):
        """Monitor the alert process and reset flags when it terminates"""
        if self.alert_process is None:
            return
        
        try:
            # Wait for the process to complete
            self.alert_process.wait()
            print(f"🔓 Alert GUI process (PID: {self.alert_pid}) has been closed")
            print("✅ System ready for new alerts")
            
        except Exception as e:
            print(f"⚠️  Warning: Error monitoring alert process: {e}")
        
        finally:
            # Reset all alert tracking variables
            self.alert_process = None
            self.alert_pid = None
            self.alert_launched = False
            print("🔄 Alert system reset - Ready for next emergency")
    
    def should_trigger_alert(self, state):
        """Enhanced logic to determine if an alert should be triggered"""
        current_time = time.time()
        
        # Don't trigger if in cooldown period
        if (current_time - self.last_alert_time) < self.ALERT_COOLDOWN:
            return False
        
        # Don't trigger if alert is already running
        if self.is_alert_process_running():
            return False
        
        # Critical states that should trigger alerts
        critical_states = ["UNCONSCIOUS", "SEVERELY_DROWSY", "MICROSLEEP", "NO_FACE_DETECTED"]
        
        if state.consciousness_level in critical_states:
            return True
        
        # Additional conditions for alert triggering
        if state.consciousness_level == "DROWSY" and state.confidence < 0.3:
            return True
        
        # Detect anomalous patterns
        if self.detect_anomalies():
            return True
        
        return False
    
    def run(self):
        """Main monitoring loop"""
        print("\n🚗 Starting Driver Monitoring System...")
        print("📹 Press 'q' to quit, 'r' to reset alert system")
        print("🔍 Monitoring driver consciousness in real-time...\n")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                state, annotated_frame = self.process_frame(frame)
                
                # Check if alert should be triggered
                if self.should_trigger_alert(state):
                    self.trigger_emergency_alert(state)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                              (annotated_frame.shape[1] - 100, annotated_frame.shape[0] - 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Driver Monitoring System - MediaPipe Edition', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Shutting down Driver Monitoring System...")
                    break
                elif key == ord('r'):
                    print("\n🔄 Resetting alert system...")
                    self.reset_alert_system()
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"driver_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"📸 Frame saved as {filename}")
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.cleanup()
    
    def reset_alert_system(self):
        """Manually reset the alert system"""
        try:
            # Kill any existing alert processes
            self.kill_existing_alert_processes()
            
            # Reset all tracking variables
            self.alert_process = None
            self.alert_pid = None
            self.alert_launched = False
            self.last_alert_time = 0
            
            print("✅ Alert system reset successfully")
            print("🔓 Ready to trigger new alerts")
            
        except Exception as e:
            print(f"⚠️  Warning: Error resetting alert system: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up resources...")
        
        # Release camera
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            print("📹 Camera released")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        print("🖼️  Windows closed")
        
        # Kill any running alert processes
        try:
            self.kill_existing_alert_processes()
            print("🔒 Alert processes terminated")
        except Exception as e:
            print(f"⚠️  Warning: Error terminating alert processes: {e}")
        
        # Clean up temporary files
        try:
            if os.path.exists('alert_data.json'):
                os.remove('alert_data.json')
                print("🗑️  Temporary files cleaned")
        except Exception as e:
            print(f"⚠️  Warning: Error cleaning temporary files: {e}")
        
        print("✅ Cleanup completed successfully")

def main():
    """Main function to run the Driver Monitoring System"""
    try:
        # Initialize the system
        dms = DriverMonitoringSystem(camera_id=0)
        
        # Run the monitoring system
        dms.run()
        
    except Exception as e:
        print(f"❌ Failed to initialize Driver Monitoring System: {e}")
        print("🔧 Please check your camera connection and dependencies")

if __name__ == "__main__":
    main()