"""
Drowsiness detection module.
Detects eye closure and yawning using facial landmarks.
"""

import numpy as np
from scipy.spatial import distance
from config.settings import Config
from utils.logger import log


class DrowsinessDetector:
    """Detect drowsiness using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)."""
    
    # MediaPipe landmark indices for facial features
    # Left eye landmarks: 33, 160, 158, 133, 153, 144
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Right eye landmarks: 362, 385, 387, 263, 373, 380
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Mouth landmarks for yawn detection
    MOUTH_INDICES = [61, 291, 0, 17, 269, 405, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    
    def __init__(self):
        """Initialize drowsiness detector."""
        # Counters for consecutive frames
        self.eye_closure_counter = 0
        self.yawn_counter = 0
        
        # CRITICAL FIX: Start at 0, not 100!
        self.drowsiness_score = 0.0
        
        # Alert state
        self.is_drowsy = False
        self.is_yawning = False
        self.alert_level = "NORMAL"
        
        # Statistics
        self.total_eye_closures = 0
        self.total_yawns = 0
        
        # Track if face is detected
        self.face_detected = False
        self.no_face_counter = 0
        
        # CRITICAL FIX: Track blink vs prolonged closure
        self.last_eye_state = "OPEN"  # "OPEN" or "CLOSED"
        self.blink_counter = 0
        
        log.info("Drowsiness detector initialized with score=0")
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks (list): List of 6 eye landmark coordinates [(x, y), ...]
        
        Returns:
            float: Eye aspect ratio
        """
        if len(eye_landmarks) != 6:
            return 0.0
        
        # Vertical distances
        vertical1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        vertical2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # EAR formula
        if horizontal == 0:
            return 0.0
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) for yawn detection.
        
        Args:
            mouth_landmarks (list): List of mouth landmark coordinates
        
        Returns:
            float: Mouth aspect ratio
        """
        if len(mouth_landmarks) < 14:
            return 0.0
        
        try:
            # Use specific mouth landmarks for better detection
            # Vertical distances (top to bottom of mouth)
            vertical1 = distance.euclidean(mouth_landmarks[2], mouth_landmarks[10])
            vertical2 = distance.euclidean(mouth_landmarks[4], mouth_landmarks[8])
            
            # Horizontal distance (left to right of mouth)
            horizontal = distance.euclidean(mouth_landmarks[0], mouth_landmarks[1])
            
            if horizontal == 0:
                return 0.0
            
            # MAR formula
            mar = (vertical1 + vertical2) / (2.0 * horizontal)
            return mar
        except Exception as e:
            log.debug(f"Error calculating MAR: {e}")
            return 0.0
    
    def detect_drowsiness(self, all_landmarks):
        """
        Main drowsiness detection function.
        
        Args:
            all_landmarks (list): All 468 facial landmarks [(x, y), ...]
        
        Returns:
            dict: Detection results
        """
        # Handle no face detection properly
        if not all_landmarks or len(all_landmarks) < 468:
            self.face_detected = False
            self.no_face_counter += 1
            
            # Rapidly decrease drowsiness score when no face detected
            if self.no_face_counter > 3:
                self.drowsiness_score = max(0, self.drowsiness_score - 10.0)
                self.alert_level = "NORMAL"
                self.eye_closure_counter = 0
                self.yawn_counter = 0
            
            return self._get_default_result()
        
        # Face detected, reset counter
        self.face_detected = True
        self.no_face_counter = 0
        
        # Extract eye landmarks
        left_eye = [all_landmarks[i] for i in self.LEFT_EYE_INDICES]
        right_eye = [all_landmarks[i] for i in self.RIGHT_EYE_INDICES]
        
        # Extract mouth landmarks
        mouth = [all_landmarks[i] for i in self.MOUTH_INDICES if i < len(all_landmarks)]
        
        # Calculate ratios
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self.calculate_mouth_aspect_ratio(mouth)
        
        # ============================================================
        # EYE CLOSURE DETECTION (FIXED)
        # ============================================================
        eyes_closed = False
        current_eye_state = "CLOSED" if avg_ear < Config.EAR_THRESHOLD else "OPEN"
        
        # Detect BLINKS vs PROLONGED CLOSURE
        if current_eye_state == "CLOSED":
            self.eye_closure_counter += 1
            
            # ONLY count as drowsiness if closed for LONG time
            if self.eye_closure_counter >= Config.EAR_CONSEC_FRAMES:
                eyes_closed = True
                self.is_drowsy = True
                # Increase drowsiness score
                self.drowsiness_score = min(
                    Config.DROWSINESS_SCORE_MAX,
                    self.drowsiness_score + Config.DROWSINESS_SCORE_INCREMENT_EYES
                )
        else:
            # Eyes are OPEN
            # CRITICAL FIX: Only count closures that were PROLONGED, not blinks
            if self.eye_closure_counter >= Config.EAR_CONSEC_FRAMES:
                # This was a PROLONGED closure (drowsiness event)
                self.total_eye_closures += 1
                log.warning(f"Prolonged eye closure detected! Total: {self.total_eye_closures}")
            
            # Reset counter
            self.eye_closure_counter = 0
            self.is_drowsy = False
        
        # Update last state
        self.last_eye_state = current_eye_state
        
        # ============================================================
        # YAWN DETECTION (FIXED)
        # ============================================================
        yawning = False
        
        # CRITICAL FIX: Only detect yawn if:
        # 1. Mouth is VERY wide open (high MAR)
        # 2. Eyes are OPEN (avg_ear > threshold)
        # 3. MAR is significantly above normal talking range
        
        if mar > Config.MAR_THRESHOLD and avg_ear > Config.EAR_THRESHOLD:
            # Mouth is wide open AND eyes are open
            self.yawn_counter += 1
            
            if self.yawn_counter >= Config.MAR_CONSEC_FRAMES:
                yawning = True
                self.is_yawning = True
                # Increase drowsiness score
                self.drowsiness_score = min(
                    Config.DROWSINESS_SCORE_MAX,
                    self.drowsiness_score + Config.DROWSINESS_SCORE_INCREMENT_YAWN
                )
        else:
            # Mouth not wide open OR eyes closed
            if self.yawn_counter >= Config.MAR_CONSEC_FRAMES:
                # This was a valid yawn
                self.total_yawns += 1
                log.warning(f"Yawn detected! Total: {self.total_yawns}")
            
            # Reset counter
            self.yawn_counter = 0
            self.is_yawning = False
        
        # ============================================================
        # DROWSINESS SCORE DECAY (FIXED)
        # ============================================================
        # CRITICAL: Decay score when user is alert
        if not eyes_closed and not yawning:
            # User is alert - decrease score RAPIDLY
            self.drowsiness_score = max(0, self.drowsiness_score - Config.DROWSINESS_SCORE_DECAY)
        
        # ============================================================
        # DETERMINE ALERT LEVEL
        # ============================================================
        if self.drowsiness_score >= Config.ALERT_LEVEL_DANGER:
            self.alert_level = "DANGER"
        elif self.drowsiness_score >= Config.ALERT_LEVEL_WARNING:
            self.alert_level = "WARNING"
        else:
            self.alert_level = "NORMAL"
        
        return {
            'ear': avg_ear,
            'mar': mar,
            'eyes_closed': eyes_closed,
            'yawning': yawning,
            'drowsiness_score': self.drowsiness_score,
            'alert_level': self.alert_level,
            'eye_closure_counter': self.eye_closure_counter,
            'yawn_counter': self.yawn_counter,
            'total_eye_closures': self.total_eye_closures,
            'total_yawns': self.total_yawns,
            'left_eye_landmarks': left_eye,
            'right_eye_landmarks': right_eye,
            'mouth_landmarks': mouth,
            'face_detected': self.face_detected
        }
    
    def _get_default_result(self):
        """Return default result when no face detected."""
        return {
            'ear': 0.0,
            'mar': 0.0,
            'eyes_closed': False,
            'yawning': False,
            'drowsiness_score': self.drowsiness_score,
            'alert_level': self.alert_level,
            'eye_closure_counter': 0,
            'yawn_counter': 0,
            'total_eye_closures': self.total_eye_closures,
            'total_yawns': self.total_yawns,
            'left_eye_landmarks': [],
            'right_eye_landmarks': [],
            'mouth_landmarks': [],
            'face_detected': False
        }
    
    def reset(self):
        """Reset all counters and scores."""
        self.eye_closure_counter = 0
        self.yawn_counter = 0
        self.drowsiness_score = 0.0  # CRITICAL: Reset to 0
        self.is_drowsy = False
        self.is_yawning = False
        self.alert_level = "NORMAL"
        self.total_eye_closures = 0
        self.total_yawns = 0
        self.face_detected = False
        self.no_face_counter = 0
        self.last_eye_state = "OPEN"
        self.blink_counter = 0
        log.info("Drowsiness detector reset")