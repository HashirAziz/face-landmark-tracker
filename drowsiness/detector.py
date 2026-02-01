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
    
    # Mouth landmarks for yawn detection: 61, 291, 0, 17, 269, 405
    MOUTH_INDICES = [61, 291, 0, 17, 269, 405, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    
    def __init__(self):
        """Initialize drowsiness detector."""
        # Counters for consecutive frames
        self.eye_closure_counter = 0
        self.yawn_counter = 0
        
        # Drowsiness score (0-100)
        self.drowsiness_score = 0
        
        # Alert state
        self.is_drowsy = False
        self.is_yawning = False
        self.alert_level = "NORMAL"  # NORMAL, WARNING, DANGER
        
        # Statistics
        self.total_eye_closures = 0
        self.total_yawns = 0
        
        # FIX #1 & #2: Track if face is detected
        self.face_detected = False
        self.no_face_counter = 0
        
        log.info("Drowsiness detector initialized")
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        where p1-p6 are the eye landmark points
        
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
        
        MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
        
        Args:
            mouth_landmarks (list): List of mouth landmark coordinates
        
        Returns:
            float: Mouth aspect ratio
        """
        if len(mouth_landmarks) < 8:
            return 0.0
        
        # Vertical distances (mouth height at different points)
        vertical1 = distance.euclidean(mouth_landmarks[2], mouth_landmarks[10])
        vertical2 = distance.euclidean(mouth_landmarks[4], mouth_landmarks[8])
        vertical3 = distance.euclidean(mouth_landmarks[6], mouth_landmarks[12])
        
        # Horizontal distance (mouth width)
        horizontal = distance.euclidean(mouth_landmarks[0], mouth_landmarks[1])
        
        if horizontal == 0:
            return 0.0
        
        # MAR formula
        mar = (vertical1 + vertical2 + vertical3) / (2.0 * horizontal)
        return mar
    
    def detect_drowsiness(self, all_landmarks):
        """
        Main drowsiness detection function.
        
        Args:
            all_landmarks (list): All 468 facial landmarks [(x, y), ...]
        
        Returns:
            dict: Detection results
        """
        # FIX #1 & #2: Handle no face detection properly
        if not all_landmarks or len(all_landmarks) < 468:
            self.face_detected = False
            self.no_face_counter += 1
            
            # Rapidly decrease drowsiness score when no face detected
            if self.no_face_counter > 5:  # After 5 frames
                self.drowsiness_score = max(0, self.drowsiness_score - 5.0)
                self.alert_level = "NORMAL"
            
            return self._get_default_result()
        
        # FIX #1: Face detected, reset counter
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
        
        # Detect eye closure
        eyes_closed = False
        if avg_ear < Config.EAR_THRESHOLD:
            self.eye_closure_counter += 1
            if self.eye_closure_counter >= Config.EAR_CONSEC_FRAMES:
                eyes_closed = True
                self.is_drowsy = True
                # FIX #2: Only increment if actually drowsy
                self.drowsiness_score = min(
                    Config.DROWSINESS_SCORE_MAX,
                    self.drowsiness_score + Config.DROWSINESS_SCORE_INCREMENT_EYES
                )
        else:
            if self.eye_closure_counter >= Config.EAR_CONSEC_FRAMES:
                self.total_eye_closures += 1
                log.warning(f"Eye closure detected! Total: {self.total_eye_closures}")
            self.eye_closure_counter = 0
            self.is_drowsy = False
        
        # FIX #3: Improved yawn detection with better thresholding
        yawning = False
        # Only detect yawn if mouth is SIGNIFICANTLY open
        if mar > Config.MAR_THRESHOLD and avg_ear > 0.2:  # Eyes must be open for valid yawn
            self.yawn_counter += 1
            if self.yawn_counter >= Config.MAR_CONSEC_FRAMES:
                yawning = True
                self.is_yawning = True
                self.drowsiness_score = min(
                    Config.DROWSINESS_SCORE_MAX,
                    self.drowsiness_score + Config.DROWSINESS_SCORE_INCREMENT_YAWN
                )
        else:
            if self.yawn_counter >= Config.MAR_CONSEC_FRAMES:
                self.total_yawns += 1
                log.warning(f"Yawn detected! Total: {self.total_yawns}")
            self.yawn_counter = 0
            self.is_yawning = False
        
        # FIX #2: Faster decay when alert (more aggressive)
        if not eyes_closed and not yawning:
            self.drowsiness_score = max(0, self.drowsiness_score - Config.DROWSINESS_SCORE_DECAY)
        
        # Determine alert level
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
            'face_detected': self.face_detected  # FIX #1: Add face detection flag
        }
    
    def _get_default_result(self):
        """Return default result when no face detected."""
        # FIX #1: Clear status when no face
        return {
            'ear': 0.0,
            'mar': 0.0,
            'eyes_closed': False,  # FIX #1: Show as false when no face
            'yawning': False,       # FIX #1: Show as false when no face
            'drowsiness_score': self.drowsiness_score,
            'alert_level': self.alert_level,
            'eye_closure_counter': 0,
            'yawn_counter': 0,
            'total_eye_closures': self.total_eye_closures,
            'total_yawns': self.total_yawns,
            'left_eye_landmarks': [],
            'right_eye_landmarks': [],
            'mouth_landmarks': [],
            'face_detected': False  # FIX #1: Explicitly mark as no face
        }
    
    def reset(self):
        """Reset all counters and scores."""
        self.eye_closure_counter = 0
        self.yawn_counter = 0
        self.drowsiness_score = 0
        self.is_drowsy = False
        self.is_yawning = False
        self.alert_level = "NORMAL"
        self.total_eye_closures = 0
        self.total_yawns = 0
        self.face_detected = False
        self.no_face_counter = 0
        log.info("Drowsiness detector reset")