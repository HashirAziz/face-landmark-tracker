"""
Phone usage detection module.
Detects ACTUAL phone usage by requiring hand to be very close to face/ear.
"""

import numpy as np
from scipy.spatial import distance
from config.settings import Config
from utils.logger import log


class PhoneDetector:
    """Detect phone usage - only when hand is VERY close to face (like holding phone to ear)."""
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC = 1
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    def __init__(self):
        """Initialize phone detector."""
        # Detection state
        self.phone_detected = False
        self.phone_detection_counter = 0
        
        # Statistics
        self.total_phone_detections = 0
        self.current_phone_duration = 0  # Frames
        self.longest_phone_duration = 0  # Frames
        
        log.info("Phone detector initialized")

    def calculate_hand_to_face_distance(self, hand_landmarks, face_bbox):
        """
        Calculate precise distance from hand to face.
        
        Args:
            hand_landmarks (list): Hand landmarks
            face_bbox (tuple): Face bounding box
        
        Returns:
            float: Distance ratio (0.0 = touching face, higher = farther)
        """
        if not hand_landmarks or not face_bbox:
            return 999.0
        
        # Get face dimensions
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1
        
        # Find closest point on hand to face
        min_dist = 999999
        for landmark in hand_landmarks:
            x, y = landmark
            
            # Distance to face edges
            dist_to_left = abs(x - face_x1)
            dist_to_right = abs(x - face_x2)
            dist_to_top = abs(y - face_y1)
            dist_to_bottom = abs(y - face_y2)
            
            # Minimum distance to any face edge
            dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            
            if dist < min_dist:
                min_dist = dist
        
        # Normalize by face size
        face_size = max(face_width, face_height)
        distance_ratio = min_dist / face_size if face_size > 0 else 999.0
        
        return distance_ratio

    def is_hand_at_ear_position(self, hand_landmarks, face_bbox):
        """
        Check if hand is at typical phone-to-ear position.
        
        Args:
            hand_landmarks (list): Hand landmarks
            face_bbox (tuple): Face bounding box
        
        Returns:
            bool: True if hand is at ear position
        """
        if not hand_landmarks or not face_bbox:
            return False
        
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        face_center_y = (face_y1 + face_y2) / 2
        
        # Get hand center
        hand_x = np.mean([p[0] for p in hand_landmarks])
        hand_y = np.mean([p[1] for p in hand_landmarks])
        
        # Check if hand is at ear level (same height as face center)
        # Allow ±30% of face height tolerance
        vertical_tolerance = (face_y2 - face_y1) * 0.3
        is_at_ear_height = abs(hand_y - face_center_y) < vertical_tolerance
        
        # Check if hand is beside face (not in front)
        is_beside_face = (hand_x < face_x1) or (hand_x > face_x2)
        
        return is_at_ear_height and is_beside_face

    def detect_phone_usage(self, hands_detected, hand_landmarks_list, face_landmarks, face_bbox, frame_shape=None):
        """
        Main phone detection function.
        """
        phone_usage_detected = False
        confidence = 0.0
        detection_reasons = []
        
        # No hands detected = reset session
        if hands_detected == 0 or not hand_landmarks_list:
            self.phone_detection_counter = 0
            self.current_phone_duration = 0
            self.phone_detected = False
            return self._get_result(False, 0.0, [], face_landmarks)
        
        best_confidence = 0.0
        
        for hand_landmarks in hand_landmarks_list:
            if not hand_landmarks:
                continue
            
            hand_confidence = 0.0
            
            # CRITICAL CHECK 1: Hand proximity to face
            distance_ratio = self.calculate_hand_to_face_distance(hand_landmarks, face_bbox)
            
            if distance_ratio < 0.3:
                hand_confidence += 0.6
                detection_reasons.append(f"Hand very close (dist: {distance_ratio:.2f})")
            
            # CRITICAL CHECK 2: Vertical alignment with ear
            if self.is_hand_at_ear_position(hand_landmarks, face_bbox):
                hand_confidence += 0.4
                detection_reasons.append("Hand at ear position")
            
            if hand_confidence > best_confidence:
                best_confidence = hand_confidence
        
        confidence = best_confidence
        
        # Determine if phone usage is detected based on threshold
        if confidence >= Config.PHONE_DETECTION_THRESHOLD:
            self.phone_detection_counter += 1
            
            # Confirmed phone usage after consecutive frames
            if self.phone_detection_counter >= Config.PHONE_CONSEC_FRAMES:
                phone_usage_detected = True
                self.phone_detected = True
                self.current_phone_duration += 1
                
                # Update statistics
                if self.current_phone_duration > self.longest_phone_duration:
                    self.longest_phone_duration = self.current_phone_duration
        else:
            # Reset if not detected, but log if a session just ended
            if self.phone_detection_counter >= Config.PHONE_CONSEC_FRAMES:
                self.total_phone_detections += 1
                duration_seconds = self.current_phone_duration / 30.0  # assuming ~30 fps
                log.warning(f"Phone usage ended! Total uses: {self.total_phone_detections}, "
                            f"Duration: {duration_seconds:.1f} seconds")
            
            self.phone_detection_counter = 0
            self.phone_detected = False
            self.current_phone_duration = 0
        
        return self._get_result(phone_usage_detected, confidence, detection_reasons, face_landmarks)

    def _get_result(self, detected, confidence, reasons, face_landmarks):
        """Format detection result."""
        return {
            'phone_detected': detected,
            'confidence': confidence,
            'detection_reasons': reasons,
            'phone_detection_counter': self.phone_detection_counter,
            'total_phone_detections': self.total_phone_detections,
            'current_duration': self.current_phone_duration,
            'longest_duration': self.longest_phone_duration,
            'phone_bbox': None,  # Always None now
            'head_tilted': False,
            'tilt_angle': 0.0
        }

    def reset(self):
        """Full reset of the detector."""
        self.phone_detected = False
        self.phone_detection_counter = 0
        self.total_phone_detections = 0
        self.current_phone_duration = 0
        self.longest_phone_duration = 0
        log.info("Phone detector reset")