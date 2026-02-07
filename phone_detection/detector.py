"""
Phone usage detection module.
Detects when driver is using a mobile phone by analyzing hand positions.
"""

import numpy as np
from scipy.spatial import distance
from config.settings import Config
from utils.logger import log


class PhoneDetector:
    """Detect phone usage using hand position analysis."""
    
    # MediaPipe hand landmark indices
    WRIST = 0
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
        
        # Tracking
        self.last_detection_time = 0
        
        log.info("Phone detector initialized")
    
    def is_hand_near_face(self, hand_landmarks, face_bbox, frame_shape):
        """
        Check if hand is positioned near face (phone usage position).
        
        Args:
            hand_landmarks (list): Hand landmark coordinates [(x, y), ...]
            face_bbox (tuple): Face bounding box (x1, y1, x2, y2)
            frame_shape (tuple): Frame dimensions (height, width)
        
        Returns:
            tuple: (is_near, distance_ratio)
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return False, 0.0
        
        if not face_bbox:
            return False, 0.0
        
        # Get face center and dimensions
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        face_center_x = (face_x1 + face_x2) / 2
        face_center_y = (face_y1 + face_y2) / 2
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1
        
        # Get hand center (average of all landmarks)
        hand_x = np.mean([p[0] for p in hand_landmarks])
        hand_y = np.mean([p[1] for p in hand_landmarks])
        
        # Calculate distance from hand to face center
        dist_to_face = distance.euclidean(
            (hand_x, hand_y),
            (face_center_x, face_center_y)
        )
        
        # Normalize by face size
        face_size = max(face_width, face_height)
        distance_ratio = dist_to_face / face_size if face_size > 0 else 999
        
        # CRITICAL FIX: More lenient threshold
        # Hand is "near face" if within 2.0x face size (increased from 1.5x)
        is_near = distance_ratio < 2.0
        
        return is_near, distance_ratio
    
    def is_hand_elevated(self, hand_landmarks, face_bbox):
        """
        Check if hand is elevated (above chest level, near face height).
        
        Args:
            hand_landmarks (list): Hand landmark coordinates
            face_bbox (tuple): Face bounding box (x1, y1, x2, y2)
        
        Returns:
            bool: True if hand is elevated
        """
        if not hand_landmarks or not face_bbox:
            return False
        
        # Get wrist position (base of hand)
        wrist_y = hand_landmarks[self.WRIST][1]
        
        # Get face bottom
        face_bottom = face_bbox[3]
        
        # Hand is elevated if wrist is at or above face bottom
        # CRITICAL FIX: More lenient - allow hand slightly below face
        tolerance = (face_bbox[3] - face_bbox[1]) * 0.5  # 50% of face height
        
        return wrist_y < (face_bottom + tolerance)
    
    def is_vertical_orientation(self, hand_landmarks):
        """
        Check if hand is in vertical orientation (typical phone holding).
        
        Args:
            hand_landmarks (list): Hand landmark coordinates
        
        Returns:
            bool: True if hand appears vertical
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return False
        
        try:
            wrist = hand_landmarks[self.WRIST]
            middle_tip = hand_landmarks[self.MIDDLE_TIP]
            
            # Calculate vertical distance vs horizontal distance
            vertical_dist = abs(middle_tip[1] - wrist[1])
            horizontal_dist = abs(middle_tip[0] - wrist[0])
            
            # Hand is vertical if vertical distance > horizontal distance
            # CRITICAL FIX: More lenient ratio
            return vertical_dist > horizontal_dist * 0.6
            
        except:
            return False
    
    def detect_phone_usage(self, hands_detected, hand_landmarks_list, face_landmarks, face_bbox, frame_shape=None):
        """
        Main phone detection function.
        
        Args:
            hands_detected (int): Number of hands detected
            hand_landmarks_list (list): List of hand landmarks for each hand
            face_landmarks (list): Face landmarks
            face_bbox (tuple): Face bounding box
            frame_shape (tuple): Frame dimensions
        
        Returns:
            dict: Detection results
        """
        phone_usage_detected = False
        confidence = 0.0
        detection_reasons = []
        
        # No hands detected = no phone usage
        if hands_detected == 0 or not hand_landmarks_list:
            # Decrease counter
            if self.phone_detection_counter > 0:
                self.phone_detection_counter = 0
            
            if self.current_phone_duration > 0:
                self.current_phone_duration = 0
            
            self.phone_detected = False
            return self._get_result(False, 0.0, [], face_landmarks)
        
        # Check each detected hand
        for hand_landmarks in hand_landmarks_list:
            if not hand_landmarks:
                continue
            
            # CRITICAL CHECK 1: Hand near face (50% confidence)
            is_near, dist_ratio = self.is_hand_near_face(hand_landmarks, face_bbox, frame_shape)
            if is_near:
                confidence += 0.5
                detection_reasons.append(f"Hand near face (dist: {dist_ratio:.2f})")
            
            # CRITICAL CHECK 2: Hand elevated (30% confidence)
            is_elevated = self.is_hand_elevated(hand_landmarks, face_bbox)
            if is_elevated:
                confidence += 0.3
                detection_reasons.append("Hand elevated")
            
            # CRITICAL CHECK 3: Vertical orientation (20% confidence)
            is_vertical = self.is_vertical_orientation(hand_landmarks)
            if is_vertical:
                confidence += 0.2
                detection_reasons.append("Vertical orientation")
        
        # CRITICAL FIX: Lower threshold for detection
        # Changed from 0.6 to 0.4 for easier detection
        if confidence >= Config.PHONE_DETECTION_THRESHOLD:
            self.phone_detection_counter += 1
            
            # CRITICAL: Confirmed phone usage after consecutive frames
            # This implements the 2-3 second delay you requested
            if self.phone_detection_counter >= Config.PHONE_CONSEC_FRAMES:
                phone_usage_detected = True
                self.phone_detected = True
                self.current_phone_duration += 1
                
                # Update statistics
                if self.current_phone_duration > self.longest_phone_duration:
                    self.longest_phone_duration = self.current_phone_duration
        else:
            # Reset if not detected
            if self.phone_detection_counter >= Config.PHONE_CONSEC_FRAMES:
                # Phone usage session ended - INCREMENT COUNTER
                self.total_phone_detections += 1
                log.warning(f"Phone usage detected! Total uses: {self.total_phone_detections}, Duration: {self.current_phone_duration} frames ({self.current_phone_duration/30:.1f} seconds)")
            
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
            'head_tilted': False,
            'tilt_angle': 0.0
        }
    
    def reset(self):
        """Reset phone detector."""
        self.phone_detected = False
        self.phone_detection_counter = 0
        self.total_phone_detections = 0
        self.current_phone_duration = 0
        self.longest_phone_duration = 0
        self.last_detection_time = 0
        log.info("Phone detector reset")