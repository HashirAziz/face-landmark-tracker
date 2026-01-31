"""
Facial landmark tracking and processing.
MediaPipe provides 468 facial landmarks including detailed eye, nose, mouth features.
"""

import numpy as np
from utils.logger import log


class LandmarkTracker:
    """Process and track facial landmarks from MediaPipe."""
    
    def __init__(self):
        """Initialize landmark tracker."""
        # MediaPipe returns 468 landmarks
        # Key landmark indices (for reference):
        # 0-16: Face oval
        # 33, 133, 159, 145: Left eye
        # 362, 263, 386, 374: Right eye
        # 1, 4: Nose tip
        # 61, 291: Left/Right mouth corners
        # 0, 17, 152, 377: Face outline
        
        self.key_landmark_indices = {
            'left_eye': [33, 133, 159, 145],
            'right_eye': [362, 263, 386, 374],
            'nose': [1, 4],
            'mouth': [61, 291, 0, 17],
            'face_oval': list(range(0, 17))
        }
    
    def process_landmarks(self, landmarks):
        """
        Process raw landmarks.
        
        Args:
            landmarks (list): List of landmark tuples [(x, y), ...]
        
        Returns:
            dict: Processed landmark data
        """
        if landmarks is None or len(landmarks) == 0:
            return None
        
        # Extract key landmarks for easier access
        key_landmarks = {}
        
        for name, indices in self.key_landmark_indices.items():
            key_landmarks[name] = [
                landmarks[idx] for idx in indices if idx < len(landmarks)
            ]
        
        return {
            'all_landmarks': landmarks,
            'key_landmarks': key_landmarks,
            'num_landmarks': len(landmarks)
        }
    
    def scale_landmarks(self, landmarks, scale_x, scale_y):
        """
        Scale landmarks from processed frame to original frame size.
        
        Args:
            landmarks (list): Landmarks in processed frame coordinates
            scale_x (float): X scaling factor
            scale_y (float): Y scaling factor
        
        Returns:
            list: Scaled landmarks
        """
        if landmarks is None or len(landmarks) == 0:
            return None
        
        scaled_landmarks = [
            (int(x * scale_x), int(y * scale_y))
            for x, y in landmarks
        ]
        
        return scaled_landmarks
    
    def get_landmark_statistics(self, landmarks):
        """
        Calculate statistics from landmarks (e.g., face size, center).
        
        Args:
            landmarks (list): Facial landmarks [(x, y), ...]
        
        Returns:
            dict: Statistics
        """
        if landmarks is None or len(landmarks) == 0:
            return None
        
        # Convert to numpy array for easier calculations
        points = np.array(landmarks)
        
        # Calculate face center (average of all landmarks)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        # Calculate face bounds
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        
        face_width = max_x - min_x
        face_height = max_y - min_y
        
        return {
            'center': (center_x, center_y),
            'width': face_width,
            'height': face_height,
            'bounds': (min_x, min_y, max_x, max_y),
            'num_landmarks': len(landmarks)
        }
    
    def filter_landmarks_for_display(self, landmarks, step=5):
        """
        Filter landmarks to reduce visual clutter.
        Returns every Nth landmark for cleaner visualization.
        
        Args:
            landmarks (list): All landmarks
            step (int): Take every Nth landmark
        
        Returns:
            list: Filtered landmarks
        """
        if landmarks is None or len(landmarks) == 0:
            return []
        
        return landmarks[::step]