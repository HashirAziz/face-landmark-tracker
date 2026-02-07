"""
Face detection using MediaPipe library (Google's solution).
MediaPipe provides fast, accurate face detection with 468 facial landmarks.

Note: This detector is for FACE detection only.
Hand detection is handled separately in the main app using MediaPipe Hands.
"""

import cv2
import numpy as np
import mediapipe as mp
from config.settings import Config
from utils.logger import log


class FaceDetector:
    """Face detection and landmark tracking using MediaPipe."""
    
    def __init__(self):
        """Initialize face detector."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = None
        self.is_initialized = False
        
    def initialize(self):
        """
        Initialize MediaPipe FaceMesh model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            log.info("Initializing MediaPipe Face Mesh detector...")
            
            # Create FaceMesh detector
            # static_image_mode=False for video (faster tracking)
            # max_num_faces: maximum number of faces to detect
            # refine_landmarks: include iris landmarks (more detailed)
            # min_detection_confidence: minimum confidence for detection
            # min_tracking_confidence: minimum confidence for tracking
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=Config.MAX_NUM_FACES,
                refine_landmarks=True,
                min_detection_confidence=Config.DETECTION_CONFIDENCE,
                min_tracking_confidence=Config.TRACKING_CONFIDENCE
            )
            
            self.is_initialized = True
            log.info("Face detector initialized successfully")
            log.info(f"Max faces: {Config.MAX_NUM_FACES}, Detection confidence: {Config.DETECTION_CONFIDENCE}")
            
            return True
            
        except Exception as e:
            log.error(f"Failed to initialize face detector: {e}")
            return False
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
        
        Returns:
            list: List of detected faces with landmarks
        """
        if not self.is_initialized:
            log.warning("Detector not initialized")
            return []
        
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.face_mesh.process(rgb_frame)
            
            # Extract face data
            faces = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    faces.append({
                        'landmarks': face_landmarks,
                        'frame_shape': frame.shape
                    })
            
            return faces
            
        except Exception as e:
            log.error(f"Error during face detection: {e}")
            return []
    
    def get_face_info(self, face, frame_width, frame_height):
        """
        Extract information from detected face.
        
        Args:
            face (dict): Face data from MediaPipe
            frame_width (int): Frame width
            frame_height (int): Frame height
        
        Returns:
            dict: Face information (bbox, landmarks, confidence)
        """
        landmarks = face['landmarks']
        
        # Convert normalized landmarks to pixel coordinates
        landmark_points = []
        x_coords = []
        y_coords = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            landmark_points.append((x, y))
            x_coords.append(x)
            y_coords.append(y)
        
        # Calculate bounding box from landmarks
        x_min = max(0, min(x_coords) - 10)
        y_min = max(0, min(y_coords) - 10)
        x_max = min(frame_width, max(x_coords) + 10)
        y_max = min(frame_height, max(y_coords) + 10)
        
        bbox = (x_min, y_min, x_max, y_max)
        
        # MediaPipe doesn't provide explicit confidence, so we use 1.0
        # (it only returns faces it's confident about)
        confidence = 1.0
        
        return {
            'bbox': bbox,
            'landmarks': landmark_points,
            'landmark_connections': self.mp_face_mesh.FACEMESH_TESSELATION,
            'confidence': confidence,
            'num_landmarks': len(landmark_points)
        }
    
    def release(self):
        """Release detector resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.is_initialized = False
            log.info("Face detector released")