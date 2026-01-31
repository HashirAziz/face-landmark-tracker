"""
Configuration settings for the face landmark tracker.
All configurable parameters are centralized here.
"""

import os
from pathlib import Path


class Config:
    """Central configuration class for the application."""
    
    # ============================================================
    # PROJECT PATHS
    # ============================================================
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # ============================================================
    # CAMERA SETTINGS
    # ============================================================
    CAMERA_ID = 0  # Default webcam (0 = primary, 1 = secondary)
    FRAME_WIDTH = 1280  # Original capture width
    FRAME_HEIGHT = 720  # Original capture height
    
    # Processing resolution (resize for speed)
    PROCESS_WIDTH = 640  # Lower = faster, but less accurate
    PROCESS_HEIGHT = 480
    
    # ============================================================
    # FACE DETECTION SETTINGS (MediaPipe)
    # ============================================================
    DETECTION_CONFIDENCE = 0.5  # Min confidence threshold (0.0 - 1.0)
    TRACKING_CONFIDENCE = 0.5   # Min tracking confidence
    MAX_NUM_FACES = 5  # Maximum number of faces to detect
    
    # ============================================================
    # VISUALIZATION SETTINGS
    # ============================================================
    # Bounding box
    BBOX_COLOR = (0, 255, 0)  # Green in BGR
    BBOX_THICKNESS = 2
    
    # Landmarks
    LANDMARK_COLOR = (0, 0, 255)  # Red in BGR
    LANDMARK_RADIUS = 2
    LANDMARK_THICKNESS = -1  # Filled circle
    
    # Connections (lines between landmarks)
    CONNECTION_COLOR = (255, 0, 0)  # Blue in BGR
    CONNECTION_THICKNESS = 1
    DRAW_CONNECTIONS = True  # Draw lines connecting landmarks
    
    # FPS display
    FPS_COLOR = (255, 255, 0)  # Cyan in BGR
    FPS_POSITION = (10, 30)
    FPS_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FPS_SCALE = 1.0
    FPS_THICKNESS = 2
    
    # ============================================================
    # PERFORMANCE SETTINGS
    # ============================================================
    TARGET_FPS = 30
    
    # ============================================================
    # LOGGING
    # ============================================================
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)