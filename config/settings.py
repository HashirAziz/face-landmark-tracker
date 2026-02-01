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
    SOUNDS_DIR = PROJECT_ROOT / "sounds"
    
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
    MAX_NUM_FACES = 2  # For drowsiness detection, focus on driver only
    
    # ============================================================
    # DROWSINESS DETECTION SETTINGS
    # ============================================================
    # Eye Aspect Ratio (EAR) thresholds
    EAR_THRESHOLD = 0.28  # Below this = eyes closed
    EAR_CONSEC_FRAMES = 20  # Consecutive frames for drowsiness (20 frames ≈ 0.67 seconds at 30 FPS)
    
    # Mouth Aspect Ratio (MAR) thresholds
    MAR_THRESHOLD = 0.55  # Above this = yawning
    MAR_CONSEC_FRAMES = 15  # Consecutive frames for yawn detection
    
    # Drowsiness score
    DROWSINESS_SCORE_MAX = 100  # Maximum drowsiness score
    DROWSINESS_SCORE_DECAY = 0.5  # How fast score decreases per frame when alert
    DROWSINESS_SCORE_INCREMENT_EYES = 2.0  # Score increase per frame when eyes closed
    DROWSINESS_SCORE_INCREMENT_YAWN = 1.5  # Score increase per frame when yawning
    
    # Alert levels
    ALERT_LEVEL_WARNING = 30  # Yellow warning
    ALERT_LEVEL_DANGER = 60   # Red danger
    
    # ============================================================
    # AUDIO ALERT SETTINGS
    # ============================================================
    ENABLE_AUDIO_ALERTS = True
    AUDIO_ALERT_COOLDOWN = 3.0  # Seconds between audio alerts (prevent spam)
    AUDIO_VOLUME = 1.0  # 0.0 to 1.0
    
    # ============================================================
    # VISUALIZATION SETTINGS
    # ============================================================
    # Bounding box
    BBOX_COLOR = (0, 255, 0)  # Green in BGR
    BBOX_THICKNESS = 3
    
    # Landmarks
    LANDMARK_COLOR = (0, 0, 255)  # Red in BGR
    LANDMARK_RADIUS = 2
    LANDMARK_THICKNESS = -1  # Filled circle
    
    # Eye landmarks (highlight when analyzing)
    EYE_COLOR = (255, 0, 255)  # Magenta
    EYE_THICKNESS = 1
    
    # Mouth landmarks (highlight when analyzing)
    MOUTH_COLOR = (0, 255, 255)  # Yellow
    MOUTH_THICKNESS = 1
    
    # Alert colors
    COLOR_NORMAL = (0, 255, 0)    # Green
    COLOR_WARNING = (0, 165, 255)  # Orange
    COLOR_DANGER = (0, 0, 255)     # Red
    
    # FPS display
    FPS_COLOR = (255, 255, 255)  # White in BGR
    FPS_POSITION = (10, 30)
    FPS_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FPS_SCALE = 0.7
    FPS_THICKNESS = 2
    
    # Dashboard settings
    SHOW_DASHBOARD = True
    DASHBOARD_POSITION = (10, 60)
    DASHBOARD_FONT_SCALE = 0.6
    DASHBOARD_LINE_SPACING = 25
    
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
        cls.SOUNDS_DIR.mkdir(parents=True, exist_ok=True)