"""
Configuration settings for the driver safety monitoring system.
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
    CAMERA_ID = 0
    FRAME_WIDTH = 1000
    FRAME_HEIGHT = 1000
    PROCESS_WIDTH = 640
    PROCESS_HEIGHT = 480
    
    # ============================================================
    # FACE DETECTION SETTINGS
    # ============================================================
    DETECTION_CONFIDENCE = 0.5
    TRACKING_CONFIDENCE = 0.5
    MAX_NUM_FACES = 1
    
    # ============================================================
    # HAND DETECTION SETTINGS
    # ============================================================
    ENABLE_HAND_DETECTION = True
    HAND_DETECTION_CONFIDENCE = 0.5
    HAND_TRACKING_CONFIDENCE = 0.5
    MAX_NUM_HANDS = 2
    
    # ============================================================
    # DROWSINESS DETECTION SETTINGS
    # ============================================================
    EAR_THRESHOLD = 0.21
    EAR_CONSEC_FRAMES = 30
    MAR_THRESHOLD = 0.75
    MAR_CONSEC_FRAMES = 20
    DROWSINESS_SCORE_MAX = 100
    DROWSINESS_SCORE_DECAY = 3.0
    DROWSINESS_SCORE_INCREMENT_EYES = 2.5
    DROWSINESS_SCORE_INCREMENT_YAWN = 1.5
    ALERT_LEVEL_WARNING = 35
    ALERT_LEVEL_DANGER = 65
    
    # ============================================================
    # PHONE DETECTION SETTINGS
    # ============================================================
    PHONE_DETECTION_THRESHOLD = 0.6  
    PHONE_CONSEC_FRAMES = 150  # 5 seconds at 30 FPS
    PHONE_SCORE_INCREMENT = 5.0
    PHONE_ALERT_THRESHOLD = 50
    
    # ============================================================
    # AUDIO ALERT SETTINGS
    # ============================================================
    ENABLE_AUDIO_ALERTS = True
    AUDIO_ALERT_COOLDOWN = 1.0
    CONTINUOUS_ALARM_ENABLED = True
    CONTINUOUS_ALARM_THRESHOLD = 70
    AUDIO_VOLUME = 1.0
    BEEP_FREQUENCY = 1000
    BEEP_DURATION = 0.5
    
    # ============================================================
    # VISUALIZATION SETTINGS (ORGANIZED FOR NO OVERLAP)
    # ============================================================
    # Colors (BGR Format for OpenCV)
    BBOX_COLOR = (0, 255, 0)      # Green
    BBOX_THICKNESS = 2
    LANDMARK_COLOR = (0, 0, 255)  # Red
    LANDMARK_RADIUS = 2
    LANDMARK_THICKNESS = -1
    
    EYE_COLOR = (255, 255, 0)     # Cyan (Replaced Magenta)
    EYE_THICKNESS = 1
    MOUTH_COLOR = (0, 255, 255)   # Yellow
    MOUTH_THICKNESS = 1
    HAND_COLOR = (255, 128, 0)    # Blue-Orange
    HAND_THICKNESS = 2
    
    COLOR_NORMAL = (0, 255, 0)    # Green
    COLOR_WARNING = (0, 165, 255) # Orange
    COLOR_DANGER = (0, 0, 255)    # Red
    COLOR_PHONE = (255, 255, 0)   # Cyan (Replaced Magenta)
    
    # Main Banner
    SHOW_NORMAL_BANNER = True
    NORMAL_BANNER_COLOR = (0, 200, 0)
    
    # FPS Text Positioning
    FPS_COLOR = (255, 255, 255)
    FPS_POSITION = (10, 30) 
    FPS_FONT = 0
    FPS_SCALE = 0.5
    FPS_THICKNESS = 2
    
    # Dashboard Text Positioning
    SHOW_DASHBOARD = True
    DASHBOARD_POSITION = (10, 110)  
    DASHBOARD_FONT_SCALE = 0.55     
    DASHBOARD_LINE_SPACING = 20     
    
    # ============================================================
    # PERFORMANCE SETTINGS
    # ============================================================
    TARGET_FPS = 120
    
    # ============================================================
    # LOGGING
    # ============================================================
    LOG_LEVEL = "INFO"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.SOUNDS_DIR.mkdir(parents=True, exist_ok=True)