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
    MAX_NUM_FACES = 1  # For drowsiness detection, focus on driver only
    
    # ============================================================
    # HAND DETECTION SETTINGS (MediaPipe)
    # ============================================================
    ENABLE_HAND_DETECTION = True  # Enable phone detection
    HAND_DETECTION_CONFIDENCE = 0.5
    HAND_TRACKING_CONFIDENCE = 0.5
    MAX_NUM_HANDS = 2
    
    # ============================================================
    # DROWSINESS DETECTION SETTINGS
    # ============================================================
    # Eye Aspect Ratio (EAR) thresholds
    EAR_THRESHOLD = 0.21  # Below this = eyes closed
    EAR_CONSEC_FRAMES = 30  # Consecutive frames for drowsiness
    
    # Mouth Aspect Ratio (MAR) thresholds
    MAR_THRESHOLD = 0.75  # Above this = yawning
    MAR_CONSEC_FRAMES = 20  # Consecutive frames for yawn detection
    
    # Drowsiness score
    DROWSINESS_SCORE_MAX = 100  # Maximum drowsiness score
    DROWSINESS_SCORE_DECAY = 3.0  # How fast score decreases when alert
    DROWSINESS_SCORE_INCREMENT_EYES = 2.5  # Score increase when eyes closed
    DROWSINESS_SCORE_INCREMENT_YAWN = 1.5  # Score increase when yawning
    
    # Alert levels
    ALERT_LEVEL_WARNING = 35  # Yellow warning
    ALERT_LEVEL_DANGER = 65   # Red danger
    
    # ============================================================
    # PHONE DETECTION SETTINGS (FIXED!)
    # ============================================================
    # CRITICAL FIX: Lowered threshold from 0.6 to 0.4 for easier detection
    PHONE_DETECTION_THRESHOLD = 0.4  # Confidence threshold (0.0 - 1.0)
    
    # CRITICAL: This controls the 2-3 second delay
    # At 30 FPS: 60 frames = 2 seconds, 90 frames = 3 seconds
    PHONE_CONSEC_FRAMES = 60  # 2 seconds at 30 FPS (CHANGED from 15)
    
    # Phone usage scoring
    PHONE_SCORE_INCREMENT = 5.0  # Score increase per frame when phone detected
    PHONE_ALERT_THRESHOLD = 50  # Trigger alert at this score
    
    # ============================================================
    # AUDIO ALERT SETTINGS
    # ============================================================
    ENABLE_AUDIO_ALERTS = True
    
    # Continuous alarm settings
    AUDIO_ALERT_COOLDOWN = 1.0  # Seconds between beeps
    CONTINUOUS_ALARM_ENABLED = True  # Enable continuous alarm mode
    CONTINUOUS_ALARM_THRESHOLD = 70  # Start continuous alarm at this score
    
    # Sound settings - LOUD and URGENT
    AUDIO_VOLUME = 1.0  # Maximum volume
    BEEP_FREQUENCY = 1000  # Hz - higher pitch = more urgent
    BEEP_DURATION = 0.5  # seconds
    
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
    
    # Eye landmarks
    EYE_COLOR = (255, 0, 255)  # Magenta
    EYE_THICKNESS = 1
    
    # Mouth landmarks
    MOUTH_COLOR = (0, 255, 255)  # Yellow
    MOUTH_THICKNESS = 1
    
    # Hand landmarks (for phone detection)
    HAND_COLOR = (255, 128, 0)  # Orange
    HAND_THICKNESS = 2
    
    # Alert colors
    COLOR_NORMAL = (0, 255, 0)    # Green
    COLOR_WARNING = (0, 165, 255)  # Orange
    COLOR_DANGER = (0, 0, 255)     # Red
    COLOR_PHONE = (255, 0, 255)    # Magenta for phone alert
    
    # Banner settings
    SHOW_NORMAL_BANNER = True  # Show green banner when normal
    NORMAL_BANNER_COLOR = (0, 200, 0)  # Dark green
    
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