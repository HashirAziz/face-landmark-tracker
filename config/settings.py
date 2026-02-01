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
    # DROWSINESS DETECTION SETTINGS
    # ============================================================
    # Eye Aspect Ratio (EAR) thresholds
    EAR_THRESHOLD = 0.25  # Below this = eyes closed
    EAR_CONSEC_FRAMES = 20  # Consecutive frames for drowsiness (20 frames ≈ 0.67 seconds at 30 FPS)
    
    # FIX #3: Improved MAR threshold for better yawn detection
    MAR_THRESHOLD = 0.7  # Above this = yawning (increased from 0.6 for more accuracy)
    MAR_CONSEC_FRAMES = 15  # Consecutive frames for yawn detection
    
    # FIX #2: Faster decay to prevent score staying at 100%
    DROWSINESS_SCORE_MAX = 100  # Maximum drowsiness score
    DROWSINESS_SCORE_DECAY = 2.0  # Increased from 0.5 - faster recovery when alert
    DROWSINESS_SCORE_INCREMENT_EYES = 3.0  # Increased from 2.0 - faster increase when drowsy
    DROWSINESS_SCORE_INCREMENT_YAWN = 2.0  # Increased from 1.5
    
    # Alert levels
    ALERT_LEVEL_WARNING = 30  # Yellow warning
    ALERT_LEVEL_DANGER = 60   # Red danger
    
    # ============================================================
    # AUDIO ALERT SETTINGS
    # ============================================================
    ENABLE_AUDIO_ALERTS = True
    
    # FIX #4: Continuous alarm settings
    AUDIO_ALERT_COOLDOWN = 1.0  # Reduced from 3.0 - beep every 1 second when drowsy
    CONTINUOUS_ALARM_ENABLED = True  # Enable continuous alarm mode
    CONTINUOUS_ALARM_THRESHOLD = 70  # Start continuous alarm at this score
    
    # FIX #4: Louder, more urgent sound settings
    AUDIO_VOLUME = 1.0  # Increased from 0.7 to maximum volume
    BEEP_FREQUENCY = 1000  # Increased from 800 Hz - higher pitch = more urgent
    BEEP_DURATION = 0.5  # Increased from 0.3 seconds - longer beep
    
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
    
    # FIX #5: Banner settings
    SHOW_NORMAL_BANNER = True  # Show green "ALERT" banner when normal
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