"""
Video capture module for handling webcam input.
"""

import cv2
import numpy as np
from config.settings import Config
from utils.logger import log


class VideoCapture:
    """Handles webcam capture and frame preprocessing."""
    
    def __init__(self):
        """Initialize video capture."""
        self.cap = None
        self.is_opened = False
        
    def start(self, camera_id=None):
        """
        Start video capture from webcam.
        
        Args:
            camera_id (int, optional): Camera device ID
        
        Returns:
            bool: True if successful, False otherwise
        """
        if camera_id is None:
            camera_id = Config.CAMERA_ID
        
        log.info(f"Initializing camera {camera_id}...")
        
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            log.error(f"Failed to open camera {camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
        
        self.is_opened = True
        log.info("Camera initialized successfully")
        
        return True
    
    def read_frame(self):
        """
        Read a frame from the camera.
        
        Returns:
            tuple: (success, original_frame, processed_frame)
                - success (bool): Whether frame was read successfully
                - original_frame (np.ndarray): Original resolution frame
                - processed_frame (np.ndarray): Resized frame for processing
        """
        if not self.is_opened:
            log.warning("Camera not opened")
            return False, None, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            log.warning("Failed to read frame")
            return False, None, None
        
        # Resize for faster processing
        processed_frame = cv2.resize(
            frame,
            (Config.PROCESS_WIDTH, Config.PROCESS_HEIGHT)
        )
        
        return True, frame, processed_frame
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            log.info("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()