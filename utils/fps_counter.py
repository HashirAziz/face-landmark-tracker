"""
FPS (Frames Per Second) counter for performance monitoring.
"""

import time
from collections import deque


class FPSCounter:
    """Calculate and smooth FPS over a time window."""
    
    def __init__(self, window_size=30):
        """
        Initialize FPS counter.
        
        Args:
            window_size (int): Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.frame_count = 0
    
    def update(self):
        """Update FPS calculation with new frame."""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_count += 1
    
    def get_fps(self):
        """
        Calculate current FPS.
        
        Returns:
            float: Current FPS value
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        # Calculate FPS based on time window
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff > 0:
            fps = (len(self.frame_times) - 1) / time_diff
            return fps
        return 0.0
    
    def reset(self):
        """Reset FPS counter."""
        self.frame_times.clear()
        self.start_time = time.time()
        self.frame_count = 0