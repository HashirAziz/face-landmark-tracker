"""
Main application - Real-Time Face Detection & Landmark Tracking.

This is the entry point of the application.
Uses MediaPipe for fast, accurate face detection with 468 facial landmarks.
"""

import cv2
import sys
from config.settings import Config
from camera.video_capture import VideoCapture
from face_detection.detector import FaceDetector
from landmarks.landmark_tracker import LandmarkTracker
from utils.fps_counter import FPSCounter
from utils.visualization import (
    draw_bounding_box,
    draw_landmarks,
    draw_fps,
    draw_no_face_message
)
from utils.logger import log


class FaceLandmarkApp:
    """Main application class orchestrating all components."""
    
    def __init__(self):
        """Initialize application components."""
        self.camera = VideoCapture()
        self.detector = FaceDetector()
        self.tracker = LandmarkTracker()
        self.fps_counter = FPSCounter()
        
        # Create necessary directories
        Config.create_directories()
    
    def initialize(self):
        """
        Initialize all components.
        
        Returns:
            bool: True if all components initialized successfully
        """
        log.info("=" * 60)
        log.info("Face Landmark Tracker - MediaPipe Edition")
        log.info("=" * 60)
        
        # Initialize camera
        if not self.camera.start():
            log.error("Failed to initialize camera")
            return False
        
        # Initialize face detector
        if not self.detector.initialize():
            log.error("Failed to initialize face detector")
            self.camera.release()
            return False
        
        log.info("All components initialized successfully")
        log.info("=" * 60)
        
        return True
    
    def process_frame(self, original_frame, processed_frame):
        """
        Process a single frame: detect faces and draw landmarks.
        
        Args:
            original_frame (np.ndarray): Original resolution frame for display
            processed_frame (np.ndarray): Resized frame for processing
        
        Returns:
            np.ndarray: Processed frame with visualizations
        """
        # Detect faces on the processed (smaller) frame
        faces = self.detector.detect_faces(processed_frame)
        
        # Calculate scaling factors
        h_orig, w_orig = original_frame.shape[:2]
        h_proc, w_proc = processed_frame.shape[:2]
        scale_x = w_orig / w_proc
        scale_y = h_orig / h_proc
        
        # If no faces detected
        if len(faces) == 0:
            draw_no_face_message(original_frame)
        else:
            # Process each detected face
            for face in faces:
                # Get face information
                face_info = self.detector.get_face_info(face, w_proc, h_proc)
                
                # Scale bounding box to original frame size
                bbox = face_info['bbox']
                scaled_bbox = (
                    int(bbox[0] * scale_x),
                    int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x),
                    int(bbox[3] * scale_y)
                )
                
                # Draw bounding box
                draw_bounding_box(
                    original_frame,
                    scaled_bbox,
                    face_info['confidence']
                )
                
                # Scale landmarks
                scaled_landmarks = self.tracker.scale_landmarks(
                    face_info['landmarks'],
                    scale_x,
                    scale_y
                )
                
                # Filter landmarks for cleaner display (show every 5th landmark)
                # Comment this line to show ALL 468 landmarks
                display_landmarks = self.tracker.filter_landmarks_for_display(
                    scaled_landmarks,
                    step=10
                )
                
                # Draw landmarks
                # Note: MediaPipe has many connection lines, which can be cluttered
                # Set Config.DRAW_CONNECTIONS = False in settings.py for cleaner look
                draw_landmarks(
                    original_frame,
                    display_landmarks,
                    connections=None  # Set to face_info['landmark_connections'] to draw all connections
                )
                
                # Optional: Get landmark statistics (for debugging)
                # stats = self.tracker.get_landmark_statistics(scaled_landmarks)
                # log.debug(f"Face center: {stats['center']}, Size: {stats['width']}x{stats['height']}")
        
        return original_frame
    
    def run(self):
        """Main application loop."""
        if not self.initialize():
            log.error("Initialization failed. Exiting...")
            return
        
        log.info("Starting main loop. Press 'q' to quit.")
        log.info("TIP: Adjust DRAW_CONNECTIONS in config/settings.py for visual preferences")
        
        try:
            while True:
                # Read frame from camera
                ret, original_frame, processed_frame = self.camera.read_frame()
                
                if not ret:
                    log.warning("Failed to capture frame. Retrying...")
                    continue
                
                # Process frame
                output_frame = self.process_frame(original_frame, processed_frame)
                
                # Update FPS counter
                self.fps_counter.update()
                fps = self.fps_counter.get_fps()
                
                # Draw FPS on frame
                draw_fps(output_frame, fps)
                
                # Display frame
                cv2.imshow("Face Landmark Tracker - MediaPipe", output_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log.info("Quit signal received")
                    break
                elif key == ord('s'):
                    # Save screenshot (bonus feature)
                    filename = f"screenshot_{int(self.fps_counter.frame_count)}.png"
                    cv2.imwrite(filename, output_frame)
                    log.info(f"Screenshot saved: {filename}")
                
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        
        except Exception as e:
            log.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        log.info("Cleaning up resources...")
        self.camera.release()
        self.detector.release()
        cv2.destroyAllWindows()
        log.info("Application terminated successfully")


def main():
    """Entry point of the application."""
    app = FaceLandmarkApp()
    app.run()


if __name__ == "__main__":
    main()