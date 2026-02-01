"""
Main application - Drowsiness Detection System.

Real-time detection of driver drowsiness using:
- Eye Aspect Ratio (EAR) for eye closure detection
- Mouth Aspect Ratio (MAR) for yawn detection
- Visual and audio alerts
"""

import cv2
import sys
from config.settings import Config
from camera.video_capture import VideoCapture
from face_detection.detector import FaceDetector
from landmarks.landmark_tracker import LandmarkTracker
from drowsiness.detector import DrowsinessDetector
from alerts.alert_system import AlertSystem
from utils.fps_counter import FPSCounter
from utils.visualization import (
    draw_bounding_box,
    draw_eye_landmarks,
    draw_mouth_landmarks,
    draw_fps,
    draw_dashboard,
    draw_no_face_message
)
from utils.logger import log


class DrowsinessDetectionApp:
    """Main application class for drowsiness detection system."""
    
    def __init__(self):
        """Initialize application components."""
        self.camera = VideoCapture()
        self.detector = FaceDetector()
        self.tracker = LandmarkTracker()
        self.drowsiness_detector = DrowsinessDetector()
        self.alert_system = AlertSystem()
        self.fps_counter = FPSCounter()
        
        # Create necessary directories
        Config.create_directories()
    
    def initialize(self):
        """
        Initialize all components.
        
        Returns:
            bool: True if all components initialized successfully
        """
        log.info("=" * 70)
        log.info("DROWSINESS DETECTION SYSTEM")
        log.info("Real-time Eye Closure & Yawn Detection")
        log.info("=" * 70)
        
        # Initialize camera
        if not self.camera.start():
            log.error("Failed to initialize camera")
            return False
        
        # Initialize face detector
        if not self.detector.initialize():
            log.error("Failed to initialize face detector")
            self.camera.release()
            return False
        
        log.info("=" * 70)
        log.info("DETECTION THRESHOLDS:")
        log.info(f"  Eye Aspect Ratio (EAR) Threshold: {Config.EAR_THRESHOLD}")
        log.info(f"  Mouth Aspect Ratio (MAR) Threshold: {Config.MAR_THRESHOLD}")
        log.info(f"  Eye Closure Frames: {Config.EAR_CONSEC_FRAMES}")
        log.info(f"  Yawn Frames: {Config.MAR_CONSEC_FRAMES}")
        log.info("=" * 70)
        log.info("CONTROLS:")
        log.info("  Press 'q' to quit")
        log.info("  Press 's' to save screenshot")
        log.info("  Press 'r' to reset statistics")
        log.info("=" * 70)
        
        return True
    
    def process_frame(self, original_frame, processed_frame):
        """
        Process a single frame: detect faces, analyze drowsiness, and draw visualizations.
        
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
        
        # Default drowsiness data (no face detected)
        drowsiness_data = None
        
        # If no faces detected
        if len(faces) == 0:
            draw_no_face_message(original_frame)
            # Use default drowsiness data
            drowsiness_data = self.drowsiness_detector._get_default_result()
        else:
            # Process the first face (driver)
            face = faces[0]
            
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
            
            # Scale landmarks to original frame
            scaled_landmarks = self.tracker.scale_landmarks(
                face_info['landmarks'],
                scale_x,
                scale_y
            )
            
            # DROWSINESS DETECTION
            drowsiness_data = self.drowsiness_detector.detect_drowsiness(scaled_landmarks)
            
            # Draw eye landmarks
            left_eye = [(int(x * scale_x), int(y * scale_y)) 
                       for x, y in drowsiness_data['left_eye_landmarks']]
            right_eye = [(int(x * scale_x), int(y * scale_y)) 
                        for x, y in drowsiness_data['right_eye_landmarks']]
            
            draw_eye_landmarks(
                original_frame,
                left_eye,
                right_eye,
                drowsiness_data['eyes_closed']
            )
            
            # Draw mouth landmarks
            mouth = [(int(x * scale_x), int(y * scale_y)) 
                    for x, y in drowsiness_data['mouth_landmarks']]
            
            draw_mouth_landmarks(
                original_frame,
                mouth,
                drowsiness_data['yawning']
            )
        
        # Draw alert overlay (works even without face detected)
        if drowsiness_data:
            self.alert_system.draw_alert_overlay(original_frame, drowsiness_data)
            
            # Trigger alerts
            self.alert_system.trigger_alert(drowsiness_data)
            
            # Draw statistics dashboard
            draw_dashboard(original_frame, drowsiness_data)
        
        return original_frame
    
    def run(self):
        """Main application loop."""
        if not self.initialize():
            log.error("Initialization failed. Exiting...")
            return
        
        log.info("Starting drowsiness detection. Stay alert!")
        
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
                cv2.imshow("Drowsiness Detection System", output_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    log.info("Quit signal received")
                    break
                
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"drowsiness_screenshot_{int(self.fps_counter.frame_count)}.png"
                    cv2.imwrite(filename, output_frame)
                    log.info(f"Screenshot saved: {filename}")
                
                elif key == ord('r'):
                    # Reset statistics
                    self.drowsiness_detector.reset()
                    log.info("Statistics reset")
                
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
        self.alert_system.cleanup()
        cv2.destroyAllWindows()
        
        # Print final statistics
        log.info("=" * 70)
        log.info("SESSION SUMMARY:")
        log.info(f"  Total Eye Closures: {self.drowsiness_detector.total_eye_closures}")
        log.info(f"  Total Yawns: {self.drowsiness_detector.total_yawns}")
        log.info(f"  Total Frames: {self.fps_counter.frame_count}")
        log.info("=" * 70)
        log.info("Application terminated successfully")


def main():
    """Entry point of the application."""
    app = DrowsinessDetectionApp()
    app.run()


if __name__ == "__main__":
    main()