"""
Main application - Driver Safety Monitoring System.

Real-time detection of:
- Eye closure (drowsiness)
- Yawning (drowsiness)
- Phone usage (distraction)
"""

import cv2
import sys
import mediapipe as mp
from config.settings import Config
from camera.video_capture import VideoCapture
from face_detection.detector import FaceDetector
from landmarks.landmark_tracker import LandmarkTracker
from drowsiness.detector import DrowsinessDetector
from phone_detection.detector import PhoneDetector
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


class DriverSafetyApp:
    """Main application class for driver safety monitoring system."""
    
    def __init__(self):
        """Initialize application components."""
        self.camera = VideoCapture()
        self.detector = FaceDetector()
        self.tracker = LandmarkTracker()
        self.drowsiness_detector = DrowsinessDetector()
        self.phone_detector = PhoneDetector()
        self.alert_system = AlertSystem()
        self.fps_counter = FPSCounter()
        
        # Initialize MediaPipe Hands for phone detection
        self.mp_hands = None
        self.hands = None
        if Config.ENABLE_HAND_DETECTION:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=Config.MAX_NUM_HANDS,
                min_detection_confidence=Config.HAND_DETECTION_CONFIDENCE,
                min_tracking_confidence=Config.HAND_TRACKING_CONFIDENCE
            )
        
        # Create necessary directories
        Config.create_directories()
    
    def initialize(self):
        """
        Initialize all components.
        
        Returns:
            bool: True if all components initialized successfully
        """
        log.info("=" * 70)
        log.info("DRIVER SAFETY MONITORING SYSTEM")
        log.info("Real-time Drowsiness & Phone Detection")
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
        log.info(f"  Eye Aspect Ratio (EAR): {Config.EAR_THRESHOLD}")
        log.info(f"  Mouth Aspect Ratio (MAR): {Config.MAR_THRESHOLD}")
        log.info(f"  Eye Closure Frames: {Config.EAR_CONSEC_FRAMES}")
        log.info(f"  Yawn Frames: {Config.MAR_CONSEC_FRAMES}")
        
        if Config.ENABLE_HAND_DETECTION:
            log.info(f"  Phone Detection: {Config.PHONE_DETECTION_THRESHOLD} (confidence)")
            log.info(f"  Phone Frames: {Config.PHONE_CONSEC_FRAMES}")
        
        log.info("=" * 70)
        log.info("CONTROLS:")
        log.info("  Press 'q' to quit")
        log.info("  Press 's' to save screenshot")
        log.info("  Press 'r' to reset statistics")
        log.info("=" * 70)
        
        return True
    
    def detect_hands(self, frame):
        """
        Detect hands in frame using MediaPipe.
        
        Args:
            frame (np.ndarray): Input frame (BGR)
        
        Returns:
            tuple: (num_hands, hand_landmarks_list)
        """
        if not Config.ENABLE_HAND_DETECTION or not self.hands:
            return 0, []
        
        try:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            # Extract hand landmarks
            if results.multi_hand_landmarks:
                hand_landmarks_list = []
                h, w = frame.shape[:2]
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Convert normalized coordinates to pixel coordinates
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append((x, y))
                    hand_landmarks_list.append(landmarks)
                
                return len(hand_landmarks_list), hand_landmarks_list
            
            return 0, []
            
        except Exception as e:
            log.error(f"Error detecting hands: {e}")
            return 0, []
    
    def draw_hand_landmarks(self, frame, hand_landmarks_list):
        """
        Draw hand landmarks on frame.
        
        Args:
            frame (np.ndarray): Input frame
            hand_landmarks_list (list): List of hand landmarks
        """
        if not hand_landmarks_list:
            return
        
        for hand_landmarks in hand_landmarks_list:
            # Draw hand skeleton
            for i, point in enumerate(hand_landmarks):
                cv2.circle(frame, point, 3, Config.HAND_COLOR, -1)
            
            # Draw connections (simplified)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            ]
            
            for start_idx, end_idx in connections:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    cv2.line(frame, start, end, Config.HAND_COLOR, 2)
    
    def draw_phone_alert(self, frame, phone_data):
        """
        Draw phone detection alert overlay.
        
        Args:
            frame (np.ndarray): Input frame
            phone_data (dict): Phone detection results
        """
        if not phone_data['phone_detected']:
            return
        
        height, width = frame.shape[:2]
        
        # Draw MAGENTA banner for phone usage
        color = Config.COLOR_PHONE
        message = "PHONE USAGE DETECTED! FOCUS ON DRIVING LEAVE THE PHONE!"
        banner_height = 90
        
        # Draw semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw message
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        text_size = cv2.getTextSize(message, font, font_scale, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = banner_height // 2 + text_size[1] // 2
        
        # Draw text with outline
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (0, 0, 0), 5)
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (255, 255, 255), 3)
    
    def draw_phone_status(self, frame, phone_data):
        """
        Draw phone detection status.
        
        Args:
            frame (np.ndarray): Input frame
            phone_data (dict): Phone detection results
        """
        height, width = frame.shape[:2]
        x = width - 220
        y = 120  # Below drowsiness indicators
        
        # Phone status
        phone_status = "DETECTED" if phone_data['phone_detected'] else "NOT DETECTED"
        phone_color = Config.COLOR_PHONE if phone_data['phone_detected'] else Config.COLOR_NORMAL
        cv2.putText(frame, f"Phone: {phone_status}", (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, phone_color, 2)
        
        # Confidence
        if Config.ENABLE_HAND_DETECTION:
            cv2.putText(frame, f"Conf: {phone_data['confidence']:.2f}", (x, y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def update_dashboard(self, drowsiness_data, phone_data):
        """
        Update dashboard with phone detection stats.
        
        Args:
            drowsiness_data (dict): Drowsiness detection data
            phone_data (dict): Phone detection data
        
        Returns:
            dict: Combined stats for dashboard
        """
        combined_stats = drowsiness_data.copy()
        combined_stats['phone_detections'] = phone_data['total_phone_detections']
        combined_stats['phone_duration'] = phone_data['current_duration']
        return combined_stats
    
    def process_frame(self, original_frame, processed_frame):
        """
        Process a single frame: detect faces, hands, analyze drowsiness and phone usage.
        
        Args:
            original_frame (np.ndarray): Original resolution frame for display
            processed_frame (np.ndarray): Resized frame for processing
        
        Returns:
            np.ndarray: Processed frame with visualizations
        """
        # Detect faces
        faces = self.detector.detect_faces(processed_frame)
        
        # Detect hands (for phone detection)
        num_hands, hand_landmarks_list = self.detect_hands(processed_frame)
        
        # Calculate scaling factors
        h_orig, w_orig = original_frame.shape[:2]
        h_proc, w_proc = processed_frame.shape[:2]
        scale_x = w_orig / w_proc
        scale_y = h_orig / h_proc
        
        # Default data
        drowsiness_data = None
        phone_data = None
        face_bbox = None
        
        # If no faces detected
        if len(faces) == 0:
            draw_no_face_message(original_frame)
            drowsiness_data = self.drowsiness_detector._get_default_result()
            phone_data = self.phone_detector._get_result(False, 0.0, [], None)
        else:
            # Process the first face (driver)
            face = faces[0]
            
            # Get face information
            face_info = self.detector.get_face_info(face, w_proc, h_proc)
            
            # Scale bounding box
            bbox = face_info['bbox']
            scaled_bbox = (
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y)
            )
            face_bbox = scaled_bbox
            
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
            
            # PHONE DETECTION (if hands detected)
            if num_hands > 0:
                # Scale hand landmarks
                scaled_hand_landmarks = []
                for hand_landmarks in hand_landmarks_list:
                    scaled_hand = [(int(x * scale_x), int(y * scale_y)) for x, y in hand_landmarks]
                    scaled_hand_landmarks.append(scaled_hand)
                
                # Detect phone usage
                phone_data = self.phone_detector.detect_phone_usage(
                    num_hands,
                    hand_landmarks_list,  # Use unscaled for detection
                    face_info['landmarks'],
                    bbox
                )
                
                # Draw phone bounding box if detected
                if phone_data and phone_data.get('phone_bbox'):
                    phone_bbox = phone_data['phone_bbox']
                    x1, y1, x2, y2 = phone_bbox
                    
                    # Draw magenta box around phone
                    cv2.rectangle(original_frame, (x1, y1), (x2, y2),
                                 Config.PHONE_BBOX_COLOR, Config.PHONE_BBOX_THICKNESS)
                    
                    # Draw "Phone" label
                    label = "Phone"
                    label_size = cv2.getTextSize(label, Config.FPS_FONT, 0.6, 2)[0]
                    cv2.rectangle(original_frame, (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0] + 10, y1), Config.PHONE_BBOX_COLOR, -1)
                    cv2.putText(original_frame, label, (x1 + 5, y1 - 5),
                               Config.FPS_FONT, 0.6, (255, 255, 255), 2)
                
                # Draw hand landmarks
                self.draw_hand_landmarks(original_frame, scaled_hand_landmarks)
            else:
                # No hands detected
                phone_data = self.phone_detector._get_result(False, 0.0, [], face_info['landmarks'])
        
        # PRIORITY ALERT SYSTEM
        # Phone usage has HIGHEST priority
        if phone_data and phone_data['phone_detected']:
            self.draw_phone_alert(original_frame, phone_data)
            # Still show drowsiness overlay below
            if drowsiness_data:
                self.alert_system.draw_alert_overlay(original_frame, drowsiness_data)
        elif drowsiness_data:
            # Show drowsiness alert if no phone
            self.alert_system.draw_alert_overlay(original_frame, drowsiness_data)
        
        # Draw status indicators
        if phone_data:
            self.draw_phone_status(original_frame, phone_data)
        
        # Trigger audio alerts
        if drowsiness_data:
            self.alert_system.trigger_alert(drowsiness_data)
        
        if phone_data and phone_data['phone_detected']:
            # Additional phone alert (could use different sound)
            self.alert_system.trigger_alert({'alert_level': 'DANGER', 'drowsiness_score': 100})
        
        # Draw combined dashboard
        if drowsiness_data and phone_data:
            combined_stats = self.update_dashboard(drowsiness_data, phone_data)
            draw_dashboard(original_frame, combined_stats)
        
        return original_frame
    
    def run(self):
        """Main application loop."""
        if not self.initialize():
            log.error("Initialization failed. Exiting...")
            return
        
        log.info("Starting driver safety monitoring. Stay safe!")
        
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
                cv2.imshow("Driver Safety Monitoring System", output_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    log.info("Quit signal received")
                    break
                
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"safety_screenshot_{int(self.fps_counter.frame_count)}.png"
                    cv2.imwrite(filename, output_frame)
                    log.info(f"Screenshot saved: {filename}")
                
                elif key == ord('r'):
                    # Reset statistics
                    self.drowsiness_detector.reset()
                    self.phone_detector.reset()
                    log.info("All statistics reset")
                
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
        
        if self.hands:
            self.hands.close()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        log.info("=" * 70)
        log.info("SESSION SUMMARY:")
        log.info(f"  Total Eye Closures: {self.drowsiness_detector.total_eye_closures}")
        log.info(f"  Total Yawns: {self.drowsiness_detector.total_yawns}")
        log.info(f"  Total Phone Detections: {self.phone_detector.total_phone_detections}")
        log.info(f"  Longest Phone Duration: {self.phone_detector.longest_phone_duration} frames")
        log.info(f"  Total Frames: {self.fps_counter.frame_count}")
        log.info("=" * 70)
        log.info("Application terminated successfully. Drive safe!")


def main():
    """Entry point of the application."""
    app = DriverSafetyApp()
    app.run()


if __name__ == "__main__":
    main()