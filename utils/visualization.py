"""
Visualization utilities for drawing bounding boxes and landmarks.
Includes drowsiness and phone detection visualizations.
"""

import cv2
import numpy as np
from config.settings import Config


def draw_bounding_box(frame, bbox, confidence=None):
    """
    Draw bounding box around detected face.
    
    Args:
        frame (np.ndarray): Input frame
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        confidence (float, optional): Detection confidence score
    
    Returns:
        np.ndarray: Frame with bounding box drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        Config.BBOX_COLOR,
        Config.BBOX_THICKNESS
    )
    
    # Draw confidence score if provided
    if confidence is not None:
        label = f"Conf: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(
            label,
            Config.FPS_FONT,
            0.5,
            1
        )
        
        # Background for text
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            Config.BBOX_COLOR,
            -1
        )
        
        # Text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            Config.FPS_FONT,
            0.5,
            (0, 0, 0),
            1
        )
    
    return frame


def draw_landmarks(frame, landmarks, connections=None):
    """
    Draw facial landmarks on frame.
    
    Args:
        frame (np.ndarray): Input frame
        landmarks (list): List of landmark points [(x, y), ...]
        connections (list, optional): List of landmark connection pairs
    
    Returns:
        np.ndarray: Frame with landmarks drawn
    """
    if landmarks is None or len(landmarks) == 0:
        return frame
    
    # Draw each landmark point
    for point in landmarks:
        cv2.circle(
            frame,
            point,
            Config.LANDMARK_RADIUS,
            Config.LANDMARK_COLOR,
            Config.LANDMARK_THICKNESS
        )
    
    return frame


def draw_eye_landmarks(frame, left_eye, right_eye, eyes_closed=False):
    """
    Draw eye landmarks with special highlighting.
    
    Args:
        frame (np.ndarray): Input frame
        left_eye (list): Left eye landmarks
        right_eye (list): Right eye landmarks
        eyes_closed (bool): Whether eyes are closed
    
    Returns:
        np.ndarray: Frame with eye landmarks drawn
    """
    color = Config.COLOR_DANGER if eyes_closed else Config.EYE_COLOR
    
    # Draw left eye
    if left_eye and len(left_eye) >= 6:
        points = np.array(left_eye, dtype=np.int32)
        cv2.polylines(frame, [points], True, color, Config.EYE_THICKNESS)
        for point in left_eye:
            cv2.circle(frame, point, 2, color, -1)
    
    # Draw right eye
    if right_eye and len(right_eye) >= 6:
        points = np.array(right_eye, dtype=np.int32)
        cv2.polylines(frame, [points], True, color, Config.EYE_THICKNESS)
        for point in right_eye:
            cv2.circle(frame, point, 2, color, -1)
    
    return frame


def draw_mouth_landmarks(frame, mouth_landmarks, yawning=False):
    """
    Draw mouth landmarks with special highlighting.
    
    Args:
        frame (np.ndarray): Input frame
        mouth_landmarks (list): Mouth landmarks
        yawning (bool): Whether person is yawning
    
    Returns:
        np.ndarray: Frame with mouth landmarks drawn
    """
    if not mouth_landmarks or len(mouth_landmarks) < 8:
        return frame
    
    color = Config.COLOR_WARNING if yawning else Config.MOUTH_COLOR
    
    # Draw mouth contour
    points = np.array(mouth_landmarks, dtype=np.int32)
    cv2.polylines(frame, [points], True, color, Config.MOUTH_THICKNESS)
    
    # Draw landmarks
    for point in mouth_landmarks:
        cv2.circle(frame, point, 2, color, -1)
    
    return frame


def draw_fps(frame, fps_value):
    """
    Draw FPS counter on frame.
    
    Args:
        frame (np.ndarray): Input frame
        fps_value (float): Current FPS
    
    Returns:
        np.ndarray: Frame with FPS displayed
    """
    fps_text = f"FPS: {fps_value:.1f}"
    
    cv2.putText(
        frame,
        fps_text,
        Config.FPS_POSITION,
        Config.FPS_FONT,
        Config.FPS_SCALE,
        Config.FPS_COLOR,
        Config.FPS_THICKNESS
    )
    
    return frame


def draw_dashboard(frame, combined_data):
    """
    Draw statistics dashboard with drowsiness AND phone detection.
    
    Args:
        frame (np.ndarray): Input frame
        combined_data (dict): Combined detection data
    
    Returns:
        np.ndarray: Frame with dashboard drawn
    """
    if not Config.SHOW_DASHBOARD:
        return frame
    
    x, y = Config.DASHBOARD_POSITION
    font = Config.FPS_FONT
    scale = Config.DASHBOARD_FONT_SCALE
    spacing = Config.DASHBOARD_LINE_SPACING
    color = (255, 255, 255)
    thickness = 1
    
    # Statistics
    stats = [
        f"Alert: {combined_data['alert_level']}",
        f"Eye Closures: {combined_data['total_eye_closures']}",
        f"Yawns: {combined_data['total_yawns']}",
    ]
    
    # Add phone stats if available
    if 'phone_detections' in combined_data:
        stats.append(f"Phone Uses: {combined_data['phone_detections']}")
    
    # Draw semi-transparent background
    bg_height = len(stats) * spacing + 20
    bg_width = 250
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 20), (x + bg_width, y + bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw text
    for i, stat in enumerate(stats):
        cv2.putText(
            frame,
            stat,
            (x, y + i * spacing),
            font,
            scale,
            color,
            thickness
        )
    
    return frame


def draw_no_face_message(frame):
    """
    Draw 'No Face Detected' message on frame.
    
    Args:
        frame (np.ndarray): Input frame
    
    Returns:
        np.ndarray: Frame with message
    """
    message = "No Face Detected"
    
    # Get text size for centering
    text_size, _ = cv2.getTextSize(
        message,
        Config.FPS_FONT,
        1.0,
        2
    )
    
    # Calculate center position
    height, width = frame.shape[:2]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    
    # Draw text
    cv2.putText(
        frame,
        message,
        (x, y),
        Config.FPS_FONT,
        1.0,
        (0, 0, 255),  # Red
        2
    )
    
    return frame