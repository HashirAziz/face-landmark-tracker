"""
Visualization utilities for drawing bounding boxes and landmarks.
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
    
    # Draw connections first (so they appear behind points)
    if connections and Config.DRAW_CONNECTIONS:
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    Config.CONNECTION_COLOR,
                    Config.CONNECTION_THICKNESS
                )
    
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