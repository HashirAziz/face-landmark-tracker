"""
Alert system for drowsiness detection.
Provides visual and audio warnings.
"""

import time
import pygame
import cv2
import numpy as np
from pathlib import Path
from config.settings import Config
from utils.logger import log


class AlertSystem:
    """Manage visual and audio alerts for drowsiness detection."""
    
    def __init__(self):
        """Initialize alert system."""
        self.last_audio_alert_time = 0
        self.audio_initialized = False
        
        # Initialize pygame mixer for audio
        if Config.ENABLE_AUDIO_ALERTS:
            try:
                pygame.mixer.init()
                self.audio_initialized = True
                log.info("Audio alert system initialized")
            except Exception as e:
                log.warning(f"Could not initialize audio: {e}")
                self.audio_initialized = False
        
        # Create simple beep sound programmatically (if sound files don't exist)
        self.beep_sound = None
        if self.audio_initialized:
            self._create_beep_sound()
    
    def _create_beep_sound(self):
        """Create a simple beep sound programmatically."""
        try:
            # Create a simple beep using numpy
            sample_rate = 22050
            duration = 0.3  # seconds
            frequency = 1800  # Hz
            
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Apply envelope to avoid clicks
            envelope = np.exp(-5 * t)
            wave = wave * envelope
            
            # Convert to 16-bit PCM
            wave = (wave * 32767).astype(np.int16)
            
            # Create stereo sound
            stereo_wave = np.column_stack([wave, wave])
            
            # Create pygame sound
            self.beep_sound = pygame.sndarray.make_sound(stereo_wave)
            self.beep_sound.set_volume(Config.AUDIO_VOLUME)
            
            log.info("Beep sound created successfully")
        except Exception as e:
            log.warning(f"Could not create beep sound: {e}")
            self.beep_sound = None
    
    def play_alert(self, alert_level):
        """
        Play audio alert based on alert level.
        
        Args:
            alert_level (str): "NORMAL", "WARNING", or "DANGER"
        """
        if not self.audio_initialized or not self.beep_sound:
            return
        
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_audio_alert_time < Config.AUDIO_ALERT_COOLDOWN:
            return
        
        if alert_level == "DANGER":
            # Play rapid beeps for danger
            try:
                self.beep_sound.play()
                self.last_audio_alert_time = current_time
                log.warning("DANGER alert played")
            except Exception as e:
                log.error(f"Error playing alert: {e}")
        
        elif alert_level == "WARNING":
            # Play single beep for warning
            try:
                self.beep_sound.play()
                self.last_audio_alert_time = current_time
                log.info("WARNING alert played")
            except Exception as e:
                log.error(f"Error playing alert: {e}")
    
    def draw_alert_overlay(self, frame, drowsiness_data):
        """
        Draw visual alert overlay on frame.
        
        Args:
            frame (np.ndarray): Input frame
            drowsiness_data (dict): Drowsiness detection results
        
        Returns:
            np.ndarray: Frame with alert overlay
        """
        alert_level = drowsiness_data['alert_level']
        drowsiness_score = drowsiness_data['drowsiness_score']
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Draw alert banner if not normal
        if alert_level != "NORMAL":
            # Determine color and message
            if alert_level == "DANGER":
                color = Config.COLOR_DANGER
                message = "DROWSINESS ALERT! TAKE A BREAK!"
                banner_height = 80
            else:  # WARNING
                color = Config.COLOR_WARNING
                message = "⚠️ Warning: Stay Alert ⚠️"
                banner_height = 60
            
            # Draw semi-transparent banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw message
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(message, font, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = banner_height // 2 + text_size[1] // 2
            
            # Draw text with outline
            cv2.putText(frame, message, (text_x, text_y), font, 1.0, (0, 0, 0), 4)
            cv2.putText(frame, message, (text_x, text_y), font, 1.0, (255, 255, 255), 2)
        
        # Draw drowsiness score bar
        self._draw_score_bar(frame, drowsiness_score)
        
        # Draw status indicators
        self._draw_status_indicators(frame, drowsiness_data)
        
        return frame
    
    def _draw_score_bar(self, frame, score):
        """
        Draw drowsiness score bar.
        
        Args:
            frame (np.ndarray): Input frame
            score (float): Drowsiness score (0-100)
        """
        height, width = frame.shape[:2]
        
        # Bar dimensions
        bar_width = 200
        bar_height = 20
        bar_x = width - bar_width - 20
        bar_y = 20
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Fill based on score
        fill_width = int((score / Config.DROWSINESS_SCORE_MAX) * bar_width)
        
        # Color based on level
        if score >= Config.ALERT_LEVEL_DANGER:
            color = Config.COLOR_DANGER
        elif score >= Config.ALERT_LEVEL_WARNING:
            color = Config.COLOR_WARNING
        else:
            color = Config.COLOR_NORMAL
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Label
        label = f"Drowsiness: {int(score)}%"
        cv2.putText(frame, label, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_status_indicators(self, frame, drowsiness_data):
        """
        Draw status indicators for eyes and mouth.
        
        Args:
            frame (np.ndarray): Input frame
            drowsiness_data (dict): Drowsiness detection results
        """
        height, width = frame.shape[:2]
        
        # Position
        x = width - 220
        y = 60
        
        # Eyes status
        eye_status = "CLOSED" if drowsiness_data['eyes_closed'] else "OPEN"
        eye_color = Config.COLOR_DANGER if drowsiness_data['eyes_closed'] else Config.COLOR_NORMAL
        cv2.putText(frame, f"Eyes: {eye_status}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 2)
        
        # Yawn status
        yawn_status = "YAWNING" if drowsiness_data['yawning'] else "NORMAL"
        yawn_color = Config.COLOR_WARNING if drowsiness_data['yawning'] else Config.COLOR_NORMAL
        cv2.putText(frame, f"Mouth: {yawn_status}", (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 2)
        
        # EAR value
        cv2.putText(frame, f"EAR: {drowsiness_data['ear']:.3f}", (x, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # MAR value
        cv2.putText(frame, f"MAR: {drowsiness_data['mar']:.3f}", (x, y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def trigger_alert(self, drowsiness_data):
        """
        Trigger appropriate alerts based on drowsiness data.
        
        Args:
            drowsiness_data (dict): Drowsiness detection results
        """
        alert_level = drowsiness_data['alert_level']
        
        # Play audio alert
        if alert_level in ["WARNING", "DANGER"]:
            self.play_alert(alert_level)
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.audio_initialized:
            try:
                pygame.mixer.quit()
                log.info("Audio system cleaned up")
            except:
                pass