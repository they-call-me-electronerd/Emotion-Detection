"""
Real-Time Facial Emotion Detection System
==========================================
Main application that combines face detection, preprocessing, and emotion prediction
to create a real-time emotion detection system using webcam input.

Author: AI Engineer
Date: January 2026
Version: 1.0.0

Usage:
    python main.py [--camera CAMERA_ID] [--confidence THRESHOLD]
    
    --camera: Camera device ID (default: 0)
    --confidence: Minimum confidence threshold to display prediction (default: 0.5)
"""

import cv2
import sys
import os
import argparse
import time
from typing import Optional

# Add src directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from face_detection import FaceDetector
from emotion_prediction import EmotionPreprocessor
from emotion_model import EmotionDetector


class EmotionDetectionSystem:
    """
    Complete real-time emotion detection system.
    
    This class integrates face detection, preprocessing, and emotion prediction
    to provide a seamless real-time emotion detection experience.
    
    Attributes:
        face_detector (FaceDetector): Face detection module
        preprocessor (EmotionPreprocessor): Face preprocessing module
        emotion_detector (EmotionDetector): Emotion prediction module
        confidence_threshold (float): Minimum confidence to display predictions
        show_fps (bool): Whether to display FPS counter
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 cascade_path: Optional[str] = None,
                 confidence_threshold: float = 0.2,
                 show_fps: bool = True):
        """
        Initialize the emotion detection system.
        
        Args:
            model_path (str, optional): Path to emotion model
            cascade_path (str, optional): Path to Haar Cascade file
            confidence_threshold (float): Minimum confidence threshold (default: 0.5)
            show_fps (bool): Display FPS counter (default: True)
        """
        print("=" * 60)
        print("Real-Time Facial Emotion Detection System")
        print("=" * 60)
        
        # Initialize components
        try:
            print("\n[1/3] Initializing face detector...")
            self.face_detector = FaceDetector(cascade_path)
            
            print("\n[2/3] Initializing emotion preprocessor...")
            self.preprocessor = EmotionPreprocessor(target_size=(48, 48))
            
            print("\n[3/3] Initializing emotion detector...")
            self.emotion_detector = EmotionDetector(model_path)
            
            print("\n" + "=" * 60)
            print("[SUCCESS] System initialized successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize system: {str(e)}")
            raise
        
        self.confidence_threshold = confidence_threshold
        self.show_fps = show_fps
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 30  # Track last 30 frames for FPS calculation
    
    def draw_emotion_info(self, frame, x, y, w, h, emotion, confidence, color_override=None):
        """
        Draw bounding box and emotion information on frame.
        
        Args:
            frame: Video frame
            x, y, w, h: Face bounding box coordinates
            emotion: Predicted emotion
            confidence: Prediction confidence
        """
        # Get color for this emotion (or override for low-confidence display)
        color = color_override or self.emotion_detector.get_emotion_color(emotion)
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare emotion text
        emotion_text = f"{emotion}: {confidence:.2%}"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            emotion_text, font, font_scale, thickness
        )
        
        # Draw background rectangle for text
        text_x = x
        text_y = y - 10
        
        # Ensure text stays within frame
        if text_y - text_height - 5 < 0:
            text_y = y + h + text_height + 5
        
        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + 5),
            color,
            -1  # Filled rectangle
        )
        
        # Draw emotion text
        cv2.putText(
            frame,
            emotion_text,
            (text_x + 2, text_y),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness,
            cv2.LINE_AA
        )
        
        # Draw small confidence bar
        bar_width = w
        bar_height = 5
        bar_x = x
        bar_y = y + h + 5
        
        # Background bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
        # Confidence bar
        conf_width = int(bar_width * confidence)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + conf_width, bar_y + bar_height),
            color,
            -1
        )
    
    def draw_statistics(self, frame, num_faces, fps):
        """
        Draw system statistics on frame.
        
        Args:
            frame: Video frame
            num_faces: Number of detected faces
            fps: Current FPS
        """
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green
        thickness = 2
        
        cv2.putText(frame, f"Faces Detected: {num_faces}", (20, 35),
                   font, font_scale, color, thickness)
        
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65),
                       font, font_scale, color, thickness)
        
        cv2.putText(frame, "Press 'Q' to quit", (20, 90),
                   font, 0.5, (255, 255, 255), 1)
    
    def calculate_fps(self):
        """Calculate current FPS based on recent frame times."""
        if len(self.frame_times) < 2:
            return 0.0
        
        # Calculate average time per frame
        avg_time = sum(self.frame_times) / len(self.frame_times)
        
        # Convert to FPS
        if avg_time > 0:
            return 1.0 / avg_time
        return 0.0
    
    def process_frame(self, frame):
        """
        Process a single frame: detect faces, predict emotions, and annotate.
        
        Args:
            frame: Input video frame
        
        Returns:
            Annotated frame
        """
        start_time = time.time()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            try:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess face
                preprocessed_face = self.preprocessor.preprocess_face(face_roi)
                
                # Predict emotion
                emotion, confidence, _ = self.emotion_detector.predict_emotion(preprocessed_face)
                
                # Always display the top prediction.
                # If confidence is low, draw it in gray so it doesn't look like a confident result.
                if confidence >= self.confidence_threshold:
                    self.draw_emotion_info(frame, x, y, w, h, emotion, confidence)
                else:
                    self.draw_emotion_info(frame, x, y, w, h, emotion, confidence, color_override=(128, 128, 128))
                
            except Exception as e:
                print(f"[WARNING] Error processing face: {str(e)}")
                continue
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        fps = self.calculate_fps()
        
        # Draw statistics
        self.draw_statistics(frame, len(faces), fps)
        
        return frame
    
    def run(self, camera_id: int = 0, window_name: str = "Emotion Detection"):
        """
        Run the real-time emotion detection system.
        
        Args:
            camera_id (int): Camera device ID (default: 0)
            window_name (str): Window title
        """
        print(f"\n[INFO] Starting camera (ID: {camera_id})...")
        
        # Initialize video capture
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot access camera {camera_id}")
            print("[INFO] Please check:")
            print("  1. Camera is connected")
            print("  2. Camera is not being used by another application")
            print("  3. Try a different camera ID (0, 1, 2, etc.)")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[SUCCESS] Camera started successfully")
        print("\n" + "=" * 60)
        print("CONTROLS:")
        print("  Q - Quit application")
        print("  S - Save current frame")
        print("=" * 60 + "\n")
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                
                if not ret:
                    print("[ERROR] Failed to grab frame")
                    break
                
                frame_count += 1
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n[INFO] Quit signal received")
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save current frame
                    filename = f"emotion_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"[INFO] Frame saved as: {filename}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            print(f"\n[INFO] Processed {frame_count} frames")
            print("[INFO] Cleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Application closed successfully")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Facial Emotion Detection System"
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--confidence', '-conf',
        type=float,
        default=0.2,
        help='Minimum confidence threshold (0-1, default: 0.2)'
    )
    
    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Disable FPS counter'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to custom emotion model (.h5 file)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate confidence threshold
    if not 0 <= args.confidence <= 1:
        print("[ERROR] Confidence threshold must be between 0 and 1")
        return
    
    try:
        # Initialize system
        system = EmotionDetectionSystem(
            model_path=args.model,
            confidence_threshold=args.confidence,
            show_fps=not args.no_fps
        )
        
        # Run the system
        system.run(camera_id=args.camera)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {str(e)}")
        print("\n[HELP] Required files:")
        print("  1. data/models/emotion_model.h5 - Pre-trained emotion detection model")
        print("  2. data/cascades/haarcascade_frontalface_default.xml - Face detector (optional)")
        print("\n[INFO] Please ensure these files are in place and try again.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to start application: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
