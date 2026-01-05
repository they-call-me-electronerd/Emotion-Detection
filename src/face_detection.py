"""
Face Detection Module
=====================
This module handles face detection using OpenCV's Haar Cascade classifier.
It provides functionality to load the cascade and detect faces in video frames.

Author: AI Engineer
Date: January 2026
"""

import cv2
import os
from typing import List, Tuple, Optional


class FaceDetector:
    """
    A class to handle face detection using Haar Cascade classifier.
    
    Attributes:
        cascade_path (str): Path to the Haar Cascade XML file
        face_cascade (cv2.CascadeClassifier): Loaded Haar Cascade classifier
    """
    
    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the FaceDetector with Haar Cascade classifier.
        
        Args:
            cascade_path (str, optional): Path to Haar Cascade XML file.
                                         If None, uses default OpenCV path.
        
        Raises:
            FileNotFoundError: If cascade file is not found
        """
        if cascade_path is None:
            # Try to use the cascade from the data/cascades folder
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cascade_path = os.path.join(base_dir, 'data', 'cascades', 'haarcascade_frontalface_default.xml')
            
            # If not found, try OpenCV's built-in cascade
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Haar Cascade file not found at: {cascade_path}\n"
                "Please ensure the haarcascade_frontalface_default.xml file is in the data/cascades folder."
            )
        
        self.cascade_path = cascade_path
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load Haar Cascade from: {cascade_path}")
        
        print(f"[INFO] Face detector initialized successfully")
        print(f"[INFO] Using cascade: {cascade_path}")
    
    def detect_faces(self, frame: cv2.Mat, scale_factor: float = 1.1, 
                    min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given frame.
        
        Args:
            frame (cv2.Mat): Input image/frame (can be color or grayscale)
            scale_factor (float): Parameter specifying how much the image size is reduced 
                                 at each image scale (default: 1.1)
            min_neighbors (int): Parameter specifying how many neighbors each candidate 
                               rectangle should have to retain it (default: 5)
            min_size (tuple): Minimum possible object size. Objects smaller than this are ignored.
        
        Returns:
            List[Tuple[int, int, int, int]]: List of tuples containing (x, y, w, h) 
                                            coordinates of detected faces
        
        Example:
            >>> detector = FaceDetector()
            >>> faces = detector.detect_faces(frame)
            >>> print(f"Detected {len(faces)} faces")
        """
        # Convert to grayscale if the frame is in color
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect faces in the grayscale frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert numpy array to list of tuples for easier handling
        return [tuple(face) for face in faces]
    
    def draw_faces(self, frame: cv2.Mat, faces: List[Tuple[int, int, int, int]], 
                   color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> cv2.Mat:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            frame (cv2.Mat): Input frame to draw on
            faces (List[Tuple]): List of face coordinates (x, y, w, h)
            color (Tuple[int, int, int]): BGR color for the rectangle (default: blue)
            thickness (int): Thickness of the rectangle border (default: 2)
        
        Returns:
            cv2.Mat: Frame with drawn rectangles
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        return frame
    
    def get_face_roi(self, frame: cv2.Mat, face_coords: Tuple[int, int, int, int]) -> cv2.Mat:
        """
        Extract the region of interest (ROI) for a detected face.
        
        Args:
            frame (cv2.Mat): Input frame
            face_coords (Tuple[int, int, int, int]): Face coordinates (x, y, w, h)
        
        Returns:
            cv2.Mat: Extracted face region
        """
        x, y, w, h = face_coords
        return frame[y:y+h, x:x+w]


# Convenience function for quick face detection
def detect_faces_simple(frame: cv2.Mat, cascade_path: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
    """
    Simple function to detect faces without creating a FaceDetector object.
    
    Args:
        frame (cv2.Mat): Input frame
        cascade_path (str, optional): Path to Haar Cascade file
    
    Returns:
        List[Tuple[int, int, int, int]]: List of detected face coordinates
    """
    detector = FaceDetector(cascade_path)
    return detector.detect_faces(frame)


if __name__ == "__main__":
    # Test the face detector with webcam
    print("Testing Face Detector...")
    print("Press 'q' to quit")
    
    try:
        detector = FaceDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Cannot access webcam")
            exit()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw rectangles
            detector.draw_faces(frame, faces)
            
            # Display count
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
