"""
Emotion Prediction Preprocessing Module
========================================
This module handles preprocessing of detected face images for emotion prediction.
It includes grayscale conversion, resizing, normalization, and reshaping operations.

Author: AI Engineer
Date: January 2026
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class EmotionPreprocessor:
    """
    A class to handle preprocessing of face images for emotion detection.
    
    The preprocessing pipeline includes:
    1. Grayscale conversion (if needed)
    2. Resizing to target dimensions (48x48 for most emotion models)
    3. Normalization (pixel values to 0-1 range)
    4. Reshaping for CNN input (adding batch and channel dimensions)
    
    Attributes:
        target_size (Tuple[int, int]): Target dimensions for face images
        normalize (bool): Whether to normalize pixel values
    """
    
    def __init__(self, target_size: Tuple[int, int] = (48, 48), normalize: bool = True):
        """
        Initialize the EmotionPreprocessor.
        
        Args:
            target_size (Tuple[int, int]): Target size for face images (default: 48x48)
            normalize (bool): Whether to normalize pixel values to 0-1 range (default: True)
        """
        self.target_size = target_size
        self.normalize = normalize
        
        print(f"[INFO] Emotion preprocessor initialized")
        print(f"[INFO] Target size: {target_size}")
        print(f"[INFO] Normalization: {normalize}")
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single face image.
        
        Args:
            face_image (np.ndarray): Input face image (can be grayscale or color)
        
        Returns:
            np.ndarray: Preprocessed face image ready for model input
                       Shape: (1, height, width, 1) for grayscale CNN models
        
        Raises:
            ValueError: If input image is invalid
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid input image: image is None or empty")
        
        # Step 1: Convert to grayscale if needed
        gray_face = self._to_grayscale(face_image)
        
        # Step 2: Resize to target dimensions
        resized_face = self._resize(gray_face)
        
        # Step 3: Normalize pixel values
        if self.normalize:
            normalized_face = self._normalize(resized_face)
        else:
            normalized_face = resized_face
        
        # Step 4: Reshape for CNN input
        reshaped_face = self._reshape_for_cnn(normalized_face)
        
        return reshaped_face
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale if it's in color.
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Grayscale image
        """
        if len(image.shape) == 3:
            # Color image (BGR or RGB)
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        
        # Already grayscale
        return image
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Resized image
        """
        if image.shape[:2] != self.target_size:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        return image
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to 0-1 range.
        
        Args:
            image (np.ndarray): Input image with pixel values 0-255
        
        Returns:
            np.ndarray: Normalized image with pixel values 0-1
        """
        return image.astype('float32') / 255.0
    
    def _reshape_for_cnn(self, image: np.ndarray) -> np.ndarray:
        """
        Reshape image for CNN model input.
        Adds batch dimension and channel dimension.
        
        Args:
            image (np.ndarray): Input image (height, width)
        
        Returns:
            np.ndarray: Reshaped image (1, height, width, 1)
        """
        # Expand dimensions: (height, width) -> (1, height, width, 1)
        return np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
    
    def preprocess_batch(self, face_images: list) -> np.ndarray:
        """
        Preprocess a batch of face images.
        
        Args:
            face_images (list): List of face images
        
        Returns:
            np.ndarray: Batch of preprocessed images (batch_size, height, width, 1)
        """
        preprocessed_faces = []
        
        for face_image in face_images:
            try:
                preprocessed = self.preprocess_face(face_image)
                preprocessed_faces.append(preprocessed[0])  # Remove batch dimension
            except Exception as e:
                print(f"[WARNING] Failed to preprocess face: {str(e)}")
                continue
        
        if not preprocessed_faces:
            return np.array([])
        
        return np.array(preprocessed_faces)


# Convenience functions for quick preprocessing
def preprocess_face_simple(face_image: np.ndarray, 
                          target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """
    Simple function to preprocess a face without creating a preprocessor object.
    
    Args:
        face_image (np.ndarray): Input face image
        target_size (Tuple[int, int]): Target size (default: 48x48)
    
    Returns:
        np.ndarray: Preprocessed face image
    """
    preprocessor = EmotionPreprocessor(target_size=target_size)
    return preprocessor.preprocess_face(face_image)


def preprocess_for_display(face_image: np.ndarray, 
                           target_size: Tuple[int, int] = (96, 96)) -> np.ndarray:
    """
    Preprocess face for display purposes (without normalization).
    
    Args:
        face_image (np.ndarray): Input face image
        target_size (Tuple[int, int]): Target size for display
    
    Returns:
        np.ndarray: Resized grayscale face image
    """
    preprocessor = EmotionPreprocessor(target_size=target_size, normalize=False)
    
    # Convert to grayscale
    gray = preprocessor._to_grayscale(face_image)
    
    # Resize
    resized = preprocessor._resize(gray)
    
    return resized


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing Emotion Preprocessor...")
    
    try:
        # Create a dummy face image
        dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Initialize preprocessor
        preprocessor = EmotionPreprocessor()
        
        # Preprocess the face
        preprocessed = preprocessor.preprocess_face(dummy_face)
        
        print(f"[SUCCESS] Preprocessing test passed")
        print(f"[INFO] Input shape: {dummy_face.shape}")
        print(f"[INFO] Output shape: {preprocessed.shape}")
        print(f"[INFO] Output dtype: {preprocessed.dtype}")
        print(f"[INFO] Output range: [{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
        
        # Test with actual webcam (optional)
        test_webcam = input("\nTest with webcam? (y/n): ").lower() == 'y'
        
        if test_webcam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("[ERROR] Cannot access webcam")
                exit()
            
            print("Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Take a small region from center
                h, w = frame.shape[:2]
                face_roi = frame[h//4:3*h//4, w//4:3*w//4]
                
                # Preprocess
                preprocessed = preprocessor.preprocess_face(face_roi)
                
                # Convert back for display
                display_face = (preprocessed[0, :, :, 0] * 255).astype(np.uint8)
                display_face = cv2.resize(display_face, (200, 200))
                
                # Show original and preprocessed
                cv2.imshow('Original ROI', face_roi)
                cv2.imshow('Preprocessed (48x48)', display_face)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
