"""
Emotion Model Module
====================
This module handles loading the pre-trained CNN emotion detection model
and provides functionality to predict emotions from preprocessed face images.

Author: AI Engineer
Date: January 2026
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
except ImportError:
    print("[ERROR] TensorFlow not installed. Please run: pip install tensorflow")
    raise


class EmotionDetector:
    """
    A class to handle emotion detection using a pre-trained CNN model.
    
    The model expects preprocessed face images (48x48 grayscale) and
    predicts one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
    
    Attributes:
        model_path (str): Path to the pre-trained model (.h5 file)
        model (keras.Model): Loaded Keras model
        emotion_labels (List[str]): List of emotion class labels
        emotion_colors (Dict[str, Tuple]): BGR colors for each emotion
    """
    
    # Define emotion labels (standard FER-2013 dataset order)
    EMOTION_LABELS = [
        'Angry',      # 0
        'Disgust',    # 1
        'Fear',       # 2
        'Happy',      # 3
        'Sad',        # 4
        'Surprise',   # 5
        'Neutral'     # 6
    ]
    
    # Define colors for each emotion (BGR format for OpenCV)
    EMOTION_COLORS = {
        'Angry': (0, 0, 255),        # Red
        'Disgust': (0, 255, 0),      # Green
        'Fear': (128, 0, 128),       # Purple
        'Happy': (0, 255, 255),      # Yellow
        'Sad': (255, 0, 0),          # Blue
        'Surprise': (255, 165, 0),   # Orange
        'Neutral': (128, 128, 128)   # Gray
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the EmotionDetector with a pre-trained model.
        
        Args:
            model_path (str, optional): Path to the .h5 model file.
                                       If None, looks in the models folder.
        
        Raises:
            FileNotFoundError: If model file is not found
            ValueError: If model fails to load
        """
        if model_path is None:
            # Default path: models/emotion_model.h5
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'emotion_model.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                "Please ensure the emotion_model.h5 file is in the models folder.\n"
                "You can download a pre-trained model or train your own using the FER-2013 dataset."
            )
        
        self.model_path = model_path
        self.emotion_labels = self.EMOTION_LABELS
        self.emotion_colors = self.EMOTION_COLORS
        
        # Load the model
        print(f"[INFO] Loading emotion detection model...")
        print(f"[INFO] Model path: {model_path}")
        
        try:
            self.model = load_model(model_path, compile=False)
            print(f"[SUCCESS] Model loaded successfully")
            
            # Display model information
            self._display_model_info()
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def _display_model_info(self):
        """Display information about the loaded model."""
        try:
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
            
            print(f"[INFO] Model input shape: {input_shape}")
            print(f"[INFO] Model output shape: {output_shape}")
            print(f"[INFO] Number of emotion classes: {len(self.emotion_labels)}")
            print(f"[INFO] Emotions: {', '.join(self.emotion_labels)}")
        except Exception as e:
            print(f"[WARNING] Could not display model info: {str(e)}")
    
    def predict_emotion(self, preprocessed_face: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from a preprocessed face image.
        
        Args:
            preprocessed_face (np.ndarray): Preprocessed face image
                                           Shape: (1, 48, 48, 1) or (48, 48, 1)
        
        Returns:
            Tuple[str, float, np.ndarray]: 
                - Predicted emotion label
                - Confidence score (0-1)
                - Full prediction array with probabilities for all emotions
        
        Raises:
            ValueError: If input shape is invalid
        """
        # Validate input shape
        if preprocessed_face.ndim == 3:
            # Add batch dimension if missing
            preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
        
        if preprocessed_face.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {preprocessed_face.shape[0]}")
        
        # Make prediction
        try:
            predictions = self.model.predict(preprocessed_face, verbose=0)
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        emotion_label = self.emotion_labels[predicted_class]
        
        return emotion_label, confidence, predictions[0]
    
    def predict_top_emotions(self, preprocessed_face: np.ndarray, 
                            top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top K predicted emotions with their confidence scores.
        
        Args:
            preprocessed_face (np.ndarray): Preprocessed face image
            top_k (int): Number of top predictions to return (default: 3)
        
        Returns:
            List[Tuple[str, float]]: List of (emotion, confidence) tuples
        """
        emotion, confidence, predictions = self.predict_emotion(preprocessed_face)
        
        # Get top K predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_emotions = [(self.emotion_labels[idx], float(predictions[idx])) 
                       for idx in top_indices]
        
        return top_emotions
    
    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """
        Get the BGR color associated with an emotion.
        
        Args:
            emotion (str): Emotion label
        
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        return self.emotion_colors.get(emotion, (255, 255, 255))  # Default: white
    
    def predict_batch(self, preprocessed_faces: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predict emotions for a batch of faces.
        
        Args:
            preprocessed_faces (np.ndarray): Batch of preprocessed faces
                                            Shape: (batch_size, 48, 48, 1)
        
        Returns:
            List[Tuple[str, float]]: List of (emotion, confidence) tuples for each face
        """
        if preprocessed_faces.size == 0:
            return []
        
        predictions = self.model.predict(preprocessed_faces, verbose=0)
        
        results = []
        for pred in predictions:
            predicted_class = np.argmax(pred)
            confidence = float(pred[predicted_class])
            emotion_label = self.emotion_labels[predicted_class]
            results.append((emotion_label, confidence))
        
        return results
    
    def get_emotion_distribution(self, preprocessed_face: np.ndarray) -> Dict[str, float]:
        """
        Get the probability distribution across all emotions.
        
        Args:
            preprocessed_face (np.ndarray): Preprocessed face image
        
        Returns:
            Dict[str, float]: Dictionary mapping emotions to their probabilities
        """
        _, _, predictions = self.predict_emotion(preprocessed_face)
        
        emotion_dist = {
            emotion: float(predictions[idx]) 
            for idx, emotion in enumerate(self.emotion_labels)
        }
        
        return emotion_dist


# Convenience function for quick emotion prediction
def predict_emotion_simple(preprocessed_face: np.ndarray, 
                          model_path: Optional[str] = None) -> Tuple[str, float]:
    """
    Simple function to predict emotion without creating an EmotionDetector object.
    
    Args:
        preprocessed_face (np.ndarray): Preprocessed face image
        model_path (str, optional): Path to model file
    
    Returns:
        Tuple[str, float]: (emotion_label, confidence)
    """
    detector = EmotionDetector(model_path)
    emotion, confidence, _ = detector.predict_emotion(preprocessed_face)
    return emotion, confidence


if __name__ == "__main__":
    # Test the emotion detector
    print("Testing Emotion Detector...\n")
    
    try:
        # Initialize detector
        detector = EmotionDetector()
        
        # Create a dummy preprocessed face (simulating preprocessed input)
        dummy_face = np.random.rand(1, 48, 48, 1).astype('float32')
        
        print("\n[TEST] Predicting emotion for dummy face...")
        emotion, confidence, predictions = detector.predict_emotion(dummy_face)
        
        print(f"[SUCCESS] Prediction completed")
        print(f"[RESULT] Predicted emotion: {emotion}")
        print(f"[RESULT] Confidence: {confidence:.2%}")
        
        # Test top emotions
        print("\n[TEST] Getting top 3 emotions...")
        top_emotions = detector.predict_top_emotions(dummy_face, top_k=3)
        for i, (emo, conf) in enumerate(top_emotions, 1):
            print(f"  {i}. {emo}: {conf:.2%}")
        
        # Test emotion distribution
        print("\n[TEST] Getting emotion distribution...")
        distribution = detector.get_emotion_distribution(dummy_face)
        for emotion, prob in distribution.items():
            bar = '█' * int(prob * 50)
            print(f"  {emotion:10s} {prob:.2%} {bar}")
        
        print("\n[SUCCESS] All tests passed!")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {str(e)}")
        print("\n[INFO] To use this emotion detector, you need to:")
        print("  1. Train a CNN model on the FER-2013 dataset, or")
        print("  2. Download a pre-trained model")
        print("  3. Place the model file as: models/emotion_model.h5")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
