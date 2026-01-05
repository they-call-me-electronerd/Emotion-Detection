"""
Model Setup Helper Script
=========================
This script helps you download or set up the emotion detection model.
Since we cannot distribute pre-trained models directly, this script
provides guidance and automation where possible.

Author: AI Engineer
Date: January 2026
"""

import os
import sys


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_model_exists():
    """Check if the emotion model already exists."""
    model_path = os.path.join('models', 'emotion_model.h5')
    return os.path.exists(model_path)


def print_download_instructions():
    """Print instructions for downloading a pre-trained model."""
    print_header("Download Pre-trained Emotion Model")
    
    print("You have several options to obtain a pre-trained emotion detection model:\n")
    
    print("OPTION 1: Download from Kaggle")
    print("-" * 70)
    print("1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Download the FER-2013 dataset")
    print("3. Look for pre-trained models in the 'Code' section")
    print("4. Download a .h5 model file")
    print("5. Place it in the 'models/' folder as 'emotion_model.h5'\n")
    
    print("OPTION 2: Use GitHub Models")
    print("-" * 70)
    print("Search GitHub for 'fer2013 emotion model .h5'")
    print("Popular repositories:")
    print("  - https://github.com/priya-dwivedi/face_and_emotion_detection")
    print("  - https://github.com/atulapra/Emotion-detection")
    print("Download the .h5 file and place it in 'models/'\n")
    
    print("OPTION 3: Use Pre-trained from Various Sources")
    print("-" * 70)
    print("  - Google Drive shared models")
    print("  - Academic paper implementations")
    print("  - Online tutorials with downloadable models\n")
    
    print("IMPORTANT:")
    print("-" * 70)
    print("  ✓ Model must be trained on FER-2013 dataset (48x48 images)")
    print("  ✓ Model should output 7 emotion classes")
    print("  ✓ Expected input shape: (None, 48, 48, 1)")
    print("  ✓ Expected output shape: (None, 7)")
    print("  ✓ Save as: models/emotion_model.h5\n")


def print_training_instructions():
    """Print instructions for training your own model."""
    print_header("Train Your Own Emotion Model")
    
    print("If you want to train your own model:\n")
    
    print("STEP 1: Get the FER-2013 Dataset")
    print("-" * 70)
    print("Download from: https://www.kaggle.com/datasets/msambare/fer2013")
    print("The dataset contains ~35,000 grayscale 48x48 face images\n")
    
    print("STEP 2: Create Training Script")
    print("-" * 70)
    print("We've included a sample training script template.")
    print("See 'train_model.py' in the project root.\n")
    
    print("STEP 3: Train the Model")
    print("-" * 70)
    print("  python train_model.py")
    print("\nThis will:")
    print("  - Load the FER-2013 dataset")
    print("  - Build a CNN architecture")
    print("  - Train for ~50 epochs")
    print("  - Save the model as models/emotion_model.h5\n")
    
    print("STEP 4: Expected Training Time")
    print("-" * 70)
    print("  - CPU: 2-4 hours")
    print("  - GPU: 15-30 minutes")
    print("  - Expected accuracy: 60-70%\n")


def create_sample_training_script():
    """Create a sample training script for users who want to train their own model."""
    training_script = '''"""
Sample Emotion Model Training Script
====================================
This is a template for training your own emotion detection model.
Modify as needed for your specific requirements.

Requirements:
- FER-2013 dataset downloaded and extracted
- TensorFlow/Keras installed
- Sufficient RAM (8GB+) and GPU recommended
"""

import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Configuration
DATASET_PATH = 'fer2013/fer2013.csv'  # Update this path
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def load_fer2013_data(csv_path):
    """Load FER-2013 dataset from CSV file."""
    print("Loading FER-2013 dataset...")
    
    df = pd.read_csv(csv_path)
    
    # Extract pixels and emotions
    X = []
    y = []
    
    for index, row in df.iterrows():
        pixels = np.array(row['pixels'].split(), dtype='float32')
        pixels = pixels.reshape(IMG_SIZE, IMG_SIZE)
        X.append(pixels)
        y.append(row['emotion'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape and normalize
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X / 255.0
    
    # Convert labels to categorical
    y = keras.utils.to_categorical(y, NUM_CLASSES)
    
    return X, y


def build_emotion_model():
    """Build CNN model for emotion detection."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fully Connected
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model


def train_model():
    """Main training function."""
    print("="*70)
    print("  Emotion Detection Model Training")
    print("="*70)
    
    # Load data
    X, y = load_fer2013_data(DATASET_PATH)
    print(f"Dataset loaded: {X.shape[0]} images")
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(X_train)
    
    # Build model
    print("\\nBuilding model...")
    model = build_emotion_model()
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/emotion_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    print("\\nStarting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\\nEvaluating model...")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    print("\\nModel saved as: models/emotion_model.h5")


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    train_model()
'''
    
    with open('train_model.py', 'w') as f:
        f.write(training_script)
    
    print("[INFO] Created 'train_model.py' - Sample training script")


def main():
    """Main function."""
    print_header("Emotion Detection Model Setup Helper")
    
    # Check if model exists
    if check_model_exists():
        print("✓ Good news! Model file found: models/emotion_model.h5")
        print("  You're all set to run the emotion detection system!")
        print("\n  Run: cd src && python main.py")
        return
    
    print("✗ Model file not found: models/emotion_model.h5")
    print("  You need to obtain a pre-trained emotion detection model.\n")
    
    while True:
        print("What would you like to do?\n")
        print("  1. View download instructions (use existing model)")
        print("  2. View training instructions (train your own)")
        print("  3. Create sample training script")
        print("  4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print_download_instructions()
        elif choice == '2':
            print_training_instructions()
        elif choice == '3':
            create_sample_training_script()
            print_training_instructions()
        elif choice == '4':
            print("\nExiting. Good luck with your emotion detection project!")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")
    
    print("\n" + "="*70)
    print("  For more information, see README.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
