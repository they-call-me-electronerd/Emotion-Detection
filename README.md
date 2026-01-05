# Real-Time Facial Emotion Detection System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.8%2B-green)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13%2B-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A production-ready, real-time facial emotion detection system built with Python, OpenCV, and Deep Learning. This system detects human faces from a webcam feed and classifies their emotions into 7 categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Dataset Information](#-dataset-information)
- [Output Preview](#-output-preview)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- ✅ **Real-time face detection** using OpenCV Haar Cascade
- ✅ **7 emotion classifications**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- ✅ **Live webcam integration** with smooth video processing
- ✅ **Confidence scoring** for each emotion prediction
- ✅ **Color-coded emotions** for easy visual identification
- ✅ **FPS counter** for performance monitoring
- ✅ **Multi-face detection** - detects and analyzes multiple faces simultaneously
- ✅ **Modular architecture** - clean, maintainable, and extensible code
- ✅ **Error handling** - robust error handling for production use
- ✅ **Screenshot capability** - save frames with detected emotions
- ✅ **Cross-platform** - works on Windows, Linux, and macOS

---

## 🏗️ System Architecture

The system consists of four main components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Webcam Input (640x480)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Face Detection (Haar Cascade)                  │
│  • Converts frame to grayscale                              │
│  • Detects faces using cascade classifier                   │
│  • Returns bounding box coordinates (x, y, w, h)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Face Preprocessing Pipeline                    │
│  1. Grayscale conversion                                    │
│  2. Resize to 48x48 pixels                                  │
│  3. Normalize pixel values (0-1 range)                      │
│  4. Reshape for CNN input (1, 48, 48, 1)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Emotion Prediction (CNN Model)                     │
│  • Loads pre-trained emotion_model.h5                       │
│  • Predicts emotion class (7 categories)                    │
│  • Returns emotion label and confidence score               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Display & Annotation                           │
│  • Draws bounding boxes (color-coded by emotion)            │
│  • Displays emotion label and confidence                    │
│  • Shows FPS and face count                                 │
│  • Outputs annotated video frame                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **OpenCV** | 4.8+ | Video capture, face detection, image processing |
| **TensorFlow/Keras** | 2.13+ | Deep learning framework for CNN model |
| **NumPy** | 1.24+ | Numerical computations and array operations |
| **Haar Cascade** | - | Pre-trained face detection classifier |

---

## 📁 Project Structure

```
emotion-detection/
│
├── models/
│   └── emotion_model.h5              # Pre-trained CNN emotion model (required)
│
├── cascades/
│   └── haarcascade_frontalface_default.xml  # Haar Cascade (optional, uses OpenCV's built-in)
│
├── src/
│   ├── face_detection.py             # Face detection module
│   ├── emotion_prediction.py         # Face preprocessing module
│   ├── emotion_model.py              # Emotion prediction module
│   └── main.py                       # Main application entry point
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

### Module Descriptions

#### 1. **face_detection.py**
- Implements `FaceDetector` class
- Loads Haar Cascade classifier
- Detects faces in video frames
- Returns bounding box coordinates
- Provides utility functions for drawing rectangles

#### 2. **emotion_prediction.py**
- Implements `EmotionPreprocessor` class
- Handles face preprocessing pipeline:
  - Grayscale conversion
  - Resizing to 48x48 pixels
  - Pixel normalization (0-1 range)
  - Reshaping for CNN input

#### 3. **emotion_model.py**
- Implements `EmotionDetector` class
- Loads pre-trained CNN model (.h5 file)
- Defines 7 emotion labels
- Predicts emotion from preprocessed face
- Returns emotion label, confidence, and full probability distribution

#### 4. **main.py**
- Main application orchestrator
- Integrates all modules
- Handles webcam video capture
- Processes frames in real-time
- Displays annotated output
- Handles user input and cleanup

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (built-in or external)
- Pre-trained emotion detection model (emotion_model.h5)

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd emotion-detection

# Or simply navigate to your project directory
cd "c:\Users\saksh\Downloads\Emotion Detection"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Obtain Pre-trained Model

You need a pre-trained emotion detection model. You have two options:

#### Option A: Download Pre-trained Model
- Download a pre-trained FER-2013 emotion model from:
  - [Kaggle FER-2013 Models](https://www.kaggle.com/datasets/msambare/fer2013)
  - [GitHub Pre-trained Models](https://github.com/search?q=fer2013+model)
- Place the `.h5` file in the `models/` directory as `emotion_model.h5`

#### Option B: Train Your Own Model
- Download the FER-2013 dataset
- Train a CNN model using TensorFlow/Keras
- Save the trained model as `models/emotion_model.h5`

### Step 5: Verify Installation

```bash
# Test individual modules
cd src
python face_detection.py       # Test face detection
python emotion_prediction.py   # Test preprocessing
python emotion_model.py        # Test emotion model (requires model file)
```

---

## 🚀 Usage

### Basic Usage

Run the application with default settings:

```bash
cd src
python main.py
```

This will:
- Open the default webcam (camera ID: 0)
- Use confidence threshold of 0.5
- Display FPS counter

### Advanced Usage

```bash
# Use a different camera
python main.py --camera 1

# Set custom confidence threshold
python main.py --confidence 0.7

# Disable FPS counter
python main.py --no-fps

# Use custom model path
python main.py --model "../models/custom_model.h5"

# Combine multiple options
python main.py --camera 1 --confidence 0.6 --no-fps
```

### Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--camera` | `-c` | Camera device ID (0, 1, 2, etc.) | 0 |
| `--confidence` | `-conf` | Minimum confidence threshold (0-1) | 0.5 |
| `--no-fps` | - | Disable FPS counter | False |
| `--model` | `-m` | Path to custom emotion model | models/emotion_model.h5 |

### Keyboard Controls

While the application is running:

| Key | Action |
|-----|--------|
| **Q** | Quit the application |
| **S** | Save current frame as image |

---

## 🔍 How It Works

### 1. **Video Capture**
The system captures real-time video from your webcam at 640x480 resolution.

### 2. **Face Detection**
Each frame is converted to grayscale and processed by the Haar Cascade classifier to detect faces. The classifier returns bounding box coordinates (x, y, width, height) for each detected face.

### 3. **Face Preprocessing**
For each detected face:
- Extract the face region (ROI)
- Convert to grayscale (if not already)
- Resize to 48x48 pixels (standard input size for FER-2013 models)
- Normalize pixel values to 0-1 range
- Reshape to (1, 48, 48, 1) for CNN input

### 4. **Emotion Prediction**
The preprocessed face is fed into the CNN model, which outputs probabilities for each of the 7 emotion classes. The class with the highest probability is selected as the predicted emotion.

### 5. **Visualization**
The system draws:
- Color-coded bounding boxes around each face
- Emotion label with confidence score
- Confidence bar below each face
- System statistics (face count, FPS)

### Emotion Color Coding

| Emotion | Color | BGR Value |
|---------|-------|-----------|
| Angry | Red | (0, 0, 255) |
| Disgust | Green | (0, 255, 0) |
| Fear | Purple | (128, 0, 128) |
| Happy | Yellow | (0, 255, 255) |
| Sad | Blue | (255, 0, 0) |
| Surprise | Orange | (255, 165, 0) |
| Neutral | Gray | (128, 128, 128) |

---

## 📊 Dataset Information

### FER-2013 Dataset

The emotion detection model is typically trained on the **FER-2013 (Facial Expression Recognition 2013)** dataset:

- **Size**: ~35,000 grayscale face images
- **Resolution**: 48x48 pixels
- **Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Split**: Training, Validation, and Test sets
- **Source**: [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

### Model Architecture (Typical)

A standard CNN architecture for emotion detection:

```
Input (48x48x1)
    ↓
Conv2D (32 filters, 3x3) → ReLU → BatchNorm
    ↓
Conv2D (64 filters, 3x3) → ReLU → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (128 filters, 3x3) → ReLU → BatchNorm
    ↓
Conv2D (128 filters, 3x3) → ReLU → BatchNorm → MaxPool → Dropout
    ↓
Flatten
    ↓
Dense (256) → ReLU → Dropout
    ↓
Dense (128) → ReLU → Dropout
    ↓
Dense (7) → Softmax
    ↓
Output (7 emotion probabilities)
```

---

## 📸 Output Preview

### Console Output
```
============================================================
Real-Time Facial Emotion Detection System
============================================================

[1/3] Initializing face detector...
[INFO] Face detector initialized successfully
[INFO] Using cascade: cascades/haarcascade_frontalface_default.xml

[2/3] Initializing emotion preprocessor...
[INFO] Emotion preprocessor initialized
[INFO] Target size: (48, 48)
[INFO] Normalization: True

[3/3] Initializing emotion detector...
[INFO] Loading emotion detection model...
[INFO] Model path: models/emotion_model.h5
[SUCCESS] Model loaded successfully
[INFO] Model input shape: (None, 48, 48, 1)
[INFO] Model output shape: (None, 7)
[INFO] Number of emotion classes: 7
[INFO] Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

============================================================
[SUCCESS] System initialized successfully!
============================================================

[INFO] Starting camera (ID: 0)...
[SUCCESS] Camera started successfully

============================================================
CONTROLS:
  Q - Quit application
  S - Save current frame
============================================================
```

### Video Output Window

The application displays a video window with:
- **Green bounding boxes** around detected faces
- **Emotion labels** with confidence percentages
- **Confidence bars** showing prediction strength
- **Statistics panel** (top-left):
  - Number of faces detected
  - Current FPS
  - Exit instructions

**Example Annotations:**
```
[Happy: 87.3%]  ← Yellow box around smiling face
[Sad: 72.1%]    ← Blue box around sad face
[Neutral: 65.4%] ← Gray box around neutral face

Statistics Panel:
Faces Detected: 3
FPS: 28.4
Press 'Q' to quit
```

---

## 🔧 Troubleshooting

### Issue 1: Camera Not Opening

**Error**: `Cannot access camera 0`

**Solutions**:
- Check if camera is connected and enabled
- Try a different camera ID: `python main.py --camera 1`
- Close other applications using the camera
- Check camera permissions in system settings

### Issue 2: Model File Not Found

**Error**: `Model file not found at: models/emotion_model.h5`

**Solutions**:
- Ensure you have downloaded/trained a model
- Place the `.h5` file in the `models/` directory
- Check file name is exactly `emotion_model.h5`
- Use `--model` flag to specify custom path

### Issue 3: Import Errors

**Error**: `ModuleNotFoundError: No module named 'cv2'`

**Solutions**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install opencv-python tensorflow numpy
```

### Issue 4: TensorFlow Warnings

**Issue**: Many TensorFlow warnings appearing

**Solution**: Warnings are suppressed in the code, but you can also:
```bash
# Set environment variable before running
export TF_CPP_MIN_LOG_LEVEL=2  # Linux/Mac
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
```

### Issue 5: Low FPS Performance

**Symptoms**: FPS < 10

**Solutions**:
- Reduce video resolution in `main.py`:
  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  ```
- Use a lighter model
- Close other applications
- Consider using GPU acceleration (tensorflow-gpu)

### Issue 6: No Faces Detected

**Symptoms**: "Faces Detected: 0" always shown

**Solutions**:
- Ensure good lighting
- Face the camera directly
- Adjust detection parameters in `face_detection.py`:
  ```python
  faces = detector.detect_faces(frame, scale_factor=1.05, min_neighbors=3)
  ```

---

## 🚀 Future Enhancements

### Short-term Improvements
- [ ] Add emotion history tracking and visualization
- [ ] Implement emotion statistics and analytics
- [ ] Add support for video file input (not just webcam)
- [ ] Create GUI with tkinter/PyQt for easier interaction
- [ ] Add audio feedback for detected emotions
- [ ] Export results to CSV/JSON format

### Medium-term Enhancements
- [ ] Implement age and gender detection
- [ ] Add face recognition with emotion tracking per person
- [ ] Create a web-based interface using Flask/FastAPI
- [ ] Implement real-time emotion heatmaps
- [ ] Add multi-threading for better performance
- [ ] Support for batch processing of images

### Long-term Goals
- [ ] Upgrade to more advanced models (ResNet, EfficientNet)
- [ ] Implement transfer learning for domain-specific emotions
- [ ] Add support for micro-expressions detection
- [ ] Deploy as a cloud service with API
- [ ] Mobile app integration (iOS/Android)
- [ ] Real-time emotion analytics dashboard

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation as needed

---

## 📝 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2026 AI Engineer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📚 References

- **FER-2013 Dataset**: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **TensorFlow Documentation**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **Haar Cascade Paper**: Viola, P., & Jones, M. (2001). "Rapid object detection using a boosted cascade of simple features"

---

## 👨‍💻 Author

**Sakshyam Bastakoti**  
*Expert in Computer Vision and Deep Learning*

---

## 🙏 Acknowledgments

- Thanks to the creators of the FER-2013 dataset
- OpenCV community for excellent documentation
- TensorFlow team for the powerful deep learning framework
- All contributors who help improve this project

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [Issues](https://github.com/your-repo/issues) page
3. Open a new issue with detailed information

---

## ⭐ Star This Project

If you find this project useful, please consider giving it a star! It helps others discover the project and motivates continued development.

---

**Happy Emotion Detecting! 😊😢😠😲😨😐🤢**
