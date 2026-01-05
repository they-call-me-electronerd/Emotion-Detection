# Quick Start Guide - Emotion Detection System

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (1 minute)

```bash
# Open PowerShell/Command Prompt in project directory
cd "c:\Users\saksh\Downloads\Emotion Detection"

# Install required packages
pip install -r requirements.txt
```

### Step 2: Download Haar Cascade (30 seconds)

The Haar Cascade file is optional - the system will use OpenCV's built-in cascade if not found.

To manually download:
```bash
# The file is already included with OpenCV installation
# No action needed!
```

### Step 3: Get Emotion Model (3 minutes)

**Option A: Run the setup helper**
```bash
python setup_model.py
```

**Option B: Manual download**
1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Download a pre-trained FER-2013 emotion model (.h5 file)
3. Place it in: `models/emotion_model.h5`

**Option C: Quick download (if you have a direct link)**
```bash
# Example - replace with actual URL
# curl -o models/emotion_model.h5 "YOUR_MODEL_URL"
```

### Step 4: Run the Application (30 seconds)

```bash
cd src
python main.py
```

**That's it!** Your webcam should open with real-time emotion detection.

---

## 📋 Pre-flight Checklist

Before running, make sure you have:

- ✅ Python 3.8+ installed
- ✅ Webcam connected and working
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Model file at `models/emotion_model.h5`

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **S** | Save screenshot |

---

## ⚡ Quick Commands

```bash
# Run with default settings
cd src
python main.py

# Use different camera
python main.py --camera 1

# Adjust confidence threshold
python main.py --confidence 0.7

# Hide FPS counter
python main.py --no-fps
```

---

## 🔧 Troubleshooting

### Camera not opening?
```bash
# Try different camera ID
python main.py --camera 1
```

### Model not found?
```bash
# Run setup helper
python setup_model.py
```

### Import errors?
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 📖 Full Documentation

For complete documentation, see [README.md](README.md)

---

## 🆘 Need Help?

1. Check [README.md](README.md) for detailed documentation
2. Run `python setup_model.py` for model setup help
3. See the troubleshooting section in README.md

---

**Happy Emotion Detecting! 😊**
