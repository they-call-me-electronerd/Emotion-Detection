"""
Automated Setup Script for Emotion Detection System
===================================================
This script automates the setup process for the Real-Time Facial
Emotion Detection System.

It will:
1. Check Python version
2. Create/verify directory structure
3. Install dependencies
4. Download Haar Cascade if needed
5. Verify installation
6. Provide next steps

Author: AI Engineer
Date: January 2026
"""

import os
import sys
import subprocess
import platform


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-"*70)


def check_python_version():
    """Check if Python version is compatible."""
    print_step(1, 6, "Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8+ is required")
        print("   Please upgrade Python and try again")
        return False
    
    print("✓ Python version is compatible")
    return True


def verify_directory_structure():
    """Verify and create necessary directories."""
    print_step(2, 6, "Verifying Directory Structure")
    
    directories = ['models', 'cascades', 'src']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}/")
        else:
            print(f"✓ Directory exists: {directory}/")
    
    return True


def install_dependencies():
    """Install required Python packages."""
    print_step(3, 6, "Installing Dependencies")
    
    print("Installing packages from requirements.txt...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "requirements.txt",
            "--upgrade"
        ])
        print("\n✓ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ ERROR: Failed to install dependencies")
        print("   Try running manually: pip install -r requirements.txt")
        return False


def check_opencv_cascade():
    """Check if Haar Cascade is available."""
    print_step(4, 6, "Checking Haar Cascade Availability")
    
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if os.path.exists(cascade_path):
            print(f"✓ OpenCV built-in cascade found")
            print(f"  Location: {cascade_path}")
            return True
        else:
            print("⚠ OpenCV cascade not found (unusual)")
            print("  The system will try to use the cascades folder")
            return True
    except ImportError:
        print("❌ OpenCV not properly installed")
        return False


def check_tensorflow():
    """Verify TensorFlow installation."""
    print_step(5, 6, "Verifying TensorFlow Installation")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU support available: {len(gpus)} GPU(s) detected")
        else:
            print("ℹ GPU not detected - will use CPU (slower but functional)")
        
        return True
    except ImportError:
        print("❌ TensorFlow not properly installed")
        return False


def check_model_file():
    """Check if emotion model file exists."""
    print_step(6, 6, "Checking Emotion Model")
    
    model_path = os.path.join('models', 'emotion_model.h5')
    
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
        
        # Check file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        
        return True
    else:
        print(f"⚠ Model file not found: {model_path}")
        print("\n  You need to obtain a pre-trained emotion detection model.")
        print("  Run: python setup_model.py")
        print("  Or see QUICKSTART.md for instructions")
        return False


def print_summary(results):
    """Print setup summary."""
    print_header("Setup Summary")
    
    all_passed = all(results.values())
    
    for step, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {step}")
    
    print()
    
    if all_passed:
        print("🎉 SUCCESS! Setup completed successfully!")
        print("\nYou're ready to run the emotion detection system!")
    else:
        print("⚠ INCOMPLETE: Some setup steps failed or need attention")
        print("\nPlease resolve the issues above before running the system")


def print_next_steps(model_exists):
    """Print next steps for the user."""
    print_header("Next Steps")
    
    if not model_exists:
        print("📥 STEP 1: Get the Emotion Model")
        print("   Option A: Run the setup helper")
        print("      python setup_model.py")
        print()
        print("   Option B: Download manually")
        print("      See QUICKSTART.md for download links")
        print()
    
    print("🚀 STEP 2: Run the Application")
    print("   cd src")
    print("   python main.py")
    print()
    print("📖 STEP 3: Read the Documentation")
    print("   See README.md for full documentation")
    print("   See QUICKSTART.md for quick reference")
    print()


def test_imports():
    """Test if all required modules can be imported."""
    print_header("Testing Module Imports")
    
    modules = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow'
    }
    
    all_imported = True
    
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError:
            print(f"❌ Failed to import {name}")
            all_imported = False
    
    return all_imported


def main():
    """Main setup function."""
    print_header("Emotion Detection System - Automated Setup")
    
    print("This script will set up your emotion detection environment.")
    print("Please wait while we check and install requirements...\n")
    
    # Track results
    results = {}
    
    # Run setup steps
    results['Python Version'] = check_python_version()
    
    if not results['Python Version']:
        print("\n[CRITICAL] Cannot proceed without compatible Python version")
        return
    
    results['Directory Structure'] = verify_directory_structure()
    results['Dependencies'] = install_dependencies()
    
    # Only proceed with checks if dependencies were installed
    if results['Dependencies']:
        results['Module Imports'] = test_imports()
        results['Haar Cascade'] = check_opencv_cascade()
        results['TensorFlow'] = check_tensorflow()
    else:
        results['Module Imports'] = False
        results['Haar Cascade'] = False
        results['TensorFlow'] = False
    
    model_exists = check_model_file()
    results['Emotion Model'] = model_exists
    
    # Print summary
    print_summary(results)
    
    # Print next steps
    print_next_steps(model_exists)
    
    print_header("Setup Complete")
    
    print("For help:")
    print("  - See README.md for full documentation")
    print("  - See QUICKSTART.md for quick start guide")
    print("  - Run 'python setup_model.py' for model setup help")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Setup interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
