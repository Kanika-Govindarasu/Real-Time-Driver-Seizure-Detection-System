Real-Time Driver Seizure Detection System

This project is a computer vision-based driver monitoring system that detects signs of seizures, drowsiness, and yawning in real-time using a webcam. It uses Dlib, OpenCV, Mediapipe, and Pygame to analyze facial and hand landmarks, and provides alerts through visual messages and audio warnings.

Features
--------
- Drowsiness detection using Eye Aspect Ratio (EAR)
- Yawning detection using Mouth Aspect Ratio (MAR)
- Seizure-like hand movement detection based on rapid hand oscillations
- Real-time facial and hand landmark tracking
- Audio alerts for drowsiness, yawning, and seizure events

Requirements
------------
Install the following Python packages before running the project:

    pip install opencv-python dlib scipy pygame mediapipe numpy

Also, download the Dlib facial landmarks model file:
- shape_predictor_68_face_landmarks.dat  
  Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Project Structure
-----------------
project/
│
├── shape_predictor_68_face_landmarks.dat   # Dlib facial landmarks model
├── alert_audio.mp3                         # Custom alert audio (change path as needed)
└── driver_detection.py                     # Main script

How it Works
------------
1. Facial Detection:
   - Detects the driver's face and extracts 68 facial landmarks.
   - Calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
   - Triggers alerts for drowsiness or yawning based on thresholds.

2. Hand Movement Detection:
   - Tracks hand landmarks using Mediapipe.
   - Calculates centroid movement to detect seizure-like rapid oscillations.
   - Triggers seizure alert if movement exceeds a threshold for a fixed duration.

3. Audio Alerts:
   - Uses Pygame to play a sound whenever drowsiness, yawning, or seizure is detected.

Usage
-----
1. Make sure your webcam is connected.
2. Ensure shape_predictor_68_face_landmarks.dat is in the project directory.
3. Update the path to your alert audio file in the script.
4. Run the script:

    python driver_detection.py

5. Press 'q' to quit the application.

Notes
-----
- Tune the thresholds (EAR, MAR, velocity, etc.) based on testing and accuracy needs.
- Ensure proper lighting conditions for reliable detection.
- Audio file must be in a supported format (e.g., .mp3, .wav) and the path must be correct.

Disclaimer
----------
This is a prototype system for educational and research purposes. It is not certified for real-world driver safety applications. Use it responsibly.


Created by:
Kanika G - LinkedIn: https://www.linkedin.com/in/kanika-g
Megha Bharathi B - LinkedIn: https://www.linkedin.com/in/meghabharathib
