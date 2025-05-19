import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
import mediapipe as mp
from collections import deque
import os

print("Starting script...")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if eye_cascade.empty():
    print("Error: Could not load Haar cascade.")
    exit()

# EAR and MAR functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Hand movement analysis
def analyze_hand_movement(history, threshold=50.0, window_size=20):
    if len(history) < window_size:
        return False
    # Compute centroid velocity
    velocities = []
    for i in range(1, len(history)):
        dx = history[i][0] - history[i-1][0]
        dy = history[i][1] - history[i-1][1]
        velocity = np.sqrt(dx**2 + dy**2)
        velocities.append(velocity)
    # Check for rapid oscillations
    avg_velocity = np.mean(velocities)
    return avg_velocity > threshold

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
SEIZURE_VELOCITY_THRESHOLD = 50.0  # Pixels per frame (adjust as needed)
CONSECUTIVE_FRAMES = 20
SEIZURE_FRAMES = 30  # Longer for seizures to avoid false positives
counter_ear = 0
counter_mar = 0
counter_seizure = 0
alert_triggered_ear = False
alert_triggered_mar = False
alert_triggered_seizure = False

# Hand movement history
hand_history = deque(maxlen=30)

# Initialize pygame mixer
pygame.mixer.init()

# Play sound function
def play_alert(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"Error: Audio file not found at {file_path}")
            return
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Sound error: {e}")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(gray)

    # Hand detection
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Track centroid (e.g., wrist: landmark 0)
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
            hand_history.append((cx, cy))
            # Analyze movement
            if analyze_hand_movement(hand_history, SEIZURE_VELOCITY_THRESHOLD):
                counter_seizure += 1
                if counter_seizure >= SEIZURE_FRAMES:
                    cv2.putText(frame, "ALERT: Seizure Detected!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    if not alert_triggered_seizure:
                        print("Playing seizure alert...")
                        play_alert("E:\\real_time_driver_detection\\real_time_driver_detection\\Metal Gear Solid Alert - QuickSounds.com.mp3")
                        alert_triggered_seizure = True
            else:
                counter_seizure = 0
                alert_triggered_seizure = False
    else:
        counter_seizure = 0
        alert_triggered_seizure = False

    # Face detection
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        for (x, y) in left_eye + right_eye + mouth:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Drowsiness/Seizure
        if ear < EAR_THRESHOLD:
            counter_ear += 1
            if counter_ear >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "ALERT: Drowsiness/Seizure!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not alert_triggered_ear:
                    print("Playing drowsiness alert...")
                    play_alert("E:\\real_time_driver_detection\\real_time_driver_detection\\Metal Gear Solid Alert - QuickSounds.com.mp3")
                    alert_triggered_ear = True
        else:
            counter_ear = 0
            alert_triggered_ear = False

        # Yawning
        if mar > MAR_THRESHOLD:
            counter_mar += 1
            if counter_mar >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "ALERT: Yawning Detected!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if not alert_triggered_mar:
                    print("Playing yawning alert...")
                    play_alert("E:\\real_time_driver_detection\\real_time_driver_detection\\Metal Gear Solid Alert - QuickSounds.com.mp3")
                    alert_triggered_mar = True
        else:
            counter_mar = 0
            alert_triggered_mar = False

    # Optional Haar cascade for eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    cv2.imshow('Real-Time Driver Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()