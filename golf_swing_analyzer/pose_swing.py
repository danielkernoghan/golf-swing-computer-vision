import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# Load video
video_path = "videos/my_swing.mp4"  # Replace with your video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]

        # Calculate angles
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(hip, shoulder, elbow)

        # Display angles on screen
        cv2.putText(frame, f'Elbow: {int(elbow_angle)}', (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f'Shoulder: {int(shoulder_angle)}', (50,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Example simple feedback
        if elbow_angle < 90:
            cv2.putText(frame, 'Elbow too bent', (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if shoulder_angle < 70:
            cv2.putText(frame, 'Rotate shoulders more', (50,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow('Golf Swing Analyzer', frame)

    if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
