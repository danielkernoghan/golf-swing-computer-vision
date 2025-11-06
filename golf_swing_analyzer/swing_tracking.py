import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)
alpha = 0.5
angle_history = {}
window_size = 15
shoulder_angles = deque(maxlen=window_size)
swing_phase = 'Setup'
frame_count = 0

good_angles = {
    'left_elbow': (90, 160),
    'right_elbow': (90, 160),
    'left_shoulder': (60, 120),
    'right_shoulder': (60, 120),
    'left_hip': (70, 130),
    'right_hip': (70, 130)
}

while cap.isOpened():
    ret, frame = cap.read()
  
    frame_count += 1
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    joint_angles = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        coords = {}
        for lm_name in ['LEFT_SHOULDER', 'RIGHT_SHOULDER',
                        'LEFT_ELBOW', 'RIGHT_ELBOW',
                        'LEFT_WRIST', 'RIGHT_WRIST',
                        'LEFT_HIP', 'RIGHT_HIP',
                        'LEFT_KNEE', 'RIGHT_KNEE',
                        'LEFT_ANKLE', 'RIGHT_ANKLE']:
            lm = getattr(mp_pose.PoseLandmark, lm_name)
            coords[lm_name] = [landmarks[lm.value].x * frame.shape[1],
                               landmarks[lm.value].y * frame.shape[0]]

        # Calculate angles
        joint_angles['left_elbow'] = calculate_angle(coords['LEFT_SHOULDER'], coords['LEFT_ELBOW'], coords['LEFT_WRIST'])
        joint_angles['right_elbow'] = calculate_angle(coords['RIGHT_SHOULDER'], coords['RIGHT_ELBOW'], coords['RIGHT_WRIST'])
        joint_angles['left_shoulder'] = calculate_angle(coords['LEFT_HIP'], coords['LEFT_SHOULDER'], coords['LEFT_ELBOW'])
        joint_angles['right_shoulder'] = calculate_angle(coords['RIGHT_HIP'], coords['RIGHT_SHOULDER'], coords['RIGHT_ELBOW'])
        joint_angles['left_hip'] = calculate_angle(coords['LEFT_SHOULDER'], coords['LEFT_HIP'], coords['LEFT_KNEE'])
        joint_angles['right_hip'] = calculate_angle(coords['RIGHT_SHOULDER'], coords['RIGHT_HIP'], coords['RIGHT_KNEE'])

        for joint, angle in joint_angles.items():
            prev = angle_history.get(joint, angle)
            smoothed = alpha * angle + (1 - alpha) * prev
            angle_history[joint] = smoothed
            joint_angles[joint] = smoothed

        # Swing phases
        left_shoulder_angle = joint_angles['left_shoulder']
        shoulder_angles.append(left_shoulder_angle)
        if len(shoulder_angles) == window_size:
            max_angle = max(shoulder_angles)
            min_angle = min(shoulder_angles)

            if left_shoulder_angle >= max_angle * 0.98:
                swing_phase = 'Top Backswing'
            elif left_shoulder_angle <= min_angle * 1.02 and swing_phase == 'Top Backswing':
                swing_phase = 'Impact'
            elif swing_phase == 'Impact' and left_shoulder_angle > min_angle + 5:
                swing_phase = 'Follow-Through'

        cv2.putText(frame, f'Phase: {swing_phase}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        for i, (joint, angle) in enumerate(joint_angles.items()):
            min_angle, max_angle = good_angles[joint]
            color = (0, 255, 0) if min_angle <= angle <= max_angle else (0, 0, 255)
            cv2.putText(frame, f'{joint}: {int(angle)}', (30, 60 + 25*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))

    cv2.imshow('AI Golf Swing Analyzer', frame)

    if cv2.waitKey(10) & 0xFF == 27:    # ESC key
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
