import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

image_path = "walking.jpg"
image = cv2.imread(image_path)

with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Pose Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
