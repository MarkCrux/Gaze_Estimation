import cv2
from gaze.gaze_estimate import GazeEstimate

gaze = GazeEstimate()
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    gaze.process(image)
    pose_direction = gaze.pose_direction()
    eye_ratio = gaze.eye_direction()

    image = gaze.annotated_frame(pose_direction, eye_ratio)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow('Gaze Estimation', image)
    if cv2.waitKey(50) & 0xFF == 27:
        break
cap.release()
