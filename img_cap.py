import cv2

# For webcam input:
cap = cv2.VideoCapture(0)
i = 0
while cap.isOpened():
    input()
    success, image = cap.read()
    cv2.imwrite(str(i) + '.jpg', image)
    i += 1

cap.release()