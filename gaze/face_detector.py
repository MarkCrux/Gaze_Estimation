import mediapipe as mp
import cv2
import numpy as np

class FaceDetector():

     def __init__(self):
         self.model = mp.solutions.face_mesh.FaceMesh(
             max_num_faces=1,
             refine_landmarks=True,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5
         )
         self.img_h = None
         self.img_w = None
         self.img_c = None

     def detect_face(self, image):
         self.img_h, self.img_w, self.img_c = image.shape
         # get the results from mediapipe
         return self.model.process(image)

     def face_points(self, landmarks):
         face_3d = []
         face_2d = []
         if landmarks.multi_face_landmarks:
             for face_landmarks in landmarks.multi_face_landmarks:
                 for idx, mesh_lm in enumerate(face_landmarks.landmark):
                     # point [1]: nosetip, [33]: righteye, [263]: lefteye, [61, 291]: lip, [199]: chin
                     if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 1:
                         x, y = int(mesh_lm.x * self.img_w), int(mesh_lm.y * self.img_h)
                         # Get the 2D Coordinates
                         face_2d.append([x, y])
                         # Get the 3D Coordinates
                         face_3d.append([x, y, mesh_lm.z])

                 # convert the coordinates to numpy array
         return (np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64))

     def eye_points(self, landmarks):
        left_eye = []
        right_eye = []
        if landmarks.multi_face_landmarks:
            for face_landmarks in landmarks.multi_face_landmarks:
                for idx, mesh_lm in enumerate(face_landmarks.landmark):
                    # left eye including iris center
                    if idx == 263 or idx == 362 or idx == 473:
                        x, y = int(mesh_lm.x * self.img_w), int(mesh_lm.y * self.img_h)
                        # Get the 2D Coordinates
                        left_eye.append([x, y])
                    # right eye including iris center
                    elif idx == 33 or idx == 133 or idx == 468:
                        x, y = int(mesh_lm.x * self.img_w), int(mesh_lm.y * self.img_h)
                        # Get the 2D Coordinates
                        right_eye.append([x, y])
        return (left_eye, right_eye)