#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 22:15:49 2023

@author: ywei9
"""

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    # convert the color space from BGR to RGB
    # Flip the image horizontally for a selfie-view display.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # get the results from mediapipe
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # get the image size
    img_h, img_w, img_c = image.shape
    # print(img_h, img_w)
    # input()
    # placeholder for 2d and 3d coordinates
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        for idx, mesh_lm in enumerate(face_landmarks.landmark):
            # point [1]: nosetip, [33]: righteye, [263]: lefteye, [61, 291]: lip, [199]: chin
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (mesh_lm.x * img_w, mesh_lm.y * img_h)
                    nose_3d = (mesh_lm.x * img_w, mesh_lm.y * img_h, mesh_lm.z * 3000)

                x, y = int(mesh_lm.x * img_w), int(mesh_lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, mesh_lm.z])

        # convert the coordinates to numpy array
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # camera matrix, this is from the camera calibration.
        focal_len = 1 * img_w
        cam_matrix = np.array([[focal_len, 0, img_h / 2],
                                   [0, focal_len, img_w / 2],
                                   [0, 0, 1]])
        # cam_matrix = np.array([[962, 0, 637.2057099],
        #                        [0, 962, 341.4072304],
        #                        [0, 0, 1]])
        # cam_matrix = np.array([[828.19743084, 0, 639.18321009]
        #                        [0, 832.74775276, 320.47891732]
        #                        [0, 0, 1]])

        # the distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
  #       dist_matrix = np.array([[-2.07893259e-01, 1.98102422e+00, -4.81576742e-03, -7.27050556e-04,
  # -5.40033789e+00]])

        # solvepnp to get the rotation and transform vector
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # rotation matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        # estimate the gaze direction by using the head pose
        if y < -10:
            text = "Looking Left"
        elif y > 10:
            text = "Looking Right"
        elif x < -10:
            text = "Looking Down"
        elif x > 10:
            text = "Looking Up"
        else:
            text = "Forward"

        # Display the nose direction
        # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        cv2.line(image, p1, p2, (255, 0, 0), 3)

        # Add the text on the image
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Gaze Estimation', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
