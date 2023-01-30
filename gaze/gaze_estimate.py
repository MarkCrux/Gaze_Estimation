#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
from gaze.face_detector import FaceDetector
from gaze.cam_cali import Calibration
import numpy as np


class GazeEstimate(object):

    def __init__(self):
        self.frame = None
        self.face_mesh = None
        self.face_points_2d = None
        self.face_points_3d = None
        self.left_eye = None
        self.right_eye = None
        self.file = './img/*.jpg' # chess board calibration images
        self.chess_size = (9, 7) # chess board size
        self._face_detector = FaceDetector()


    def process(self, frame):
        self.frame = frame
        self.face_mesh = self._face_detector.detect_face(frame)
        self.face_points_2d, self.face_points_3d = self._face_detector.face_points(self.face_mesh)
        self.left_eye, self.right_eye = self._face_detector.eye_points(self.face_mesh)

    def pose_direction(self):

        # camera matrix, this is from the camera calibration.

        # The below codes are for calibrating the pinhole camera.
        # We don't use these because you may have a different camera, so we will use a common camera matrix
        # cal = Calibration(self.file, self.chess_size)
        # cam_matrix, dist_matrix = cal.cam_mtx()

        # The comment setting of the camera matrix can also demonstrate the gaze estimation demo.
        focal_len = 1 * self.frame.shape[1]
        cam_matrix = np.array([[focal_len, 0, self.frame.shape[0] / 2],
                               [0, focal_len, self.frame.shape[1] / 2],
                               [0, 0, 1]])
        # the distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # solvepnp to get the rotation and transform vector
        success, rot_vec, trans_vec = cv2.solvePnP(self.face_points_3d, self.face_points_2d, cam_matrix, dist_matrix)

        # rotation matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        return [x, y, z]

    def eye_direction(self):
        left_eye_w = abs(self.left_eye[0][0]-self.left_eye[1][0]) # left eye width
        left_iris = abs(self.left_eye[2][0]-self.left_eye[1][0]) # iris center to left eye point
        left_ratio = left_iris / (abs(left_eye_w - 10) + 1e-5)

        right_eye_w = abs(self.right_eye[0][0] - self.right_eye[1][0]) # right eye width
        right_iris = abs(self.right_eye[2][0] - self.right_eye[0][0]) # iris center to right eye point
        right_ratio = right_iris / (abs(right_eye_w - 10) + 1e-5)

        return (left_ratio + right_ratio)/2 # average move ratio for two eyes


    def annotated_frame(self, pose, eye):
        image = self.frame.copy()
        # estimate the gaze direction by using the head pose and eye movement together
        eye_angle = ((eye - 0.35) / (0.65 - 0.35)) * 20 - 10
        x, y_pose, z = pose[:]
        y = y_pose + eye_angle
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

        # Add the text on the image
        color_1 = (0, 255, 0)
        color_2 = (0, 0, 255)

        cv2.circle(image, self.left_eye[2], 3, color_1, -1)
        cv2.circle(image, self.right_eye[2], 3, color_1, -1)

        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color_1, 2)
        cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_2, 2)
        cv2.putText(image, "y_pose: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_2, 2)
        cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color_2, 2)
        cv2.putText(image, "eye_angle: " + str(np.round(eye_angle, 2)), (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        return image
