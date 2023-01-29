import numpy as np
import cv2 as cv
import glob
class Calibration():
    def __init__(self, file, size):
        self.chess_size = size
        self.file = file
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

    def cam_mtx(self):
        images = glob.glob(self.file)

        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            # (9, 7) is the chessboard size
            ret, corners = cv.findChessboardCorners(gray, self.chess_size, None)
            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((self.chess_size[0] * self.chess_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.chess_size[0], 0:self.chess_size[1]].T.reshape(-1, 2)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        return np.array(mtx), np.array(dist)


#
# chess_size = (9, 7)
# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((chess_size[0]*chess_size[1],3), np.float32)
# objp[:,:2] = np.mgrid[0:chess_size[0],0:chess_size[1]].T.reshape(-1,2)
#
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
#
# images = glob.glob('./img/*.jpg')
#
# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     # (9, 7) is the chessboard size
#     ret, corners = cv.findChessboardCorners(gray, chess_size, None)
#
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)
#         # Draw and display the corners
#         cv.drawChessboardCorners(img, chess_size, corners2, ret)
#         cv.imshow('img', img)
#         print(img.shape)
#         cv.waitKey(1000)
# cv.destroyAllWindows()
#
# # calibration
# print(gray.shape[::-1])
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print("ret:", ret)
# print("mtx:", mtx)
# print("dist:", dist)
#
# img = cv.imread('./img/3.jpg')
# h,  w = img.shape[:2]
# print(h, w)
#
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# print(roi)
# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult1.png', dst)
#
# # undistort
# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult2.png', dst)
#
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error
# print( "total error: {}".format(mean_error/len(objpoints)) )
