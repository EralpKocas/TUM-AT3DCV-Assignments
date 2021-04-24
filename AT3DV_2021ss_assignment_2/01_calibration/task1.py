import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.1)

objp = np.zeros((7 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)
objp = np.multiply(objp, 20)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('col/*.png')
gray = []
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 9), None, flags=cv2.CALIB_CB_FILTER_QUADS)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 9), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1500)


cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                   None, None,
                                                   flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)

print(mtx, ret)

print(dist)
