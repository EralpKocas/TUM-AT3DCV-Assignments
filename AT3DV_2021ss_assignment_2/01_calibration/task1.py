import cv2
import numpy as np
import glob


def calibrate_camera(filename, show_imgs):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 0.001)

    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)
    #objp = np.multiply(objp, 20)

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(filename+'/*.png'))
    gray = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 9), None, flags=cv2.CALIB_CB_FILTER_QUADS)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)

            if show_imgs:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (7, 9), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(1500)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                       None, None,
                                                       flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)

    return objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs


filename = 'ir1'
# filename = 'ir2'
# filename = 'col'
show_imgs = True

objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs = calibrate_camera(filename, show_imgs)

print(mtx, ret)

print(dist)
