import cv2
import numpy as np
import glob


def calibrate_camera(filename, show_imgs):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 0.001)

    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)
    #objp = np.multiply(objp, 20)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = sorted(glob.glob(filename+'/*.png'))

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


filename_ir1 = 'ir1'
filename_ir2 = 'ir2'
filename_col = 'col'
show_imgs = False

objpoints_ir1, imgpoints_ir1, ret_ir1, mtx_ir1, dist_ir1, rvecs_ir1, tvecs_ir1 = \
    calibrate_camera(filename_ir1, show_imgs)

objpoints_ir2, imgpoints_ir2, ret_ir2, mtx_ir2, dist_ir2, rvecs_ir2, tvecs_ir2 = \
    calibrate_camera(filename_ir2, show_imgs)

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 0.000001)


#mtx_ir1 = np.array([[388.425466037048, 0.0, 321.356734811229],
#                    [0.0, 387.559412128232, 244.543659354387],
#                    [0.0, 0.0, 1.000000]])

#dist_ir1 = np.array([0.00143845958426059, -0.00410315309358759, 0.0, 0.0, 0.0])

#mtx_ir2 = np.array([[390.034619271096, 0.0, 321.390633361907],
#                    [0.0, 389.119919973996, 244.648608218415],
#                    [0.0, 0.0, 1.000000]])
#dist_ir2 = np.array([0.00241762888488943, -0.00118610336539317, 0.0, 0.0, 0.0])


ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints_ir1, imgpoints_ir1, imgpoints_ir2, mtx_ir1, dist_ir1,
                        mtx_ir2, dist_ir2, (640, 480),
                        criteria=criteria,
                        flags=cv2.CALIB_FIX_INTRINSIC)

print('fundamental matrix from ir1 to ir2')
print(F)

print('rotation matrix')
print(R)

print('translation vector')
print(T)

print(ret)


#objpoints_col, imgpoints_col, ret_col, mtx_col, dist_col, rvecs_col, tvecs_col = calibrate_camera(filename_col, show_imgs)

#ret_col, cameraMatrix1_col, distCoeffs1_col, cameraMatrix2_col, distCoeffs2_col, R_col, T_col, E_col, F_col = \
#    cv2.stereoCalibrate(objpoints_col, imgpoints_col, imgpoints_ir1, mtx_col, dist_col,
#                        mtx_ir1, dist_ir1, (640, 480),
#                        criteria=criteria,
#                        flags=cv2.CALIB_FIX_INTRINSIC)

#print('fundamental matrix from col to ir1')
#print(F_col)

#print('rotation matrix')
#print(R_col)

#print('translation vector')
#print(T_col)

#print(ret_col)
