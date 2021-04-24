import importlib
import cv2
import numpy as np

camera_calibration = importlib.import_module('camera_calibration')

filename_ir1 = 'ir1'
filename_ir2 = 'ir2'
filename_col = 'col'
show_imgs = False

objpoints_ir1, imgpoints_ir1, ret_ir1, mtx_ir1, dist_ir1, rvecs_ir1, tvecs_ir1 = \
    camera_calibration.calibrate_camera(filename_ir1, show_imgs)

objpoints_ir2, imgpoints_ir2, ret_ir2, mtx_ir2, dist_ir2, rvecs_ir2, tvecs_ir2 = \
    camera_calibration.calibrate_camera(filename_ir2, show_imgs)

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 0.00001)

# print(len(objpoints_ir1))
# print(len(objpoints_ir2))
# print(len(imgpoints_ir1))
# print(len(imgpoints_ir2))

# mtx_ir1 = np.array([[388.425466037048, 0.0, 321.356734811229],
#                [0.0, 387.559412128232, 244.543659354387],
#                [0.0, 0.0, 1.000000]])
# dist_ir1 = np.array([0.00143845958426059,-0.00410315309358759,0.0,0.0,0.0])

# mtx_ir2 = np.array([[390.034619271096, 0.0, 321.390633361907],
#                [0.0, 389.119919973996, 244.648608218415],
#                [0.0, 0.0, 1.000000]])
# dist_ir2 = np.array([0.00241762888488943,-0.00118610336539317,0.0,0.0,0.0])

# config
flags = 0
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_ZERO_TANGENT_DIST
flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_FIX_K1
# flags |= cv2.CALIB_FIX_K2
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5
# flags |= cv2.CALIB_FIX_K6

ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints_ir1, imgpoints_ir1, imgpoints_ir2, mtx_ir1, dist_ir1,
                        mtx_ir2, dist_ir2, (640, 480),
                        criteria=criteria,
                        flags=flags)

print('fundamental matrix from ir1 to ir2')
print(F)

print('rotation matrix')
print(R)

print('translation vector')
print(T)

print(ret)


objpoints_col, imgpoints_col, ret_col, mtx_col, dist_col, rvecs_col, tvecs_col = \
    camera_calibration.calibrate_camera(filename_col, show_imgs)

ret_col, cameraMatrix1_col, distCoeffs1_col, cameraMatrix2_col, distCoeffs2_col, R_col, T_col, E_col, F_col = \
    cv2.stereoCalibrate(objpoints_col, imgpoints_col, imgpoints_ir1, mtx_col, dist_col,
                        mtx_ir1, dist_ir1, (640, 480),
                        criteria=criteria,
                        flags=flags)

print('fundamental matrix from col to ir1')
print(F_col)

print('rotation matrix')
print(R_col)

print('translation vector')
print(T_col)

print(ret_col)
