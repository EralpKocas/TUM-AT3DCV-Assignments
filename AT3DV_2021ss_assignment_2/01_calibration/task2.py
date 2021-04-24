import importlib
import cv2

camera_calibration = importlib.import_module('camera_calibration')

filename_ir1 = 'ir1'
filename_ir2 = 'ir2'
# filename = 'col'
show_imgs = False

objpoints_ir1, imgpoints_ir1, ret_ir1, mtx_ir1, dist_ir1, rvecs_ir1, tvecs_ir1 = \
    camera_calibration.calibrate_camera(filename_ir1, show_imgs)

objpoints_ir2, imgpoints_ir2, ret_ir2, mtx_ir2, dist_ir2, rvecs_ir2, tvecs_ir2 = \
    camera_calibration.calibrate_camera(filename_ir2, show_imgs)

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 0.0001)

print(len(objpoints_ir1))
print(len(objpoints_ir2))
print(len(imgpoints_ir1))
print(len(imgpoints_ir2))

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
