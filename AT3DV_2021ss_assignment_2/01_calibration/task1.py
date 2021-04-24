import importlib

camera_calibration = importlib.import_module('camera_calibration')

filename = 'ir1'
# filename = 'ir2'
# filename = 'col'
show_imgs = False

objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs = camera_calibration.calibrate_camera(filename, show_imgs)

print(mtx, ret)

print(dist)
