import cv2
import numpy as np


def undistort_pipe(filename, imgname, mtx, dist, w, h):
    img = cv2.imread(filename + imgname)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imwrite(filename + 'undist/' + imgname, dst)


K_col = np.array([[662.593701688052, 0.0, 324.857607968018],
                 [0.0, 658.422641634482, 224.715217487322],
                 [0.0, 0.0, 1.000000]])
d_col = np.array([0.155208391239907, -0.360250096753537, 0.0, 0.0, 0.0])

K_ir1 = np.array([[388.425466037048, 0.0, 321.356734811229],
                 [0.0, 387.559412128232, 244.543659354387],
                 [0.0, 0.0, 1.000000]])
d_ir1 = np.array([0.00143845958426059, -0.00410315309358759, 0.0, 0.0, 0.0])

K_ir2 = np.array([[390.034619271096, 0.0, 321.390633361907],
                 [0.0, 389.119919973996, 244.648608218415],
                 [0.0, 0.0, 1.000000]])
d_ir2 = np.array([0.00241762888488943, -0.00118610336539317, 0.0, 0.0, 0.0])


def step1():
    undistort_pipe('col/', '0000000039.png', K_col, d_col, 640, 480)
    undistort_pipe('col/', '0000000040.png', K_col, d_col, 640, 480)
    undistort_pipe('ir1/', '0000000039.png', K_ir1, d_ir1, 640, 480)
    undistort_pipe('ir1/', '0000000040.png', K_ir1, d_ir1, 640, 480)
    undistort_pipe('ir2/', '0000000039.png', K_ir2, d_ir2, 640, 480)
    undistort_pipe('ir2/', '0000000040.png', K_ir2, d_ir2, 640, 480)


step1()

