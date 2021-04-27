import cv2
import numpy as np
from matplotlib import pyplot as plt


def undistort_pipe(filename, imgname, mtx, dist, w, h):
    img = cv2.imread(filename + imgname)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imwrite(filename + 'undist/' + imgname, dst)


def rectify_pipe(K1, D1, K2, D2, R, T, img_ir1, img_ir2):
    output = cv2.stereoRectify(K1, D1, K2, D2, (640, 480), R, T)

    R1 = output[0]
    R2 = output[1]
    P1 = output[2]
    P2 = output[3]

    map11, map12 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (640, 480), cv2.CV_8UC1)
    map21, map22 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (640, 480), cv2.CV_8UC1)

    img_ir1_rectified = cv2.remap(img_ir1, map11, map12, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    img_ir2_rectified = cv2.remap(img_ir2, map21, map22, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    ir1_rectified_gray = cv2.cvtColor(img_ir1_rectified, cv2.COLOR_BGR2GRAY)
    ir2_rectified_gray = cv2.cvtColor(img_ir2_rectified, cv2.COLOR_BGR2GRAY)
    return ir1_rectified_gray, ir2_rectified_gray


def disparity_pipe(ir1_rectified_gray, ir2_rectified_gray):
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=21)
    disparity_BM = stereo.compute(ir1_rectified_gray, ir2_rectified_gray)
    return disparity_BM


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

R = np.array([[0.999999506646425, -3.18339774658664e-05, 0.000992820983631579],
                [3.15905844318835e-05, 0.999999969447414, 0.000245167709718939],
                [-0.000992828757961677, -0.000245136224969466, 0.999999477099508]])
T = np.array([-49.9430087222935, 0.0126441058712290, -0.0678600809461142])


def task1():
    undistort_pipe('col/', '0000000029.png', K_col, d_col, 640, 480)
    undistort_pipe('col/', '0000000031.png', K_col, d_col, 640, 480)
    undistort_pipe('col/', '0000000032.png', K_col, d_col, 640, 480)
    undistort_pipe('col/', '0000000038.png', K_col, d_col, 640, 480)
    undistort_pipe('col/', '0000000039.png', K_col, d_col, 640, 480)

    undistort_pipe('ir1/', '0000000029.png', K_ir1, d_ir1, 640, 480)
    undistort_pipe('ir1/', '0000000031.png', K_ir1, d_ir1, 640, 480)
    undistort_pipe('ir1/', '0000000032.png', K_ir1, d_ir1, 640, 480)
    undistort_pipe('ir1/', '0000000038.png', K_ir1, d_ir1, 640, 480)
    undistort_pipe('ir1/', '0000000039.png', K_ir1, d_ir1, 640, 480)

    undistort_pipe('ir2/', '0000000029.png', K_ir2, d_ir2, 640, 480)
    undistort_pipe('ir2/', '0000000031.png', K_ir2, d_ir2, 640, 480)
    undistort_pipe('ir2/', '0000000032.png', K_ir2, d_ir2, 640, 480)
    undistort_pipe('ir2/', '0000000038.png', K_ir2, d_ir2, 640, 480)
    undistort_pipe('ir2/', '0000000039.png', K_ir2, d_ir2, 640, 480)

    imgs = ['0000000029.png', '0000000031.png', '0000000032.png', '0000000038.png', '0000000039.png']
    x = 0
    for img in imgs:
        img_ir1 = cv2.imread('ir1/undist/' + img)
        img_ir2 = cv2.imread('ir2/undist/' + img)
        ir1_rectified_gray, ir2_rectified_gray = rectify_pipe(K_ir1, d_ir1, K_ir2, d_ir2, R, T, img_ir1, img_ir2)
        disparity = disparity_pipe(ir1_rectified_gray, ir2_rectified_gray)
        np.save('disparity_' + str(x), disparity)
        x += 1


task1()
disp = np.load('disparity_0.npy')
plt.imshow(disp, "hot")
plt.colorbar()
plt.show()
