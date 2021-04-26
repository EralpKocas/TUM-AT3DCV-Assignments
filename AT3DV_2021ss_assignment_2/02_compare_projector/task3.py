import cv2
import numpy as np
from matplotlib import pyplot as plt

K_ir1 = np.array([[388.425466037048, 0.0, 321.356734811229],
                [0.0, 387.559412128232, 244.543659354387],
                [0.0, 0.0, 1.000000]])
d_ir1 = np.array([0.00143845958426059,-0.00410315309358759,0.0,0.0,0.0])

K_ir2 = np.array([[390.034619271096, 0.0, 321.390633361907],
                [0.0, 389.119919973996, 244.648608218415],
                [0.0, 0.0, 1.000000]])
d_ir2 = np.array([0.00241762888488943,-0.00118610336539317,0.0,0.0,0.0])


F = np.array([[-6.87111220925229e-11, -7.77039351599509e-07, 0.000153509631502188],
                [4.48903589546165e-07, -8.11678056642449e-08, -0.128989836785960],
                [-7.72939005753161e-05, 0.128617945871685, 0.0713708982802110]])

R = np.array([[0.999999506646425, -3.18339774658664e-05, 0.000992820983631579],
                [3.15905844318835e-05, 0.999999969447414, 0.000245167709718939],
                [-0.000992828757961677, -0.000245136224969466, 0.999999477099508]])
T = np.array([-49.9430087222935, 0.0126441058712290, -0.0678600809461142])



img_ir1 = cv2.imread('ir1/undist/0000000039.png')
img_ir2 = cv2.imread('ir2/undist/0000000039.png')

output = cv2.stereoRectify(K_ir1, d_ir1, K_ir2, d_ir2, (640, 480), R, T)

R1 = output[0]
R2 = output[1]
P1 = output[2]
P2 = output[3]

map11, map12 = cv2.initUndistortRectifyMap(K_ir1, d_ir1, R1, P1, (640, 480), cv2.CV_8UC1)
map21, map22 = cv2.initUndistortRectifyMap(K_ir2, d_ir2, R2, P2, (640, 480), cv2.CV_8UC1)

img_ir1_rectified = cv2.remap(img_ir1, map11, map12, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
img_ir2_rectified = cv2.remap(img_ir2, map21, map22, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

ir1_rectified_gray = cv2.cvtColor(img_ir1_rectified, cv2.COLOR_BGR2GRAY)
ir2_rectified_gray = cv2.cvtColor(img_ir2_rectified, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=21)
disparity_BM = stereo.compute(ir1_rectified_gray, ir2_rectified_gray)
plt.imshow(disparity_BM, "hot")
plt.colorbar()
plt.show()
