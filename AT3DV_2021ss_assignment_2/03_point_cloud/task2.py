import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv2 import aruco


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

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def read_disparity_map(filename):
    return np.load(filename)


def rectify_pipe(K1, D1, K2, D2, R, T, img_ir1, img_ir2):
    output = cv2.stereoRectify(K1, D1, K2, D2, (640, 480), R, T)

    R1 = output[0]
    R2 = output[1]
    P1 = output[2]
    P2 = output[3]
    Q = output[4]

    map11, map12 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (640, 480), cv2.CV_8UC1)
    map21, map22 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (640, 480), cv2.CV_8UC1)

    img_ir1_rectified = cv2.remap(img_ir1, map11, map12, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    img_ir2_rectified = cv2.remap(img_ir2, map21, map22, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    ir1_rectified_gray = cv2.cvtColor(img_ir1_rectified, cv2.COLOR_BGR2GRAY)
    ir2_rectified_gray = cv2.cvtColor(img_ir2_rectified, cv2.COLOR_BGR2GRAY)
    return ir1_rectified_gray, ir2_rectified_gray, Q


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    print('%s saved' % fn)


def task2_1_write_ply():
    imgs = ['0000000029.png', '0000000031.png', '0000000032.png', '0000000038.png', '0000000039.png']
    x = 0
    for img in imgs:
        # disparity_map = read_disparity_map('disparity_estimations/disparity_' + str(x) + '.npy')
        disparity_map = read_disparity_map('disparity_' + str(x) + '.npy')
        img_ir1 = cv2.imread('ir1/undist/' + img)
        img_ir2 = cv2.imread('ir2/undist/' + img)
        _, _, Q = rectify_pipe(K_ir1, d_ir1, K_ir2, d_ir2, R, T, img_ir1, img_ir2)
        mask = np.logical_and(disparity_map > 200, disparity_map < 800)
        img_3d = cv2.reprojectImageTo3D(disparity_map, Q)
        colors = cv2.cvtColor(img_ir1, cv2.COLOR_BGR2RGB)
        img_3d = img_3d[mask]
        colors = colors[mask]
        write_ply('out_individual_'+str(x)+'.ply', img_3d, colors)
        x += 1


def task2_2_get_poses_camera_aruco(filename):
    img = cv2.imread('ir1/'+filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
    img_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)

    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, K_ir1, d_ir1)

    for i in range(len(tvecs)):
        rvec = rvecs[i]
        tvec = tvecs[i]
        aruco.drawAxis(img_markers, K_ir1, d_ir1, rvec, tvec, 0.1)

    plt.figure()
    plt.imshow(img_markers)
    plt.show()


def task2_3_aruco_reference_frame(filename, x):
    disparity_map = read_disparity_map('disparity_' + str(x) + '.npy')
    img_ir1 = cv2.imread('ir1/undist/' + filename)
    img_ir2 = cv2.imread('ir2/undist/' + filename)
    _, _, Q = rectify_pipe(K_ir1, d_ir1, K_ir2, d_ir2, R, T, img_ir1, img_ir2)
    mask = np.logical_and(disparity_map > 200, disparity_map < 800)
    img_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    colors = cv2.cvtColor(img_ir1, cv2.COLOR_BGR2RGB)
    img_3d = img_3d[mask]
    colors = colors[mask]

    img = cv2.imread('ir1/'+filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)

    rvecs_markers, tvecs_markers, _ = aruco.estimatePoseSingleMarkers(corners, 7.0, K_ir1, d_ir1)

    rvecs_markers, _ = cv2.Rodrigues(rvecs_markers)
    h_matrix = np.identity(4)
    h_matrix[0:3, 0:3] = rvecs_markers
    h_matrix[0:3, 3] = tvecs_markers
    h_matrix = np.linalg.inv(h_matrix)

    img_3d = np.append(img_3d, np.ones((img_3d.shape[0], 1)), axis=1).T
    img_3d = np.matmul(h_matrix, img_3d)
    img_3d = img_3d[0:3, :].T
    write_ply('out_' + str(x) + '.ply', img_3d, colors)


task2_1_write_ply()
x = 0
imgs = ['0000000029.png', '0000000031.png', '0000000032.png', '0000000038.png', '0000000039.png']
for img in imgs:
    task2_2_get_poses_camera_aruco(img)
    task2_3_aruco_reference_frame(img, x)
    x += 1

# 3.2.4
# Due to the its frontal view, the floor is recovered worse than the wall.
# It mainly focuses on the objects front also includes the seen parts of wall.
# It would recover worse if IR support was not present.
