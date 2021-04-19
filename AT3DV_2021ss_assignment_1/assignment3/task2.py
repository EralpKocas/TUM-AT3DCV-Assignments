import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utilities import *

#################################################################
###        Read dataset  :  Don't modify anything             ###
#################################################################

# read data from camera 0
img_0 = plt.imread("data_for_task2/rgb_full/0.png")[:, :, :3]

depth_0 = plt.imread("data_for_task2/depth_full/0.png")[:, :, 0]
depth_0 = depth_0 * (6-0.1) + 0.1 # rescale depth range from [0,1] to [0.1,4]

pose_0 = np.loadtxt("data_for_task2/pose_cam/0.txt")
pose_0_inv = np.linalg.inv(pose_0)
K_0 = np.loadtxt("data_for_task2/intrinsic_cam/0.txt")
K_0_inv = np.linalg.inv(K_0)

# read data from camera 0
img_1 = plt.imread("data_for_task2/rgb_full/1.png")[:, :, :3]
depth_1 = plt.imread("data_for_task2/depth_full/1.png")[:, :, 0]
depth_1 = depth_1 * (6-0.1) + 0.1 # rescale depth range from [0,1] to [0.1,4]

pose_1 = np.loadtxt("data_for_task2/pose_cam/1.txt")
pose_1_inv = np.linalg.inv(pose_1)

K_1 = np.loadtxt("data_for_task2/intrinsic_cam/1.txt")
K_1_inv = np.linalg.inv(K_1)


######################################################################
###     To do 1: Copy the code from To do 1 into double for loop   ###
###              and implement naive full image warping            ###
######################################################################

# Forward warping using naive double for loop

if True:
    
    img_fwd = np.zeros((480,640,3))

    # factor = 
    
    for x in range(640):
        for y in range(480):
        
            ##############################################################
            ###              Code from To do 1 goes here               ###
            ###   terms like factor can be calculated before the loop  ###
            ###   for the better performance here                      ###
            ##############################################################
            point_0 = [x, y, 1]
            depth = depth_0[y, x]

            #### Step 1 ~ Step 8
            
            #### incase you use external library to calculate factor,
            #### place the factor calculation outside of loop

            ####  step 1. Convert image coordinate into camera_0's coordinate
            ####          system of z = 1 by using inverse intrinsic

            cam_0 = np.matmul(K_0_inv, point_0)

            ####  step 2. Scale the coordinate by depth to bring it to 3D point
            ####          and make it as homogeneous 3D coordinate (i.e. [x,y,z,1])

            cam_d = cam_0 * depth
            cam_d = np.append(cam_d, 1)

            ####  step 3. Camera's extrinsic is recorded in pyrender's orientation
            ####          Re-orient the camera's orientation to pyrender's orientation
            ####          by mutliplying correct rotational factor

            r_euler = R.from_euler('x', 180, degrees=True)
            r_matrix = r_euler.as_matrix()
            rot_x = np.identity(4)
            rot_x[:3, :3] = r_matrix

            pyrender_orient_d = np.matmul(rot_x, cam_d)

            ####  step 4. Express the point into world coordinate system by using
            ####          pose of camera_0 Here, pose_0's direction is *Cam0 -> World*

            world_d = np.matmul(pose_0, pyrender_orient_d)

            ####  step 5. Express the point into camera_1's coordinate system.
            ####          Here, pose_1's direction is *Cam1 -> World*

            cam_1_coord = np.matmul(pose_1_inv, world_d)

            ####  step 6. Re-orient the camera to pyrender's orientation
            ####          by mutliplying correct rotational factor

            pyrender_orient_d = np.matmul(rot_x, cam_1_coord)

            ####  step 7. Discard the homogeneous point (i.e. [x,y,z,1] -> [x,y,z]
            ####          and divid by it's z to make z = 1 (i.e. [x',y',1])
            ####          --> now point is expressed in plane in front of camera
            ####              with distance of z = 1

            cam_plane_point_1 = pyrender_orient_d[0:3] / pyrender_orient_d[2]

            ####  step 8. Express the point into pixel coordinate system by using
            ####          intrinsic K

            point_1 = np.matmul(K_1, cam_plane_point_1)
            
            #### Bilinear interpolation function you implemented is specialized 
            #### in inverse warping. so, here you will use nn interpolation instead.
            
            # nearest_neighbor
            nearest_neighbor = np.round(point_1).astype(int)
            
            # skip coordinate outside of image boundary
            x_, y_, _ = nearest_neighbor
            if x_ >= 640 or x < 0 or y_ >= 480 or y_ < 0:
                continue
            img_fwd[y_, x_] = img_0[y, x]

            
    plt.figure()
    plt.imshow(img_fwd)
    plt.show()


######################################################################
###    To do 3: To do 2 results in lots of holes in the image.     ###
###             Reuse To do 2's loops but modify the direction of  ###
###             waring into inverse (inverse warping)              ###
######################################################################

# Inverse warping using naive double for loop and bilinear interpolation

if True:
    img_inv = np.zeros((480, 640, 3))

    for x in range(640):
        for y in range(480):
            
            # inverse warping goes from camera_1 to camera_0
            # while forward warping goes from camera_0 to camera_1
            point_1 = [x, y, 1]
            depth = depth_1[y, x]
            
            ##################################################################
            ### Copy paste the loop from To do 2 and change the direction  ###
            ##################################################################

            cam_0 = np.matmul(K_1_inv, point_1)

            ####  step 2. Scale the coordinate by depth to bring it to 3D point
            ####          and make it as homogeneous 3D coordinate (i.e. [x,y,z,1])

            cam_d = cam_0 * depth
            cam_d = np.append(cam_d, 1)

            ####  step 3. Camera's extrinsic is recorded in pyrender's orientation
            ####          Re-orient the camera's orientation to pyrender's orientation
            ####          by mutliplying correct rotational factor

            r_euler = R.from_euler('x', 180, degrees=True)
            r_matrix = r_euler.as_matrix()
            rot_x = np.identity(4)
            rot_x[:3, :3] = r_matrix

            pyrender_orient_d = np.matmul(rot_x, cam_d)

            ####  step 4. Express the point into world coordinate system by using
            ####          pose of camera_0 Here, pose_0's direction is *Cam0 -> World*

            world_d = np.matmul(pose_1, pyrender_orient_d)

            ####  step 5. Express the point into camera_1's coordinate system.
            ####          Here, pose_1's direction is *Cam1 -> World*

            cam_1_coord = np.matmul(pose_0_inv, world_d)

            ####  step 6. Re-orient the camera to pyrender's orientation
            ####          by mutliplying correct rotational factor

            pyrender_orient_d = np.matmul(rot_x, cam_1_coord)

            ####  step 7. Discard the homogeneous point (i.e. [x,y,z,1] -> [x,y,z]
            ####          and divid by it's z to make z = 1 (i.e. [x',y',1])
            ####          --> now point is expressed in plane in front of camera
            ####              with distance of z = 1

            cam_plane_point_1 = pyrender_orient_d[0:3] / pyrender_orient_d[2]

            ####  step 8. Express the point into pixel coordinate system by using
            ####          intrinsic K

            ### Should get corresponding point in image0

            point_0 = np.matmul(K_0, cam_plane_point_1)

            # your function :
            img_inv[y, x] = bilinear_interpolation_per_pixel(point_0[:2], img_0)
            
            # incase you want to test out nearest neighbor, try this
            '''
            nearest_neighbor = np.round(point_0).astype(int)
            x_, y_,_  = nearest_neighbor
            if x_ >= 640 or x < 0 or y_ >= 480 or y_ < 0:
                continue
            img_inv[y,x] = img_0[y_,x_]
            '''

    plt.figure()
    plt.imshow(img_inv)
    plt.show()


######################################################################
###    To do 4: Implement the vectorized bilinear interpolation    ###
###             in assignmen3_utilities.py                         ###
######################################################################

# Tip : It's always eay to reshape coordinates into shape = (n_dim, n_coordinates)
#       before applying the matrix multiplication, then reshape back to original shape!
#
#       i.e. homogeneous 2d coord for image shape = (480,640,3), transform matrix shape = (3,3)
#
#       1. reshape coordinates into (3,480*640) then multiply transform 
#          - matmul((3,3),(3,480*640)) -> resulting shape = (3,480*640)
#
#       2. reshape the result back to the original shape
#          - reshape to (480,640,3)

if True:
    h, w, _ = img_1.shape
    cam_1_pixel_grid = generate_homogeneous_grid(h, w)  # shape = [480,640,3]
    
    cam_1_pixel_grid = cam_1_pixel_grid.reshape(w*h, 3).T
    
    # all homogeneous 2D grid (480,640,3) is now expressed to [3,480*640]
    # to make vectorized operation much simpler to implement
    #
    #    i.e. transformed = np.matmul(transform_matrix, cam_1_pixel_grid)

    ########################################################################
    ###  Todo : implement vertorized pipeline for inverse image warping  ###
    ########################################################################
    
    ####  step 1. Convert image1 grid into camera_1,s coordinate 
    ####          system of z = 1 by using inverse intrinsic
    ####          (result shape = [3,480*640])

    cam_1 = np.matmul(K_1_inv, cam_1_pixel_grid)

    ####  step 2. Scale the coordinate by depth to bring it to 3D point
    ####          and make it as homogeneous 3D coordinate (i.e. [x,y,z,1])
    ####          (result shape = [4,480*640])

    depth_1 = np.reshape(depth_1, (1, cam_1.shape[1]))

    cam_d = np.multiply(cam_1, depth_1)

    cam_d = np.append(cam_d, np.ones((1, cam_d.shape[1])), axis=0)

    ####  step 3. Re-orient the camera to pyrender's orientation
    ####          by mutliplying correct rotational factor
    ####          (result shape = [4,480*640])

    r_euler = R.from_euler('x', 180, degrees=True)
    r_matrix = r_euler.as_matrix()
    rot_x = np.identity(4)
    rot_x[:3, :3] = r_matrix

    pyrender_orient_d = np.matmul(rot_x, cam_d)

    ####  step 4. Express the point into world coordinate system by using
    ####          pose of camera_1 Here, pose_1's direction is *Cam1 -> World*
    ####          (result shape = [4,480*640])

    world_d = np.matmul(pose_1, pyrender_orient_d)

    ####  step 5. Express the point into camera_0's coordinate system.
    ####          Here, pose_0's direction is *Cam0 -> World*
    ####          (result shape = [4,480*640])

    cam_1_coord = np.matmul(pose_0_inv, world_d)

    ####  step 6. Re-orient the camera to pyrender
    ####          by mutliplying correct rotational factor
    ####          (result shape = [4,480*640])

    pyrender_orient_d = np.matmul(rot_x, cam_1_coord)

    ####  step 7. Discard the homogeneous point (i.e. [x,y,z,1] -> [x,y,z]
    ####          and divid by it's z to make z = 1 (i.e. [x',y',1])
    ####          --> now point is expressed in plane in front of camera0
    ####              with distance of z = 1
    ####          (result shape = [3,480*640])

    z_axis = pyrender_orient_d[2, :]
    cam_plane_point_1 = pyrender_orient_d[0:3, :] / z_axis

    ####  step 8. Exress the grid in camera0 into pixel grid of camera0
    ####          by using intrinsic K
    ####          (result shape = [3,480*640])

    cam_0_pixel_grid = np.matmul(K_0, cam_plane_point_1)

    #### For input of verctorized bilinear interpolation function, we will
    #### reshape the pixel grid into shape of [640*480,2]
    cam_0_pixel_grid = cam_0_pixel_grid.T[:, :2]
    
    #### output of the bilinear interpolation fuction will be in shape of
    #### [480*640,3] (flattened RGB image)
    cam_0_warped = bilinear_interpolation_per_grid(cam_0_pixel_grid, img_0)
    
    #### Reshape back flattened image form for imshow
    ####(result shape = [480,640,3])
    cam_0_warped = cam_0_warped.reshape((h, w, 3))
    plt.figure()
    plt.imshow(cam_0_warped)
    plt.show()
    
