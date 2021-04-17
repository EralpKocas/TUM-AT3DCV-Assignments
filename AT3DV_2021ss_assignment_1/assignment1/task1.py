import pyrender, trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# scene
scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[3.0, 3.0, 3.0])
r = pyrender.OffscreenRenderer(640, 480)

# camera
K = np.array([[1066.7780,         0, 312.9869],
              [        0, 1067.4870, 241.3109],
              [        0,         0,        1]])
   
camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.001, zfar=3)
scene.add(camera)

# mesh
trimesh_obj = trimesh.load('data_for_task1/003_cracker_box/textured.obj')
mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
mesh_node = scene.add(mesh)


##########################################################################
###     To do : Find out coordinate system orientation of Pyrender     ###
##########################################################################

####  step 1. Check out the plot. with z = 1m VS -1m
####          Which direction does Pyrender point for z axis?

# seems like negative z since we can see object in that pose.

# render positive z
mesh_pose = np.identity(4)
mesh_pose[:3, 3] = [0, 0, 2]
scene.set_pose(mesh_node, mesh_pose)
color_pos_z, depth = r.render(scene)

# render negative z
mesh_pose = np.identity(4)
mesh_pose[:3, 3] = [0, 0, -2]
scene.set_pose(mesh_node, mesh_pose)
color_neg_z, depth = r.render(scene)

# compare the plot
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(color_pos_z)
plt.subplot(1, 2, 2)
plt.imshow(color_neg_z)
plt.show()

####  step 2. Then place the object in front of the camera (from step 1)
####          either (0,0,2) or (0,0,-2) depending on the visibility
####          and test with x,y axis in the similar way to figure out which
####          direction does Pyrender point for x,y axis

#### Tip : Make sure to set x,y with relatively smaller number (i.e. 0.2)
####       So that the rendering doesn't go out of image boundary
 
# render positive x

mesh_pose = np.identity(4)
mesh_pose[:3, 3] = [0.2, 0, -2]
scene.set_pose(mesh_node, mesh_pose)
color_pos_x, depth = r.render(scene)


# render negative x

mesh_pose = np.identity(4)
mesh_pose[:3, 3] = [-0.2, 0, -2]
scene.set_pose(mesh_node, mesh_pose)
color_neg_x, depth = r.render(scene)


# compare the plot. Which direction is x?

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(color_pos_x)
plt.subplot(1, 2, 2)
plt.imshow(color_neg_x)
plt.show()

# x is to the right

# render positive y


mesh_pose = np.identity(4)
mesh_pose[:3, 3] = [0, 0.2, -2]
scene.set_pose(mesh_node, mesh_pose)
color_pos_y, depth = r.render(scene)

# render negative y

mesh_pose = np.identity(4)
mesh_pose[:3, 3] = [0, -0.2, -2]
scene.set_pose(mesh_node, mesh_pose)
color_neg_y, depth = r.render(scene)


# compare the plot. Which direction is y?

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(color_pos_y)
plt.subplot(1, 2, 2)
plt.imshow(color_neg_y)
plt.show()

# y is to the up

####  Now you know the orientation of Pyrender :)

#      y
#      ^
#      |
#      |
#      |---->x
#      -
#     -
#   <
#  z

##########################################################################
###  To do : Figure out transform factor between Linemod and Pyrender  ###
##########################################################################
# - hint : orientation of YCB dataset is already in the slide

# mesh pose from YCB dataset.
mesh_pose = np.identity(4)
mesh_pose[:3, :] = np.loadtxt("data_for_task1/000001-color-0.txt", delimiter=" ")

####  step 1. figure out transform factor between Pyrender and YCB dataset
####          and modify the mesh pose

# factor 

# positive z is forward
# x is to the left
# y is to the down

# rotate 180 degree around x.

r_euler = R.from_euler('x', 180, degrees=True)
r_matrix = r_euler.as_matrix()
rot_x = np.identity(4)
rot_x[:3, :3] = r_matrix

# modify mesh_pose using factor

mesh_pose = np.matmul(rot_x, mesh_pose)

scene.set_pose(mesh_node, mesh_pose)

#############################################################################
### To do : augment the rendered obj on the image, ** WITHOUT FOR LOOP ** ###
#############################################################################

####  step 1. Generate mask for the object using one of rendered images
####          (hint : You can directly use conditional statement on numpy array!)

color, depth = r.render(scene)

plt.figure()
plt.imshow(color)
plt.show()

# calculate mask **without for loop**.
mask2 = np.array(color < 1)

mask3 = np.array(np.nonzero(color))

mask = np.where(color < 1, color, 1)
ones = np.ones(mask.shape)
mask = ones - mask
####  step 2. Augment the rendred object on the image using mask.

img = plt.imread("data_for_task1/000001-color.png")

# mask out image and add rendring **without for loop**.

img = img * mask2

#img[mask2 == 0] = color[mask2 == 0]

plt.figure()
plt.imshow(img)
plt.show()


