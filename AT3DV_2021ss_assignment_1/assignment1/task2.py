import pyrender, trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# scene
scene = pyrender.Scene(bg_color=[0,0,0], ambient_light=[3.0,3.0,3.0])
r = pyrender.OffscreenRenderer(1920,1080)

# camera
K = np.array([[1081.37,       0,  959.5],
              [      0, 1081.37,  539.5],
              [      0,       0,       1]])
   
camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.001, zfar=3)
scene.add(camera)

# mesh
trimesh_obj = trimesh.load('data_for_task2/turtle/geometry.ply')
mesh = pyrender.Mesh.from_trimesh(trimesh_obj)

# mesh pose
mesh_pose = np.identity(4)
mesh_pose[:3, :] = np.loadtxt("data_for_task2/000299-color.txt", delimiter=" ")

##########################################################################
###   To do : Figure out transform factor between Laval and Pyrender   ###
##########################################################################

####  step 1. figure out transform factor between Pyrender and Laval dataset
####          and modify the mesh pose 
####          (hint : See where the turtle located in the RGB image, and
####                  compare with corresponding translation!)

# factor 

# no rotation is needed.
# z is backward
# x is to the left
# y is to the up
# so, same as pyrender.

#      y
#      ^
#      |
#      |
#      |---->x
#      -
#     -
#   <
#  z

                
# modify mesh_pose using factor


mesh_node = scene.add(mesh, pose=mesh_pose)


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

mask = np.nonzero(depth)

####  step 2. Augment the rendred object on the image using mask.

img = plt.imread("data_for_task2/000299-color.png")

# mask out image and add rendring **without for loop**

img[mask] = 0  # masking

plt.figure()
plt.imshow(img)
plt.show()

img[mask] += color[mask] / 255.  # adding rendering

plt.figure()
plt.imshow(img)
plt.show()
