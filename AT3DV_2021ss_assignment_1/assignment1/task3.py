import pyrender, trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# scene
scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
r = pyrender.OffscreenRenderer(640, 480)

# camera
K = np.array([[572.411,       0,  325.26],
              [      0, 573.570, 242.048],
              [      0,       0,       1]])
   
camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.001, zfar=3)
scene.add(camera)

# mesh
trimesh_obj = trimesh.load('data_for_task3/cat/cat.ply')
mesh = pyrender.Mesh.from_trimesh(trimesh_obj)

# mesh pose
mesh_pose = np.identity(4)
mesh_pose[:3, :] = np.loadtxt("data_for_task3/0.txt", delimiter=" ")

##########################################################################
###  To do : Figure out transform factor between Linemod and Pyrender  ###
##########################################################################

####  step 1. figure out transform factor between Pyrender and Unity
####          and modify the mesh pose 
####          (hint : Unity's orientation is already mentioned in the slides.
####                  Modify transform factor according to Pyrender's orientation)

# factor_pose


r_euler = R.from_euler('x', 180, degrees=True)
r_matrix = r_euler.as_matrix()
rot_x = np.identity(4)
rot_x[:3, :3] = r_matrix

# factor_mesh

elt_wise = np.array([[-1, 1, 1, 1],
                     [1, -1, -1, -1],
                     [-1, 1, 1, 1],
                     [-1, 1, 1, 1]])

# modify mesh_pose using factor

mesh_pose = np.matmul(rot_x, mesh_pose)
mesh_pose = np.multiply(elt_wise, mesh_pose)

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

img = plt.imread("data_for_task3/0.png")[:, :, :3]

# mask out image and add rendring **without for loop**.

img[mask] = 0  # masking

plt.figure()
plt.imshow(img)
plt.show()

img[mask] += color[mask] / 255.  # adding rendering

plt.figure()
plt.imshow(img)
plt.show()
