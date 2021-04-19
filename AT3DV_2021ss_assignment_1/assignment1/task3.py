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

flip_ori_matrix = np.identity(4)
flip_ori_matrix[1, 1] = -1

flip_mesh_matrix = np.identity(4)
flip_mesh_matrix[0, 0] = -1

r_euler = R.from_euler('x', 180, degrees=True)
r_matrix = r_euler.as_matrix()
rot_x = np.identity(4)
rot_x[:3, :3] = r_matrix

# factor_mesh

factor_mesh = np.matmul(flip_ori_matrix, rot_x)
factor_mesh = np.matmul(factor_mesh, flip_mesh_matrix)
print(factor_mesh)

# modify mesh_pose using factor

mesh_pose = np.matmul(factor_mesh, mesh_pose)

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

mask2 = np.array(color < 1)

####  step 2. Augment the rendred object on the image using mask.

img = plt.imread("data_for_task3/0.png")[:, :, :3]

# mask out image and add rendring **without for loop**.

img = img * mask2

plt.figure()
plt.imshow(img)
plt.show()
