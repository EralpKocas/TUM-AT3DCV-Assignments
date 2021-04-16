import pyrender,trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# scene
scene = pyrender.Scene(bg_color=[0,0,0], ambient_light=[1.0,1.0,1.0])
r = pyrender.OffscreenRenderer(640,480)

# light
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)

scene.add(light,pose=np.identity(4))

# lumbar mesh
trimesh_lumbar = trimesh.load('dataset_for_task2/SK55M_texturized_bonemesh.ply')
lumbar_mesh = pyrender.Mesh.from_trimesh(trimesh_lumbar)
lumbar_node = scene.add(lumbar_mesh)

# tip marker mesh
sm = trimesh.creation.uv_sphere(radius=0.0075)
sm.visual.vertex_colors = [1.0, 0.0, 0.0]
tip_mesh = pyrender.Mesh.from_trimesh(sm)
tip_node = scene.add(tip_mesh)

######################################################################
###     To do 1: Setup intirnsic matrix K obtained from task 1     ###
######################################################################

# camera intrinsic K from task 1

# K = 




camera = pyrender.IntrinsicsCamera(fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],znear=0.001,zfar=3)
scene.add(camera,pose=np.identity(4))

######################################################################
###   To do 3: Calculate "lumbar_to_world_camera" pose and         ###
###            "tip_to_world_camera" pose                          ###
######################################################################

# lumbar_to_marker_pose
lumbar_to_marker = np.loadtxt("dataset_for_task2/pose_lumbar/0.txt",delimiter=" ")
# marker to world pose
lumbar_marker_to_world_cam = np.loadtxt("dataset_for_task2/pose_phantom/0.txt",delimiter=" ")

###  step 1. calculate "lumbar_to_to_world_cam" pose 




scene.set_pose(lumbar_node,pose=lumbar_to_world_cam)


# tooltip_to_marker_pose
tooltip_to_marker = np.loadtxt("dataset_for_task2/pose_tooltip/0.txt",delimiter=" ")
# marker to world pose
tooltip_marker_to_world_cam = np.loadtxt("dataset_for_task2/pose_tool/0.txt",delimiter=" ")

###  step 1. calculate "tooltip_to_to_world_cam" pose 




scene.set_pose(tip_node,pose=tooltip_to_world_cam)

color,depth = r.render(scene)

# load image
img = plt.imread("dataset_for_task2/rgb_full/0.png")[:,:,:3]

##############################################################################
### To do 4 : augment the rendered obj on the image ** WITHOUT FOR LOOP ** ###
##############################################################################

# mask = 
# img_aug =

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_aug)
plt.show()

