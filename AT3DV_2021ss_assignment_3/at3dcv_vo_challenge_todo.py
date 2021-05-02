#Only intentend for academic use for the course "Advanced Topics in 3D Computer Vision" at TUM in SS2021
#Do not distribute or share!!!

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
#HINT: Needed for Pose Graph Optimization
#from posegraphoptimizer import PoseGraphOptimizer, getGraphNodePose

# Util function
def T_from_R_t(R, t):
    R = np.array(R).reshape(3, 3)
    t = np.array(t).reshape(3)
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[0, 3] = t[0]
    T[1, 3] = t[1]
    T[2, 3] = t[2]
    T[3, 3] = 1
    return T


def draw_feature_tracked(second_frame, first_frame,
                        second_keypoints, first_keypoints,
                        color_line=(0, 255, 0), color_circle=(255, 0, 0)):
    mask_bgr = np.zeros_like(cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR))
    frame_bgr = cv2.cvtColor(second_frame, cv2.COLOR_GRAY2BGR)
    for i, (second, first) in enumerate(zip(second_keypoints, first_keypoints)):
        a, b = second.ravel()
        c, d = first.ravel()
        mask_bgr = cv2.line(mask_bgr, (int(a), int(b)), (int(c), int(d)), color_line, 1)
        frame_bgr = cv2.circle(frame_bgr, (int(a), int(b)), 3, color_circle, 1)
    return cv2.add(frame_bgr, mask_bgr)


def getGT(file_context, frame_id):
    ss = file_context[frame_id].strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    return [x, y, z]


class Camera:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


# Major functions for VO computation
class VO:
    def __init__(self, camera):
        self.camera = camera
        self.focal = self.camera.fx
        self.center = (self.camera.cx, self.camera.cy)

        self.curr_R = None
        self.curr_t = None

        self.T = None
        self.relative_T = None

        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    def featureTracking(self, curr_frame, old_frame, old_kps):
        # ToDo
        # Not: There is a optical flow method in OpenCV that can help ;) input the old_kps and track them

        opt_flow_params = dict(winSize=(25, 25),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 0.001))
        curr_kps, matches, err = cv2.calcOpticalFlowPyrLK(old_frame, curr_frame, old_kps, None, **opt_flow_params)

        ###

        # Remove nono-matched keypoints
        matches = matches.reshape(matches.shape[0])
        return curr_kps[matches == 1], old_kps[matches == 1], matches

    def featureMatching(self, curr_frame, old_frame, orb=True):
        if orb:
            # ToDo
            # Hint: again, OpenCV is your friend ;) Tip: maybe you want to improve the feature matching by only taking the best matches...
            orb_det = cv2.ORB_create()
            kp1, des1 = orb_det.detectAndCompute(curr_frame, None)
            kp2, des2 = orb_det.detectAndCompute(old_frame, None)

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)

            matches = sorted(matches, key=lambda x: x.distance)

            ###
        else:  # use SIFT
            # ToDo
            # Hint: Have you heared about the Ratio Test for SIFT?

            sift_det = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift_det.detectAndCompute(curr_frame, None)
            kp2, des2 = sift_det.detectAndCompute(old_frame, None)

            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])

            matches = good
            ###
        if orb:
            kp1_match = np.array([kp1[mat.queryIdx].pt for mat in matches])
            kp2_match = np.array([kp2[mat.trainIdx].pt for mat in matches])
        else:
            kp1_match = np.array([kp1[mat[0].queryIdx].pt for mat in matches])
            kp2_match = np.array([kp2[mat[0].trainIdx].pt for mat in matches])

        return kp1_match, kp2_match, matches

    def initialize(self, first_frame, sceond_frame, of=True, orb=False):
        if of:
            first_keypoints = self.detector.detect(first_frame)
            first_keypoints = np.array([x.pt for x in first_keypoints], dtype=np.float32)
            second_keypoints_matched, first_keypoints_matched, _ = self.featureTracking(sceond_frame, first_frame,
                                                                                        first_keypoints)
        else:
            second_keypoints_matched, first_keypoints_matched, _ = self.featureMatching(sceond_frame, first_frame,
                                                                                        orb=orb)

        # ToDo
        # Hint: Remember the lecture: given the matched keypoints you can compute the Essential matrix and from E you can recover R and t...
        camera_K = np.identity(3)
        camera_K[0, 0] = self.camera.fx
        camera_K[1, 1] = self.camera.fy
        camera_K[0, 2] = self.camera.cx
        camera_K[1, 2] = self.camera.cy
        E, mask = cv2.findEssentialMat(second_keypoints_matched, first_keypoints_matched, camera_K,
                                            cv2.RANSAC, 0.999, 1.0, 1000)

        output = cv2.recoverPose(E, second_keypoints_matched, first_keypoints_matched, camera_K, mask=mask)
        self.curr_R = output[1]
        self.curr_t = output[2]
        ###

        self.relative_T = T_from_R_t(self.curr_R, self.curr_t)
        self.T = self.relative_T
        return second_keypoints_matched, first_keypoints_matched

    def processFrame(self, curr_frame, old_frame, old_kps, of=True, orb=False):

        if of:
            curr_kps_matched, old_kps_matched, matches = self.featureTracking(curr_frame, old_frame,
                                                                                           old_kps)
        else:
            curr_kps_matched, old_kps_matched, matches = self.featureMatching(curr_frame, old_frame,
                                                                                           orb=orb)

        # ToDo
        # Hint: Here we only do the naive way and do everything based on Epipolar Geometry (Essential Matrix). No need for PnP in this tutorial
        camera_K = np.identity(3)
        camera_K[0, 0] = self.camera.fx
        camera_K[1, 1] = self.camera.fy
        camera_K[0, 2] = self.camera.cx
        camera_K[1, 2] = self.camera.cy
        E, mask = cv2.findEssentialMat(curr_kps_matched, old_kps_matched, camera_K,
                                       cv2.RANSAC, 0.999, 1.0, 1000)
        mask = np.multiply(mask, 255)
        output = cv2.recoverPose(E, curr_kps_matched, old_kps_matched, camera_K, mask=mask)

        R = output[1]
        t = output[2]
        ###

        inliners = len(mask[mask == 255])

        if (inliners > 20):
            self.relative_T = T_from_R_t(R, t)
            self.curr_t = self.curr_t + self.curr_R.dot(t)
            self.curr_R = R.dot(self.curr_R)
            self.T = T_from_R_t(self.curr_R, self.curr_t)

        # Get new KPs if too few
        if (old_kps_matched.shape[0] < 1000):
            curr_kps_matched = self.detector.detect(curr_frame)
            curr_kps_matched = np.array([x.pt for x in curr_kps_matched], dtype=np.float32)
        return curr_kps_matched, old_kps_matched



def main():
    argument = argparse.ArgumentParser()
    argument.add_argument("--o", help="use ORB", action="store_true")
    argument.add_argument("--f", help="use Optical Flow", action="store_false")
    argument.add_argument("--l", help="use Loop Closure for PGO", action="store_true")
    args = argument.parse_args()
    orb = args.o
    of = args.f
    loop_closure = args.l

    #Hard-coded Loop closure estimates (Needed for PGO); We only take these 2 for now
    lc_ids = [1572, 3529]
    lc_dict = {1572: 125, 3529: 553}

    #HINT: Adapt path
    image_dir = os.path.realpath("../../dataset/kitti/00/image_0/")
    pose_path = os.path.realpath("../../dataset/kitti/00.txt")

    with open(pose_path) as f:
        poses_context = f.readlines()

    image_list = []
    for file in os.listdir(image_dir):
        if file.endswith("png"):
            image_list.append(image_dir + '/' + file)

    image_list.sort()

    # Initial VisualOdometry Object
    camera = Camera(1241.0, 376.0, 718.8560,
                    718.8560, 607.1928, 185.2157)
    vo = VO(camera)
    traj_plot = np.zeros((1000,1000,3), dtype=np.uint8)

    # ToDo (PGO)
    #Hint: Initialize Pose Graph Optimizer
    # Hint: have a look in the PGO class and what methods are provided. The first frame should be static (addPriorFactor)


    ###


    first = 0
    second = first + 3  # For wider baseline with better initialization...
    first_frame = cv2.imread(image_list[first], 0)
    second_frame = cv2.imread(image_list[second], 0)

    second_keypoints, first_keypoints = vo.initialize(first_frame, second_frame, of=of, orb=orb)

    # ToDo (PGO)
    # Hint: fill the Pose Graph: There is a difference between the absolute pose and the relative pose



    ###


    old_frame = second_frame
    old_kps = second_keypoints

    for index in range(second+1, len(image_list)):
        curr_frame = cv2.imread(image_list[index], 0)
        true_pose = getGT(poses_context, index)
        true_x, true_y = int(true_pose[0])+290, int(true_pose[2])+90

        curr_kps, old_kps = vo.processFrame(curr_frame, old_frame, old_kps, of=of, orb=orb)

        # ToDo (PGO)
        # Hint: keep filling new poses



        ###

        if loop_closure:
            if index in lc_ids:
                loop_idx = lc_dict[index]
                print("Loop: ", PGO.curr_node_idx, loop_idx)

                # ToDo (PGO)
                # Hint: just use Identity pose for Loop Closure np.eye(4)



                ###

                #Plot trajectory after PGO
                for k in range(index):
                    try:
                        pose_trans, pose_rot = getGraphNodePose(PGO.graph_optimized, k)
                        print(pose_trans)
                        print(pose_rot)
                        cv2.circle(traj_plot, (int(pose_trans[0])+290, int(pose_trans[2])+90), 1, (255, 0, 255), 5)
                    except:
                        #catch error for first few missing poses...
                        print("Pose not available for frame # ", k)


        #Utilities for Drawing
        curr_t = vo.curr_t

        if(index > 2):
            x, y, z = curr_t[0], curr_t[1], curr_t[2]
        else:
            x, y, z = 0., 0., 0.
        odom_x, odom_y = int(x)+290, int(z)+90

        cv2.circle(traj_plot, (odom_x,odom_y), 1, (index*255/4540,255-index*255/4540,0), 1)
        cv2.circle(traj_plot, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj_plot, (10, 20), (600, 60), (0,0,0), -1)
        text = "FrameID: %d  Coordinates: x=%1fm y=%1fm z=%1fm"%(index,x,y,z)
        cv2.putText(traj_plot, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.imshow('Trajectory', traj_plot)
        show_image = draw_feature_tracked(curr_frame, old_frame,
                                         curr_kps, old_kps)
        cv2.imshow('Mono', show_image)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break



        # Update old data
        old_frame = curr_frame
        old_kps = curr_kps

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
