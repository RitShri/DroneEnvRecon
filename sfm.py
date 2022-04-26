import cv2
import numpy as np
import collections
import json
import open3d as o3d

import CalibrationHelpers as calib
from ARImagePoseTracker import ProjectPoints, ComputePoseFromHomography, compute_fundamental, compute_fundamental_normalized, FilterByEpipolarConstraint, get_M, in_front_of_both_cameras

feature_detector = cv2.BRISK_create(octaves=5)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

fisheye = cv2.VideoCapture('./data/recording5/fisheye_video.avi')
DIM=(960, 540)
K, D, roi, new_intrinsics = calib.LoadCalibrationData('calib')
K_inv = np.linalg.inv(K)

first_frame = None
ff = None
    
count = collections.Counter()
num_frames = 0
matrix_matches = {}
rotations = []
translations = []
ref_r = None
ref_t = None
reference_keypoints = None
while(fisheye.isOpened()):
    ret, frame = fisheye.read()
    
    if ret:
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        nk = K.copy()
        nk[0,0]=K[0,0]/2
        nk[1,1]=K[1,1]/2
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, DIM, cv2.CV_32FC1)
        undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        
        # sets the first frame as my (0,0)
        if ff == None:
            first_frame = undistorted_img
            ff = True
            reference_keypoints, reference_descriptors = feature_detector.detectAndCompute(first_frame, None)
            matches = matcher.match(reference_descriptors, reference_descriptors)

            referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
             # convert positions from pixels to meters
            SCALE = 1 #TODO: this is the scale of our reference image: 0.1m x 0.1m
            RES = 1 #TODO: 
            referencePoints = SCALE*referencePoints/RES
        
            # compute homography
            ret, ref_r, ref_t = ComputePoseFromHomography(new_intrinsics,referencePoints,
                                            referencePoints)
        else:
            current_keypoints, current_descriptors = feature_detector.detectAndCompute(undistorted_img, None)
            matches = matcher.match(reference_descriptors, current_descriptors)

            referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
             # convert positions from pixels to meters
            SCALE = 1 #TODO: this is the scale of our reference image: 0.1m x 0.1m
            RES = 1 #TODO: 
            referencePoints = SCALE*referencePoints/RES
        
            imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                    for m in matches])
            # compute homography
#            ret_new, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
#                                            imagePoints)
            F, mask = cv2.findFundamentalMat(referencePoints, imagePoints, cv2.FM_RANSAC, 0.1, 0.99)
            E = K.T.dot(F).dot(K)
            U, S, Vt = np.linalg.svd(E)
            W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
            
            first_inliers = []
            second_inliers = []
            for i in range(len(mask)):
                if mask[i]:
                    # normalize and homogenize the image coordinates
                    first_inliers.append(K_inv.dot([referencePoints[i][0], referencePoints[i][1], 1.0]))
                    second_inliers.append(K_inv.dot([imagePoints[i][0], imagePoints[i][1], 1.0]))
                    
            R = U.dot(W).dot(Vt)
            T = U[:, 2]
            
            if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                # Second choice: R = U * W * Vt, T = -u_3
                T = - U[:, 2]
#                print(num_frames, "Reached 1")
                if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
#                    print(num_frames, "Reached 2")
                    # Third choice: R = U * Wt * Vt, T = u_3
                    R = U.dot(W.T).dot(Vt)
                    T = U[:, 2]

                    if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
#                        print(num_frames, "Reached 3")
                        # Fourth choice: R = U * Wt * Vt, T = -u_3
                        T = - U[:, 2]
                        
            ret_new = in_front_of_both_cameras(first_inliers, second_inliers, R, T)
            if ret_new:
                relative_rotation = np.matmul(R, ref_r.T)
                relative_translation = T - np.dot(np.matmul(R, ref_r.T), ref_t)

                # epipolar constraints
                # calculating depth from feature matches.
                for i in matches:
                   count[i.queryIdx] += 1

#                feature_tracks = []
#                for i in count.most_common():
#                    # if a feature has shown up in atleast half of the frames
#                    if i[1] >= num_frames / 2: # number of frames / 2
#                        feature_tracks.append(i[0])

                match1 = []

                for i in matches:
#                    if i.queryIdx in feature_tracks:
                    match1.append(i)
                    if i.queryIdx in matrix_matches:
                        matrix_matches[i.queryIdx] += [(i,reference_keypoints,current_keypoints,relative_rotation,relative_translation)]
                    else:
                        matrix_matches[i.queryIdx] = [(i,reference_keypoints,current_keypoints,relative_rotation,relative_translation)]


                inlier_mask = FilterByEpipolarConstraint(K,
                                                        match1,
                                                        reference_keypoints,
                                                        current_keypoints,
                                                        relative_rotation,
                                                        relative_translation)
                 # TODO: change the current image to the next image 
#                M = get_M(K, matrix_matches)
#                W,U,Vt = cv2.SVDecomp(M)
#                depths = Vt[-1,:]/Vt[-1,-1]
##
#                your_pointCloud = []
#                count2 = 0
#                fx = K[0][0]
#                fy = K[1][1]
#                cx = K[0][2]
#                cy = K[1][2]
#                for i in matrix_matches:
#                    (u1,v1) = (reference_keypoints[i]).pt
#                    x1 = np.array([(u1 - cx)/fx, (v1 - cy)/fy,1])
##                    x1 = np.array([u1, v1,depths[count2]])
#                    your_pointCloud.append(np.multiply(depths[count2], x1))
#                    count2 += 1
##
##
#                your_pointCloud = np.array(your_pointCloud)
##                print(np.min(your_pointCloud))
#                print(len(your_pointCloud[:,2]<100) , len(your_pointCloud[:,2]> -10))
#                your_pointCloud = your_pointCloud[(your_pointCloud[:,2]<100) & (your_pointCloud[:,2]>-10)]
##
#                #part 3.10
#                pcd = o3d.geometry.PointCloud()
#                pcd.points = o3d.utility.Vector3dVector(your_pointCloud)
##                print(len(pcd.points))
#                o3d.visualization.draw_geometries([pcd]) # want to assign color based on the pixel taken
        num_frames += 1
    else:
        print('Cant read the video , Exit!')
        break
    keyCode = cv2.waitKey(25) & 0xFF
    # Stop the program on the ESC key or 'q'
    if keyCode == 27 or keyCode == ord('q'):
        break
        
M = get_M(K, matrix_matches)
W,U,Vt = cv2.SVDecomp(M)
depths = Vt[-1,:]/Vt[-1,-1]
    
    
your_pointCloud = []
count2 = 0
fx = K[0][0]
fy = K[1][1]
cx = K[0][2]
cy = K[1][2]
for i in matrix_matches:
    (u1,v1) = (reference_keypoints[i]).pt
    x1 = np.array([(u1 - cx)/fx, (v1 - cy)/fy,1])
    your_pointCloud.append(np.multiply(depths[count2],x1))
    count2 += 1
    
your_pointCloud = np.array(your_pointCloud)
your_pointCloud = your_pointCloud[(your_pointCloud[:,2]<100) & (your_pointCloud[:,2]>-10)]


#part 3.10
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(your_pointCloud)
o3d.visualization.draw_geometries([pcd])
fisheye.release()
cv2.destroyAllWindows()


