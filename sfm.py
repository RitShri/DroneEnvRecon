import cv2
import numpy as np
import collections
import json

import CalibrationHelpers as calib
from ARImagePoseTracker import ProjectPoints, ComputePoseFromHomography, compute_fundamental, compute_fundamental_normalized, FilterByEpipolarConstraint

feature_detector = cv2.BRISK_create(octaves=5)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

fisheye = cv2.VideoCapture('./data/recording5/fisheye_video.avi')
DIM=(960, 540)
K, D, roi, new_intrinsics = calib.LoadCalibrationData('calib')

first_frame = None
ff = None

count = collections.Counter()
num_frames = 0
while(fisheye.isOpened()):
    ret, frame = fisheye.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nk = K.copy()
        nk[0,0]=K[0,0]/2
        nk[1,1]=K[1,1]/2
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, DIM, cv2.CV_32FC1)
        undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # sets the first frame as my (0,0)
        if ff == None:
            print(True)
            first_frame = undistorted_img
            ff = True
            reference_keypoints, reference_descriptors = feature_detector.detectAndCompute(first_frame, None)
        else:
            current_keypoints, current_descriptors = feature_detector.detectAndCompute(undistorted_img, None)
            matches = matcher.match(reference_descriptors, current_descriptors)
            
            # epipolar constraints
            # calculating depth from feature matches.
            
#            # I will be using feature tracks
#            for i in matches:
#                count[i.queryIdx] += 1
#
#            feature_tracks = []
#            for i in count.most_common():
#                # if a feature has shown up in atleast half of the frames
#                if i[1] >= num_frames / 2: # number of frames / 2
#                    feature_tracks.append(i[0])
#
#            # I want to get the depth of the points. Find the Z value and plot it.
#            FilterByEpipolarConstraint
            
            #
#        num_frames += 1
    else:
        print('Cant read the video , Exit!')
        break
    keyCode = cv2.waitKey(25) & 0xFF
    # Stop the program on the ESC key or 'q'
    if keyCode == 27 or keyCode == ord('q'):
        break
fisheye.release()
cv2.destroyAllWindows()
