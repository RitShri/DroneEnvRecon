import cv2
import numpy as np
import json

import CalibrationHelpers as calib
from ARImagePoseTracker import ProjectPoints, ComputePoseFromHomography, compute_fundamental, compute_fundamental_normalized, eight_point_algorithm

num_image = 0
max_image = 8
fisheye = cv2.imread("calib_fish/"+"calib_image_fish_" + str(num_image) + ".png")
zed = cv2.imread("calib_zed/"+"calib_image_zed_" + str(num_image) + ".png")

DIM=(960, 540)
K, D, roi, new_intrinsics = calib.LoadCalibrationData('calib')
f = open('calib/zed_left_calib.json')
zed_calib = json.load(f)
K_zed_l, D_zed_l = zed_calib['K'], zed_calib['D']
f.close()

while(num_image<max_image):
    num_image += 1

    frame = fisheye
    zedframe = zed

    fisheye = cv2.imread("calib_fish/"+"calib_image_fish_" + str(num_image) + ".png")
    zed = cv2.imread("calib_zed/"+"calib_image_zed_" + str(num_image) + ".png")
    
    if ret and zedret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zedgray = cv2.cvtColor(zedframe, cv2.COLOR_BGR2GRAY)
        
#        cv2.imshow("Distorted", gray)
        
        # based on online solution for fisheye calibration
        nk = K.copy()
        nk[0,0]=K[0,0]/2
        nk[1,1]=K[1,1]/2
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, DIM, cv2.CV_32FC1)
        undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#        # once we have the undistorted fisheye camera we can get the R* and T* between the fishey
        fisheyereference = cv2.resize(undistorted_img,DIM)
        zedreference = cv2.resize(zedgray,(960*2, 540))
        zed_left = zedreference[:, 0:960]
        zed_right = zedreference[:, 960:]
#
        ret_fisheye, corners_fisheye = cv2.findChessboardCorners(fisheyereference, (9,6))
        ret_zed, corners_zed = cv2.findChessboardCorners(zed_left, (9,6))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        imgpoints_fish = []
        imgpoints_zed = []

        if ret_fisheye and ret_zed:
            print(len(corners_fisheye), len(corners_zed))
            corners2_fish = cv2.cornerSubPix(fisheyereference,corners_fisheye,(11,11),(-1,-1),criteria)
            imgpoints_fish.append(corners2_fish)
#
            corners2_zed = cv2.cornerSubPix(zed_left,corners_zed,(11,11),(-1,-1),criteria)
            imgpoints_zed.append(corners2_zed)
#
#            # Draw and display the corners
#            # print(corners2_fish, corners2_zed)
#            # img_fish = cv2.drawChessboardCorners(fisheyereference, (6,9), corners2_fish,ret_fisheye)
#            # cv2.imshow('Chess Corners Fisheye',img_fish)
#
#            # img_zed = cv2.drawChessboardCorners(zed_left, (6,9), corners2_zed,ret_zed)
#            # cv2.imshow('Chess Corners Zed',img_zed)
#            # cv2.waitKey(500)

            Rs, Ts = eight_point_algorithm(corners2_fish, corners2_zed, K, K_zed_l)
            print(Rs, Ts)
            
            F = compute_fundamental(corners_fisheye, corners_zed)
#            # E = np.dot(np.dot(K_prime.T,F),K)
            print(F)
#
#
#
#       
    else:
        print('Cant read the video , Exit!')
        break
    keyCode = cv2.waitKey(25) & 0xFF
    # Stop the program on the ESC key or 'q'
    if keyCode == 27 or keyCode == ord('q'):
        break

 
fisheye.release()
zed.release()
cv2.destroyAllWindows()
