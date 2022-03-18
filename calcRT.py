import cv2
import numpy as np

#import CalibrationHelpers as calib
#
from ARImagePoseTracker import ProjectPoints, ComputePoseFromHomography, compute_fundamental, compute_fundamental_normalized

# TODO: add a terminal script to input which recording to read
fisheye = cv2.VideoCapture('./data/recording6/fisheye_video.avi')
zed = cv2.VideoCapture('./data/recording6/zed_video.avi')

DIM=(640, 480)
## 160 Fisheye
#K=np.array([[395.2474957410931, 0.0, 313.5019461730335], [0.0, 527.3954916199217, 196.37742657771022], [0.0, 0.0, 1.0]])
#D=np.array([[-0.03798599445910202], [0.020273631739096985], [-0.00890488243952009], [-0.03915599454217283]])

# 200 Fisheye
orig_K=np.array([[205.24532788472575, 0.0, 318.9682550623967], [0.0, 273.75289634645463, 239.22518618551558], [0.0, 0.0, 1.0]])

# main difference from the original is the change in orig_K1[0,2] and orig_K1[1,2]
# which should just shift the center of the camera
shifted_K=np.array([[205.24532788472575, 0.0, 500], [0.0, 273.75289634645463, 275], [0.0, 0.0, 1.0]])
D=np.array([[-0.014415679703456747], [0.00032319201754507073], [-0.0021461891034443544], [0.00032634649968791885]])


while(fisheye.isOpened() and zed.isOpened()):
#while(zed.isOpened()):
    ret, frame = fisheye.read()
    zedret, zedframe = zed.read()
    
    if ret and zedret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zedgray = cv2.cvtColor(zedframe, cv2.COLOR_BGR2GRAY)
        
        # cv2.imshow("Distorted", gray)
        
        # print("Need to choose which calibration matrix to use")
        # Using the OpenCV result for fisheye calibration
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(orig_K, D, DIM, np.eye(3), balance=0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(orig_K, D, np.eye(3), new_K, DIM, cv2.CV_32FC1)
        undistorted_img0 = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("(0) OpenCV result for camera calibration", undistorted_img0)
        
        # based on online solution for fisheye calibration
        nk = orig_K.copy()
        nk[0,0]=orig_K[0,0]/2
        nk[1,1]=orig_K[1,1]/2
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(orig_K, D, np.eye(3), nk, DIM, cv2.CV_32FC1)
        undistorted_img1 = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("(1) Scaled OpenCV calibration result", undistorted_img1)
        
        # I manually shifted the K
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(shifted_K, D, DIM, np.eye(3), balance=0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(shifted_K, D, np.eye(3), new_K, DIM, cv2.CV_32FC1)
        undistorted_img2 = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("(2) Shifted K", undistorted_img2)
        
        nk = shifted_K.copy()
        nk[0,0]=shifted_K[0,0]/3
        nk[1,1]=shifted_K[1,1]/3
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(shifted_K, D, np.eye(3), nk, DIM, cv2.CV_32FC1)
        undistorted_img3 = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("(3) Scaled Shifted K", undistorted_img3)

#        val = input("Enter your value (options: 0,1,2,3): ")
#        val = int(val)
#        if val == 0:
#            undistorted_img = undistorted_img0
#        elif val == 1:
#            undistorted_img = undistorted_img1
#        elif val == 2:
#            undistorted_img = undistorted_img2
#        elif val == 3:
#            undistorted_img = undistorted_img3
#        else:
#            print("Enter a value between 0 and 3")
#
#        # once we have the undistorted fisheye camera we can get the R* and T* between the fisheye and zed
#        fisheyereference = cv2.resize(undistorted_img,DIM)
#        zedreference = cv2.resize(zedgray,(640*2, 480))
#        zed_left = zedreference[:, 0:640]
#        zed_right = zedreference[:, 640:]
#
#        ret_fisheye, corners_fisheye = cv2.findChessboardCorners(fisheyereference, (9,6),None)
#        ret_zed, corners_zed = cv2.findChessboardCorners(zed_left, (9,6),None)
#        # print(corners_fisheye, corners_zed)
#
#        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#        imgpoints_fish = []
#        imgpoints_zed = []
#
#        if ret_fisheye and ret_zed:
#            # print("reached here")
#            corners2_fish = cv2.cornerSubPix(fisheyereference,corners_fisheye,(11,11),(-1,-1),criteria)
#            imgpoints_fish.append(corners2_fish)
#
#            corners2_zed = cv2.cornerSubPix(zed_left,corners_zed,(11,11),(-1,-1),criteria)
#            imgpoints_zed.append(corners2_zed)
#
#            # Draw and display the corners
#            # print(corners2_fish, corners2_zed)
#            # img_fish = cv2.drawChessboardCorners(fisheyereference, (6,9), corners2_fish,ret_fisheye)
#            # cv2.imshow('Chess Corners Fisheye',img_fish)
#
#            # img_zed = cv2.drawChessboardCorners(zed_left, (6,9), corners2_zed,ret_zed)
#            # cv2.imshow('Chess Corners Zed',img_zed)
#            # cv2.waitKey(500)
#
#            F = compute_fundamental(corners_fish, corners_zed)
#            # E = np.dot(np.dot(K_prime.T,F),K)
#            print(F)
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
