import cv2
import numpy as np
import json

import CalibrationHelpers as calib
from ARImagePoseTracker import ProjectPoints, ComputePoseFromHomography, compute_fundamental, compute_fundamental_normalized

fisheye = cv2.VideoCapture('./data/recording5/fisheye_video.avi')
DIM=(960, 540)
K, D, roi, new_intrinsics = calib.LoadCalibrationData('calib')

first_frame = None
while(fisheye.isOpened()):
    ret, frame = fisheye.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nk = K.copy()
        nk[0,0]=K[0,0]/2
        nk[1,1]=K[1,1]/2
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, DIM, cv2.CV_32FC1)
        undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        if first_frame.all() == None:
            print(True)
            first_frame = undistorted_img
        
        cv2.imshow("gray", undistorted_img)
    else:
        print('Cant read the video , Exit!')
        break
    keyCode = cv2.waitKey(25) & 0xFF
    # Stop the program on the ESC key or 'q'
    if keyCode == 27 or keyCode == ord('q'):
        break
fisheye.release()
cv2.destroyAllWindows()
