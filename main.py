import cv2
import numpy as np

#import CalibrationHelpers as calib
#
#from ARImagePoseTracker import ProjectPoints, ComputePoseFromHomography

# TODO: add a terminal script to input which recording to read
fisheye = cv2.VideoCapture('./data/recording2/fisheye_video.avi')
zed = cv2.VideoCapture('./data/recording2/test.avi')

DIM=(640, 480)
K=np.array([[395.2474957410931, 0.0, 313.5019461730335], [0.0, 527.3954916199217, 196.37742657771022], [0.0, 0.0, 1.0]])
D=np.array([[-0.03798599445910202], [0.020273631739096985], [-0.00890488243952009], [-0.03915599454217283]])

while(fisheye.isOpened() and zed.isOpened()):
#while(zed.isOpened()):
    ret, frame = fisheye.read()
    zedret, zedframe = zed.read()
    
    if ret and zedret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zedgray = cv2.cvtColor(zedframe, cv2.COLOR_BGR2GRAY)
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("undistorted", undistorted_img)
        cv2.imshow("Distorted", gray)
        
        # TODO: add a terminal script to change the resolution from terminal
#        RES = 480
        fisheyereference = cv2.resize(frame,DIM)
        zedreference = cv2.resize(zedframe,(640*2, 480))
        # For Visualizations
#        cv2.imshow('frame', gray)
#        cv2.imshow('zedframe', zedgray)
        feature_detector = cv2.BRISK_create(octaves=5)
        f_reference_keypoints, f_reference_descriptors = feature_detector.detectAndCompute(fisheyereference, None)
        z_reference_keypoints, z_reference_descriptors = feature_detector.detectAndCompute(zedreference, None)
        
        f_keypoint_visualization = cv2.drawKeypoints(fisheyereference,f_reference_keypoints,outImage=np.array([]),flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        z_keypoint_visualization = cv2.drawKeypoints(zedreference,z_reference_keypoints,outImage=np.array([]),flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        #Wondering if the reshaping will cause some affine transformations
        # For Visualizations
#        cv2.imshow("Keypoints",z_keypoint_visualization)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
