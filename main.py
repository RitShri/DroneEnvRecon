import cv2
import numpy as np

#import CalibrationHelpers as calib
#
#from ARImagePoseTracker import ProjectPoints, ComputePoseFromHomography

# TODO: add a terminal script to input which recording to read
fisheye = cv2.VideoCapture('./data/recording4/fisheye_video.avi')
zed = cv2.VideoCapture('./data/recording4/zed4.avi')

DIM=(640, 480)
## 160 Fisheye
#K=np.array([[395.2474957410931, 0.0, 313.5019461730335], [0.0, 527.3954916199217, 196.37742657771022], [0.0, 0.0, 1.0]])
#D=np.array([[-0.03798599445910202], [0.020273631739096985], [-0.00890488243952009], [-0.03915599454217283]])

# 200 Fisheye
K=np.array([[203.89411335875053, 0.0, 317.91346556362936],
            [0.0, 271.52940003242463, 240.70402221094517],
            [0.0, 0.0, 1.0]])
D=np.array([[-0.024122176699239672], [0.011100015685128415], [-0.005106773595069454], [0.0002034975302124561]])


while(fisheye.isOpened() and zed.isOpened()):
#while(zed.isOpened()):
    ret, frame = fisheye.read()
    zedret, zedframe = zed.read()
    
    if ret and zedret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zedgray = cv2.cvtColor(zedframe, cv2.COLOR_BGR2GRAY)
        
        # TODO: need to undistort the fisheye here
#        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), balance=0)
#        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_32FC1)
        # and then remap:
#        undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


#        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
#        undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#        cv2.imshow("undistorted", undistorted_img)
#        cv2.imshow("Distorted", gray)
        
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
        
        #reference then current
        matches = matcher.match(z_reference_descriptors, f_reference_descriptors)
        match_visualization = cv2.drawMatches(
            zedreference, z_reference_keypoints,
            fisheyereference,
            f_reference_keypoints, matches, 0,
            # matchesMask =inlier_mask, #this applies your inlier filter
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        match_visualization = cv2.drawMatches(zedreference, z_reference_keypoints, fisheyereference,
            f_reference_keypoints, matches, 0, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
        cv2.imshow('EpipolarConstraint',match_visualization)
    else:
        print('Cant read the video , Exit!')
        break
    keyCode = cv2.waitKey(25) & 0xFF
    # Stop the program on the ESC key or 'q'
    if keyCode == 27 or keyCode == ord('q'):
        break
        
# may want to apply later
def FilterByEpipolarConstraint(intrinsics, matches, points1, points2, Rx1, Tx1,
                               threshold = 0.01):
    E = np.cross(Tx1,Rx1,axisa=0,axisb=0)
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    
    inlier_mask = []
    
    for i in matches:
        #print(i.imgIdx, i.trainIdx, i.queryIdx)
        u_v1 = points1[i.queryIdx]
        u_v2 = points2[i.trainIdx]

        (u1,v1) = u_v1.pt
        (u2,v2) = u_v2.pt
        
        x1 = np.array([(u1 - cx)/fx, (v1 - cy)/fy,1])
        x2 = np.array([(u2 - cx)/fx, (v2 - cy)/fy,1])
        
        m = (abs(x2.T @ E @ x1) < threshold).astype(int)
        
        inlier_mask.append(m)

    return np.array(inlier_mask)
 
fisheye.release()
zed.release()
cv2.destroyAllWindows()
