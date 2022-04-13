import cv2
import numpy as np
import json

import CalibrationHelpers as calib
from ARImagePoseTracker import find_3d_points

def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        second = np.append(second,1)
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, first[1] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True

num_image = 0
max_image = 52 #change this num
fisheye = cv2.imread("calib_fish/"+"calib_image_fish_" + str(num_image) + ".png")
zed = cv2.imread("calib_zed/"+"calib_image_zed_" + str(num_image) + ".png")

DIM=(960, 540)
K, D, roi, new_intrinsics = calib.LoadCalibrationData('calib')
K_inv_fish = np.linalg.inv(K)
#print(roi)
f = open('calib/zed_left_calib.json')
zed_calib = json.load(f)
K_zed_l, D_zed_l = zed_calib['K'], zed_calib['D']
K_inv_zed = np.linalg.inv(K_zed_l)
f.close()

r_t_dict =  {}
best_error = float("inf")
bestR = None
bestT = None
while(num_image<max_image):
    num_image += 1
    
    frame = fisheye
    zedframe = zed
    
    fisheye = cv2.imread("calib_fish/"+"calib_image_fish_" + str(num_image) + ".png")
    zed = cv2.imread("calib_zed/"+"calib_image_zed_" + str(num_image) + ".png")
        
    #print(fisheye)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    zedgray = cv2.cvtColor(zedframe, cv2.COLOR_BGR2GRAY)


#        cv2.imshow("Distorted", gray)

# based on online solution for fisheye calibration
    nk = K.copy()
    nk[0,0]=K[0,0]/2
    nk[1,1]=K[1,1]/2
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_32FC1)
    undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#        # once we have the undistorted fisheye camera we can get the R* and T* between the fishey
    #print(undistorted_img.shape)
    #undistorted_img = undistorted_img[roi[0]:roi[2]][roi[1]:roi[3]]
    
    fisheyereference = cv2.resize(undistorted_img,DIM)
    zedreference = cv2.resize(zedgray,(960, 540))
    zed_left = zedreference[:, 0:960]
    zed_right = zedreference[:, 960:]
#
    #cv2.imshow("fish", fisheyereference)
    #cv2.imshow("fish_undistort", fisheye)
    #cv2.imshow("zed", zed_left)
    flags = 0
    flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
    flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    #print(flags)
    ret_fisheye, corners_fisheye = cv2.findChessboardCorners(fisheye, (6,9))
    ret_zed, corners_zed = cv2.findChessboardCorners(zedreference, (6,9))
       
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    print("fish", ret_fisheye)
    print("zed", ret_zed)    
    if ret_fisheye and ret_zed:
        print(len(corners_fisheye), len(corners_zed))
        corners2_fish = cv2.cornerSubPix(fisheyereference,corners_fisheye,(11,11),(-1,-1),criteria)
        
        corners2_zed = cv2.cornerSubPix(zed_left,corners_zed,(11,11),(-1,-1),criteria)
        
        #F, mask = cv2.findFundamentalMat(corners_fisheye, corners_zed, cv2.FM_LMEDS)
        F, mask = cv2.findFundamentalMat(corners_fisheye, corners_zed, cv2.FM_RANSAC, 0.1, 0.99)
        E = np.dot(np.dot(np.array(K_zed_l).T, F), K)
#       # E = np.dot(np.dot(K_prime.T,F),K)
        #print(E)
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the fundamental matrix
        first_inliers = []
        second_inliers = []
        corners_fisheye = corners_fisheye.reshape((54,2))
        corners_zed = corners_zed.reshape((54,2))

        first_inliers = []
        second_inliers = []
        for i in range(len(mask)):
            if mask[i]:
                # normalize and homogenize the image coordinates
                first_inliers.append(K_inv_fish.dot([corners_fisheye[i][0], corners_fisheye[i][1], 1.0]))
                second_inliers.append(K_inv_zed.dot([corners_zed[i][0], corners_zed[i][1], 1.0]))
        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in front of both cameras
        # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not in_front_of_both_cameras(corners_fisheye, corners_zed, R, T):

            # Second choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]
            if not in_front_of_both_cameras(corners_fisheye, corners_zed, R, T):

                # Third choice: R = U * Wt * Vt, T = u_3
                R = U.dot(W.T).dot(Vt)
                T = U[:, 2]

                if not in_front_of_both_cameras(corners_fisheye, corners_zed, R, T):

                    # Fourth choice: R = U * Wt * Vt, T = -u_3
                    T = - U[:, 2]
            #print("R", R)
            #print("T", T)
            points1, points2 = corners_fisheye, corners_zed
            
            P1 = np.zeros((3,4))
            P1[:3, :3] = np.eye(3)
            P1 = K @ P1
            temp = np.zeros((3,4))
            temp[:3, :3] = R
            temp[:, 3] = T
            P2 = K_zed_l @ temp
            
            point_3d, reconstrction_error = find_3d_points(points1, points2, P1, P2)
            if(reconstrction_error<best_error):
                best_error = reconstrction_error
                bestR = R
                bestT = T
        
        # Calculate the 3D point
        # Get the R and T and find the reprojection error

    '''
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, D, K_zed_l, D_zed_l, fisheye.shape[:2], R, T, alpha=1.0)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(K, D, R1, K_zed_l, fisheye.shape[:2], cv2.CV_32F)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(K_zed_l, D_zed_l, R2, K, zedreference.shape[:2], cv2.CV_32F)
    img_rect1 = cv2.remap(first_img, mapx1, mapy1, cv2.INTER_LINEAR)
    img_rect2 = cv2.remap(second_img, mapx2, mapy2, cv2.INTER_LINEAR)

    # draw the images side by side
    total_size = (max(img_rect1.shape[0], img_rect2.shape[0]), img_rect1.shape[1] + img_rect2.shape[1], 3)
    img = np.zeros(total_size, dtype=np.uint8)
    img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
    img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

    # draw horizontal lines every 25 px accross the side by side image
    for i in range(20, img.shape[0], 25):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

    cv2.imshow('rectified', img)
    cv2.waitKey(0)
    '''

    # Stop the program on the ESC key or 'q'
    keyCode = cv2.waitKey(10)
    if keyCode == 27 or keyCode == ord('q'):
        break

cv2.destroyAllWindows()
print(bestR)
print(bestT)
print(best_error)
r_t_dict["R"] = bestR.tolist()
r_t_dict["T"] = bestT.tolist()
r_t_dict["ERR"] = best_error
json_data = {}
json_data['R_T_DICT'] = r_t_dict
json_string = json.dumps(json_data)

with open('zed_time.json', 'w') as  outfile:
     outfile.write(json_string)
