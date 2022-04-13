import cv2
import numpy as np
import CalibrationHelpers as calib
import numpy as np

def find_3d_points(points1, points2, P1, P2):
    def construct_Ab(points, P):
        x, y = points
 
        A = np.array([[P[0,0] - P[2,0] * x, P[0, 1] - P[2,1] * x, P[0,2] - P[2,2] * x,  P[0,3] - P[2,3] * x],
                      [P[1,0] - P[2,0] * y, P[1, 1] - P[2,1] * y, P[1,2] - P[2,2] * y,  P[1,3] - P[2,3] * y]])

#        b = np.array([[P[0,3] - P[2,3] * x],
#                      [P[1,3] - P[2,3] * x]])
        return A #, b

    points3D = []
    rec_err = []
    for p1, p2 in zip(points1, points2):
    
        A1 = construct_Ab(p1, P1)
        A2 = construct_Ab(p2, P2)
        
#        A = np.array([A1,A2]).reshape((4,3))
#        b = np.array([b1,b2]).reshape((4,1))

        A = np.array([A1,A2]).reshape((4,4))
        u_A, s_A, vh_A = np.linalg.svd(A)
        pt3d = vh_A[-1,:]
        pt3d = pt3d / pt3d[3] # Normalize by last number in column (should be 1)
        
#        point3d = np.linalg.lstsq(A,b, rcond=None)[0]
#        point3d = np.append(point3d,1)
        point3d = pt3d
        points3D.append(point3d)
#        print(pt3d)
        
        # project the 3D points onto 2D
#        print(P1)
#        print(P2)
        
        reproj1 = P1 @ point3d
        reproj2 = P2 @ point3d
        reproj1 = reproj1[:2] / reproj1[2]
        reproj2 = reproj2[:2] / reproj2[2]
        # TODO this is not working
#        print(reproj1, p1)
#        print(reproj2, p2)

        res_1 = np.linalg.norm(reproj1[:2]-p1)
        res_2 = np.linalg.norm(reproj2[:2]-p2)
        rec_err.append(res_1)
        rec_err.append(res_2)
    
        
    return np.array(points3D), np.sum(rec_err) / len(rec_err)


def calibrate_pt(K, pts):
    K_inv = np.linalg.inv(K)

    pts_tilde = np.dot((K_inv, np.concatenate([pts, np.ones([pts.shape[0], 1])], axis=1).T)).T
    pts_tilde = pts_tilde[:, :2]

    return pts_tilde

def hat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def unhat(Ahat):
    a3 = -Ahat[0,1]
    a2 = Ahat[0,2]
    a1 = -Ahat[1,2]
    return np.array([a1, a2, a3])

def eight_point_algorithm(pts0, pts1, K_fish, K_zed):
    """
    Implement the eight-point algorithm
    Args:
        pts0 (np.ndarray): shape (num_matched_points, 2)
        pts1 (np.ndarray): shape (num_matched_points, 2)
        K (np.ndarray): 3x3 intrinsic matrix
    Returns:
        Rs (list): a list of possible rotation matrices
        Ts (list): a list of possible translation matrices
    """
    # apply k inverse to points
    pts0_tilde = calibrate_pt(K_fish, pts0)
    pts1_tilde = calibrate_pt(K_zed, pts1)

    A = np.array([[x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
                    for ((x1, y1), (x2, y2)) in zip(pts0_tilde, pts1_tilde)])
    
    # SVD
    u_A, s_A, vh_A = np.linalg.svd(A)

    # Take last column of V
    Es = vh_A[-1,:]

    # Unstack Es
    E = np.array([
            [Es[0],Es[3],Es[6]],
            [Es[1],Es[4],Es[7]],
            [Es[2],Es[5],Es[8]]
        ])

    # Re-project aka eigenvalues are 1, 1, 0
    u_E, s_E, vh_E = np.linalg.svd(E)
    s_E = np.diag((1, 1, 0)) # Re-project
    # Don't need to calculate this
    E = np.dot(np.dot(u_E,s_E),vh_E)

    def Rz(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta),  0],
            [0,             0,              1]
        ])

    Ts = [
        unhat(np.dot(np.dot(np.dot(u_E,Rz(np.pi/2)),s_E),u_E.T)),
        unhat(np.dot(np.dot(np.dot(u_E,Rz(np.pi/2)),s_E),u_E.T)),
        unhat(np.dot(np.dot(np.dot(u_E,Rz(-np.pi/2)),s_E),u_E.T)),
        unhat(np.dot(np.dot(np.dot(u_E,Rz(-np.pi/2)),s_E),u_E.T))
    ]
    Rs = [
        np.dot(np.dot(u_E,Rz(np.pi/2).T),vh_E),
        np.dot(np.dot(u_E,Rz(-np.pi/2).T),vh_E),
        np.dot(np.dot(u_E,Rz(np.pi/2).T),vh_E),
        np.dot(np.dot(u_E,Rz(-np.pi/2).T),vh_E),
    ]

    return Rs, Ts
    
def compute_fundamental(x1,x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = dot(U,dot(diag(S),V))
    
    return F/F[2,2]

def compute_fundamental_normalized(x1,x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the normalized 8 point algorithm. """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = mean(x1[:2],axis=1)
    S1 = sqrt(2) / std(x1[:2])
    T1 = array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = mean(x2[:2],axis=1)
    S2 = sqrt(2) / std(x2[:2])
    T2 = array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = dot(T2,x2)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = dot(T1.T,dot(F,T2))

    return F/F[2,2]
    
# This function is yours to complete
# it should take in a set of 3d points and the intrinsic matrix
# rotation matrix(R) and translation vector(T) of a camera
# it should return the 2d projection of the 3d points onto the camera defined
# by the input parameters    
def ProjectPoints(points3d, new_intrinsics, R, T):
    
    # your code here!
    x = points3d.shape[0]
    y = points3d.shape[1]
    points = np.zeros((x, y + 1))
    points[0:x,0:y] = points3d
    points[0:x,y] = np.ones(y+1).T

    A = np.zeros((3,4))
    A[0:3,0:3] = R
    A[0:3,3] = T.T

    points2d = np.dot(np.dot(new_intrinsics, A), points.T).T

    x = points2d.shape[0]
    y = points2d.shape[1]
    s = 1/points2d[0:x,y-1]
    
    points2d = (np.dot(points2d.T,np.diag(s)).T)[0:x,0:2]
    
    return points2d
    
# This function will render a cube on an image whose camera is defined
# by the input intrinsics matrix, rotation matrix(R), and translation vector(T)
def renderCube(img_in, new_intrinsics, R, T):
    # Setup output image
    img = np.copy(img_in)

    # We can define a 10cm cube by 4 sets of 3d points
    # these points are in the reference coordinate frame
    scale = 0.1
    face1 = np.array([[0,0,0],[0,0,scale],[0,scale,scale],[0,scale,0]],
                     np.float32)
    face2 = np.array([[0,0,0],[0,scale,0],[scale,scale,0],[scale,0,0]],
                     np.float32)
    face3 = np.array([[0,0,scale],[0,scale,scale],[scale,scale,scale],
                      [scale,0,scale]],np.float32)
    face4 = np.array([[scale,0,0],[scale,0,scale],[scale,scale,scale],
                      [scale,scale,0]],np.float32)
    # using the function you write above we will get the 2d projected 
    # position of these points
    face1_proj = ProjectPoints(face1, new_intrinsics, R, T)
    # this function simply draws a line connecting the 4 points
    img = cv2.polylines(img, [np.int32(face1_proj)], True, 
                              tuple([255,0,0]), 3, cv2.LINE_AA) 
    # repeat for the remaining faces
    face2_proj = ProjectPoints(face2, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face2_proj)], True, 
                              tuple([0,255,0]), 3, cv2.LINE_AA) 
    
    face3_proj = ProjectPoints(face3, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face3_proj)], True, 
                              tuple([0,0,255]), 3, cv2.LINE_AA) 
    
    face4_proj = ProjectPoints(face4, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face4_proj)], True, 
                              tuple([125,125,0]), 3, cv2.LINE_AA) 
    return img

# This function takes in an intrinsics matrix, and two sets of 2d points
# if a pose can be computed it returns true along with a rotation and 
# translation between the sets of points. 
# returns false if a good pose estimate cannot be found
def ComputePoseFromHomography(new_intrinsics, referencePoints, imagePoints):
    # compute homography using RANSAC, this allows us to compute
    # the homography even when some matches are incorrect
    homography, mask = cv2.findHomography(referencePoints, imagePoints, 
                                          cv2.RANSAC, 5.0)
    # check that enough matches are correct for a reasonable estimate
    # correct matches are typically called inliers
    MIN_INLIERS = 30
    if(sum(mask)>MIN_INLIERS):
        # given that we have a good estimate
        # decompose the homography into Rotation and translation
        # you are not required to know how to do this for this class
        # but if you are interested please refer to:
        # https://docs.opencv.org/master/d9/dab/tutorial_homography.html
        RT = np.matmul(np.linalg.inv(new_intrinsics), homography)
        norm = np.sqrt(np.linalg.norm(RT[:,0])*np.linalg.norm(RT[:,1]))
        RT = -1*RT/norm
        c1 = RT[:,0]
        c2 = RT[:,1]
        c3 = np.cross(c1,c2)
        T = RT[:,2]
        R = np.vstack((c1,c2,c3)).T
        W,U,Vt = cv2.SVDecomp(R)
        R = np.matmul(U,Vt)
        return True, R, T
    # return false if we could not compute a good estimate
    return False, None, None
    
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

def get_M(intrinsic, matrix_matches):
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    
    total = 0
    for i in matrix_matches:
        total += len(matrix_matches[i])
    M = np.zeros((3*total, len(matrix_matches) + 1))
    
    counter1 = 0
    counter2 = 0
    for i in matrix_matches:
        for j in matrix_matches[i]:
            m = j[0]
            (u1,v1) = j[1][m.queryIdx].pt
            (u2,v2) = j[2][m.trainIdx].pt
        
            x1 = np.array([(u1 - cx)/fx, (v1 - cy)/fy,1])
            x2 = np.array([(u2 - cx)/fx, (v2 - cy)/fy,1])
            R = j[3]
            T = j[4]
            
            a = np.cross(x2, np.matmul(R,x1))
            b = np.cross(x2, T)

            M[counter2:counter2+3, counter1] = a.T
            M[counter2:counter2+3, len(matrix_matches)] = b.T
            counter2 += 3
        counter1 += 1
    
    return M
    
def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True
## Load the reference image that we will try to detect in the webcam
#reference = cv2.imread('ARTrackerImage.jpg',0)
#RES = 480
#reference = cv2.resize(reference,(RES,RES))
#
## create the feature detector. This will be used to find and describe locations
## in the image that we can reliably detect in multiple images
#feature_detector = cv2.BRISK_create(octaves=5)
## compute the features in the reference image
#reference_keypoints, reference_descriptors = \
#        feature_detector.detectAndCompute(reference, None)
## make image to visualize keypoints
#keypoint_visualization = cv2.drawKeypoints(
#        reference,reference_keypoints,outImage=np.array([]), 
#        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
## display the image
#cv2.imshow("Keypoints",keypoint_visualization)
## wait for user to press a key before proceeding
#
## create the matcher that is used to compare feature similarity
## Brisk descriptors are binary descriptors (a vector of zeros and 1s)
## Thus hamming distance is a good measure of similarity        
#matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
## Load the camera calibration matrix
#intrinsics, distortion, new_intrinsics, roi = \
#        calib.LoadCalibrationData('images')
#
## initialize video capture
## the 0 value should default to the webcam, but you may need to change this
## for your camera, especially if you are using a camera besides the default
#cap = cv2.VideoCapture(0)
#
#while True:
#    # read the current frame from the webcam
#    ret, current_frame = cap.read()
#    
#    # ensure the image is valid
#    if not ret:
#        print("Unable to capture video")
#        break
#    
#    # undistort the current frame using the loaded calibration
#    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
#                                  new_intrinsics)
#    # apply region of interest cropping
#    x, y, w, h = roi
#    current_frame = current_frame[y:y+h, x:x+w]
#    
#    # detect features in the current image
#    current_keypoints, current_descriptors = \
#        feature_detector.detectAndCompute(current_frame, None)
#    if current_descriptors is not None:
#        # match the features from the reference image to the current image
#        matches = matcher.match(reference_descriptors, current_descriptors)
#        # matches returns a vector where for each element there is a 
#        # query index matched with a train index. I know these terms don't really
#        # make sense in this context, all you need to know is that for us the 
#        # query will refer to a feature in the reference image and train will
#        # refer to a feature in the current image
#
#        # create a visualization of the matches between the reference and the
#        # current image
#        # match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
#        #                        current_keypoints, matches, 0, 
#        #                        flags=
#        #                        cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#         # set up reference points and image points
#         # here we get the 2d position of all features in the reference image
#        referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
#                                  for m in matches])
#             # convert positions from pixels to meters
#        SCALE = 0.1 # this is the scale of our reference image: 0.1m x 0.1m
#        referencePoints = SCALE*referencePoints/RES
#    
#        imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
#                                  for m in matches])
#    # compute homography
#        ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
#                                          imagePoints)
#        render_frame = current_frame
#        if(ret):
#            # compute the projection and render the cube
#            render_frame = renderCube(current_frame,new_intrinsics,R,T) 
#            
#        # display the current image frame
#        cv2.imshow('frame', render_frame)
#        k = cv2.waitKey(1)
#        if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
#            #exit
#            break
#
#    
#cv2.destroyAllWindows() 
#cap.release()
#
#
