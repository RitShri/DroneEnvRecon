import cv2
import numpy as np
import glob
import CalibrationHelpers as calib
import collections
import open3d as o3d
import matplotlib.pyplot as plt


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

# Load the reference image that we will try to detect in the webcam
reference = cv2.imread('ARTrackerImage.jpg',0)
RES = 480
reference = cv2.resize(reference,(RES,RES))

# create the feature detector. This will be used to find and describe locations
# in the image that we can reliably detect in multiple images
feature_detector = cv2.BRISK_create(octaves=5)
# compute the features in the reference image
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
# make image to visualize keypoints
keypoint_visualization = cv2.drawKeypoints(
        reference,reference_keypoints,outImage=np.array([]), 
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display the image
cv2.imshow("Keypoints",keypoint_visualization)
# wait for user to press a key before proceeding

# create the matcher that is used to compare feature similarity
# Brisk descriptors are binary descriptors (a vector of zeros and 1s)
# Thus hamming distance is a good measure of similarity        
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load the camera calibration matrix
intrinsics, distortion, new_intrinsics, roi = \
        calib.LoadCalibrationData('images')
print(intrinsics)
print(distortion)
print(new_intrinsics)
print(roi)
        

#TODO: add section to import a image
cap = glob.glob('cube/*.jpg')
rotations = []
translations = []
count = 0
for fname in cap:
    # TODO iterate over one of the images
    current_frame = cv2.imread(fname)
    
    # undistort the current frame using the loaded calibration
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    # apply region of interest cropping
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    # detect features in the current image
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)

    if current_descriptors is not None:
        # match the features from the reference image to the current image
        matches = matcher.match(reference_descriptors, current_descriptors)
        # matches returns a vector where for each element there is a 
        # query index matched with a train index. I know these terms don't really
        # make sense in this context, all you need to know is that for us the 
        # query will refer to a feature in the reference image and train will
        # refer to a feature in the current image

        # create a visualization of the matches between the reference and the
        # current image
        # match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
        #                        current_keypoints, matches, 0, 
        #                        flags=
        #                        cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
         # set up reference points and image points
         # here we get the 2d position of all features in the reference image
        referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
             # convert positions from pixels to meters
        SCALE = 0.1 # this is the scale of our reference image: 0.1m x 0.1m
        referencePoints = SCALE*referencePoints/RES
    
        imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                  for m in matches])
    # compute homography
        ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
                                          imagePoints)
        rotations.append(R)
        translations.append(T)
        render_frame = current_frame
        if(ret):
            # compute the projection and render the cube
            render_frame = renderCube(current_frame,new_intrinsics,R,T) 
            
        # display the current image frame
        cv2.imshow('frame', render_frame)
        plt.imsave("cubes"+str(count)+".jpg", render_frame)
        count += 1
        k = cv2.waitKey(1)
        if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
            #exit
            break

#part 3.4
rotations = np.array(rotations)
translations = np.array(translations)
ref_r = rotations[0]
ref_t = translations[0]
relative_rotations = []
relative_translations = []
for i in range(1,translations.shape[0]):
    relative_rotations.append(np.matmul(rotations[i], ref_r.T))
    relative_translations.append(translations[i] - np.dot(np.matmul(rotations[i], ref_r.T), ref_t))
    
# Here relative_roation has each image relative to the first image
    #at index 0 we have image 1 relative to image 0
    #at index 1 we have image 2 relative to image 0 and so on...
    #same logic is applied to the tranlations
relative_rotations = np.array(relative_rotations)
relative_translations = np.array(relative_translations)

#part 3.5

reference = cv2.imread(cap[0],0)
reference = cv2.undistort(reference, intrinsics, distortion, None,\
                                  new_intrinsics)

x, y, w, h = roi
reference = reference[y:y+h, x:x+w]
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)

c = 0

for fname in cap[1:]:

    current_frame = cv2.imread(fname,0)
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
        
    matches = matcher.match(reference_descriptors, current_descriptors)

    match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
                            current_keypoints, matches, 0, 
                            flags=
                            cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches',match_visualization)
    plt.imsave("originalMatches"+str(c)+".jpg", match_visualization)
    c += 1
    k = cv2.waitKey(1)
    if k == 27 or k==113:  
        #exit
        Break

#part 3.6

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


reference = cv2.imread(cap[0],0)
reference = cv2.undistort(reference, intrinsics, distortion, None,\
                                  new_intrinsics)

x, y, w, h = roi
reference = reference[y:y+h, x:x+w]
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
c = 0
for index, fname in enumerate(cap[1:]):

    current_frame = cv2.imread(fname,0)
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
        
    matches = matcher.match(reference_descriptors, current_descriptors)
    
    inlier_mask = FilterByEpipolarConstraint(intrinsics,
                                             matches, 
                                             reference_keypoints,
                                             current_keypoints,
                                             relative_rotations[index], 
                                             relative_translations[index])
    
    
    match_visualization = cv2.drawMatches(
        reference, reference_keypoints, 
        current_frame,
        current_keypoints, matches, 0, 
        matchesMask =inlier_mask, #this applies your inlier filter
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    #match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
    #                        current_keypoints, matches, 0, 
    #                        flags=
    #                        cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow('EpipolarConstraint',match_visualization)
    plt.imsave("epipolarconstraint"+str(c)+".jpg", match_visualization)
    c += 1
    k = cv2.waitKey(1)
    if k == 27 or k==113:  
        #exit
        Break
        
# Part 3.7
reference = cv2.imread(cap[0],0)
reference = cv2.undistort(reference, intrinsics, distortion, None,\
                                  new_intrinsics)

x, y, w, h = roi
reference = reference[y:y+h, x:x+w]
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
        
#set up the values
count = collections.Counter()

for index, fname in enumerate(cap[1:]):

    current_frame = cv2.imread(fname,0)
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
        
    matches = matcher.match(reference_descriptors, current_descriptors)
    
    for i in matches:
        count[i.queryIdx] += 1
        
feature_tracks = []
for i in count.most_common():
    if i[1] >= 4:
        feature_tracks.append(i[0]) #Todo: Code could be broken
        
matrix_matches = {}
c = 0
for index, fname in enumerate(cap[1:]):

    current_frame = cv2.imread(fname,0)
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
        
    matches = matcher.match(reference_descriptors, current_descriptors)
    
    match1 = []
    for i in matches: 
        if i.queryIdx in feature_tracks:
            match1.append(i)
            if i.queryIdx in matrix_matches:
                matrix_matches[i.queryIdx] += [(i,reference_keypoints,current_keypoints,relative_rotations[index],relative_translations[index])]
            else:
                matrix_matches[i.queryIdx] = [(i,reference_keypoints,current_keypoints,relative_rotations[index],relative_translations[index])]

    inlier_mask = FilterByEpipolarConstraint(intrinsics,
                                              match1, 
                                              reference_keypoints,
                                              current_keypoints,
                                              relative_rotations[index], 
                                              relative_translations[index])
    
    match_visualization = cv2.drawMatches(
        reference, reference_keypoints, 
        current_frame,
        current_keypoints, match1, 0, 
        matchesMask =inlier_mask, #this applies your inlier filter
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
    #                         current_keypoints, match1, 0, 
    #                         flags=
    #                         cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow('> 4 Constraints',match_visualization)
    plt.imsave("> 4 constraints"+str(c)+".jpg", match_visualization)
    c += 1
    k = cv2.waitKey(1)
    if k == 27 or k==113:  
        #exit
        break
    
# Part 3.8
def get_M(intrinsic, matrix_matches):
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    
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

M = get_M(intrinsics, matrix_matches)
W,U,Vt = cv2.SVDecomp(M)  
depths = Vt[-1,:]/Vt[-1,-1]
        
#part 3.9
your_pointCloud = []
count = 0
fx = intrinsics[0][0]
fy = intrinsics[1][1]
cx = intrinsics[0][2]
cy = intrinsics[1][2]
for i in matrix_matches:
    (u1,v1) = (reference_keypoints[i]).pt
    x1 = np.array([(u1 - cx)/fx, (v1 - cy)/fy,1])
    your_pointCloud.append(np.multiply(depths[count],x1))
    count += 1

your_pointCloud = np.array(your_pointCloud)


#part 3.10
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(your_pointCloud)
o3d.visualization.draw_geometries([pcd])

    
cv2.destroyAllWindows() 

