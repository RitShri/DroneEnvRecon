import cv2
import numpy as np
import glob
import datetime
import json
import time
import os
import sys
from pynput import keyboard

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

take_image = False
break_loop = False

def on_press(key):
    global take_image
    global break_loop
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if k in ['q']:  # keys of interest
        # self.keys.append(k)  # store it in global-like variable
        print('Key pressed: ' + k)
        break_loop = True
    if k in ['s']:
        print('Key pressed: ' + k)
        if(take_image):
            take_image = False
        else:
            time.sleep(1)
            take_image = True
        

listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread
#listener.join()  # remove if main thread is polling self.keys
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3264,
    capture_height=2464,
    display_width=960,
    display_height=540,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# This function records images from the connected camera to specified directory 
# when the "Space" key is pressed.
# directory: should be a string corresponding to the name of an existing 
# directory
def CaptureImages(directory):
	global take_image
	global break_loop
    # Open the camera for capture
    # the 0 value should default to the webcam, but you may need to change this
    # for your camera, especially if you are using a camera besides the default

	cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
	img_counter = 0
	img_to_write = []
	# Read until user quits
        last_call = time.time()
	while True:
            ret, frame = cam.read()
	    if not ret:
	        break
	    # display the current image
	    if(not take_image):
	        cv2.imshow("Display", frame)
	    # wait for 1ms or key press
	    k = cv2.waitKey(10) #k is the key pressed
	    if k == 27 or k==113 or break_loop:  #27, 113 are ascii for escape and q respectively
	        break
	    elif k == 32 or take_image: #32 is ascii for space
	        #record image
	        duration = time.time() - last_call
                if duration > 1:
                    last_call = time.time()
	            ret, frame = cam.read()
                    print("fish milli: ", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
	            img_to_write.append(frame)
	        #img_name = "calib_image_fish_{}.png".format(img_counter)
	        #cv2.imwrite(directory+'/'+img_name, frame)
	        #print("Writing: {}".format(directory+'/'+img_name))
	        
	        #take_image = False
	for frame in img_to_write:
	    img_name = "calib_image_fish_{}.png".format(img_counter)
	    cv2.imwrite(directory+'/'+img_name, frame)
	    print("Writing: {}".format(directory+'/'+img_name))
	    img_counter += 1
	cam.release()

def calibrate_images(CHECKERBOARD, directory):
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_K2+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob("./{}/*.png".format(directory))
    #print(len(images))
    #gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    index = 0
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # print(index, fname)
        index+=1
        # Find the chess board corners
        corners = None
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
        # If found, add object points, image points (after refining them)
        # print(ret)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)


    N_OK = len(objpoints)
    #print(gray.shape[::-1])
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, K, D, rvecs, tvecs = \
        cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    )
    #print(K)
    if not ret:
        print("Calibration failed, recollect images and try again")
    # if successful, compute an print reprojection error, this is a good metric
    # for how good the calibration is. If your result is greater than 1px you
    # should probably recalibrate
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], \
                                          K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    #print( "mean error: {}".format(total_error/len(objpoints)) )
    # compute the region for where we have full information and the resulting
    # intrinsic calibration matrix
    h,  w = img.shape[:2]
    new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1,\
                                                        (w,h))
    # return only the information we will need going forward

    return K, D, roi, new_intrinsics

# This function will save the calibration data to a file in the specified 
# directory
def SaveCalibrationData(directory, intrinsics, distortion, new_intrinsics, \
                        roi):
    np.savez(directory+'/calib', intrinsics=intrinsics, distortion=distortion,\
             new_intrinsics = new_intrinsics, roi=roi)
    
# This function will load the calibration data from a file in the specified 
# directory   
def LoadCalibrationData(directory):
    npzfile = np.load(directory+'/calib.npz')
    return npzfile['intrinsics'], npzfile['distortion'], \
            npzfile['new_intrinsics'], npzfile['roi']
            

