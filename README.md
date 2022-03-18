# DroneEnvRecon
Environment Reconstruction for Isaacs Drone

Calibration recordings: recording2, recording6
Environement Reconstruction recordings: recording3, recording4, recording5

Step 1. Run ```python calibration.py```
 - returns the camera calibration on fisheye camera from the images in calibration_images
 
 
 Step 2. Run ```python calcRT.py```
 - This is the environment reconstruction code. 
    The first prompt will show different camera calibration matrices applied on the fisheye camera. I am stuck on how to improve this and between which calibration matrix to choose. 
    Then the algorithm works on calcuating the R\* and t\* between the two cameras. By using the 8-point algorithm and then multiplying by K and K' 
