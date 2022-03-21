import cv2
import numpy as np
from datetime import datetime
import json
import os
import glob
import CalibrationHelpers


CHECKERBOARD = (6,9)
CaptureImages('calib')
K, D, roi, new_intrinsics = CalibrationHelpers.calibrate_images(CHECKERBOARD, 'calib')
CalibrationHelpers.SaveCalibrationData('calib', K, D, roi, new_intrinsics)

