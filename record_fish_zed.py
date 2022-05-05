import sys
import numpy as np
import pyzed.sl as sl
import cv2
import sys
import json
from datetime import datetime
from multiprocessing import Process
from subprocess import Popen, PIPE
import time
from pynput.keyboard import Key, Controller



if __name__ == "__main__":
    #processes = [Popen(['python3 zed_capture.py 0'], stdin=PIPE,shell=True), Popen(['python3 zed_capture.py 1'], stdin=PIPE,shell=True), Popen(['python calibration.py'], stdin=PIPE,shell=True)]
    processes = [Popen(['python3 zed_spatial_map.py'], stdin=PIPE,shell=True), Popen(['python calibration.py'], stdin=PIPE,shell=True)]
    '''
    time.sleep(30)
    keyboard = Controller()
    for i in range(10):
        keyboard.press('s')
        keyboard.release('s')
        time.sleep(1)
    
    keyboard.press('q')
    keyboard.release('q')
    '''
    # use s to screenshot q to kill

