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

if __name__ == "__main__":
    processes = [Popen(['python3 zed_capture.py'], stdin=PIPE,shell=True), 
		 Popen(['python calibration.py'], stdin=PIPE,shell=True)]
    time.sleep(5)
    r = ""
    i = 0
    
    time.sleep(60)
    processes[1].stdin.write("exit")
    processes[1].terminate()
    processes[0].stdin.write("exit")
    processes[0].terminate()
    # does not kill the process properly

