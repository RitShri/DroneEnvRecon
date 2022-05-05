import sys
import numpy as np
import pyzed.sl as sl
import cv2
import time
from pynput import keyboard
import datetime
import json


help_string = "[s] Save side by side image"
path = "./"

count_save = 0
dic_count = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

take_image = False
break_loop = False


point_cloud_mats = []

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
        take_image = True

r_t_dict =  {}
listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread
#listener.join()  # remove if main thread is polling self.keys

def save_point_cloud(zed, filename) :
    #print("Saving Point Cloud...")
    tmp = sl.Mat()
    critical_time = datetime.datetime.utcnow()+datetime.timedelta(seconds=1)
    critical_time.replace(microsecond=0)
    while(datetime.datetime.utcnow() < critical_time):
        continue
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    print("zed milli: ", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    point_cloud_mats.append(tmp)
    #print("Got Point Cloud in var...")
    '''
    saved = (tmp.write("calib_zed_left"+'/'+filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")
    '''
def write_point_cloud(zed, filename):
    global count_save
    for tmp in point_cloud_mats:
        saved = (tmp.write("data"+'/'+filename + str(count_save) + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
        count_save += 1
        if saved :
            print("Saved")
        else :
            print("Failed... Please check that you have permissions to write on disk")
def save_sbs_image(zed, filename) :
    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    cv2.imwrite("calib_zed"+'/'+filename, image_cv_left)
 

def process_key_event(zed, key) :
    global count_save
    global take_image
    global dic_count
    #print(take_image)
    if key == 115 or take_image:
        #save_point_cloud(zed, "calib_image_zed_" + str(count_save))	#CHANGE THIS IF U WANT PCS
	save_sbs_image(zed,"calib_image_zed_" + str(count_save)) 

        #count_save += 1
        #take_image = False

        #Get R,T per frame
        zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD) #CAMERA
        zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)

        py_translation = sl.Translation()
        translationMat = zed_pose.get_translation(py_translation).get()

        py_rotation = sl.Rotation()
        rotationMat = zed_pose.get_rotation_matrix(py_rotation).r

        #py_orientation = sl.Orientation()
        #orientationMat = zed_pose.get_orientation(py_orientation).get()
        print("Rotation {0}, Translation {1}".format(rotationMat, translationMat))
        r_t_dict[dic_count] = [rotationMat.tolist(), translationMat.tolist()]
        dic_count += 1
        
        

        #IMUOrient = zed_sensors.get_imu_data().get_pose().get_orientation().get()
        #IMUTrans = zed_sensors.get_imu_data().get_pose().get_translation().get()
        #IMURot = zed_sensors.get_imu_data().get_pose().get_rotation().get()
        #print("IMU: Orientation {1}, Translation {0}".format(IMUTrans, IMUOrient))

def print_help() :
    print(" Press 's' to save Side by side images")


def main() :

    camNum = 0
    
    # Set configuration parameters
    input_type = sl.InputType()
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    #init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_DOW
    
    cameras = sl.Camera.get_device_list()
    print(cameras)
    if len(sys.argv) >= 2:
        camNum = int(sys.argv[1])
    init.set_from_serial_number(cameras[camNum].serial_number)
	
        # Create a ZED camera object
    zed = sl.Camera()


    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Enable positional tracking with default parameters
    tracking_parameters = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_parameters)
    if (err != sl.ERROR_CODE.SUCCESS):
        exit(-1)
    global zed_pose 
    zed_pose = sl.Pose()
    global zed_sensors
    zed_sensors = sl.SensorsData()
    global runtime_parameters
    runtime_parameters = sl.RuntimeParameters()

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    key = ' '
    while key != ord("q") :
        r = ""
        
        #print(break_loop)
        #print(take_image) 
        #try:
            #if not sys.stdin.isatty():
            #    for line in sys.stdin:
            #        r += line
            #        print(line)
        #except EOFError as e:
        r = "g"
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            if(not take_image):
                zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
                image_ocv = image_zed.get_data()
                cv2.imshow("Image", image_ocv)
            key = cv2.waitKey(10)
            if(take_image):
                process_key_event(zed, key)
        if(break_loop):
            break
    write_point_cloud(zed, "calib_image_zed_" + str(camNum))
    cv2.destroyAllWindows()
    json_data = {}
    json_data['R_T_DICT'] = r_t_dict
    print(r_t_dict)
    json_string = json.dumps(json_data)
    with open('calib_zed_left/r_t_timestamps_zed.json', 'w') as  outfile:
        outfile.write(json_string)
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()
