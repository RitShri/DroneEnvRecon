import sys
import numpy as np
import pyzed.sl as sl
import cv2
from pynput import keyboard

help_string = "[s] Save side by side image"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

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
        take_image = True

listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread
#listener.join()  # remove if main thread is polling self.keys


def save_sbs_image(zed, filename) :
    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    cv2.imwrite("calib_zed"+'/'+filename, image_cv_left)
 

def process_key_event(zed, key) :
    global count_save
    global take_image
    #print(take_image)
    if key == 115 or take_image:
        save_sbs_image(zed, "calib_image_zed_" + str(count_save) + ".png")
        count_save += 1
        take_image = False

        #Get R,T per frame
        zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
        zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
        
        py_translation = sl.Translation()
        translationMat = zed_pose.get_translation(py_translation).get()
        py_rotation = sl.Rotation()
        rotationMat = zed_pose.get_rotation(py_rotation).get()
        print("Rotation {0}, Translation {1}".format(rotationMat, translationMat))


def print_help() :
    print(" Press 's' to save Side by side images")


def main() :
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

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
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            image_ocv = image_zed.get_data()
            cv2.imshow("Image", image_ocv)
            key = cv2.waitKey(10)
            process_key_event(zed, key)
        if(break_loop):
            break
    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()
