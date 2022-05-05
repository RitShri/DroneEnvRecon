

"""
    This sample shows how to capture a real-time 3D reconstruction      
    of the scene using the Spatial Mapping API. The resulting mesh      
    is displayed as a wireframe on top of the left image using OpenGL.  
    Spatial Mapping can be started and stopped with the Space Bar key
"""
import sys
import json
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import datetime

def main():
    print("Running Spatial Mapping sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.coordinate_units = sl.UNIT.METER         # Set coordinate units
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # OpenGL coordinates

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.set_from_svo_file(filepath)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Get camera parameters
    camera_parameters = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam

    pymesh = sl.FusedPointCloud()        # Current incremental mesh
    image = sl.Mat()          # Left image from camera
    pose = sl.Pose()          # Camera pose tracking data
    zed_sensors = sl.SensorsData()
    r_t_dict = {}
    dic_count = 0

    viewer = gl.GLViewer()
    viewer.init(camera_parameters, pymesh)

    spatial_mapping_parameters = sl.SpatialMappingParameters()
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
    mapping_activated = False
    last_call = time.time()             # Timestamp of last mesh request

    # Enable positional tracking
    err = zed.enable_positional_tracking()
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit()

    # Set runtime parameters
    runtime = sl.RuntimeParameters()

    while viewer.is_available():
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Update pose data (used for projection of the mesh over the current image)
            tracking_state = zed.get_position(pose)

            if mapping_activated:
                mapping_state = zed.get_spatial_mapping_state()
                # Compute elapsed time since the last call of Camera.request_spatial_map_async()
                duration = time.time() - last_call  
                # Ask for a mesh update if 500ms elapsed since last request
                if(duration > 1 and viewer.chunks_updated()):
                    last_call = time.time()
                    zed.request_spatial_map_async()
                    
                    zed.get_position(pose, sl.REFERENCE_FRAME.CAMERA)
                    zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)

                    py_translation = sl.Translation()
                    translationMat = pose.get_translation(py_translation).get()

                    py_rotation = sl.Rotation()
                    rotationMat = pose.get_rotation_matrix(py_rotation).r
                    print("zed milli: ", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])  
                    #print("Rotation {0}, Translation {1}".format(rotationMat, translationMat))
                    r_t_dict[dic_count] = [rotationMat.tolist(), translationMat.tolist()]
                    dic_count += 1

                if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_spatial_map_async(pymesh)
                    viewer.update_chunks()

                
                
            change_state = viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state)

            if change_state:
                if not mapping_activated:
                    init_pose = sl.Transform()
                    zed.reset_positional_tracking(init_pose)

                    # Configure spatial mapping parameters
                    spatial_mapping_parameters.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.LOW)
                    #spatial_mapping_parameters.use_chunk_only = True
                    spatial_mapping_parameters.save_texture = True         # Set to True to apply texture over the created mesh
                    spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

                    # Enable spatial mapping
                    zed.enable_spatial_mapping(spatial_mapping_parameters)

                    # Clear previous mesh data
                    pymesh.clear()
                    viewer.clear_current_mesh()

                    # Start timer
                    last_call = time.time()

                    mapping_activated = True
                else:
                    # Extract whole mesh
                    zed.extract_whole_spatial_map(pymesh)

                    #filter_params = sl.MeshFilterParameters()
                    #filter_params.set(sl.MESH_FILTER.MEDIUM) 
                    # Filter the extracted mesh
                    #pymesh.filter(filter_params, True)
                    viewer.clear_current_mesh()

                    # If textures have been saved during spatial mapping, apply them to the mesh
                    #if(spatial_mapping_parameters.save_texture):
                    #    print("Save texture set to : {}".format(spatial_mapping_parameters.save_texture))
                    #    pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)

                    # Save mesh as an obj file
                    json_data = {}
                    json_data['R_T_DICT'] = r_t_dict
                    print(r_t_dict)
                    json_string = json.dumps(json_data)
                    with open('data/RT_zed_spatial.json', 'w') as  outfile:
                        outfile.write(json_string)
                    filepath = "data/mesh_gen.ply"
                    status = pymesh.save(filepath, sl.MESH_FILE_FORMAT.PLY)
                    if status:
                        print("Mesh saved under " + filepath)
                    else:
                        print("Failed to save the mesh under " + filepath)
                    
                    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
                    mapping_activated = False
    
    image.free(memory_type=sl.MEM.CPU)
    pymesh.clear()
    # Disable modules and close camera
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
if __name__ == "__main__":
    main()
