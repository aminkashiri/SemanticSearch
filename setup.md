## ros pkgs setup

- If running for the first time after reboot

    $ sudo modprobe gs_usb && rosrun scout_bringup bringup_can2usb.bash && roslaunch dm3_pkg dm3.launch

- If RTNETLINK is busy OR NOT first run after reboot, run 

    $ roslaunch dm3_pkg dm3.launch





## running eval_episode 

$ python ~/projects/search/SemanticSearch/projects/real_world_ovmm/eval_episode_scout_goat.py


## Debugging

Rostopic list 

/BMS_status 

/camera/camera/align_to_color/parameter_descriptions 

/camera/camera/align_to_color/parameter_updates

/camera/camera/aligned_depth_to_color/camera_info

/camera/camera/aligned_depth_to_color/image_raw

/camera/camera/color/camera_info

/camera/camera/color/image_raw

/camera/camera/color/metadata

/camera/camera/depth/camera_info

/camera/camera/depth/image_rect_raw

/camera/camera/depth/metadata

/camera/camera/extrinsics/depth_to_color

/camera/camera/motion_module/parameter_descriptions

/camera/camera/motion_module/parameter_updates

/camera/camera/realsense2_camera_manager/bond

/camera/camera/rgb_camera/auto_exposure_roi/parameter_descriptions

/camera/camera/rgb_camera/auto_exposure_roi/parameter_updates

/camera/camera/rgb_camera/parameter_descriptions

/camera/camera/rgb_camera/parameter_updates

/camera/camera/stereo_module/auto_exposure_roi/parameter_descriptions

/camera/camera/stereo_module/auto_exposure_roi/parameter_updates

/camera/camera/stereo_module/parameter_descriptions

/camera/camera/stereo_module/parameter_updates

/cmd_vel

/diagnostics

/dlio/map_node/map


/dlio/odom_node/keyframes

/dlio/odom_node/odom

/dlio/odom_node/path

/dlio/odom_node/pointcloud/deskewed

/dlio/odom_node/pointcloud/keyframe

/dlio/odom_node/pose

/livox/imu

/livox/lidar

/odom

/rosout

/rosout_agg

/rs_status

/scout_light_control

/scout_status

/tf

/tf_static