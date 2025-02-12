import numpy as np
# robot ip
IP_ADDRESS = '***'
# gripper port
GRIPPER_PORT = '/dev/ttyUSB0'

# realsense in ee2
pose_realsense = [-0.0251, -0.0658, -0.10, 0.9995741, 0.0105889, 0.0068908, 0.0263039]
pose_realsense = np.array(pose_realsense)

# realsense in ee
pose_realsense_gripper = [-0.0251, -0.1178, -0.19, 0.9995741, 0.0105889, 0.0068908, 0.0263039]
pose_realsense_gripper = np.array(pose_realsense_gripper)
