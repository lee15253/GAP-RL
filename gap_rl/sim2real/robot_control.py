import sys
import time
import numpy as np
from queue import Queue, Empty
from threading import Thread, Lock
from scipy.spatial.transform import Rotation as R
from typing import List

import rospy
from actionlib import SimpleActionClient
from trajectory_msgs.msg import *
from geometry_msgs.msg import *
from control_msgs.msg import *
from tf2_msgs.msg import TFMessage
from cartesian_control_msgs.msg import *
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, ListControllers
from sensor_msgs.msg import JointState

from gap_rl.sim2real.gripper_control import GripperController


# set fixed joint status
HOME_POSE = [0.0387, 0.3218, 0.4269, -0.1681, 0.9856, -0.0144, -0.011]
HOME_JOINT = [-1.1672, -0.9103, -1.8854, -1.5976, 1.4559, -2.7273]
HOME_JOINT_HIGHER = [-1.16710167, -1.24511789, -0.92101025, -2.26421564, 1.48632239, -2.74679918]
threed_HOME_JOINT = [-1.27, -1.0103, -1.5, -1.7, 1.4559, -2.7273]
GRASP_POSE = [-0.4, 0, 0.4, 0, 0, 1, 0]
# GRASP_JOINT_STATUS = [
#     0.12994670867919922, -1.221276120548584, -1.9320955276489258, -1.4974048894694825, 1.604724407196045,
#     -2.9260993639575403
# ]

# Available trajectory controllers:
# 0 (joint-based): scaled_pos_joint_traj_controller, try
# 1 (joint-based): scaled_vel_joint_traj_controller, try
# 2 (joint-based): pos_joint_traj_controller, try`
# 3 (joint-based): vel_joint_traj_controller, try
# 4 (joint-based): forward_joint_traj_controller, try
# 5 (Cartesian): pose_based_cartesian_traj_controller, try
# 6 (Cartesian): joint_based_cartesian_traj_controller, try
# 7 (Cartesian): forward_cartesian_traj_controller, try

# Available servoing controllers:
# 0 (joint-based): joint_group_vel_controller, try
# 1 (Cartesian): twist_controller, try


class RobotUR:
    """Ur5e robot with robotiq gripper controller using ros packages."""

    def __init__(self, gripper_controller: GripperController, controller_type="twist_controller") -> None:
        """init robot controller.

        Args:
            a (float, optional): max_acceleration_scaling_factor. Defaults to 0.5.
            v (float, optional): max_velocity_scaling_factor. Defaults to 0.3.
        """
        self.gripper_controller = gripper_controller
        self.controller = None
        assert controller_type in ["scaled_pos_joint_traj_controller", "scaled_vel_joint_traj_controller", "pos_joint_traj_controller", "vel_joint_traj_controller",
                        "forward_joint_traj_controller", "joint_group_vel_controller",  # joint-based
                        "pose_based_cartesian_traj_controller", "joint_based_cartesian_traj_controller", "forward_cartesian_traj_controller", "twist_controller"]  # Cartesian
        self.readonly_controllers = ['joint_state_controller', 'speed_scaling_state_controller', 'force_torque_sensor_controller']
        self.controller_type = controller_type
        self.rate = rospy.Rate(5)  # 5
        # joint names in the messages
        self.msg_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                                'wrist_2_joint', 'wrist_3_joint']

        # init arm controller
        self.switch_controller(self.controller_type)

        # init gripper controller
        self.gripper_controller.activate_gripper()

        # robot state subscriber
        self._q_state = rospy.wait_for_message('joint_states', JointState, timeout=3000)
        self._tf_state = rospy.wait_for_message('tf', TFMessage, timeout=3000)
        self._ee_state = None
        # self._wrench_state = rospy.wait_for_message('wrench', WrenchStamped, timeout=3000)
        self._lock = Lock()
        self._state_thread = Thread(target=self.init_robot_state_subscriber)
        self._state_thread.start()

    def init_robot_state_subscriber(self):
        rospy.Subscriber('/joint_states', JointState, self.q_callback)
        rospy.Subscriber('/tf', TFMessage, self.tf_callback)
        # rospy.Subscriber('/wrench', WrenchStamped, self.wrench_callback)
        rospy.spin()

    def q_callback(self, msg: JointState):
        with self._lock:
            self._q_state = msg

    def tf_callback(self, msg: TFMessage):
        with self._lock:
            self._tf_state = msg

    # def wrench_callback(self, msg: Wrench):
    #     with self._lock:
    #         self._wrench_state = msg

    def get_real_state(self):
        with self._lock:
            qpos = self._q_state.position
            qvel = self._q_state.velocity
            qeffort = self._q_state.effort  # currents actually
            for transform in self._tf_state.transforms:
                if transform.child_frame_id == "tool0_controller":
                    trans = transform.transform.translation
                    rot = transform.transform.rotation
                    self._ee_state = np.stack((
                        trans.x, trans.y, trans.z,
                        rot.w, rot.x, rot.y, rot.z
                    ))
            tcp_state = self._ee_state
            # force, torque = self._wrench_state.wrench.force, self._wrench_state.wrench.torque
            # wrench = np.stack((
            #     force.x, force.y, force.z,
            #     torque.x, torque.y, torque.z
            # ))
            gripper_pos = self.gripper_controller.get_pos()
            grasp_state = self.gripper_controller.object_grasped()
            robot_state = {
                "qpos": qpos,
                "qvel": qvel,
                "qeffort": qeffort,
                "tcp_state": tcp_state,  # (pos, wxyz)
                # "wrench": wrench,  # (force, torque)
                "gripper_pos": gripper_pos,  # [0, 0.085]
                "grasp_state": grasp_state
            }
            return robot_state

    def switch_controller(self, controller_type):
        # switch to specified controller
        active_controller_name = None
        try:
            list_controllers = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
            response = list_controllers()

            # Print information only for active controllers
            for controller in response.controller:
                if controller.state == "running" and controller.name not in self.readonly_controllers:
                    print("Active Controller Name:", controller.name)
                    print("Controller Type:", controller.type)
                    print("---")
                    active_controller_name = controller.name
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        if controller_type != active_controller_name:
            rospy.wait_for_service('/controller_manager/switch_controller')
            switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
            request = SwitchControllerRequest()
            request.start_controllers = [controller_type]
            request.stop_controllers = [active_controller_name]
            request.strictness = 2
            request.start_asap = False
            request.timeout = 0.0
            switch_controller.call(request)
            print("switch to controller:", controller_type)

        print('set controller: ', controller_type)
        if self.controller_type == "twist_controller":
            self.controller = rospy.Publisher('twist_controller/command', Twist, queue_size=10)
        elif self.controller_type == "forward_joint_traj_controller":
            self.controller = SimpleActionClient('/forward_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
            self.controller.wait_for_server()
        elif self.controller_type == "pose_based_cartesian_traj_controller":
            self.controller = SimpleActionClient('/pose_based_cartesian_traj_controller/follow_cartesian_trajectory', FollowCartesianTrajectoryAction)
            self.controller.wait_for_server()
        else:
            raise NotImplementedError
        rospy.sleep(3)  # sleep 3s to build the publisher

    def arm_command(self, action):
        """
        Publishes a Twist command to the robot.
        """
        if self.controller_type == "twist_controller":
            # Create a Twist message
            cmd = Twist()
            d_pos, d_euler = tuple(action[:3]), tuple(action[3:6])
            # print(d_pos, d_euler)
            cmd.linear.x, cmd.linear.y, cmd.linear.z = d_pos
            cmd.angular.x, cmd.angular.y, cmd.angular.z = d_euler
            # Publish the Twist command
            self.controller.publish(cmd)
        elif self.controller_type == "scaled_pos_joint_traj_controller":
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = JointTrajectory()
            goal.trajectory.joint_names = self.msg_joint_names
            cur_state = self.get_real_state()
            cur_qpos = cur_state['qpos']
            # positions, velocities, accelerations, effort
            goal.trajectory.points = [0] * 4
            goal.trajectory.points[0] = JointTrajectoryPoint(positions=action, velocities=[0] * 6, time_from_start=rospy.Duration(0))
            self.controller.send_goal(goal)
            self.controller.wait_for_result()
        elif self.controller_type == "pose_based_cartesian_traj_controller":
            goal = FollowCartesianTrajectoryGoal()
            traj = CartesianTrajectory()
            point = CartesianTrajectoryPoint()
            # point.pose = action + self.get_state()["tcp_state"]

            origin_quat = R.from_euler('XYZ', [[0.051, 3.127, -0.225]], degrees=False).as_matrix()  # xyzw
            goal_pos = np.array([-0.042, -0.529, 0.312]) + action[:3]
            goal_mat = origin_quat @ (R.from_euler('XYZ', [action[3:6]], degrees=False).as_matrix())
            goal_pose = Pose()
            goal_pose.position = goal_pos
            goal_pose.orientation = R.from_matrix(goal_mat).as_quat()
            point.pose = goal_pose
            point.time_from_start = rospy.Duration(1)
            traj.points.append(point)
            goal.trajectory = traj
            # Publish the goal
            self.controller.send_goal(goal)
            self.controller.wait_for_result()
        else:
            raise NotImplementedError

        # print("++++ robot arm action published ++++")

    def gripper_command(self, pos):
        """
        :para pos: [0, 0.085], close -> open
        """
        self.gripper_controller.move(pos=pos, vel=100, force=50)
        # print("++++ gripper action published ++++")

    def execute_combined_command(self, arm_action, gripper_action):
        # t = time.time()
        self.arm_command(arm_action)
        # print("arm control time: ", time.time() - t)
        # t = time.time()
        self.gripper_command(gripper_action)
        self.rate.sleep()
        # print("gripper control time: ", time.time() - t)

    def close(self):
        rospy.signal_shutdown("Terminated by user.")
        self._state_thread.join()

        
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # "pose_based_cartesian_traj_controller", "joint_based_cartesian_traj_controller", "forward_cartesian_traj_controller", "twist_controller"
    rospy.init_node('ur_controller', anonymous=True)
    gripper_controller = GripperController(GRIPPER_PORT="/dev/ttyUSB0", sync=True)
    UR = RobotUR(gripper_controller, controller_type="twist_controller")
    print("UR controller start! ")

    arm_gripper_cmds = [
        # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.085],
        [0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0],
        [-0.02, -0.01, -0.01, 0.0, 0.0, 0.0, 0.085],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.085],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    for step in range(len(arm_gripper_cmds)):
        t = time.time()
        states = UR.get_real_state()
        print(states)

        UR.execute_combined_command(
            arm_action=arm_gripper_cmds[step][:6],
            gripper_action=arm_gripper_cmds[step][6]
        )
        states = UR.get_real_state()
        print(states)
        print(f'++++ {step}', time.time() - t)

    UR.close()

    # import actionlib
    # rospy.init_node("pub_action_test")
    #
    # client = actionlib.SimpleActionClient('/scaled_pos_joint_traj_controller/follow_joint_trajectory/', FollowJointTrajectoryAction)
    # print("Waiting for server...")
    # client.wait_for_server()
    # print("Connect to server")
