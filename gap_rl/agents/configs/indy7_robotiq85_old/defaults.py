from gap_rl import DESCRIPTION_DIR
from gap_rl.agents.controllers import *
from gap_rl.sensors.camera import CameraConfig
from gap_rl.sensors.depth_camera import StereoDepthCameraConfig


class Indy7Robotiq85oldDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = f"{DESCRIPTION_DIR}/indy7/indy_robotiq85_old.urdf"
        self.urdf_config = {}

        self.arm_joint_names = [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            # "tcp"
            
            # "shoulder_pan_joint",
            # "shoulder_lift_joint",
            # "elbow_joint",
            # "wrist_1_joint",
            # "wrist_2_joint",
            # "wrist_3_joint",
        ]
        # self.arm_stiffness = 100  # 1000
        # self.arm_damping = 20  # 50
        # self.arm_force_limit = 100  # TODO: 그냥 이걸로 고정
        # # TODO: 아래것들은 어디에 쓰는거지?
        # self.arm_delta = 0.04  # 0.05 rad/(control step)
        # self.arm_vel_delta = 0.2  # 0.2 rad/s

        self.gripper_joint_names = [
            "robotiq_2f_85_left_driver_joint",
            "robotiq_2f_85_right_driver_joint",
        ]
        self.gripper_stiffness = 1e3  # 1e3
        self.gripper_damping = 3e2  # 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "grasp_convenient_link"
        self.ee_delta = 0.01
        self.rot_bound = 0.1
        self.rot_euler_bound = 0.05  # ~2.86deg

    @property
    def controllers(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        # PD ee position
        # 안씀
        # arm_pd_ee_delta_pos = PDEEPosControllerConfig(
        #     self.arm_joint_names,
        #     -self.ee_delta,
        #     self.ee_delta,
        #     stiffness=self.arm_stiffness,
        #     damping=self.arm_damping,
        #     force_limit=self.arm_force_limit,
        #     ee_link=self.ee_link_name,
        # )
        # 안씀
        # arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
        #     self.arm_joint_names,
        #     -self.ee_delta,
        #     self.ee_delta,
        #     self.rot_bound,
        #     stiffness=self.arm_stiffness,
        #     damping=self.arm_damping,
        #     force_limit=self.arm_force_limit,
        #     ee_link=self.ee_link_name,
        # )

        arm_pd_ee_delta_pose_euler = PDEEPoseEulerControllerConfig(
            self.arm_joint_names,
            -self.ee_delta,
            self.ee_delta,
            self.rot_euler_bound,
            stiffness=100.0,
            damping=20.0,
            # force_limit=[431.97, 431.97, 197.23, 79.79, 79.79, 79.79],
            force_limit=100.0,
            cache_size=3,  # smooth action cache size
            frame="ee",
            smooth=False,  # smooth action or not
            ee_link=self.ee_link_name,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0,
            0.0425,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=0.2,
            # interpolate=True,
            normalize_action=False,
        )

        controller_configs = dict(
            # pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            # pd_ee_delta_pose=dict(
            #     arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            # ),
            pd_ee_delta_pose_euler=dict(
                arm=arm_pd_ee_delta_pose_euler, 
                gripper=gripper_pd_joint_pos
            )
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return [
            CameraConfig(
                uid="hand_realsense",
                p=[0.0, 0.0, 0.0],
                q=[1, 0, 0, 0],
                width=320,
                height=180,
                fov=0.758,
                near=0.01,
                far=5,
                actor_uid="camera_hand_link",
                hide_link=False,
            ),  # sapien camera config
            # StereoDepthCameraConfig(
            #     uid="hand_stereo",
            #     p=[0.0, 0.0, 0.0],
            #     q=[1, 0, 0, 0],
            #     width=320,
            #     height=180,
            #     ir_width=848,  # 848
            #     ir_height=480,  # 480
            #     min_depth=0.01,
            #     fov=0.758,  # 0.758, 1.137
            #     near=0.01,
            #     far=5,
            #     actor_uid="camera_hand_link",
            #     hide_link=False,
            # )
        ]
