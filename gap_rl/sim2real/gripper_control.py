import time
import numpy as np

import rospy
from gap_rl.sim2real.robotiq_2f_85_gripper import Robotiq2FingerGripper


class GripperController(Robotiq2FingerGripper):
    r"""Gripper controller using serial port

    Attention:
        pos in [0, 0.085] (cm)
        vel in [0, 100] (%)
        force in [0, 100] (%)
    """
    def __init__(self, GRIPPER_PORT="/dev/ttyUSB0", sync=False):
        """init gripper controller.

        Args:
            GRIPPER_PORT (str): gripper usb port, usually be "/dev/ttyUSB0"

        Raises:
            RuntimeError: Cannot acitvate gripper
        """
        super().__init__(comport=GRIPPER_PORT)
        self.deactivate()  # should deactivate first
        self.activate()
        self.synv = sync
        if not self.is_ready():
            raise RuntimeError('Cannot acitvate gripper')

    def is_reset(self):
        """whether gripper is reset (not activated)

        Returns:
            bool
        """
        self.getStatus()
        return super().is_reset()

    def is_moving(self):
        """whether gripper is moving.

        Returns:
            bool
        """
        self.getStatus()
        return super().is_moving()

    def is_ready(self):
        """whether gripper is ready for move.

        Returns:
            bool
        """
        self.getStatus()
        return super().is_ready()

    def is_stopped(self):
        """whether gripper is stoped.

        Returns:
            bool
        """
        self.getStatus()
        return super().is_stopped()

    def object_grasped(self):
        """whether object is grasped.

        Returns:
            bool
        """
        self.getStatus()
        return super().object_detected()

    def deactivate(self):
        """deactivate gripper."""
        print('deactivating gripper')
        super().deactivate_gripper()
        self.sendCommand()
        # wait for deactivating
        while self.getStatus() and not self.is_reset():
            time.sleep(0.1)
        print('gripper deactivated')

    def activate(self):
        """activate gripper."""
        print('activating gripper')
        super().activate_gripper()
        self.sendCommand()
        # wait for activating
        while self.getStatus() and not self.is_ready():
            time.sleep(0.1)
        print('gripper activated')

    def get_pos(self):
        """get gripper position.

        Returns:
            float, pos in [0, 0.085]
        """
        self.getStatus()
        return super().get_pos()

    def get_current(self):
        """get gripper current.

        Returns:
            float
        """
        self.getStatus()
        return super().get_current()

    def get_fault_status(self):
        """get gripper fault status.

        Returns:
            int, gFLT register
        """
        self.getStatus()
        return super().get_fault_status()

    def wait_for_gripper(self):
        """wait movement to end."""
        time.sleep(0.2)
        while self.is_moving() or not self.is_ready():
            # cur = self.get_current()
            # pos = self.get_pos()
            # print(cur, pos)
            time.sleep(0.1)
        time.sleep(0.2)

    def move(self, pos, vel=10, force=10):
        """move gripper to given pos.

        Args:
            pos (float): pos in [0, 0.085]
            vel (int, optional): gripper velocity in [0, 100]. Defaults to 10.
            force (int, optional): gripper force in [0, 100]. Defaults to 10.
        """
        if not self.synv:
            self.wait_for_gripper()
        print(f'gripper move: pos={pos} vel={vel} force={force}')
        self.goto(pos=pos, vel=vel, force=force)
        self.sendCommand()
        if not self.synv:
            self.wait_for_gripper()

    def stop(self):
        """stop gripper."""
        super().stop()
        self.sendCommand()

    def open(self, vel=10, force=10):
        """open gripper."""
        self.move(pos=0.085, vel=vel, force=force)

    def grasp(self, vel=10, force=10):
        """get grasp.

        Returns:
            bool, success
        """
        self.move(pos=0, vel=vel, force=force)
        return self.object_grasped()


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # "pose_based_cartesian_traj_controller", "joint_based_cartesian_traj_controller", "forward_cartesian_traj_controller", "twist_controller"
    rospy.init_node('ur_controller', anonymous=True)
    gripper_controller = GripperController(GRIPPER_PORT="/dev/ttyUSB0", sync=True)
    pos_list = [0.085, 0, 0.04, 0.025, 0, 0.085]
    for pos in pos_list:
        t = time.time()
        gripper_controller.move(pos=pos, vel=100, force=50)
        print("control time: ", time.time() - t)
