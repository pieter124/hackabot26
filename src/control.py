from math import sqrt
from pal.utilities.timing import QTimer

import time
import numpy as np


# CONSTANTS
RADIUS = 0.29
GAMMA = -np.pi/2
SAFE_Z = 0.15
OPEN_GRIP= 0.0
CLOSED_GRIP = 0.52
STEPS = 100
INIT_POSE = [0, np.pi, -3*np.pi/2, -np.pi/4]
TICK_RATE = 1.0 / 60.0
OPERATION_RATE = 0.5
SCAN_ERROR_THRESHOLD = 0.01
SCAN_ERROR_TIME_OUT = 5
REQUIRED_STABLE_TICKS = 10

class Control:
    SUCCESS = "SUCCESS"
    BUSY = "BUSY"
    FAILED = "FAILED"
    
    def __init__(self, arm_math, arm):
        self.arm = arm
        self.arm_math = arm_math

        self._coord = INIT_POSE
        self.gripper_position = OPEN_GRIP
        self._timer = QTimer(sampleRate=45.0, totalTime=300.0)
        
        

    def init_pose(self):
        self._coord = INIT_POSE
        self.arm.read_write_std(self._coord, self.gripper_position)
        return self.SUCCESS

    
    def ee_position_m(self, joints):
        '''Estimates the position of the effector'''
        pose, _, _ = self.arm_math.forward_kinematics(joints)
        return np.array(pose[:3], dtype=float)  # m


    def hover_to(self, x, y, z=None) -> str:
        """In meters"""        
        print(max(0, RADIUS**2 - x**2 - y**2))
        position = self.ee_position_m(self.arm.positionMeasured)
        if z is None:
            z = position[2]  # maintain current z if not specified

        i = 0.01
        _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics((x, y, z), GAMMA, self._coord)
        while num_sol < 1:
            _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics((x, y, z-i), GAMMA, self._coord)
            print(f"[CONTROL] No solutions for x:{x} y:{y} z:{z-i}")
            i += 0.01
            if i > 0.15:
                return self.FAILED

        self._coord = theta_opt

        scan_start = self._timer.get_current_time()
        scan_err_mm = np.inf
        scan_pose_xyz = self.ee_position_m(self._coord)

        stable_ticks = 0
        # from proto stacker
        while self._timer.check():
            self.arm.read_write_std(self._coord, self.gripper_position)
            measured_xyz = self.ee_position_m(self.arm.positionMeasured)
            scan_err_mm = float(np.linalg.norm(measured_xyz - scan_pose_xyz))
            if scan_err_mm <= SCAN_ERROR_THRESHOLD:
                stable_ticks += 1
                if stable_ticks >= REQUIRED_STABLE_TICKS:
                    break
            else:
                stable_ticks = 0
            if (self._timer.get_current_time() - scan_start) >= SCAN_ERROR_TIME_OUT:
                break
            self._timer.sleep()
        return self.SUCCESS


    def descend(self, pick_up_height):
        # maintain x, y and reduce z to pick-up height
        pose, _, _ = self.arm_math.forward_kinematics(self._coord)
        x, y, z_start = pose[0], pose[1], pose[2]
        
        for i in range(1, STEPS + 1):
            z = z_start + (pick_up_height - z_start) * i / STEPS
            _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics((x, y, z), GAMMA, self._coord)

            
            if num_sol < 1:
                print(f"No IK solution at step {i} for position ({x}, {y}, {z})")
                continue

            self._coord = theta_opt
            time.sleep(OPERATION_RATE / STEPS)
        self.arm.read_write_std(self._coord, self.gripper_position)

        return self.SUCCESS

    def ascend(self):
        # set the lower joint angles to extend the arm straight up, and keep the wrist angle the same
        # use a while loop and try, except with small increments to go up as high as possible until we hit an IK failure, then stop at the last valid position
        pose, _, _ = self.arm_math.forward_kinematics(self._coord)
        x, y, z = pose[0], pose[1], pose[2]
        i = 0.005
        while True:
            _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics((x, y, z + i), GAMMA, self._coord)
            if num_sol < 1:
                print(f"[CONTROL] No IK solution for ascend at ({x}, {y}, {z + i})")
                break
            self._coord = theta_opt
            i += 0.01
        self.arm.read_write_std(self._coord, self.gripper_position)
        time.sleep(OPERATION_RATE)

    def grip(self):
        print("[CONTROL] Activating Gripper (CLOSING)...")
        self.gripper_position = CLOSED_GRIP
        self.arm.read_write_std(self._coord, self.gripper_position)
        return self.SUCCESS

    def release(self):
        print("[CONTROL] Deactivating Gripper (OPENING)...")
        self.gripper_position = OPEN_GRIP
        self.arm.read_write_std(self._coord, self.gripper_position)
        time.sleep(OPERATION_RATE)
        return self.SUCCESS

