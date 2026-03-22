from math import sqrt
from pal.utilities.timing import QTimer

import time
import numpy as np


# CONSTANTS
RADIUS = 0.29
GAMMA = -np.pi/3
SAFE_Z = 0.15
OPEN_GRIP= 0.0
CLOSED_GRIP = 0.52
STEPS = 100
INIT_POSE = [0, 1.3*np.pi/2, -np.pi/2, 0]
TICK_RATE = 1.0 / 60.0
OPERATION_RATE = 0.5
SCAN_ERROR_THRESHOLD = 0.01
SCAN_ERROR_TIME_OUT = 5
REQUIRED_STABLE_TICKS = 25
TOWER_X = 0
TOWER_Y = 0.2


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
        
    def init_pose2(self):
        current = list(self.arm.positionMeasured)  # actual measured joints
        target = INIT_POSE
        steps = 100  # fewer steps = faster, more = smoother

        print(f"[CONTROL/INIT_POSE] Interpolating from {np.round(current, 3)} to {np.round(target, 3)}")

        for i in range(1, steps + 1):
            interp = [
                current[j] + (target[j] - current[j]) * i / steps
                for j in range(len(target))
            ]
            self.arm.read_write_std(interp, self.gripper_position)
            time.sleep(OPERATION_RATE / steps)

        self._coord = target
        self.arm.read_write_std(self._coord, self.gripper_position)
        print(f"[CONTROL/INIT_POSE] Done.")
        return self.SUCCESS

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
            z = SAFE_Z

        x_start, y_start, z_start = position[0], position[1], position[2]

        # Verify IK exists at target before committing to the move
        i = 0.005
        _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics((x, y, z), GAMMA, self._coord)
        while num_sol < 1:
            _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics((x, y, z - i), GAMMA, self._coord)
            print(f"[CONTROL] No solutions for x:{x} y:{y} z:{z-i}")
            i += 0.005
            if i > 0.15:
                return self.FAILED
        z -= (i - 0.005)  # use the last valid z

        # Cartesian-space interpolation from current position to target
        for step in range(1, STEPS + 1):
            t = step / STEPS
            x_i = x_start + (x - x_start) * t
            y_i = y_start + (y - y_start) * t
            z_i = z_start + (z - z_start) * t
            _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics((x_i, y_i, z_i), GAMMA, self._coord)
            if num_sol < 1:
                continue
            self._coord = theta_opt
            self.arm.read_write_std(self._coord, self.gripper_position)
            time.sleep(OPERATION_RATE / STEPS)

        scan_err_mm = np.inf
        scan_pose_xyz = self.ee_position_m(self._coord)

        stable_ticks = 0
        scan_start = self._timer.get_current_time()
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
        arm_height = self.ee_position_m(self.arm.positionMeasured)[2]
        return arm_height


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
            self.arm.read_write_std(self._coord, self.gripper_position)
            time.sleep(OPERATION_RATE / STEPS)

        return self.SUCCESS

    def hover_to_safe(self, x, y, z, steps=40) -> str:
        """
        Cartesian-space interpolation to (x, y, z).
        Rises to SAFE_Z first if current z is below it, then arcs to target.
        Guarantees the arm stays above SAFE_Z for the entire lateral move.
        """
        pose, _, _ = self.arm_math.forward_kinematics(self._coord)
        x_start, y_start, z_start = pose[0], pose[1], pose[2]

        # If below SAFE_Z, rise to SAFE_Z first before moving laterally
        if z_start < SAFE_Z:
            print(f"[CONTROL/HOVER_SAFE] Rising from z={z_start:.4f} to SAFE_Z={SAFE_Z}")
            rise_steps = 20
            for i in range(1, rise_steps + 1):
                z_i = z_start + (SAFE_Z - z_start) * i / rise_steps
                _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics(
                    (x_start, y_start, z_i), GAMMA, self._coord
                )
                if num_sol < 1:
                    print(f"[CONTROL/HOVER_SAFE] No IK on rise at z={z_i:.4f}")
                    continue
                self._coord = theta_opt
                self.arm.read_write_std(self._coord, self.gripper_position)
                time.sleep(OPERATION_RATE / rise_steps)
            x_start, y_start, z_start = x_start, y_start, SAFE_Z

        # Lateral arc at SAFE_Z to target xy, then descend to target z
        print(f"[CONTROL/HOVER_SAFE] Arcing ({x_start:.3f},{y_start:.3f}) -> ({x:.3f},{y:.3f}) at SAFE_Z")
        failed_steps = 0
        for i in range(1, steps + 1):
            t = i / steps
            x_i = x_start + (x - x_start) * t
            y_i = y_start + (y - y_start) * t
            # Hold SAFE_Z for lateral move, then blend down to z in final 30% of steps
            if t < 0.7:
                z_i = SAFE_Z
            else:
                z_i = SAFE_Z + (z - SAFE_Z) * ((t - 0.7) / 0.3)

            _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics(
                (x_i, y_i, z_i), GAMMA, self._coord
            )
            if num_sol < 1:
                failed_steps += 1
                print(f"[CONTROL/HOVER_SAFE] No IK at step {i} ({x_i:.3f},{y_i:.3f},{z_i:.3f})")
                continue

            self._coord = theta_opt
            self.arm.read_write_std(self._coord, self.gripper_position)
            time.sleep(OPERATION_RATE / steps)

        print(f"[CONTROL/HOVER_SAFE] Done. Failed steps: {failed_steps}/{steps}")
        return self.SUCCESS if failed_steps < steps // 2 else self.FAILED

    def ascend(self):
        # set the lower joint angles to extend the arm straight up, and keep the wrist angle the same
        # use a while loop and try, except with small increments to go up as high as possible until we hit an IK failure, then stop at the last valid position
        pose, _, _ = self.arm_math.forward_kinematics(self._coord)
        x, y, z = pose[0], pose[1], pose[2]

        if abs(x - TOWER_X) < 0.05 and abs(y - TOWER_Y) < 0.05:
            print(f"[CONTROL/ASCEND] Near tower, pulling back before ascend")
            _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics(
                (x, y - 0.04, z), GAMMA, self._coord)
            if num_sol > 0:
                self._coord = theta_opt
                self.arm.read_write_std(self._coord, self.gripper_position)
                time.sleep(OPERATION_RATE * 0.5)
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
        pose, _, _ = self.arm_math.forward_kinematics(self._coord)
        z = pose[2]
        time.sleep(OPERATION_RATE)
        return z

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

