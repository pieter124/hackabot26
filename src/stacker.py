"""
stacker.py - autonomous one-block-at-a-time tower stacker.
"""

import time
import numpy as np

R_CAM_TO_EE = np.array([
    [ 0,  0,  1],  # Camera +Z (forward) maps to Gripper +X (forward)
    [-1,  0,  0],  # Camera +X (right) maps to Gripper -Y (right)
    [ 0, -1,  0],  # Camera +Y (down) maps to Gripper -Z (down)
], dtype=float)

BLOCK_Z_MM        =   -45.0
TOWER_Z_MM        =   -25.0
TOWER_X_MM        = 249.2
TOWER_Y_MM        =  52.3
BLOCK_HEIGHT_MM   =  40.0
TRAVEL_Z_MM       =  80.0
GRIPPER_OPEN      =   0.0
GRIPPER_CLOSED    =   0.52
GRIP_CURRENT_THR  =   0.10
MOVE_TIMEOUT      =  10.0
POSITION_TOL_MM   =   8.0
TIMEOUT_ACCEPT_MM =  20.0
SAFE_Z_FLOOR_MM   =  -65.0


class BlockStacker:
    IDLE           = "IDLE"
    APPROACH       = "APPROACH"
    DESCEND        = "DESCEND"
    GRIP           = "GRIP"
    LIFT           = "LIFT"
    MOVE_TO_TOWER  = "MOVE_TO_TOWER"
    LOWER_TO_TOWER = "LOWER_TO_TOWER"
    RELEASE        = "RELEASE"
    RETRACT        = "RETRACT"

    def __init__(self, arm, arm_math, table_homography=None):
        self.arm = arm
        self.arm_math = arm_math
        self.table_homography = table_homography

        self.state = self.IDLE
        self.blocks_placed = 0
        self.gripper_pos = GRIPPER_OPEN
        self._holding_block = False

        self._target = None
        self._target_arm = None
        self._waypoint = None
        self._waypoint_xyz = None
        self._state_start = None
        self._dwell_until = None

    def update(self, detections: list[dict], cam_mtx=None) -> float:
        if self.state == self.IDLE:
            self._handle_idle(detections, cam_mtx)
        elif self.state == self.APPROACH:
            self._handle_move_to_waypoint(next_state=self.DESCEND)
        elif self.state == self.DESCEND:
            self._handle_move_to_waypoint(next_state=self.GRIP)
        elif self.state == self.GRIP:
            self._handle_grip()
        elif self.state == self.LIFT:
            self._handle_move_to_waypoint(next_state=self.MOVE_TO_TOWER)
        elif self.state == self.MOVE_TO_TOWER:
            self._handle_move_to_waypoint(next_state=self.LOWER_TO_TOWER)
        elif self.state == self.LOWER_TO_TOWER:
            self._handle_move_to_waypoint(next_state=self.RELEASE)
        elif self.state == self.RELEASE:
            self._handle_release()
        elif self.state == self.RETRACT:
            self._handle_move_to_waypoint(next_state=self.IDLE)
        return self.gripper_pos

    def _handle_idle(self, detections, cam_mtx):
        if not detections:
            return

        best = max(detections, key=lambda d: d["area"])
        xy_arm_mm, source = self._localize_block_xy(best, cam_mtx)
        if xy_arm_mm is None:
            return

        x_arm, y_arm = xy_arm_mm
        z_arm = BLOCK_Z_MM

        self._holding_block = False
        self._target = best
        self._target_arm = (x_arm, y_arm, z_arm)

        print(f"[STACKER] Target: {best['color']} block at arm frame "
              f"({x_arm:.1f}, {y_arm:.1f}, {z_arm:.1f}) mm via {source}")

        self._set_waypoint(x_arm, y_arm, TRAVEL_Z_MM, gamma=-np.pi / 4)
        if self._waypoint is None:
            self._abort_cycle("pickup waypoint has no IK solution")
            return
        self._transition(self.APPROACH)

    def _handle_grip(self):
        if self._dwell_until is None:
            self.gripper_pos = GRIPPER_CLOSED
            self._dwell_until = time.monotonic() + 1.5
            print("[STACKER] Gripping...")

        self._command_active_waypoint()

        if time.monotonic() >= self._dwell_until:
            current = float(self.arm.gripperCurrentMeasured)
            if current >= GRIP_CURRENT_THR:
                self._holding_block = True
                print(f"[STACKER] Grip confirmed (current={current:.3f} A)")
            else:
                print(f"[STACKER] WARNING: low gripper current ({current:.3f} A) - may have missed block")
            self._dwell_until = None
            x, y, _ = self._target_arm
            self._set_waypoint(x, y, TRAVEL_Z_MM, gamma=-np.pi / 4)
            if self._waypoint is None:
                self._abort_cycle("lift waypoint has no IK solution")
                return
            self._transition(self.LIFT)

    def _handle_release(self):
        if self._dwell_until is None:
            self.gripper_pos = GRIPPER_OPEN
            self._dwell_until = time.monotonic() + 0.3
            if self._holding_block:
                self.blocks_placed += 1
                print(f"[STACKER] Block placed. Tower height: {self.blocks_placed}")
            else:
                print("[STACKER] WARNING: release without a confirmed grip")
            self._holding_block = False

        self._command_active_waypoint()

        if time.monotonic() >= self._dwell_until:
            self._dwell_until = None
            self._set_waypoint(TOWER_X_MM, TOWER_Y_MM, TRAVEL_Z_MM, gamma=-np.pi / 4)
            if self._waypoint is None:
                self._abort_cycle("retract waypoint has no IK solution")
                return
            self._transition(self.RETRACT)

    def _handle_move_to_waypoint(self, next_state: str):
        if self._waypoint is None:
            self._abort_cycle(f"missing waypoint while in {self.state}")
            return

        self.arm.read_write_std(self._waypoint, self.gripper_pos)

        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        current_xyz = np.array(pose[:3], dtype=float) * 1000
        target_xyz  = np.array(self._waypoint_xyz, dtype=float)
        dist_mm = float(np.linalg.norm(current_xyz - target_xyz))
        timed_out = (time.monotonic() - self._state_start) > MOVE_TIMEOUT

        if dist_mm < POSITION_TOL_MM:
            if not self._setup_next_waypoint(next_state):
                self._abort_cycle(f"failed to prepare waypoint for {next_state}")
                return
            self._transition(next_state)
            return

        if timed_out:
            if dist_mm <= TIMEOUT_ACCEPT_MM:
                print(f"[STACKER] WARNING: timeout in state {self.state} "
                      f"(dist={dist_mm:.1f} mm) - accepting relaxed tolerance")
                if not self._setup_next_waypoint(next_state):
                    self._abort_cycle(f"failed to prepare waypoint for {next_state}")
                    return
                self._transition(next_state)
                return

            print(f"[STACKER] ERROR: timeout in state {self.state} "
                  f"(dist={dist_mm:.1f} mm, current={np.round(current_xyz,1)}, "
                  f"target={np.round(target_xyz,1)})")
            self._abort_cycle(f"arm did not reach waypoint for {self.state}")

    def _setup_next_waypoint(self, next_state: str) -> bool:
        x, y, z = self._target_arm if self._target_arm else (TOWER_X_MM, TOWER_Y_MM, TOWER_Z_MM)

        if next_state == self.DESCEND:
            self._set_waypoint(x, y, z, gamma=-np.pi / 2)
            return self._waypoint is not None

        if next_state == self.MOVE_TO_TOWER:
            self._set_waypoint(TOWER_X_MM, TOWER_Y_MM, TRAVEL_Z_MM, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.LOWER_TO_TOWER:
            drop_z = TOWER_Z_MM + self.blocks_placed * BLOCK_HEIGHT_MM
            self._set_waypoint(TOWER_X_MM, TOWER_Y_MM, drop_z, gamma=-np.pi / 4)
            print(f"[STACKER] Dropping at tower Z = {drop_z:.1f} mm")
            return self._waypoint is not None

        if next_state == self.IDLE:
            self._target = None
            self._target_arm = None
            self._waypoint = None
            self._waypoint_xyz = None
            return True

        return True

    def _set_waypoint(self, x_mm, y_mm, z_mm, gamma=-np.pi / 4):
        z_mm = max(z_mm, SAFE_Z_FLOOR_MM)
        pos_mm = np.array([x_mm, y_mm, z_mm], dtype=np.float64)
        _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics(
            pos_mm / 1000.0, gamma, self.arm.positionMeasured)
        if num_sol > 0:
            self._waypoint = theta_opt.flatten()
            self._waypoint_xyz = pos_mm
            print(f"[STACKER] Waypoint set: ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm")
        else:
            print(f"[STACKER] WARNING: no IK solution for ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm")
            self._waypoint = None
            self._waypoint_xyz = None

    def _command_active_waypoint(self):
        hold_joints = self._waypoint if self._waypoint is not None else self.arm.positionMeasured
        self.arm.read_write_std(hold_joints, self.gripper_pos)

    def _abort_cycle(self, reason: str):
        print(f"[STACKER] ERROR: {reason} - aborting cycle")
        self.gripper_pos = GRIPPER_OPEN
        self._holding_block = False
        self._target = None
        self._target_arm = None
        self._dwell_until = None
        
        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        # FIX: Multiply by 1000 so we don't try to move inside the base
        x_mm = float(pose[0]) * 1000.0   
        y_mm = float(pose[1]) * 1000.0   
        
        self._set_waypoint(x_mm, y_mm, TRAVEL_Z_MM, gamma=-np.pi / 4)
        if self._waypoint is not None and self.state != self.IDLE:
            self._transition(self.RETRACT)
        else:
            self._waypoint = None
            self._waypoint_xyz = None
            if self.state != self.IDLE:
                self._transition(self.IDLE)

    def _localize_block_xy(self, detection, cam_mtx):
        if self.table_homography is not None:
            pixel = np.array([detection["cx"], detection["cy"], 1.0], dtype=float)
            arm_xyw = self.table_homography @ pixel
            scale = float(arm_xyw[2])
            if abs(scale) < 1e-6:
                return None, "planar-homography"
            return (float(arm_xyw[0] / scale), float(arm_xyw[1] / scale)), "planar-homography"

        if cam_mtx is not None:
            fx = float(cam_mtx[0, 0])
            fy = float(cam_mtx[1, 1])
            cx = float(cam_mtx[0, 2])
            cy = float(cam_mtx[1, 2])
        else:
            fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0

        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        pose_mm = np.array(pose[:3], dtype=float) * 1000

        # --- THE SILVER BULLET FIX ---
        # Bypass the flipped rotation matrix and map directly to the world frame.
        # Assuming camera top points toward the robot base:
        ray_arm = np.array([
            (detection["cy"] - cy) / fy,    # Image Down maps to World Forward (+X)
            -(detection["cx"] - cx) / fx,   # Image Right maps to World Right (-Y)
            -1.0                            # Camera Forward maps to World Down (-Z)
        ])

        # Intersect the ray at the top surface of the block
        z_table = BLOCK_Z_MM + BLOCK_HEIGHT_MM / 2.0
        dz = float(ray_arm[2])
        
        if abs(dz) < 0.01:
            return None, "ray-plane"

        t = (z_table - float(pose_mm[2])) / dz
        if t < 0.05:
            print(f"[STACKER] WARNING: table intersection behind/too close to camera (t={t:.3f}), skipping")
            return None, "ray-plane"

        block_pos = pose_mm + t * ray_arm
        x_arm = float(block_pos[0])
        y_arm = float(block_pos[1])
        return (x_arm, y_arm), "ray-plane"

    def _transition(self, new_state: str):
        print(f"[STACKER] {self.state} -> {new_state}")
        self.state = new_state
        self._state_start = time.monotonic()