"""
stacker.py - autonomous one-block-at-a-time tower stacker (Claw Machine Edition).
"""

import time
import numpy as np

R_CAM_TO_EE = np.array([
    [ 0,  0,  1],  
    [-1,  0,  0],  
    [ 0, -1,  0],  
], dtype=float)

BLOCK_Z_MM        =   -35.0  
TOWER_Z_MM        =   -15.0  
TOWER_X_MM        = 200.0   
TOWER_Y_MM        =   0.0   
BLOCK_HEIGHT_MM   =  40.0
TRAVEL_Z_MM       =  80.0
GRIPPER_OPEN      =   0.0
GRIPPER_CLOSED    =   0.52
GRIP_CURRENT_THR  =   0.10
MOVE_TIMEOUT      =  10.0
POSITION_TOL_MM   =   8.0
TIMEOUT_ACCEPT_MM =  30.0
SAFE_Z_FLOOR_MM   =  -65.0

LOCALIZE_X_OFFSET_MM = 0.0
LOCALIZE_Y_OFFSET_MM = 0.0

DETECTION_FRAMES_REQUIRED = 5
TOWER_EXCLUSION_MM = 35.0   


class BlockStacker:
    # --- CLAW MACHINE STATES ---
    IDLE           = "IDLE"
    LIFT_TO_TRAVEL = "LIFT_TO_TRAVEL"  # Get to safe height first
    APPROACH_X     = "APPROACH_X"      # Move X axis only
    APPROACH_Y     = "APPROACH_Y"      # Move Y axis only
    DESCEND        = "DESCEND"         # Drop Z axis only
    GRIP           = "GRIP"
    LIFT           = "LIFT"            # Pull Z axis straight up
    MOVE_TOWER_X   = "MOVE_TOWER_X"    # Move X back to tower
    MOVE_TOWER_Y   = "MOVE_TOWER_Y"    # Move Y back to tower
    LOWER_TO_TOWER = "LOWER_TO_TOWER"  # Drop Z to tower
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
        self._detection_buffer = []  
        self._failed_positions  = []  

    def update(self, detections: list[dict], cam_mtx=None) -> float:
        s = self.state
        if s == self.IDLE:
            self._handle_idle(detections, cam_mtx)
        elif s == self.LIFT_TO_TRAVEL:
            self._handle_move_to_waypoint(next_state=self.APPROACH_X)
        elif s == self.APPROACH_X:
            self._handle_move_to_waypoint(next_state=self.APPROACH_Y)
        elif s == self.APPROACH_Y:
            self._handle_move_to_waypoint(next_state=self.DESCEND)
        elif s == self.DESCEND:
            self._handle_move_to_waypoint(next_state=self.GRIP)
        elif s == self.GRIP:
            self._handle_grip()
        elif s == self.LIFT:
            self._handle_move_to_waypoint(next_state=self.MOVE_TOWER_X)
        elif s == self.MOVE_TOWER_X:
            self._handle_move_to_waypoint(next_state=self.MOVE_TOWER_Y)
        elif s == self.MOVE_TOWER_Y:
            self._handle_move_to_waypoint(next_state=self.LOWER_TO_TOWER)
        elif s == self.LOWER_TO_TOWER:
            self._handle_move_to_waypoint(next_state=self.RELEASE)
        elif s == self.RELEASE:
            self._handle_release()
        elif s == self.RETRACT:
            self._handle_move_to_waypoint(next_state=self.IDLE)
        return self.gripper_pos

    def _handle_idle(self, detections, cam_mtx):
        if not detections:
            self._detection_buffer = []
            return

        now = time.monotonic()
        self._failed_positions = [(fx, fy, exp) for fx, fy, exp in self._failed_positions if exp > now]

        candidates = []
        for det in detections:
            xy, _ = self._localize_block_xy(det, cam_mtx)
            if xy is None:
                continue
            x, y = xy
            if float(np.hypot(x - TOWER_X_MM, y - TOWER_Y_MM)) < TOWER_EXCLUSION_MM:
                continue
            if any(float(np.hypot(x - fx, y - fy)) < 40.0 for fx, fy, _ in self._failed_positions):
                continue
            candidates.append((det, x, y))

        if not candidates:
            self._detection_buffer = []
            return

        best_det, best_x, best_y = min(candidates, key=lambda c: np.hypot(c[1], c[2]))

        if self._detection_buffer:
            last_det, last_x, last_y = self._detection_buffer[-1]
            if (best_det["color"] != last_det["color"] or
                    float(np.hypot(best_x - last_x, best_y - last_y)) > 30.0):
                self._detection_buffer = []

        self._detection_buffer.append((best_det, best_x, best_y))

        if len(self._detection_buffer) < DETECTION_FRAMES_REQUIRED:
            return

        x_arm = float(np.mean([bx for _, bx, _ in self._detection_buffer]))
        y_arm = float(np.mean([by for _, _, by in self._detection_buffer]))
        best_det = self._detection_buffer[-1][0]
        self._detection_buffer = []

        z_arm = BLOCK_Z_MM
        theta_deg = float(np.degrees(np.arctan2(y_arm, x_arm)))

        self._holding_block = False
        self._target = best_det
        self._target_arm = (x_arm, y_arm, z_arm)

        print(f"[STACKER] Target: {best_det['color']} block at ({x_arm:.1f}, {y_arm:.1f}, {z_arm:.1f}) mm")

        # Step 1: Elevate to TRAVEL_Z_MM exactly where we currently are (no X/Y movement yet)
        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        current_x = float(pose[0]) * 1000.0
        current_y = float(pose[1]) * 1000.0
        
        self._set_waypoint(current_x, current_y, TRAVEL_Z_MM, gamma=-np.pi / 4)
        if self._waypoint is None:
            self._set_waypoint(current_x, current_y, TRAVEL_Z_MM, gamma=-np.pi / 2)
            
        if self._waypoint is None:
            self._abort_cycle("Could not find IK to lift to travel height")
            return
            
        self._transition(self.LIFT_TO_TRAVEL)

    def _handle_grip(self):
        if self._dwell_until is None:
            self.gripper_pos = GRIPPER_CLOSED
            self._dwell_until = time.monotonic() + 0.5
            pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
            actual_xyz = np.array(pose[:3], dtype=float) * 1000.0
            tx, ty, tz = self._target_arm
            print(f"[STACKER] Gripping... XY-err=({actual_xyz[0]-tx:.1f}, {actual_xyz[1]-ty:.1f}) mm")

        self._command_active_waypoint()

        if time.monotonic() >= self._dwell_until:
            current = float(self.arm.gripperCurrentMeasured)
            if current >= GRIP_CURRENT_THR:
                self._holding_block = True
                print(f"[STACKER] Grip confirmed (current={current:.3f} A)")
            else:
                print(f"[STACKER] WARNING: low gripper current ({current:.3f} A)")
            self._dwell_until = None
            
            # Straight lift up!
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
                if not self._setup_next_waypoint(next_state):
                    self._abort_cycle(f"failed to prepare waypoint for {next_state}")
                    return
                self._transition(next_state)
                return
            self._abort_cycle(f"arm did not reach waypoint for {self.state}")

    def _setup_next_waypoint(self, next_state: str) -> bool:
        target_x, target_y, target_z = self._target_arm if self._target_arm else (TOWER_X_MM, TOWER_Y_MM, TOWER_Z_MM)

        # Get current real-world coordinates to isolate axis movement
        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        current_y = float(pose[1]) * 1000.0

        if next_state == self.APPROACH_X:
            # Move X axis, lock Y axis at current, lock Z at Travel Height
            self._set_waypoint(target_x, current_y, TRAVEL_Z_MM, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.APPROACH_Y:
            # Move Y axis, lock X at target, lock Z at Travel Height
            self._set_waypoint(target_x, target_y, TRAVEL_Z_MM, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.DESCEND:
            # Drop Z, lock X and Y at target
            self._set_waypoint(target_x, target_y, target_z, gamma=-np.pi / 4)
            if self._waypoint is None:
                self._set_waypoint(target_x, target_y, target_z, gamma=-np.pi / 2)
            return self._waypoint is not None

        if next_state == self.MOVE_TOWER_X:
            # Move X to tower, lock Y at current, keep Z high
            self._set_waypoint(TOWER_X_MM, current_y, TRAVEL_Z_MM, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.MOVE_TOWER_Y:
            # Move Y to tower, lock X at tower, keep Z high
            self._set_waypoint(TOWER_X_MM, TOWER_Y_MM, TRAVEL_Z_MM, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.LOWER_TO_TOWER:
            drop_z = TOWER_Z_MM + self.blocks_placed * BLOCK_HEIGHT_MM
            self._set_waypoint(TOWER_X_MM, TOWER_Y_MM, drop_z, gamma=-np.pi / 4)
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
            print(f"[STACKER] WARNING: no IK sol for ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm")
            self._waypoint = None
            self._waypoint_xyz = None

    def _command_active_waypoint(self):
        hold_joints = self._waypoint if self._waypoint is not None else self.arm.positionMeasured
        self.arm.read_write_std(hold_joints, self.gripper_pos)

    def _abort_cycle(self, reason: str):
        print(f"[STACKER] ERROR: {reason} - aborting cycle")
        self.gripper_pos = GRIPPER_OPEN
        self._holding_block = False
        self._dwell_until = None

        if self._target_arm is not None:
            fx, fy, _ = self._target_arm
            expiry = time.monotonic() + 20.0
            self._failed_positions.append((fx, fy, expiry))

        self._target = None
        self._target_arm = None
        
        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
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
            fx = float(cam_mtx[0, 0]); fy = float(cam_mtx[1, 1])
            cx = float(cam_mtx[0, 2]); cy = float(cam_mtx[1, 2])
        else:
            fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0

        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        pose_mm = np.array(pose[:3], dtype=float) * 1000

        ray_arm = np.array([
            (detection["cy"] - cy) / fy,    
            -(detection["cx"] - cx) / fx,   
            -1.0                            
        ])

        z_table = BLOCK_Z_MM + BLOCK_HEIGHT_MM / 2.0
        dz = float(ray_arm[2])
        
        if abs(dz) < 0.01:
            return None, "ray-plane"

        t = (z_table - float(pose_mm[2])) / dz
        if t < 0.05:
            return None, "ray-plane"

        block_pos = pose_mm + t * ray_arm
        x_arm = float(block_pos[0]) + LOCALIZE_X_OFFSET_MM
        y_arm = float(block_pos[1]) + LOCALIZE_Y_OFFSET_MM
        return (x_arm, y_arm), "ray-plane"

    def _transition(self, new_state: str):
        print(f"[STACKER] {self.state} -> {new_state}")
        self.state = new_state
        self._state_start = time.monotonic()