"""
stacker.py — autonomous block tower stacker.

Movement is split into explicit stages for debuggability:

  IDLE
    → ROTATE        J1 rotates to face the block (base yaw only)
    → FLY_XY        arm moves to X,Y above block at TRAVEL_Z (horizontal)
    → PRE_DESCEND   arm lowers to just above block top (momentum damping)
    → DESCEND       arm lowers straight down to BLOCK_Z (grip height)
    → GRIP          gripper closes + current check
    → LIFT          arm rises straight up to TRAVEL_Z
    → FLY_TO_TOWER  arm moves horizontally to above tower X,Y
    → LOWER_TOWER   arm lowers to drop height
    → RELEASE       gripper opens
    → RETRACT       arm rises to TRAVEL_Z, returns to IDLE

All positions are in mm throughout. IK receives metres (/ 1000).
FK returns metres — multiply by 1000 immediately after every FK call.
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Constants — tune these for your physical setup
# ---------------------------------------------------------------------------

BLOCK_Z_MM = 100.0   # ← MUST TUNE — jog arm to block and measure
TOWER_X_MM = 200.0
TOWER_Y_MM =   0.0
TOWER_Z_MM = 100.0   # ← same as BLOCK_Z_MM until tuned
BLOCK_HEIGHT_MM = 55.0

# UPGRADED: Travel Z raised to match new scan pose (350mm) to avoid dipping.
TRAVEL_Z_MM = 350.0

PRE_DESCEND_CLEARANCE_MM = 25.0

GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.52
GRIP_CURRENT_THR = 0.10

MOVE_TIMEOUT_S    = 7.0
POSITION_TOL_MM   = 10.0
TIMEOUT_ACCEPT_MM = 35.0

SAFE_Z_FLOOR_MM = 50.0
DETECTION_FRAMES_REQUIRED = 5
TOWER_EXCLUSION_MM = 40.0

LOCALIZE_X_OFFSET_MM =  0.0
LOCALIZE_Y_OFFSET_MM =  0.0

GRIP_NUDGE_MM      = 6.0
GRIP_MAX_RETRIES   = 2
FAIL_BLACKLIST_S = 20.0


# ---------------------------------------------------------------------------
# BlockStacker FSM
# ---------------------------------------------------------------------------

class BlockStacker:

    IDLE         = "IDLE"
    ROTATE       = "ROTATE"        
    FLY_XY       = "FLY_XY"       
    PRE_DESCEND  = "PRE_DESCEND"   
    DESCEND      = "DESCEND"       
    GRIP         = "GRIP"          
    LIFT         = "LIFT"          
    FLY_TO_TOWER = "FLY_TO_TOWER"  
    LOWER_TOWER  = "LOWER_TOWER"   
    RELEASE      = "RELEASE"       
    RETRACT      = "RETRACT"       

    def __init__(self, arm, arm_math, table_homography=None):
        self.arm             = arm
        self.arm_math        = arm_math
        self.table_homography = table_homography

        self.state           = self.IDLE
        self.blocks_placed   = 0
        self.gripper_pos     = GRIPPER_OPEN
        self._holding_block  = False
        self._grip_retries   = 0

        self._target         = None   
        self._target_arm     = None   

        self._waypoint       = None   
        self._waypoint_xyz   = None   

        self._state_start    = None   
        self._dwell_until    = None   

        self._detection_buffer  = []  
        self._failed_positions  = []  

    def update(self, detections: list[dict], cam_mtx=None) -> float:
        s = self.state

        if s == self.IDLE:
            self._handle_idle(detections, cam_mtx)
        elif s == self.ROTATE:
            self._handle_rotate()
        elif s == self.FLY_XY:
            self._handle_move(next_state=self.PRE_DESCEND)
        elif s == self.PRE_DESCEND:
            self._handle_move(next_state=self.DESCEND)
        elif s == self.DESCEND:
            self._handle_move(next_state=self.GRIP)
        elif s == self.GRIP:
            self._handle_grip()
        elif s == self.LIFT:
            self._handle_move(next_state=self.FLY_TO_TOWER)
        elif s == self.FLY_TO_TOWER:
            self._handle_move(next_state=self.LOWER_TOWER)
        elif s == self.LOWER_TOWER:
            self._handle_move(next_state=self.RELEASE)
        elif s == self.RELEASE:
            self._handle_release()
        elif s == self.RETRACT:
            self._handle_move(next_state=self.IDLE)

        return self.gripper_pos

    def _handle_idle(self, detections, cam_mtx):
        if not detections:
            self._detection_buffer = []
            return

        now = time.monotonic()
        self._failed_positions = [e for e in self._failed_positions if e[2] > now]

        candidates = []
        for det in detections:
            xy, source = self._localize_block_xy(det, cam_mtx)
            if xy is None:
                continue
            x, y = xy

            if np.hypot(x - TOWER_X_MM, y - TOWER_Y_MM) < TOWER_EXCLUSION_MM:
                continue

            if any(np.hypot(x - fx, y - fy) < 40.0 for fx, fy, _ in self._failed_positions):
                continue

            candidates.append((det, x, y))

        if not candidates:
            self._detection_buffer = []
            return

        best_det, best_x, best_y = min(candidates, key=lambda c: np.hypot(c[1], c[2]))

        if self._detection_buffer:
            _, last_x, last_y = self._detection_buffer[-1]
            if np.hypot(best_x - last_x, best_y - last_y) > 30.0:
                self._detection_buffer = []

        self._detection_buffer.append((best_det, best_x, best_y))

        if len(self._detection_buffer) < DETECTION_FRAMES_REQUIRED:
            return  

        x_arm = float(np.mean([bx for _, bx, _ in self._detection_buffer]))
        y_arm = float(np.mean([by for _, _, by in self._detection_buffer]))
        best_det = self._detection_buffer[-1][0]
        self._detection_buffer = []

        self._holding_block  = False
        self._grip_retries   = 0
        self._target         = best_det
        self._target_arm     = (x_arm, y_arm, BLOCK_Z_MM)

        print(f"\n[STACKER] ── New target ──────────────────────────────")
        print(f"[STACKER]   Color : {best_det['color']}")
        print(f"[STACKER]   XY    : ({x_arm:.1f}, {y_arm:.1f}) mm")
        print(f"[STACKER]   Z     : {BLOCK_Z_MM:.1f} mm  (grip height)")
        print(f"[STACKER]   θ base: {np.degrees(np.arctan2(y_arm, x_arm)):.1f} deg")
        print(f"[STACKER] ──────────────────────────────────────────────")

        self.gripper_pos = GRIPPER_OPEN

        self._set_waypoint_rotate(x_arm, y_arm)
        if self._waypoint is None:
            self._abort_cycle("ROTATE: no IK solution")
            return
        self._transition(self.ROTATE)

    def _handle_rotate(self):
        if self._waypoint is None:
            self._abort_cycle("ROTATE: waypoint lost")
            return

        self.arm.read_write_std(self._waypoint, self.gripper_pos)

        j1_current = float(self.arm.positionMeasured[0])
        j1_target  = float(self._waypoint[0])
        j1_err_deg = abs(np.degrees(j1_current - j1_target))

        timed_out = (time.monotonic() - self._state_start) > 3.0  

        if j1_err_deg < 3.0 or timed_out:
            if timed_out and j1_err_deg >= 3.0:
                print(f"[STACKER] ROTATE timeout — J1 err={j1_err_deg:.1f} deg, continuing")

            x, y, _ = self._target_arm
            self._set_waypoint(x, y, TRAVEL_Z_MM, gamma=-np.pi / 4)
            if self._waypoint is None:
                self._abort_cycle("FLY_XY: no IK solution")
                return
            self._transition(self.FLY_XY)

    def _handle_grip(self):
        if self._dwell_until is None:
            self.gripper_pos  = GRIPPER_CLOSED
            self._dwell_until = time.monotonic() + 0.8

            pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
            actual = np.array(pose[:3]) * 1000.0
            tx, ty, tz = self._target_arm
            print(f"[GRIP DIAG] actual=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) mm  "
                  f"target=({tx:.1f},{ty:.1f},{tz:.1f}) mm  "
                  f"XY-err=({actual[0]-tx:.1f},{actual[1]-ty:.1f}) mm  "
                  f"Z-err={actual[2]-tz:.1f} mm")

        self._command_active_waypoint()

        if time.monotonic() >= self._dwell_until:
            current = float(self.arm.gripperCurrentMeasured)
            print(f"[STACKER] Gripper current: {current:.3f} A")

            if current >= GRIP_CURRENT_THR:
                self._holding_block = True
                self._grip_retries  = 0
                print(f"[STACKER] ✓ Grip confirmed")
                self._dwell_until = None
                x, y, _ = self._target_arm
                self._set_waypoint(x, y, TRAVEL_Z_MM, gamma=-np.pi / 4)
                if self._waypoint is None:
                    self._abort_cycle("LIFT: no IK solution")
                    return
                self._transition(self.LIFT)

            elif self._grip_retries < GRIP_MAX_RETRIES:
                self._grip_retries += 1
                self._dwell_until = None
                x, y, _ = self._target_arm
                r     = float(np.hypot(x, y))
                angle = float(np.arctan2(y, x))
                r_new = r - GRIP_NUDGE_MM * self._grip_retries
                x_new = r_new * np.cos(angle)
                y_new = r_new * np.sin(angle)
                self._target_arm = (x_new, y_new, BLOCK_Z_MM)
                print(f"[STACKER] Grip retry {self._grip_retries}/{GRIP_MAX_RETRIES} — nudging to ({x_new:.1f},{y_new:.1f}) mm")
                self._set_waypoint(x_new, y_new, BLOCK_Z_MM, gamma=-np.pi / 4)
                if self._waypoint is None:
                    self._abort_cycle("GRIP retry: no IK solution")
                    return
                self._transition(self.DESCEND)

            else:
                print(f"[STACKER] ✗ Grip failed after {GRIP_MAX_RETRIES} retries")
                self._dwell_until = None
                self._abort_cycle("grip failed — max retries reached")

    def _handle_release(self):
        if self._dwell_until is None:
            self.gripper_pos  = GRIPPER_OPEN
            self._dwell_until = time.monotonic() + 0.4
            if self._holding_block:
                self.blocks_placed += 1
                print(f"[STACKER] ✓ Block placed. Tower height: {self.blocks_placed}")
            else:
                print("[STACKER] WARNING: release without confirmed grip")
            self._holding_block = False

        self._command_active_waypoint()

        if time.monotonic() >= self._dwell_until:
            self._dwell_until = None
            self._set_waypoint(TOWER_X_MM, TOWER_Y_MM, TRAVEL_Z_MM, gamma=-np.pi / 4)
            if self._waypoint is None:
                self._abort_cycle("RETRACT: no IK solution")
                return
            self._transition(self.RETRACT)

    def _handle_move(self, next_state: str):
        if self._waypoint is None:
            self._abort_cycle(f"missing waypoint in {self.state}")
            return

        self.arm.read_write_std(self._waypoint, self.gripper_pos)

        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        current_mm = np.array(pose[:3]) * 1000.0
        target_mm  = np.array(self._waypoint_xyz)
        dist_mm    = float(np.linalg.norm(current_mm - target_mm))
        elapsed    = time.monotonic() - self._state_start
        timed_out  = elapsed > MOVE_TIMEOUT_S

        arrived = dist_mm < POSITION_TOL_MM
        soft_ok = timed_out and dist_mm <= TIMEOUT_ACCEPT_MM

        if arrived or soft_ok:
            if not self._prepare_next_waypoint(next_state):
                self._abort_cycle(f"could not prepare waypoint for {next_state}")
                return
            self._transition(next_state)
            return

        if timed_out:
            print(f"[STACKER] ERROR: {self.state} timed out — dist={dist_mm:.1f} mm")
            self._abort_cycle(f"{self.state} failed to reach waypoint")

    def _prepare_next_waypoint(self, next_state: str) -> bool:
        x, y, z = self._target_arm if self._target_arm else (TOWER_X_MM, TOWER_Y_MM, TOWER_Z_MM)

        if next_state == self.PRE_DESCEND:
            pre_z = BLOCK_Z_MM + BLOCK_HEIGHT_MM / 2.0 + PRE_DESCEND_CLEARANCE_MM
            self._set_waypoint(x, y, pre_z, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.DESCEND:
            self._set_waypoint(x, y, z, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.FLY_TO_TOWER:
            self._set_waypoint(TOWER_X_MM, TOWER_Y_MM, TRAVEL_Z_MM, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.LOWER_TOWER:
            drop_z = TOWER_Z_MM + self.blocks_placed * BLOCK_HEIGHT_MM
            self._set_waypoint(TOWER_X_MM, TOWER_Y_MM, drop_z, gamma=-np.pi / 4)
            return self._waypoint is not None

        if next_state == self.IDLE:
            self._target     = None
            self._target_arm = None
            self._waypoint     = None
            self._waypoint_xyz = None
            return True

        return True   

    def _set_waypoint_rotate(self, x_mm, y_mm):
        theta = np.array(self.arm.positionMeasured, dtype=float).copy()
        theta[0] = float(np.arctan2(y_mm, x_mm))
        self._waypoint     = theta
        self._waypoint_xyz = None   

    def _set_waypoint(self, x_mm, y_mm, z_mm, gamma=-np.pi / 4):
        z_mm = max(z_mm, SAFE_Z_FLOOR_MM)
        pos_m = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0

        theta_seed    = np.array(self.arm.positionMeasured, dtype=float).copy()
        theta_seed[0] = float(np.arctan2(y_mm, x_mm))

        _, _, num_sol, theta_opt = self.arm_math.inverse_kinematics(pos_m, gamma, theta_seed)

        if num_sol > 0:
            self._waypoint     = theta_opt.flatten()
            self._waypoint_xyz = np.array([x_mm, y_mm, z_mm])  
        else:
            print(f"[IK] ({x_mm:.1f},{y_mm:.1f},{z_mm:.1f}) mm  γ={np.degrees(gamma):.0f}°  ✗ no solution")
            self._waypoint     = None
            self._waypoint_xyz = None

    def _command_active_waypoint(self):
        joints = (self._waypoint if self._waypoint is not None else self.arm.positionMeasured)
        self.arm.read_write_std(joints, self.gripper_pos)

    def _abort_cycle(self, reason: str):
        print(f"\n[STACKER] ✗ ABORT: {reason}")
        self.gripper_pos    = GRIPPER_OPEN
        self._holding_block = False
        self._dwell_until   = None
        self._grip_retries  = 0

        if self._target_arm is not None:
            fx, fy, _ = self._target_arm
            expiry = time.monotonic() + FAIL_BLACKLIST_S
            self._failed_positions.append((fx, fy, expiry))

        self._target     = None
        self._target_arm = None

        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        x_mm = float(pose[0]) * 1000.0
        y_mm = float(pose[1]) * 1000.0
        self._set_waypoint(x_mm, y_mm, TRAVEL_Z_MM, gamma=-np.pi / 4)
        if self._waypoint is not None and self.state != self.IDLE:
            self._transition(self.RETRACT)
        else:
            self._waypoint     = None
            self._waypoint_xyz = None
            if self.state != self.IDLE:
                self._transition(self.IDLE)

    def _localize_block_xy(self, detection, cam_mtx):
        if self.table_homography is not None:
            px = np.array([detection["cx"], detection["cy"], 1.0], dtype=float)
            w  = self.table_homography @ px
            if abs(w[2]) < 1e-6:
                return None, "homography"
            x = float(w[0] / w[2]) + LOCALIZE_X_OFFSET_MM
            y = float(w[1] / w[2]) + LOCALIZE_Y_OFFSET_MM
            return (x, y), "homography"

        if cam_mtx is not None:
            fx = float(cam_mtx[0, 0]);  fy = float(cam_mtx[1, 1])
            cx = float(cam_mtx[0, 2]);  cy = float(cam_mtx[1, 2])
        else:
            fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0

        pose, _, _ = self.arm_math.forward_kinematics(self.arm.positionMeasured)
        pose_mm = np.array(pose[:3]) * 1000.0

        ray = np.array([
            -(detection["cy"] - cy) / fy,   
            -(detection["cx"] - cx) / fx,   
            -1.0,                            
        ])

        z_target = BLOCK_Z_MM + BLOCK_HEIGHT_MM / 2.0
        dz = float(ray[2])
        if abs(dz) < 0.01:
            return None, "ray-plane"

        t = (z_target - float(pose_mm[2])) / dz
        if t < 1.0:   
            return None, "ray-plane"

        pos = pose_mm + t * ray
        x = float(pos[0]) + LOCALIZE_X_OFFSET_MM
        y = float(pos[1]) + LOCALIZE_Y_OFFSET_MM
        return (x, y), "ray-plane"

    def _transition(self, new_state: str):
        print(f"[STACKER] {self.state:15s} → {new_state}")
        self.state        = new_state
        self._state_start = time.monotonic()