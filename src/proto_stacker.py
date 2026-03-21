import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import inspect
from pal.products.qarm_mini import QArmMini
print(inspect.signature(QArmMini.__init__))
from hal.content.qarm_mini import QArmMiniFunctions
from pal.utilities.timing import QTimer

from vision import detect_blocks
from stacker import BlockStacker


SCAN_SETTLE_TOL_MM = 15.0
SCAN_SETTLE_TIMEOUT_S = 8.0

def ee_position_m(arm_math, joints):
    pose, _, _ = arm_math.forward_kinematics(joints)
    return np.array(pose[:3], dtype=float) * 1000.0  # mm


# Camera calibration.
calib_file = os.path.join(os.path.dirname(__file__), "camera_calibration.npz")
if os.path.exists(calib_file):
    calib = np.load(calib_file)
    CAM_MTX, CAM_DIST = calib["mtx"], calib["dist"]
    print("Camera calibration loaded.")
else:
    CAM_MTX, CAM_DIST = None, None
    print("No camera calibration file - running without undistort.")

# Optional planar table calibration for fixed scan-pose pickup.
table_calib_file = os.path.join(os.path.dirname(__file__), "table_calibration.npz")
if os.path.exists(table_calib_file):
    table_calib = np.load(table_calib_file)
    TABLE_H = table_calib["homography"]
    print("Table calibration loaded.")
else:
    TABLE_H = None
    print("No table calibration file - using ray-plane localization.")

# Setup.
myMiniArm = QArmMini(hardware=1, id=4)
myArmMath = QArmMiniFunctions()
timer = QTimer(sampleRate=45.0, totalTime=300.0)

scan_candidates = [
    (np.array([0.20, 0.0, 0.15]), -np.pi / 4),
    (np.array([0.18, 0.0, 0.15]), -np.pi / 4),
    (np.array([0.20, 0.0, 0.12]), -np.pi / 4),
    (np.array([0.15, 0.0, 0.15]), -np.pi / 4),
    (np.array([0.20, 0.0, 0.10]), -np.pi / 4),
]

SCAN_POSE = myMiniArm.HOME_POSE
for pos, gamma in scan_candidates:
    _, _, num_sol, theta = myArmMath.inverse_kinematics(pos, gamma, myMiniArm.HOME_POSE)
    if num_sol > 0:
        SCAN_POSE = theta.flatten()
        print(f"Scan pose selected: {np.round(pos * 1000.0, 1)} mm  gamma={np.degrees(gamma):.1f} deg")
        break
else:
    print("WARNING: no IK solution for any scan pose - using HOME_POSE")

# Startup diagnostics.
pose_home, rot_home, _ = myArmMath.forward_kinematics(myMiniArm.HOME_POSE)
from stacker import R_CAM_TO_EE as CAM_TO_EE_ROT

cam_forward_world = rot_home @ CAM_TO_EE_ROT @ np.array([0.0, 0.0, 1.0])
print(f"[DIAG] EE at home: {np.round(np.array(pose_home[:3]) * 1000.0, 1)} mm")
print(f"[DIAG] Camera forward in world frame: {np.round(cam_forward_world, 3)}")
print("       (want Z-component < 0 for downward-facing camera)")

cap = cv2.VideoCapture(1)
stacker = BlockStacker(arm=myMiniArm, arm_math=myArmMath, table_homography=TABLE_H)
img_count = 0
scan_pose_xyz = ee_position_m(myArmMath, SCAN_POSE)

# Move arm to scan pose and wait for it to settle.
print("Moving to scan pose...")
scan_start = timer.get_current_time()
scan_err_mm = np.inf
while timer.check():
    myMiniArm.read_write_std(SCAN_POSE, stacker.gripper_pos)
    measured_xyz = ee_position_m(myArmMath, myMiniArm.positionMeasured)
    scan_err_mm = float(np.linalg.norm(measured_xyz - scan_pose_xyz))
    if scan_err_mm <= SCAN_SETTLE_TOL_MM:
        break
    if (timer.get_current_time() - scan_start) >= SCAN_SETTLE_TIMEOUT_S:
        break
    timer.sleep()

measured_xyz = ee_position_m(myArmMath, myMiniArm.positionMeasured)
print(
    f"Scan pose settle: measured={np.round(measured_xyz, 1)} mm  "
    f"target={np.round(scan_pose_xyz, 1)} mm  "
    f"err={scan_err_mm:.1f} mm"
)
if scan_err_mm > SCAN_SETTLE_TOL_MM:
    print("WARNING: arm did not settle tightly at scan pose before starting.")

try:
    print("Stacker running. Press 'q' in the video window to quit.")
    while timer.check():
        detections_for_stacker = []
        ready_for_detection = False
        scan_err_mm = None

        if stacker.state == stacker.IDLE:
            myMiniArm.read_write_std(SCAN_POSE, stacker.gripper_pos)
            measured_xyz = ee_position_m(myArmMath, myMiniArm.positionMeasured)
            scan_err_mm = float(np.linalg.norm((measured_xyz - scan_pose_xyz)))
            ready_for_detection = scan_err_mm <= SCAN_SETTLE_TOL_MM

        ret, frame = cap.read()
        if ret:
            if CAM_MTX is not None:
                frame = cv2.undistort(frame, CAM_MTX, CAM_DIST)

            annotated, detections = detect_blocks(frame)

            if ready_for_detection:
                detections_for_stacker = detections
            else:
                cv2.putText(
                    annotated,
                    "Waiting for scan pose",
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,              # BUG FIX: was scan_err_mm (a float)
                    cv2.LINE_AA,
                )
                if scan_err_mm is not None:
                    cv2.putText(
                        annotated,
                        f"scan err {scan_err_mm:.1f} mm",
                        (20, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow("QArm Mini Feed", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                fname = f"snapshot_{img_count}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Snapshot saved: {fname}")
                img_count += 1
            elif key == ord("q"):
                break

        stacker.update(detections_for_stacker, cam_mtx=CAM_MTX)
        timer.sleep()

except KeyboardInterrupt:
    print("User terminated.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    myMiniArm.terminate()