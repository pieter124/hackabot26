
#finds the pal folder and adds it to the path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2  # <--- Added this for the camera
import time
import numpy as np
from pal.products.qarm_mini import QArmMini
from hal.content.qarm_mini import QArmMiniKeyboardNavigator, QArmMiniFunctions
from pal.utilities.keyboard import QKeyboard
from pal.utilities.timing   import QTimer

from vision import detect_blocks, estimate_3d_position

# Load camera calibration if available
_calib_file = os.path.join(os.path.dirname(__file__), "camera_calibration.npz")
if os.path.exists(_calib_file):
    _calib = np.load(_calib_file)
    CAM_MTX, CAM_DIST = _calib["mtx"], _calib["dist"]
    print("Camera calibration loaded.")
else:
    CAM_MTX, CAM_DIST = None, None
    print("No calibration file found — running without undistort.")

CLOSED_POSITION = 0.52

def toggle_gripper(gripper_pos):
    print("Attempt toggle")
    print(f"Gripper | "
    f"Pos: {float(myMiniArm.gripperPositionMeasured):.3f} | "
    f"Speed: {float(myMiniArm.gripperSpeedMeasured):.3f} | "
    f"Current: {float(myMiniArm.gripperCurrentMeasured):.3f} A")
    time.sleep(0.2)
    if gripper_pos == CLOSED_POSITION:
        return 0
    else:
        return CLOSED_POSITION

# -- Section A: Setup --
kbd         = QKeyboard()
myMiniArm   = QArmMini(hardware=1, id=4)
kbdNav      = QArmMiniKeyboardNavigator(keyboard=kbd, initialPose=myMiniArm.HOME_POSE)
myArmMath   = QArmMiniFunctions()
timer       = QTimer(sampleRate=30.0, totalTime=300.0)

# -- CAMERA SETUP --
# On Ubuntu, 0 is usually your laptop webcam.
# 1 or 2 is usually the QArm camera. Try different numbers if it's the wrong one!
cap = cv2.VideoCapture(1)

gripper_open = True
gripper_pos = 0
d_key_prev = False

try:
    # log_thread = threading.Thread(target=log_gripper)
    # log_thread.start()
    img_count = 0
    print("Main loop started. Press 'ESC' on the video window to stop.")
    i = 0
    while timer.check():
        i = (i + 1) % 50
        kbd.update()
        # 1. Capture a frame from the camera
        ret, frame = cap.read()
        if ret:
            if CAM_MTX is not None:
                frame = cv2.undistort(frame, CAM_MTX, CAM_DIST)
            # Show the video feed in a window
            annotated, detections = detect_blocks(frame)

            # Print any new detections to the terminal
            # for d in detections:
            #     print(f"[BLOCK] {d['color']:5s}  centre=({d['cx']:4d},{d['cy']:4d})  area={d['area']:.0f}px²")
            for d in detections:
                x, y, z = estimate_3d_position(d, cam_mtx=CAM_MTX)
                if i % 30 == 0:
                    print(f"[{d['color']:5s}] X:{x:7.1f}mm  Y:{y:7.1f}mm  Z:{z:7.1f}mm")

            #cv2.imshow("QArm Mini Feed", frame)

            # Show annotated feed
            cv2.imshow("QArm Mini Feed", annotated)

            # OPTIONAL: Save a frame if you press 's' (useful for Gemini later)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite(f"snapshot{img_count}.jpg", frame)
                print("Snapshot saved!")
                img_count += 1


        d_key_now = kbd.states[kbd.K_D]
        if d_key_now and not d_key_prev:
            gripper_pos = toggle_gripper(gripper_pos)
        d_key_prev = d_key_now  

        # myMiniArm.write_gripper_position(0.1)
        # myMiniArm.write_gripper_PWM(-0.4) # Apply 40% power to close
        # myMiniArm.write_gripper_PWM(0)    # STOP the motor so it doesn't overheat

        # 2. Forward kinematics — get current end-effector pose in base frame (metres)
        pose, rotMatrix, gamma = myArmMath.forward_kinematics(myMiniArm.positionMeasured)
        print(pose)

        # 3. If blocks detected, compute absolute position and move to first one
        target_joints = kbdNav.move_joints_with_keyboard(timer.get_sample_time(), speed=np.pi/4)
        if detections:
            d = detections[0]  # target the first detected block
            cam_x, cam_y, cam_z = estimate_3d_position(d, cam_mtx=CAM_MTX)

            # Block position in camera frame (metres): +X right, +Y down, +Z depth
            cam_pos_m = np.array([cam_x, cam_y, cam_z], dtype=np.float64) / 1000.0

            # Camera-to-EE frame rotation — adjust to match physical camera mounting.
            # Default: camera Z (depth) = EE X (forward), camera X = EE Y, camera Y = EE -Z
            R_cam_to_ee = np.array([
                [0, 0, 1],   # EE X  = camera Z (depth)
                [1, 0, 0],   # EE Y  = camera X (right)
                [0, -1, 0],  # EE Z  = camera -Y (up)
            ], dtype=np.float64)

            # Full chain: camera -> EE frame -> base frame (using FK rotation matrix)
            block_pos = pose + rotMatrix @ (R_cam_to_ee @ cam_pos_m) - np.array([0.0, 0.0, 0.07])

            # Gripper pointing straight down for pickup
            pickup_gamma = -np.pi / 2

            _, _, numSol, thetaOpt = myArmMath.inverse_kinematics(
                block_pos, pickup_gamma, myMiniArm.positionMeasured)

            if numSol > 0:
                target_joints = thetaOpt.flatten()
                if i % 30 == 0:
                    print(f"Moving to {d['color']} block at {block_pos*1000} mm")
            else:
                if i % 30 == 0:
                    print("Block out of reach — no IK solution")

        myMiniArm.read_write_std(target_joints, gripper_pos)

        # grip = input()
        timer.sleep()

except KeyboardInterrupt:
    print('Received user terminate command.')

finally:
    # log_thread.join()
    # Cleanup
    cap.release()          # Close the camera
    cv2.destroyAllWindows() # Close the video window
    myMiniArm.terminate()