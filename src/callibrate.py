#finds the pal folder and adds it to the path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2  # <--- Added this for the camera
import time
import numpy as np
from pal.products.qarm_mini import QArmMini, QArmMiniCamera
from hal.content.qarm_mini import QArmMiniKeyboardNavigator, QArmMiniFunctions
from pal.utilities.keyboard import QKeyboard
from pal.utilities.timing   import QTimer

from vision import detect_blocks

CLOSED_POSITION = 0.52
def callibrate_camera(images):
    obj_points = [] 
    img_points = [] 

    if len(images) == 0:
        print("Error: No images found! Check your file path and extension.")
    else:
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray,(4,6), None)

            if ret:
                objp = np.zeros((4*6, 3), np.float32)
                objp[:, :2] = np.mgrid[0:4, 0:6].T.reshape(-1, 2)
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                img_points.append(corners2)
                print("✅ Corners found")

                
            else:
                print(f"❌ Failed: Could not find corners in. Check the board size!")

    # Final check before calibration
    if len(obj_points) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        np.savez("camera_calibration.npz", mtx=mtx, dist=dist)
        print("Calibration successful! Saved to camera_calibration.npz")
    else:
        print("Calibration failed: No valid frames were processed.")
        
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
    images = []
    print("Main loop started. Press 'ESC' on the video window to stop.")
    i = 0
    while timer.check():
        i = (i + 1) % 100
        kbd.update()
        # 1. Capture a frame from the camera
        ret, frame = cap.read()
        if ret:
            # Show the video feed in a window
            annotated, detections = detect_blocks(frame)

            # Print any new detections to the terminal
            # for d in detections:
            #     print(f"[BLOCK] {d['color']:5s}  centre=({d['cx']:4d},{d['cy']:4d})  area={d['area']:.0f}px²")

            #cv2.imshow("QArm Mini Feed", frame)

            # Show annotated feed
            cv2.imshow("QArm Mini Feed", annotated)

            # OPTIONAL: Save a frame if you press 's' (useful for Gemini later)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                images.append(frame)
                cv2.imshow("Saved frame", frame)
                cv2.imwrite(f"sample{img_count}.png",frame)
                img_count += 1


        d_key_now = kbd.states[kbd.K_D]
        if d_key_now and not d_key_prev:
            gripper_pos = toggle_gripper(gripper_pos)
        d_key_prev = d_key_now

        # myMiniArm.write_gripper_position(0.1)
        # myMiniArm.write_gripper_PWM(-0.4) # Apply 40% power to close
        # myMiniArm.write_gripper_PWM(0)    # STOP the motor so it doesn't overheat

        # 2. Control the Arm
        # print(f"Calling with pos: {gripper_pos}")
        myMiniArm.read_write_std(
        kbdNav.move_joints_with_keyboard(timer.get_sample_time(), speed=np.pi/4), gripper_pos)

        # 3. Math (Forward Kinematics)
        pose, rotationMatrix, gamma = myArmMath.forward_kinematics(myMiniArm.positionMeasured)
        if i == 0:
            print(f"Pose: {pose}, Rot: {rotationMatrix}, Gamma: {gamma}")

        # grip = input()
        timer.sleep()

except KeyboardInterrupt:
    print('Received user terminate command.')

finally:
    # log_thread.join()
    # Cleanup
    cap.release()          # Close the camera
    cv2.destroyAllWindows() # Close the video window
    callibrate_camera(images)
    myMiniArm.terminate()