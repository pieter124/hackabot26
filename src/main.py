import csv
import sys, os
import cv2
import time
import numpy as np
from control import Control
from preprocess import preprocess_positions
from pal.products.qarm_mini import QArmMini
from hal.content.qarm_mini import QArmMiniKeyboardNavigator, QArmMiniFunctions
from pal.utilities.keyboard import QKeyboard
from pal.utilities.timing   import QTimer


# CONSTANTS
TOWER_HEIGHT = 0.0001
TOWER_X = 0
TOWER_Y = 0.2
BLOCK_HEIGHT = 0.04

SCAN_ERROR_THRESHOLD = 0.01
SCAN_ERROR_TIMEOUT = 8 


# SETUP
kbd         = QKeyboard()
myMiniArm   = QArmMini(hardware=1, id=4)
kbdNav      = QArmMiniKeyboardNavigator(keyboard=kbd, initialPose=myMiniArm.HOME_POSE)
myArmMath   = QArmMiniFunctions()
timer       = QTimer(sampleRate=30.0, totalTime=300.0)
    
# Main tower-stacking loop

def stack(positions, ctrl):
    current_height = 0
    # 1. Initialize Pose
    ctrl.init_pose()
    print("init pos")
    time.sleep(0.3)
    good_z = positions[0][2]
    for i, position in enumerate(positions):
        x, y, z = position
        print(f"\n[MAIN] --- Starting sequence for block {i+1} at ({x}, {y}, {z}) ---")

        ctrl.ascend()

        
        # 2. Hover over block
        if ctrl.hover_to(x, y) == ctrl.FAILED:
            print(f"[MAIN] ERROR: No IK solution to hover at ({x}, {y}). Skipping block.")
            continue

        # 3. Descend to block
        if ctrl.descend(good_z) == ctrl.FAILED:
            print(f"[MAIN] ERROR: Failed to descend to {z}. Skipping block.")
            continue

        time.sleep(2) 
        # 4. Grip the block
        ctrl.grip()
        # Gripping is usually instantaneous, but a short sleep or wait is safe
        time.sleep(0.7)

        ctrl.ascend()

        # 5. Hover to Tower (Drop-off point)
        if ctrl.hover_to(TOWER_X, TOWER_Y, TOWER_HEIGHT + (BLOCK_HEIGHT * current_height + 0.01)) == ctrl.FAILED:
            print(f"[MAIN] ERROR: No IK solution for tower at ({TOWER_X}, {TOWER_Y}, {TOWER_HEIGHT}).")

        # 6. Release the block
        time.sleep(0.5)
        ctrl.release()
        time.sleep(0.5)

        ctrl.ascend() # are we updating the state?

        current_height += 1
        print(f"[MAIN] Successfully stacked block {i+1}.")
            
      
      
def test_vision_estimate():
    '''Tests the depth based vision estimate'''
    # load robot camera
    
    calib_file = os.path.join(os.path.dirname(__file__), "camera_calibration.npz")
    if os.path.exists(calib_file):
        calib = np.load(calib_file)
        CAM_MTX, CAM_DIST = calib["mtx"], calib["dist"]
        print("Camera calibration loaded.")
    else:
        CAM_MTX, CAM_DIST = None, None
        print("No camera calibration file - running without undistort.")
        
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[MAIN] Unable to open camera")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Main", frame)
        if ret:
            preprocess_positions(CAM_MTX,
                                    frame,
                                    show_frame=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            

def test_harness(file:str, ctrl:Control):
    '''Loads test coords for mocked landmark positions'''
    with open(file, "r") as f:
        position_reader = csv.reader(f)
        next(position_reader)
        for row in position_reader:
            print(row)
            status = ctrl.hover_to(float(row[1]), float(row[2]))
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            if status == ctrl.FAILED:
                print(f"[TEST] Skipping ({x}, {y}, {z}) - No IK solution.")
                continue
            elif status == ctrl.BUSY:
                print(f"[TEST] Arm busy, forcing reset for ({x}, {y}, {z})")
                ctrl.hover_to(x, y) # Try one more time after reset
                

if __name__ == "__main__":
    # Get positions of blocks
    ctrl, positions = Control(myArmMath, myMiniArm), preprocess_positions
    ctrl.init_pose()

    with open("positions.csv", "r") as file:
        positions = []
        position_reader = csv.reader(file)
        next(position_reader)
        for row in position_reader:
            positions.append((float(row[1]), float(row[2]), float(row[3])))
    
    try:
        stack(positions, ctrl)
        #test_harness("positions.csv", ctrl)
    except KeyboardInterrupt:
        print("[MAIN] Program interrupted...")

    # test_vision_estimate()
    
    # test_harness("positions.txt")
    # exit()
    # positions = preprocess_positions()

    # Start stacking...
    
# # 