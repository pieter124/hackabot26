import csv
import sys, os
import cv2
import time
import numpy as np
import multiprocessing
from control import Control
from preprocess import preprocess_positions
from pal.products.qarm_mini import QArmMini
from hal.content.qarm_mini import QArmMiniKeyboardNavigator, QArmMiniFunctions
from pal.utilities.keyboard import QKeyboard
from pal.utilities.timing   import QTimer
from vision import detect_blocks_and_prongs, get_frame, load_camera_calibration, estimate_3d_position, VISION_CENTRE,calculate_camera_delta_mm


# CONSTANTS
TOWER_HEIGHT = 0.0001
TOWER_X = 0
TOWER_Y = 0.2
BLOCK_HEIGHT = 0.042
STACK_CLEARANCE = 0.003
SAFE_Z = 0.15
TABLE_Z = 0.005

SCAN_ERROR_THRESHOLD = 0.01
SCAN_ERROR_TIMEOUT = 8 
RETRACT_BASE   = 0.055
RETRACT_SCALE  = 0.01 


# SETUP
kbd         = QKeyboard()
myMiniArm   = QArmMini(hardware=1, id=4)
kbdNav      = QArmMiniKeyboardNavigator(keyboard=kbd, initialPose=myMiniArm.HOME_POSE)
myArmMath   = QArmMiniFunctions()
timer       = QTimer(sampleRate=30.0, totalTime=300.0)


def vision_loop(detections_queue, cam_mtx, cam_dist,refinement, ref_queue, height_queue):
    cap = cv2.VideoCapture(1)
    
    while True:
        frame = get_frame(cap, cam_mtx, cam_dist)
        if frame is not None:
            annotated, detections, _ = detect_blocks_and_prongs(frame)
            
            # Draw the static target crosshair on screen
            cv2.drawMarker(annotated, VISION_CENTRE, (0, 255, 255), 
                           cv2.MARKER_CROSS, 20, 2)

            # Replace queue contents with latest detections (non-blocking)
            while not detections_queue.empty():
                try:
                    detections_queue.get_nowait()
                except:
                    pass
            
            # Proceed only if we actually detected something
            if detections:
                det = detections[0]
                
                if refinement.is_set():
                    # 1. Get the depth (Z) of the block
                    if height_queue.empty():
                       depth_guess = 0.07
                    else:
                        height = height_queue.get()
                        height_queue.put(height)          # put back unchanged
                        depth_guess = height - BLOCK_HEIGHT
                    # 2. Calculate the alignment delta
                    dx_mm, dy_mm = calculate_camera_delta_mm(det, cam_mtx, depth_guess)

                    # 3. Send the delta to the robot logic
                    while not ref_queue.empty():
                            ref_queue.get_nowait()
                    ref_queue.put((dx_mm,dy_mm,0))
                    
                    # Debug Visual: Draw a line from vision center to the block
                    cv2.line(annotated, VISION_CENTRE, (det["cx"], det["cy"]), (255, 0, 255), 2)
                    cv2.putText(annotated, f"dx: {dx_mm:.1f}mm, dy: {dy_mm:.1f}mm", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                else:
                    # Get initial pose estimates
                    detections_queue.put(det)
            
            cv2.imshow("Vision", annotated)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


def stack(positions, ctrl):
    current_height = 0

    # 1. Initialize Pose
    ctrl.init_pose()
    print("init pos")
    time.sleep(1.5)
    good_z = positions[0][2]
    
    
    # for i, (x, y, z) in enumerate(positions):
    i = 0
    while True:
        refinement.clear()
        if detections_queue.empty():
            time.sleep(0.1)
            continue
        det = detections_queue.get()
        _, R_ee, _ = myArmMath.forward_kinematics(ctrl._coord)
        x, y, z = estimate_3d_position(det, CAM_MTX, ee_rotation=R_ee)
        print(f"\n[MAIN] --- Starting sequence for block {i+1} at ({x}, {y}, {z}) ---")

        while not height_queue.empty():
            height_queue.get_nowait()
        height = ctrl.ascend()
        print(f"[MAIN] Current height after ascend: {height:.3f}m")
        height_queue.put(height)

        
        # 2. Hover over block
        if ctrl.hover_to(x, y) == ctrl.FAILED:
            print(f"[MAIN] ERROR: No IK solution to hover at ({x}, {y}). Skipping block.")
            continue
        placed = False
        refinement.set()
        new_x, new_y = x, y
        attempts = 0
        while not placed:
            if ref_queue.empty():
                time.sleep(0.1)
                continue

            dx,dy,_ = ref_queue.get()
            new_x = x+  dx / 1000
            new_y = y+  dy / 1000
            # call hover_to with new_x, new_y, and maintain current z
            pos, _, _ = ctrl.arm_math.forward_kinematics(ctrl.arm.positionMeasured)
            print(f"[MAIN] Refinement delta received: dx={dx:.1f}mm, dy={dy:.1f}mm -> New target: ({new_x:.4f}, {new_y:.4f})")
            if res:= ctrl.hover_to(new_x, new_y, pos[2]) == ctrl.FAILED:
                print(f"[MAIN] ERROR: No IK solution to hover at ({new_x}, {new_y}). Skipping block.")
            else:
                while not height_queue.empty():
                    height_queue.get_nowait()
                height_queue.put(res)
                x = new_x
                y = new_y
            if abs(dx) < 7 and abs(dy) < 7:
                placed = True
            else:
                attempts += 1
                if attempts > 5:
                    ctrl.ascend()
                    break
        
        if not placed:
            print(f"[MAIN] Failed to refine position after {attempts} attempts. Moving to next block.")
            continue

        # 3. Descend to block
        input("Press Enter to descend...")
        if ctrl.descend(TABLE_Z) == ctrl.FAILED:
            print(f"[MAIN] ERROR: Failed to descend to {z}. Skipping block.")
            continue

        time.sleep(2.5) 
        # 4. Grip the block
        ctrl.grip()
        # Gripping is usually instantaneous, but a short sleep or wait is safe
        time.sleep(0.7)

        

        # 5. Hover to Tower (Drop-off point)
        drop_z = TOWER_HEIGHT + (BLOCK_HEIGHT * current_height) + STACK_CLEARANCE
        retract_offset = RETRACT_BASE + RETRACT_SCALE * current_height

        retract_x = TOWER_X - retract_offset
        retract_y = TOWER_Y     

        ctrl.ascend()
        
        if ctrl.hover_to(retract_x, retract_y, SAFE_Z) == ctrl.FAILED:
            print(f"[MAIN] ERROR: Could not reach retract position.")
            continue
        time.sleep(0.5)

        if ctrl.descend(drop_z) == ctrl.FAILED:
            print(f"[MAIN] ERROR: Could not descend to drop_z={drop_z:.4f}. Skipping.")
            continue
        print(f"[MAIN] Joints after descend (block {i+1}): {np.round(ctrl._coord, 3)}")

        time.sleep(0.5)
        if ctrl.hover_to(TOWER_X, TOWER_Y, drop_z) == ctrl.FAILED:
            print(f"[MAIN] ERROR: No IK solution for tower at ({TOWER_X}, {TOWER_Y}, {TOWER_HEIGHT}).")
            continue

        # 6. Release the block
        time.sleep(0.5)
        ctrl.release()
        time.sleep(0.5)

        ctrl.ascend() # are we updating the state?

        current_height += 1
        print(f"[MAIN] Successfully stacked block {i+1}.")
        i += 1

        ctrl.init_pose()
        time.sleep(1.5)

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
    CAM_MTX, CAM_DIST = load_camera_calibration("camera_calibration.npz")
    refinement = multiprocessing.Event()
    detections_queue = multiprocessing.Queue()
    ref_queue = multiprocessing.Queue()
    height_queue = multiprocessing.Queue()
    vision_process = multiprocessing.Process(
        target=vision_loop, args=(detections_queue, CAM_MTX, CAM_DIST,refinement,ref_queue,height_queue), daemon=True
    )
    vision_process.start()
    ctrl = Control(myArmMath, myMiniArm)
    ctrl.init_pose()

    with open("positions.csv", "r") as file:
        positions = []
        position_reader = csv.reader(file)
        next(position_reader)
        for row in position_reader:
            positions.append((float(row[1]), float(row[2]), float(row[3])))

    try:
        stack(positions, ctrl)
    except KeyboardInterrupt:
        print("[MAIN] Program interrupted...")
    finally:
        vision_process.terminate()
        vision_process.join()
        print("[MAIN] Vision process stopped.")

    # test_vision_estimate()
    
    # test_harness("positions.txt")
    # exit()
    # positions = preprocess_positions()

    # Start stacking...
    
# # 