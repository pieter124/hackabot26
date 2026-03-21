"""
measure_config.py - physically measure BLOCK_Z_MM and TOWER position.

Instructions
------------
1. Run this script.  The arm moves to home and the camera opens.
2. Use the keyboard to navigate the arm (same keys as the main script).
3. Lower the arm until the gripper just TOUCHES the table surface.
   Press B  →  records that Z as BLOCK_Z_MM.
4. Move the arm so the gripper is directly ABOVE the tower base at
   comfortable drop height, then press T  →  records X, Y as TOWER position.
5. Press Q (or Ctrl-C) to quit.  Measurements are printed at the end.
   Copy the printed values into stacker.py.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from pal.products.qarm_mini import QArmMini
from hal.content.qarm_mini import QArmMiniKeyboardNavigator, QArmMiniFunctions
from pal.utilities.keyboard import QKeyboard
from pal.utilities.timing import QTimer

kbd       = QKeyboard()
myMiniArm = QArmMini(hardware=1, id=4)
kbdNav    = QArmMiniKeyboardNavigator(keyboard=kbd, initialPose=myMiniArm.HOME_POSE)
myArmMath = QArmMiniFunctions()
timer     = QTimer(sampleRate=30.0, totalTime=600.0)

block_z_mm  = None
tower_x_mm  = None
tower_y_mm  = None
i = 0

# Blank window so cv2.waitKey works for B/T/Q keys
_canvas = np.zeros((120, 480, 3), dtype=np.uint8)
cv2.putText(_canvas, "B = table height | T = tower XY | Q = quit",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
cv2.putText(_canvas, "Move arm with keyboard arrows / WASD",
            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
cv2.imshow("measure_config", _canvas)

print("Controls: B = record table Z | T = record tower XY | Q = quit")

try:
    while timer.check():
        i = (i + 1) % 30
        kbd.update()

        pose, _, _ = myArmMath.forward_kinematics(myMiniArm.positionMeasured)
        x_mm = float(pose[0] * 1000)
        y_mm = float(pose[1] * 1000)
        z_mm = float(pose[2] * 1000)

        if i == 0:
            print(f"EE pos:  X={x_mm:7.1f}  Y={y_mm:7.1f}  Z={z_mm:7.1f} mm"
                  f"  |  recorded: BLOCK_Z={block_z_mm}  TOWER=({tower_x_mm},{tower_y_mm})")

        key = cv2.waitKey(1) & 0xFF

        if key == ord('b') or key == ord('B'):
            block_z_mm = round(z_mm, 1)
            print(f">>> BLOCK_Z_MM = {block_z_mm}  (arm touching table)")

        elif key == ord('t') or key == ord('T'):
            tower_x_mm = round(x_mm, 1)
            tower_y_mm = round(y_mm, 1)
            print(f">>> TOWER_X_MM = {tower_x_mm}  TOWER_Y_MM = {tower_y_mm}")

        elif key == ord('q') or key == ord('Q'):
            break

        target_joints = kbdNav.move_joints_with_keyboard(
            timer.get_sample_time(), speed=np.pi / 4)
        myMiniArm.read_write_std(target_joints, 0.0)
        timer.sleep()

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
    myMiniArm.terminate()

    print("\n" + "=" * 50)
    print("Copy these values into src/stacker.py:")
    print("=" * 50)
    if block_z_mm is not None:
        print(f"BLOCK_Z_MM      = {block_z_mm}")
        print(f"TOWER_Z_MM      = {block_z_mm}   # table surface = same level")
    else:
        print("BLOCK_Z_MM      = ??? (not recorded — press B next time)")
    if tower_x_mm is not None:
        print(f"TOWER_X_MM      = {tower_x_mm}")
        print(f"TOWER_Y_MM      = {tower_y_mm}")
    else:
        print("TOWER_X_MM/Y_MM = ??? (not recorded — press T next time)")
    print("=" * 50)
