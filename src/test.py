import time
from control import Control
from pal.products.qarm_mini import QArmMini
from hal.content.qarm_mini import QArmMiniFunctions

print("Connecting to arm...")
# Make sure hardware=1 and id=4 are actually correct for your setup!
myMiniArm = QArmMini(hardware=1, id=4)
myArmMath   = QArmMiniFunctions()
print("Moving to all zeros...")
# Assuming read_write_std expects a list of 4 joint angles and 1 gripper value
myMiniArm.read_write_std([0.0, 0.0, 0.0, 0.0], 0.0)
time.sleep(3)

print("Moving joint 2...")
myMiniArm.read_write_std([0.0, 0.5, 0.0, 0.0], 0.0)
time.sleep(3)

control = Control(myArmMath, myMiniArm)
control._coord = [0.0, 0.5, 0.0, 0.0]
control.grip()
time.sleep(3)

print("Test complete.")