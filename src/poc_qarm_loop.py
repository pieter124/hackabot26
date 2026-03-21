import sys, os
import cv2
import glob
import threading
import numpy as np
from pal.products.qarm_mini import QArmMini, QArmMiniCamera
from hal.content.qarm_mini import QArmMiniKeyboardNavigator, QArmMiniFunctions, DataIO
from pal.utilities.keyboard import QKeyboard
from pal.utilities.timing   import QTimer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# COLOR RANGES (HSV)
# ---------------------------------------------------------------------------
# HSV is much more robust than BGR under changing light.
# Hue is 0-179 in OpenCV, Saturation/Value are 0-255.
# Red wraps around 0/180 so it needs TWO ranges.
COLOR_RANGES = {
    "red": [
        (np.array([0,   120,  70]),  np.array([10,  255, 255])),   # lower red
        (np.array([170, 120,  70]),  np.array([180, 255, 255])),   # upper red
    ],
    "green": [
        (np.array([40,  80,  50]),   np.array([85,  255, 255])),
    ],
    "blue": [
        (np.array([100, 120,  50]),  np.array([130, 255, 255])),
    ],
}

# Visual colour for bounding boxes (BGR)
BOX_COLORS = {
    "red":   (0,   0,   255),
    "green": (0,   255,  0),
    "blue":  (255,  0,   0),
}

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
MIN_AREA    = 800    # px² — ignore tiny blobs (noise, reflections)
MAX_AREA    = 20000  # px² — ignore huge blobs (shadows, merged objects)
MIN_SOLIDITY = 0.80  # contour area / convex hull area — blocks are solid squares


def build_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """Union all HSV ranges for a colour into one binary mask."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in COLOR_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological open  → removes small specks (salt noise).
    Morphological close → fills small holes inside the block face.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def is_square_ish(contour, tolerance: float = 0.30) -> bool:
    """
    True when the bounding rectangle is roughly square.
    tolerance=0.30 allows ±30 % difference between width and height,
    which handles slight perspective skew from the overhead camera.
    """
    _, _, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    ratio = w / h
    return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)


def detect_blocks(frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Detect red / green / blue 40 mm × 40 mm blocks in *frame*.

    Returns
    -------
    annotated : np.ndarray
        A copy of frame with bounding boxes and labels drawn on it.
    detections : list[dict]
        Each entry has keys: color, cx, cy, area, bbox (x,y,w,h)
    """
    hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    annotated  = frame.copy()
    detections = []

    for color in COLOR_RANGES:
        mask = build_mask(hsv, color)
        mask = clean_mask(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_AREA <= area <= MAX_AREA):
                continue                             # wrong size

            # Solidity check: real blocks are solid; shadows/glare are not
            hull     = cv2.convexHull(cnt)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if solidity < MIN_SOLIDITY:
                continue

            if not is_square_ish(cnt):
                continue                             # not a square shape

            # ---- Passed all checks — record and draw ----
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            detections.append({
                "color": color,
                "cx":    cx,
                "cy":    cy,
                "area":  area,
                "bbox":  (x, y, w, h),
            })

            box_color = BOX_COLORS[color]

            # Bounding rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # Crosshair at centroid
            cv2.drawMarker(annotated, (cx, cy), box_color,
                           markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

            # Label: colour + pixel centroid
            label = f"{color} ({cx},{cy})"
            cv2.putText(annotated, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

    return annotated, detections


# ---------------------------------------------------------------------------
# Section A: Setup  (unchanged from your original)
# ---------------------------------------------------------------------------
kbd       = QKeyboard()
myMiniArm = QArmMini(hardware=1, id=4)
kbdNav    = QArmMiniKeyboardNavigator(keyboard=kbd, initialPose=myMiniArm.HOME_POSE)
myArmMath = QArmMiniFunctions()
timer     = QTimer(sampleRate=60.0, totalTime=300.0)
camera    = QArmMiniCamera()
data      = DataIO()

cap = cv2.VideoCapture(1)

img_count = 0

try:
    print("Main loop started. Press 'q' on the video window to stop.")

    while timer.check():
        kbd.update()

        # ---- 1. Capture frame ----
        ret, frame = cap.read()
        if ret:
            # ---- 2. Detect coloured blocks ----
            annotated, detections = detect_blocks(frame)

            # Print any new detections to the terminal
            for d in detections:
                print(f"[BLOCK] {d['color']:5s}  centre=({d['cx']:4d},{d['cy']:4d})  area={d['area']:.0f}px²")

            # Show annotated feed
            cv2.imshow("QArm Mini Feed", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                fname = f"snapshot_{img_count}.jpg"   # BUG FIX: extension outside format
                cv2.imwrite(fname, frame)
                print(f"Snapshot saved: {fname}")
                img_count += 1                         # BUG FIX: consistent variable name
            elif key == ord('q'):
                print("Quit key pressed.")
                break

        # ---- 3. Control the arm ----
        myMiniArm.read_write_std(
            kbdNav.move_joints_with_keyboard(timer.get_sample_time(), speed=np.pi/4))

        # ---- 4. Forward kinematics ----
        pose, rotationMatrix, gamma = myArmMath.forward_kinematics(myMiniArm.positionMeasured)

        timer.sleep()

except KeyboardInterrupt:
    print("Received user terminate command.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    myMiniArm.terminate()
