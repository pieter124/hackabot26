import os
import cv2
import numpy as np

BLOCK_SIZE_MM = 40.0

# CS-USB-IMX307 default intrinsics (used when no calibration file exists)
_DEFAULT_FX = 800.0
_DEFAULT_FY = 800.0
_DEFAULT_CX = 640.0
_DEFAULT_CY = 360.0

COLOR_RANGES = {
    "red": [
        (np.array([0,   80,  50]),   np.array([10,  255, 255])),
        (np.array([170, 80,  50]),   np.array([180, 255, 255])),
    ],
    "green": [
        (np.array([40,  50,  40]),   np.array([85,  255, 255])),
    ],
    "blue": [
        (np.array([100, 80,  40]),   np.array([130, 255, 255])),
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([180, 255, 50])),
    ]
}

BOX_COLORS = {
    "red":   (0,   0,   255),
    "green": (0,   255,  0),
    "blue":  (255,  0,   0),
    "black": (50,  50,  50),
}

MIN_AREA     = 800
MAX_AREA     = 120000
MIN_SOLIDITY = 0.80
EDGE_MARGIN_PX = 4
PARTIAL_MIN_COVERAGE = 0.35
VISION_CENTRE = (335,250)


def build_mask(hsv, color):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in COLOR_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def is_square_ish(contour, tolerance=0.30):
    _, _, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    ratio = w / h
    return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)


def touches_image_edge(x, y, w, h, frame_w, frame_h, margin=EDGE_MARGIN_PX):
    return {
        "left": x <= margin,
        "top": y <= margin,
        "right": (x + w) >= (frame_w - margin),
        "bottom": (y + h) >= (frame_h - margin),
    }


def estimate_block_center(x, y, w, h, frame_w, frame_h, touches_edge):
    side_px = float(max(w, h))
    cx = float(x + w / 2.0)
    cy = float(y + h / 2.0)

    if touches_edge["left"] and touches_edge["right"]:
        cx = frame_w / 2.0
    elif touches_edge["left"]:
        cx = side_px / 2.0
    elif touches_edge["right"]:
        cx = frame_w - side_px / 2.0

    if touches_edge["top"] and touches_edge["bottom"]:
        cy = frame_h / 2.0
    elif touches_edge["top"]:
        cy = side_px / 2.0
    elif touches_edge["bottom"]:
        cy = frame_h - side_px / 2.0

    cx = float(np.clip(cx, 0, frame_w - 1))
    cy = float(np.clip(cy, 0, frame_h - 1))
    return cx, cy, side_px


def is_partial_block_candidate(w, h, touches_edge):
    if not any(touches_edge.values()):
        return False

    side_px = float(max(w, h))
    if side_px <= 0:
        return False

    visible_fraction = float(min(w, h)) / side_px
    return visible_fraction >= PARTIAL_MIN_COVERAGE

def detect_blocks_and_prongs(frame):
    # Convert to HSV just once to save CPU cycles
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    annotated = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    
    # ==========================================
    # 1. DETECT BLOCKS
    # ==========================================
    block_detections = []
    block_crosshairs = []
    
    for color in COLOR_RANGES:
        mask = build_mask(hsv, color)
        mask = clean_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_AREA <= area <= MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            touches_edge = touches_image_edge(x, y, w, h, frame_w, frame_h)
            partial = is_partial_block_candidate(w, h, touches_edge)

            hull = cv2.convexHull(cnt)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if solidity < MIN_SOLIDITY:
                continue

            if not partial and not is_square_ish(cnt):
                continue

            cx, cy, side_px = estimate_block_center(x, y, w, h, frame_w, frame_h, touches_edge)
            cx_i, cy_i = int(round(cx)), int(round(cy))

            block_detections.append({
                "color": color, "cx": cx_i, "cy": cy_i, "area": area,
                "bbox": (x, y, w, h), "size_px": side_px,
                "partial": partial, "touches_edge": touches_edge,
            })

            block_crosshairs.append((cx_i, cy_i))

            # Draw block bounding box and center
            box_color = BOX_COLORS[color]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            cv2.drawMarker(annotated, (cx_i, cy_i), box_color,
                           markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
            label = f"{color}{' partial' if partial else ''} ({cx_i},{cy_i})"
            cv2.putText(annotated, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

    # ==========================================
    # 2. DETECT PRONGS
    # ==========================================
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])
    prong_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Specific morphological cleanup for prongs
    kernel = np.ones((5, 5), np.uint8)
    prong_mask = cv2.morphologyEx(prong_mask, cv2.MORPH_OPEN, kernel)
    prong_mask = cv2.morphologyEx(prong_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(prong_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_prongs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtering logic
        if w < 10 or h < 20: continue 
        if (y + h) < (frame_h * 0.7): continue # Bottom 30% only
        if (h / float(max(w, 1))) < 1.2: continue # Taller than wide
            
        detected_prongs.append((x, y, w, h))
    
    # Sort left to right
    detected_prongs = sorted(detected_prongs, key=lambda b: b[0])[:2]

    # ==========================================
    # 3. CALCULATE MIDPOINT & DRAW PRONGS
    # ==========================================
    prong_midpoint_px = None
    PRONG_COLOR = (0, 255, 255) # Bright Yellow
    
    # Draw individual prong boxes
    for (x, y, w, h) in detected_prongs:
        cv2.rectangle(annotated, (x, y), (x+w, y+h), PRONG_COLOR, 2)
        cv2.putText(annotated, "Prong", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRONG_COLOR, 2, cv2.LINE_AA)

    # Return the annotated image, block data, and the midpoints of detected blocks.
    return annotated, block_detections, block_crosshairs


def estimate_3d_position(detection, cam_mtx=None, ee_rotation=None):
    """
    Returns (X, Y, Z) in metres relative to the camera centre.

    If ee_rotation is None, returns raw camera-frame coordinates:
        +X = right, +Y = down, +Z = into the scene.
    If ee_rotation is a (3x3) rotation matrix mapping camera -> world frame,
    the result is rotated into the world/arm frame instead.

    Z is estimated from the known physical block size using the pinhole model:
        Z = fx * BLOCK_SIZE / size_px
    X, Y are back-projected from the pixel offset:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    if cam_mtx is not None:
        fx, fy, cx, cy = cam_mtx[0,0], cam_mtx[1,1], cam_mtx[0,2], cam_mtx[1,2]
    else:
        fx, fy, cx, cy = _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY

    _, _, w, h = detection["bbox"]
    if w == 0 or h == 0:
        return (0.0, 0.0, 0.0)

    # Use min(w, h): perspective foreshortening shrinks one axis when the block
    # is viewed at an angle, so the smaller dimension is the safer size estimate.
    size_px = float(min(w, h))

    # Depth from known physical size (mm -> metres)
    Z = (BLOCK_SIZE_MM / 1000.0) * (fx / size_px)

    if Z <= 0:
        return (0.0, 0.0, 0.0)

    X = (detection["cx"] - cx) * Z / fx
    Y = (detection["cy"] - cy) * Z / fy

    if ee_rotation is not None:
        point_world = ee_rotation @ np.array([X, Y, Z])
        return tuple(point_world)

    return (X, Y, Z)

def get_frame(cap, cam_mtx=None, cam_dist=None):
    ret, frame = cap.read()
    if not ret:
        return None
    if cam_mtx is not None:
        frame = cv2.undistort(frame, cam_mtx, cam_dist)
    return frame

def load_camera_calibration(calib_file):
    if calib_file is not None and os.path.exists(calib_file):
        calib = np.load(calib_file)
        print("Camera calibration loaded.")
        return calib["mtx"], calib["dist"]
    else:
        print("No camera calibration file - running without undistort.")
        return None, None
    
def calculate_camera_delta_mm(det, cam_mtx, Z):
    """
    Calculates the physical X and Y distance (in mm) the camera needs 
    to move to align the detection center with VISION_CENTRE.
    """
    if cam_mtx is not None:
        fx, fy = cam_mtx[0,0], cam_mtx[1,1]
    else:
        fx, fy = _DEFAULT_FX, _DEFAULT_FY
        
    # Calculate pixel difference
    # Positive dx means the block is to the right of the crosshair
    dx_px = det["cx"] - VISION_CENTRE[0]
    dy_px = det["cy"] - VISION_CENTRE[1]
    # print(f"[VISION] Pixel delta: dx={dx_px:.1f}px, dy={dy_px:.1f}px at estimated depth Z={Z:.3f}m")
    # Convert to physical mm
    dx_mm = (dx_px * 20) / fx
    dy_mm = (dy_px * 20) / fy
    
    return dx_mm, dy_mm