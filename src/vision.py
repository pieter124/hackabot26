import numpy as np
import cv2

# ---------------------------------------------------------------------------
# CS-USB-IMX307 Starlight camera default intrinsics
#   Sensor : Sony IMX307, 1/2.8"  (pixel pitch 2.9 µm)
#   Lens   : 2.8 mm fixed focal length
#   Default capture resolution: 1280 × 720
#
#   fx = fy = focal_length_mm / pixel_pitch_mm
#           = 2.8 mm / 0.0029 mm ≈ 965 px  (at full 1920 px width)
#   Scaled to 1280 px width: 965 * (1280/1920) ≈ 643 px
# ---------------------------------------------------------------------------
_DEFAULT_FX = 643.0
_DEFAULT_FY = 643.0
_DEFAULT_CX = 640.0   # principal point — half of 1280
_DEFAULT_CY = 360.0   # principal point — half of 720

# Real-world block face size (mm) — 40 mm × 40 mm Lego-style blocks
BLOCK_SIZE_MM = 40.0

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


def estimate_3d_position(
    detection: dict,
    cam_mtx: np.ndarray = None,
) -> tuple[float, float, float]:
    """
    Estimate the 3D position (x, y, z) of a detected block in mm,
    relative to the camera centre.

    Uses the pinhole camera model:
        Z = (real_size_mm * fx) / pixel_width
        X = (cx_pixel - principal_cx) * Z / fx
        Y = (cy_pixel - principal_cy) * Z / fy

    Parameters
    ----------
    detection : dict
        A detection entry from detect_blocks() with keys cx, cy, bbox.
    cam_mtx : np.ndarray, optional
        3x3 camera matrix from cv2.calibrateCamera. If None, uses the
        CS-USB-IMX307 default intrinsics.

    Returns
    -------
    (x, y, z) : floats, all in mm relative to camera centre.
        +X = right, +Y = down, +Z = into the scene.
    """
    if cam_mtx is not None:
        fx = cam_mtx[0, 0]
        fy = cam_mtx[1, 1]
        cx = cam_mtx[0, 2]
        cy = cam_mtx[1, 2]
    else:
        fx, fy, cx, cy = _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY

    _, _, w, h = detection["bbox"]

    if w == 0 or h == 0:
        return (0.0, 0.0, 0.0)

    # Each face is 40mm — use both axes independently and average for accuracy
    Z_from_w = (BLOCK_SIZE_MM * fx) / w
    Z_from_h = (BLOCK_SIZE_MM * fy) / h
    Z = (Z_from_w + Z_from_h) / 2.0
    X = (detection["cx"] - cx) * Z / fx
    Y = (detection["cy"] - cy) * Z / fy

    return (X, Y, Z)


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
