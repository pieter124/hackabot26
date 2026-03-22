'''Estimates where to "roughly hover to" '''
import cv2
from vision import detect_blocks, estimate_3d_position

def preprocess_positions(cam_matrix,frame,show_frame=False):
    predicated_coords = []
    # Get 2d coords of blocks in the image
    annotations, detections = detect_blocks(frame=frame)
 
    # Estimate a 3d coord for the block
    for det in detections:
        coords = estimate_3d_position(detection=det,
                                      cam_mtx=cam_matrix)
        predicated_coords.append(coords)
    if show_frame:
        cv2.imshow("Prediction",annotations)
    
    print(f"[VISION PRED] {predicated_coords}")
    return predicated_coords