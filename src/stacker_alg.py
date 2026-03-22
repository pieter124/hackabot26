import numpy as np

BLOCK_HEIGHT = 39 
SAFE_Z = 150 

# Placeholder functions (ensure these are defined in your main script)
def move_to(x, y, z): 
    print(f"Moving to {x}, {y}, {z}")
    return True

def grab_block(): 
    print("Grabbing...")
    return True

def release_block(): 
    print("Releasing...")
    return True

def schedule_blocks(targets):
    def get_sort_key(block):
        x, y, z = block
        theta = np.arctan2(y, x) 
        return (z, theta)

    return sorted(targets, key=get_sort_key)

def construct_tower(blocks: list, tower_location: tuple[float, float]):
    '''Schedules and executes a sequence of movements'''
    placement_order = schedule_blocks(blocks)
    
    # Initialize tower destination as a list [x, y, z]
    current_tower_pos = [tower_location[0], tower_location[1], 0]
    
    for block_coords in placement_order:
        bx, by, bz = block_coords
        
        # --- PICKUP SEQUENCE ---
        move_to(bx, by, SAFE_Z)      
        move_to(bx, by, bz)         
        grab_block()
        move_to(bx, by, SAFE_Z)      
        
        # --- PLACE SEQUENCE ---
        tx, ty, tz = current_tower_pos
        move_to(tx, ty, SAFE_Z)
        move_to(tx, ty, tz)          
        release_block()
        move_to(tx, ty, SAFE_Z)     
      
        current_tower_pos[2] += BLOCK_HEIGHT
