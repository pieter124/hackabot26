''' Algorithm to determine control sequence for stacking tower'''
import heapq
import numpy as np

def schedule_blocks(targets:list[tuple[int,int]]):
    '''Schedules a list of movements to pick up and place blocks in turn'''
    execution = []
    for block in targets:
        
        
        theta = np.arctan2(block[1],block[0])
        
        heapq.heappush(execution,(theta,block))
    
    return execution 
    