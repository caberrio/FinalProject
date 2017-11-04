import numpy as np
from skimage import measure

def find_centroids(src):
    
    # src : label image/matrix
    
    regionprops = measure.regionprops(src, cache = False) 
    centroids = [np.int_(obj.centroid) for obj in regionprops]   
    output_segmented = np.zeros_like(src)
    
    for ind, arr in enumerate(centroids):
        output_segmented[tuple(arr)] = 1
    
    return output_segmented
