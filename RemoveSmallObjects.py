import cv2
import numpy as np

def RemoveObjects(src, Pixels):
    
    # src : input one-layer image (numpy array)
    # Pixels : min number of pixels to keep.

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, 
    #but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    
    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = Pixels  
    
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

