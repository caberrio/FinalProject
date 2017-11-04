import numpy as np
from skimage.morphology import reconstruction, square, disk
 
def imimposemin(I, BW, conn=8):
    fm = I.copy().astype(np.float64)
 
    fm[BW == True] = -float('Inf')
    fm[BW == False] = float('Inf')
 
    if I.dtype == float:
        I_range = np.amax(I) - np.amin(I)
 
        if I_range == 0:
            h = 0.1
        else:
            h = I_range*0.001
    else:
        h = 1
 
    fp1 = I + h
 
    g = np.minimum(fp1, fm)
 
    if conn == 8:
        selem = square(3)
    elif conn == 4:
        selem = disk(1)
 
    if I.dtype == float:
        J = reconstruction(1 - fm, 1 - g, selem=selem)
        J = 1 - J
    else:
        J = reconstruction(255 - fm, 255 - g, method='dilation', selem=selem)
        J = 255 - J
    J[BW] = 0
 
    return J