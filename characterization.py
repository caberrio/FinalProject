from skimage import measure

def properties(src):
    #src : matrix/image segmentaded
    #properties : Measure properties of labeled image regions.
    
    regionprops = measure.regionprops(src, cache = False)  # cache = False doesn't calculate the properties until you call them
    
    areas = [r.area for r in regionprops]
    solidities = [r.solidity for r in regionprops]
    _,NumObjects = measure.label(src,return_num =True)
    
    return areas, solidities, NumObjects
