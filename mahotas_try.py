import cv2
import mahotas

from skimage import filters, util, img_as_ubyte, feature, exposure , segmentation, measure, color
from skimage.color import rgb2grey
from skimage.morphology import disk, reconstruction, local_maxima, remove_small_objects

import matplotlib.pyplot as plt

from scipy import ndimage as ndi

import numpy as np

import Imadjust
import RemoveSmallObjects


img = cv2.imread('Snap-16290_Rhodamine_1.tif',0)
#img1 = cv2.imread('Snap-16290_Rhodamine_1.tif')
#Grey_image = img_as_ubyte(rgb2grey(img1))
#img_grey = cv2.cvtColor( img1, cv2.COLOR_RGB2GRAY)
#cv2.imshow( "image", img );         
print("image")
plt.imshow(img, cmap = 'gray')
plt.show()

# Clear Borders

clear_borders_img = mahotas.labeled.remove_bordering(img)
plt.imshow(clear_borders_img)
plt.show()
# Imadjust

#image_imadjust = Imadjust.imadjust(img)
#image_contrast = mahotas.stretch(img)
image_imadjust = exposure.rescale_intensity(img)
#cv2.imshow( "imadjust", image_imadjust );         
print("imadjust")
plt.imshow(image_imadjust)
plt.show()

# Gradient Sobel
sobel = filters.sobel(image_imadjust)
#cv2.imshow( "gradient (sobel)", sobel );         
print("gradient (sobel)")
plt.imshow(sobel)
plt.show()

# opening-by-reconstruction 

erosion = cv2.erode(image_imadjust,disk(1))
Image_by_reconstruction_opening = reconstruction(erosion,image_imadjust)
Image_by_reconstruction_opening_uint8 = Image_by_reconstruction_opening.astype(np.uint8)
#cv2.imshow( "erosion", Image_by_reconstruction_opening_uint8 );         
print("erosion")
plt.imshow(Image_by_reconstruction_opening_uint8)
plt.show()

# opening closing by reconstruction

dilation = cv2.dilate(Image_by_reconstruction_opening_uint8,disk(1))

#------------------------
x = 255 - dilation
y = 255 - Image_by_reconstruction_opening_uint8
Image_by_reconstruction_open_close_resta = reconstruction(x,y)
Image_by_reconstruction_open_close_resta = 255 - Image_by_reconstruction_open_close_resta
Image_by_reconstruction_open_close_resta_uint8 = Image_by_reconstruction_open_close_resta.astype(np.uint8)
#cv2.imshow( "dilation by (255-)", Image_by_reconstruction_open_close_resta_uint8 );         
print("dilation by (255-)")
plt.imshow(Image_by_reconstruction_open_close_resta_uint8)
plt.show()

#------------------------
#Image_by_reconstruction_open_close = reconstruction(cv2.bitwise_not(dilation),cv2.bitwise_not(Image_by_reconstruction_opening))
#Image_by_reconstruction_open_close = cv2.bitwise_not(Image_by_reconstruction_open_close)
#plt.imshow(Image_by_reconstruction_open_close)
#plt.show()


xx = util.invert(dilation)
yy = util.invert(Image_by_reconstruction_opening_uint8)
Image_by_reconstruction_open_close_util = reconstruction(xx,yy)
Image_by_reconstruction_open_close_util_uint8 = util.invert(Image_by_reconstruction_open_close_util.astype(np.uint8))
#cv2.imshow( "dilation by util", Image_by_reconstruction_open_close_util_uint8 );         
print("dilation by util")
plt.imshow(Image_by_reconstruction_open_close_util_uint8)
plt.show()

# Regional max

#RegMax_ndi = ndi.filters.maximum_filter(Image_by_reconstruction_open_close_util, footprint = np.ones((3,3),dtype=np.float64))
#RegMax_ndi = (Image_by_reconstruction_open_close_util == RegMax_ndi)
#print("Regional max by ndi")
#plt.imshow(RegMax_ndi)
#plt.show()

RegMax_skimage = local_maxima(Image_by_reconstruction_open_close_util_uint8) 
#cv2.imshow( "Regional max by skimage", RegMax_skimage );        
RegMax_skimage = RegMax_skimage.astype(bool) 
print("Regional max by skimage")
plt.imshow(RegMax_skimage)
plt.show()

# Remove small objects from binary image

RegMax_remove_skimage = remove_small_objects(RegMax_skimage,min_size= 5)
#cv2.imshow( "Regional max remove by skimage", RegMax_remove_skimage.astype(np.uint8) );         
print("Regional max remove by skimage")
plt.imshow(RegMax_remove_skimage)
plt.show()

resta = np.logical_xor(RegMax_skimage , RegMax_remove_skimage)
print("resta ")
plt.imshow(resta)
plt.show()

# Compute Background Markers
Thresholded = filters.threshold_otsu(Image_by_reconstruction_open_close_util_uint8) ## 0.2686
binary_threshold = util.invert(Image_by_reconstruction_open_close_util_uint8 <= Thresholded)
print("binary_threshold")
plt.imshow(binary_threshold)
plt.show()

add_Regional_Thresholded = RegMax_remove_skimage + binary_threshold  
#cv2.imshow( "Thresholded opening-closing by reconstruction ", Sum_images.astype(np.uint8) );         
print("Thresholded opening-closing by reconstruction ")
plt.imshow(add_Regional_Thresholded)
plt.show()

#---------------------

#logical_not_add = np.logical_not(add_Regional_Thresholded)
#distance = ndi.morphology.distance_transform_edt(logical_not_add)

distance_normal = ndi.morphology.distance_transform_edt(add_Regional_Thresholded)

#distance_stretch = mahotas.stretch(distance)
distance_stretch_normal = mahotas.stretch(distance_normal)

#surface = distance_stretch.max() - distance_stretch
surface_normal = distance_stretch_normal.max() - distance_stretch_normal


#A large connectivity will suppress local maximum that are close together 
#into a single maximum, but this will require more memory and time to run. 
#A small connectivity will preserve local maximum that are close together,
#but this can lead to oversegmentation.

footprint = np.ones((8,8))     # <-- connectiviy

#peaks = mahotas.regmax(distance_stretch, footprint)
peaks_normal = mahotas.regmax(distance_stretch_normal, footprint)

#markers, NObjects = mahotas.label(peaks, np.ones((16, 16)))

#### 
# label_mahotas : This is also called connected component labeled, where the connectivity 
# is defined by the structuring element Bc.
###
markers_normal, NObjects_normal = mahotas.label(peaks_normal, footprint)  
#watershed_surface, WL = mahotas.cwatershed(surface, markers ,return_lines = True)
watershed_normal, WL_normal = mahotas.cwatershed(surface_normal, markers_normal ,return_lines = True)

"""print("watershed not(add_regional)")
plt.imshow(watershed_surface)
plt.show()

print("watershed not(add_regional) lines")
plt.imshow(WL)
plt.show()"""

print("watershed (add_regional)")
plt.imshow(watershed_normal)
plt.show()

print("watershed (add_regional) lines")
plt.imshow(WL_normal)
plt.show()

watershed_multiplication = watershed_normal * add_Regional_Thresholded
print("multiplication watershed")
plt.imshow(watershed_multiplication)
plt.show()

### 
# label skimage: Label connected regions of an integer array
###
watershed = measure.label(watershed_multiplication)
print("watershed connected regions")
plt.imshow(watershed )
plt.show()

color_watershed=color.label2rgb(watershed)
print("colored watershed label matrix")
plt.imshow(color_watershed )
plt.show()


"""import random
from matplotlib import colors as c
colors = map(plt.cm.jet,range(0, 256, 4))
random.shuffle(colors)
colors[0] = (0.,0.,0.,1.)
rmap = c.ListedColormap(colors)
print("colored watershed label matrix with random")
plt.imshow(watershed, cmap=rmap)
plt.show()"""
