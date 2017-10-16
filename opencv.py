import cv2

from skimage import filters, util, img_as_ubyte, feature
from skimage.color import rgb2grey
from skimage.morphology import disk, reconstruction, local_maxima, remove_small_objects, watershed

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

# Imadjust

image_imadjust = Imadjust.imadjust(img)
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

RegMax_skimage = local_maxima(Image_by_reconstruction_open_close_resta_uint8) 
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

resta = RegMax_skimage - RegMax_remove_skimage
print("resta ")
plt.imshow(resta)
plt.show()

#RegMax_remove_code= RemoveSmallObjects.RemoveObjects(RegMax_skimage,5)
#print("Regional max remove by code")
#plt.imshow(RegMax_remove_code)
#plt.show()

# Compute Background Markers
Thresholded = filters.threshold_otsu(Image_by_reconstruction_open_close_resta_uint8) ## 0.2686
binary_threshold = util.invert(Image_by_reconstruction_open_close_resta_uint8 <= Thresholded)
#cv2.imshow( "binary_threshold", binary_threshold.astype(np.uint8) );         
print("binary_threshold")
plt.imshow(binary_threshold)
plt.show()

ret2,th2 = cv2.threshold(Image_by_reconstruction_open_close_resta_uint8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

add_Regional_Thresholded = np.logical_or(RegMax_remove_skimage,binary_threshold)
#cv2.imshow( "Thresholded opening-closing by reconstruction ", Sum_images.astype(np.uint8) );         
print("Thresholded opening-closing by reconstruction ")
plt.imshow(add_Regional_Thresholded)
plt.show()


#  Compute the Watershed Transform of the Segmentation Function
distance = ndi.morphology.distance_transform_edt(np.logical_not(add_Regional_Thresholded))
local_maxi = feature.peak_local_max(distance, labels = np.logical_not(add_Regional_Thresholded), footprint=np.ones((3, 3)), indices=False)
markers = ndi.label(local_maxi)[0]
watershed_trans = watershed(-distance ,markers, mask= np.logical_not(add_Regional_Thresholded))
print("Watershed")
plt.imshow(watershed_trans)
plt.show() 

watershed_riged_lines = watershed_trans == 0
print("Watershed ridge lines")
plt.imshow(watershed_riged_lines)
plt.show() 




