import cv2
from skimage import filters, util
from skimage.morphology import disk, reconstruction, local_maxima, remove_small_objects
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import Imadjust
import RemoveSmallObjects


img = cv2.imread('Snap-16290_Rhodamine_1.tif',0)
print("grey image")
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# Clear Borders


# Imadjust

image_imadjust = Imadjust.imadjust(img)
print("imadjust")
plt.imshow(image_imadjust)
plt.show()

# Gradient Sobel
sobel = filters.sobel(image_imadjust)
print("gradient (sobel)")
plt.imshow(sobel)
plt.show()

# opening-by-reconstruction 

erosion = cv2.erode(image_imadjust,disk(1))
Image_by_reconstruction_opening = reconstruction(erosion,image_imadjust)
Image_by_reconstruction_opening = 255 - Image_by_reconstruction_opening
Image_by_reconstruction_opening_uint8 = Image_by_reconstruction_opening.astype(np.uint8)
print("erosion")
plt.imshow(Image_by_reconstruction_opening_uint8)
plt.show()

# opening closing by reconstruction

dilation = cv2.dilate(Image_by_reconstruction_opening_uint8,disk(1))

#------------------------
x = 255 - dilation
y = 255 - Image_by_reconstruction_opening_uint8
Image_by_reconstruction_open_close_resta = reconstruction(x,y)
#Image_by_reconstruction_open_close_resta = 255 - Image_by_reconstruction_open_close_resta
Image_by_reconstruction_open_close_resta = Image_by_reconstruction_open_close_resta
Image_by_reconstruction_open_close_resta_uint8 = Image_by_reconstruction_open_close_resta.astype(np.uint8)
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
Image_by_reconstruction_open_close_util = util.invert(Image_by_reconstruction_open_close_util)
Image_by_reconstruction_open_close_util_uint8 = Image_by_reconstruction_open_close_util.astype(np.uint8)
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
print("Regional max by skimage")
plt.imshow(RegMax_skimage)
plt.show()

# Remove small objects from binary image

RegMax_remove_skimage = remove_small_objects(RegMax_skimage, 5)
print("Regional max remove by skimage")
plt.imshow(RegMax_remove_skimage)
plt.show()

#RegMax_remove_code= RemoveSmallObjects.RemoveObjects(RegMax_skimage,5)
#print("Regional max remove by code")
#plt.imshow(RegMax_remove_code)
#plt.show()

# Compute Background Markers
Thresholded = filters.threshold_otsu(Image_by_reconstruction_open_close_resta_uint8) ## 0.2686

ret2,th2 = cv2.threshold(Image_by_reconstruction_open_close_resta_uint8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
