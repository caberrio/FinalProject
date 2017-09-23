# Scikit modules

from skimage import io, filters, util, exposure, segmentation, img_as_ubyte
from skimage.color import rgb2grey
from skimage import img_as_float
from skimage.morphology import disk, erosion, reconstruction, dilation , local_maxima


# Matplotlib modules   

import matplotlib.pyplot as plt

# Scipy modules

from scipy import ndimage as ndi

# PIL modules

from PIL import Image

# Numpy modules

import numpy as np
np.set_printoptions(threshold=np.nan)
# Scripts


# Read Image and convert to grey 
im = io.imread('Snap-16290_Rhodamine_1.tif')

io.imshow(im)
plt.figure()

#Grey_image = np.floor(255*rgb2grey(im))
Grey_image = img_as_ubyte(rgb2grey(im))
io.imshow(Grey_image)
plt.figure()

# Past grey image to array

Grey_array = np.array(Grey_image)

# Contrast stretching

print("Contrast stretching")
Contrast_image = exposure.rescale_intensity(Grey_image)
io.imshow(Contrast_image)
plt.figure()

# SOMEHOW CLEAN BORDERS


# Edge detection
print("Edge detection")
sobel = filters.sobel(Contrast_image)
io.imshow(sobel)
plt.figure()

# Gradient magnitude 

Sobel_gradient_array = filters.rank.gradient(sobel,disk(1))

# Compute the opening-by-reconstruction (erode)
print("Compute the opening-by-reconstruction (erode)")
Image_erosion = erosion(Contrast_image,disk(1))
Image_by_reconstruction_opening = reconstruction(Image_erosion,Contrast_image)
io.imshow(Image_by_reconstruction_opening)
plt.figure()

# Compute the closing-by-reconstruction (dilation)
print("Compute the closing-by-reconstruction (dilation)")
Image_dilation = dilation(Image_by_reconstruction_opening,disk(1))
Image_by_reconstruction_closing = reconstruction(util.invert(Image_dilation),util.invert(Image_by_reconstruction_opening))
Image_by_reconstruction_closing = util.invert(Image_by_reconstruction_closing)
io.imshow(Image_by_reconstruction_closing)
plt.figure()

# Regional max

print("REGIONAL MAX")

Regional_max = local_maxima(Image_by_reconstruction_closing)
io.imshow(Regional_max)
