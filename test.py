import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


def hist(image, bins=32):
    fig, ax = plt.subplots(1, 1)
    ax.hist(image.ravel(), bins=bins, range=[0, 256])
    ax.set_xlim(0, 256)

def circle_points(resolution, center, radius):    
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """   
    radians = np.linspace(0, 2*np.pi, resolution)    
    c = center[1] + radius*np.cos(radians) #polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T# Exclude last point because a closed path should not have duplicate points


def multiotsu(image):
    thresholds = filters.threshold_multiotsu(image, classes=3)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    # Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='jet')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()



image = color.rgb2gray(io.imread('DJI_0829.jpg'))


# Try all unsupervised thresholding methods
# filters.try_all_threshold(image)

multiotsu(image)
 # Using the threshold values, we generate the three regions.
# regions = np.digitize(image, bins=thresholds)
# all_labels = measure.label(regions)

plt.show()

