import matplotlib.pyplot as plt
from pylab import imread, imshow, gray, mean
import numpy as np
from skimage.segmentation import slic


def get_slic_image_segments(image, num_segments=100):
    return slic(image, n_segments=num_segments)


if __name__ == "__main__":
    img = imread('/home/jvarley/data/rgbd-dataset/apple/apple_1/apple_1_1_95_crop.png')
    segments = get_slic_image_segments(img)
    plt.imshow(segments)
    plt.show()