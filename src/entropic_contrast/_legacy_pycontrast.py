# Copyright (C) 2017 Ashish Gupta
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author Ashish Gupta
# version 0.1.0

from __future__ import print_function

# build-in modules
import os

# third-party modules
from skimage import io
from skimage import exposure
from skimage import color
from skimage import img_as_float
import numpy as np
from sklearn.metrics import mutual_info_score as mis
import matplotlib.pyplot as plt


def __compute_contrast_quality_for_image(input_image, num_bins=128):
    """
    Computes a score of the quality of contrast in input image based on divergence
    from an intensity equalized histogram.

    We compute the mutual information (MI) (a measure of entropy) between histogram of intensity
    of image and its contrast equalized histogram.
    MI is not real metric, but a symmetric and non-negative similarity measures that
    takes high values for similar images. Negative values are also possible.

    Intuitively, mutual information measures the information that histogram_image and histogram_equlized
    share: it measures how much knowing one of these variables reduces uncertainty about the other.

    The Entropy is defined as:

    .. math::

        H(X) = - \sum_i p(g_i) * ln(p(g_i)
    with :math:`p(g_i)` being the probability of the images intensity value :math:`g_i`.

    Assuming two histograms :math:`R` and :math:`T`, the mutual information is then computed by comparing the
    histogram entropy values (i.e. a measure how well-structured the common histogram is).
    The distance metric is then calculated as follows:

    .. math::

        MI(R,T) = H(R) + H(T) - H(R,T) = H(R) - H(R|T) = H(T) - H(T|R)

    A maximization of the mutual information is equal to a minimization of the joint
    entropy.

    :param input_image : 2-D array
    :param num_bins :  integer : the number of bins in histogram, it has a small scaling effect on
    the mutual information score since it slightly modifies the shape of the histogram
    :return: quality of contrast : float
    :raises: argument error if input image data is corrupted
    """

    # Check dimensions of input image
    # If image dimensions is 2, then it is a gray-scale image
    # First convert input to RGB image
    if input_image.shape == 2:
        input_image = color.gray2rgb(input_image)

    # Convert the RGB image to HSV. Exposure is primarily correlated with Value rather
    # than Hue and Saturation
    image_hsv = color.rgb2hsv(input_image)

    # The intensity channel is third in HSV format image
    v_channel = image_hsv[:, :, 2]

    # compute the contrast equalized array of intensity channel of image
    v_channel_equalized = exposure.equalize_hist(v_channel, nbins=num_bins)

    # compute the histogram of intensity channel
    v_channel_histogram, histogram_bin_edges = np.histogram(img_as_float(v_channel), bins=num_bins, density=True)

    # compute the histogram of contrast equalized intensity channel
    v_channel_equalized_histogram, _ = np.histogram(img_as_float(v_channel_equalized), bins=num_bins, density=True, range=(histogram_bin_edges[0], histogram_bin_edges[-1]))

    # compute the mutual information based contrast quality measure
    return mis(v_channel_histogram, v_channel_equalized_histogram)


def contrast_quality(image_url):
    """
    Computes the contrast quality of image file
    :param image_url: string : path to image file resource
    :return: float : contrast quality of input image
    """
    try:
        image = io.imread(image_url)
        return __compute_contrast_quality_for_image(image)
    except IOError as err:
        # modify print function in case of incompatibility between Python 2.x and 3.x
        print(err)


def contrast_quality_collection(image_folder_url, output_folder=None):
    """
    Computes the contrast quality of all image files in directory URL provided
    :param image_folder_url: URL of image data folder
    :return: URL of log file to which contrast quality is recorded
    """
    # Check if provided folder URL exists
    if not os.path.exists(image_folder_url):
        # modify print function in case of incompatibility between Python 2.x and 3.x
        print('{} does not exist.'.format(image_folder_url))
        return None

    # output file path
    if output_folder:
        result_file_url = os.path.join(output_folder, 'contrast_quality.log')
    else:
        result_file_url = os.path.join(image_folder_url, '../', 'contrast_quality.txt')

    with open(result_file_url, 'w') as output_log_file:
        # process only the valid image files in directory
        file_list = os.listdir(image_folder_url)
        # increment valid image file extensions as required
        valid_extensions = ['jpg', 'jpeg', 'png', 'pgm', 'bmp']

        for file in file_list:
            if file.split('.')[-1] in valid_extensions:
                image_url = os.path.join(image_folder_url, file)
                cq = contrast_quality(image_url)
                # echo to console
                # print('%s,%08.6f\n' % (image_url, cq))
                output_log_file.write('%s,%08.6f\n' % (file, cq))

    return result_file_url


def demo():
    """
    Demonstrate contrast_quality measure for sample photographs

    The results of estimated contrast quality are consistent with the expectation of
    contrast quality of each of the photographs

    :return: None
    """

    # photograph samples, some of the image sets contain the same image with different exposures
    # or contrast enhancement processing
    image_set_1 = ['./examples/retinex1.jpg', './examples/retinex2.jpg', './examples/retinex3.jpg', './examples/retinex4.jpg']
    image_set_2 = ['./examples/under1.jpg', './examples/over1.jpg', './examples/correct1.jpg']
    image_set_3 = ['examples/3894541598_bb37af2dcd_o.jpg', 'examples/18056401685_c5b313e712_o.jpg']

    image_set_list = [image_set_1, image_set_2, image_set_3]

    for image_set in image_set_list:
        num_images = len(image_set)
        fig = plt.figure(figsize=(num_images*5, 8))
        for i in range(num_images):
            plt.subplot(1, num_images, i+1)
            image_url = image_set[i]
            image = io.imread(image_url)
            cq = contrast_quality(image_url)
            plt.imshow(image)
            plt.title('contrast Q : %08.6f' % cq)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # run demo on image files in examples directory
    demo()

    # run demo on all image files in a folder (in this case the examples directory)
    contrast_quality_collection('./examples/')