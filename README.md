# entropic_image_contrast_measure
This project develops a novel measure of contrast in image based on entropy distribution of pixel color values. The aim is to be effective in measuring subtle changes in exposure level of the same visual scene.

The program computes a score of the quality of contrast in input image based on divergence from an intensity equalized histogram. We compute the mutual information (MI) (a measure of entropy) between histogram of intensity of image and its contrast equalized histogram. MI is not real metric, but a symmetric and non-negative similarity measures that takes high values for similar images. Negative values are also possible. Intuitively, mutual information measures the information that histogram_image and histogram_equlized share: it measures how much knowing one of these variables reduces uncertainty about the other.

### Entropy
The Entropy is defined as:
.. math::
    H(X) = - \sum_i p(g_i) * ln(p(g_i)
with :math:`p(g_i)` being the probability of the images intensity value :math:`g_i`.
Assuming two histograms :math:`R` and :math:`T`, the mutual information is then computed by comparing the
histogram entropy values (i.e. a measure how well-structured the common histogram is).
The distance metric is then calculated as follows:
.. math::
    MI(R,T) = H(R) + H(T) - H(R,T) = H(R) - H(R|T) = H(T) - H(T|R)

A maximization of the mutual information is equal to a minimization of the joint entropy.
:param input_image : 2-D array
:param num_bins :  integer : the number of bins in histogram, it has a small scaling effect on the mutual information score since it slightly modifies the shape of the histogram
:return: quality of contrast : float
:raises: argument error if input image data is corrupted

### Usage

$ python pycontrast.py

Edit the code to change the location of the input images and the output text file.

### Block diagram
<img src="https://github.com/ashish-code/entropic_image_contrast_measure/blob/master/contrast%20score%20research%20plan.jpg" width="800">

#### Example

I compared the performance of our approach against state-of-art method called 'Generalized Contrast Function' that utilizes local feature descriptor based measure of contrast value in image. The images below are progressively improving exposure and contrast. My method was able to consistently estimate the contrast score better than GCF.

<img src="https://github.com/ashish-code/entropic_image_contrast_measure/blob/master/contrast_comparison1.png">

##### Other sample results

<img src="https://github.com/ashish-code/entropic_image_contrast_measure/blob/master/contrast_comparison2.png" width="1000">

<img src="https://github.com/ashish-code/entropic_image_contrast_measure/blob/master/contrast_comparison3.png" width="600">
