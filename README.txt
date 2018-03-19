=============
pycontrast
=============

This program will compute contrast quality of photographs

=============
Requirements
=============
External libraries required are:
numpy
matplotlib
skimage
sklearn

These libraries are typically bundled with distributions like Anaconda 
If not available on client system, it is highly recommended to install it from (https://www.anaconda.com/download/)
All major and most minor third party libraries are amenable to Anaconda's package management system

Alternatively, manually install dependencies:
pip install numpy
pip install matplotlib
pip install -U scikit-learn
pip install -U scikit-image

Use sudo pip * if installation of python library requires root priveleges

============
Usage
============
Unzip folder to desired directory in local system. Change directory to unzipped directory

>>> python pycontrast.py
will run a demo and should display series of images (there may be compatibility issues with matplotlib and X11 on some systems)
demo will also compute and record contrast quality of images in ./examples/ sub-directory in a text file: ./contrast_quality.txt

In a python scipt, import pycontrast.py as a module:
from pycontrast import contrast_quality

Then run the function 'contrast_quality' on an image file.

