"""
entropic_contrast
=================

Reference-free image contrast quality measurement based on Mutual Information.

This package implements the **Entropic Contrast Measure (ECM)**, a no-reference
image quality metric that quantifies contrast by measuring the divergence between
an image's intensity histogram and the histogram of its contrast-equalized
counterpart.

The key insight: a well-exposed image already has a near-uniform intensity
distribution, so equalising it changes the histogram very little.  Mutual
Information (MI) captures this similarity — high MI means the image is already
well-contrasted; low MI indicates under- or over-exposure.

Published context
-----------------
Related to the methods described in:

    A. Gupta, "Entropic Image Contrast Measure", University of Surrey, 2017.

Submodules
----------
core
    :func:`~entropic_contrast.core.contrast_quality` — score a single image.
    :func:`~entropic_contrast.core.contrast_quality_batch` — score a directory.
    Internal helper :func:`~entropic_contrast.core._compute_contrast_score`.
metrics
    :func:`~entropic_contrast.metrics.mutual_information` — MI between two
    discrete histograms using the marginal-entropy definition.
    :func:`~entropic_contrast.metrics.normalised_mutual_information` — NMI
    variant bounded in [0, 1].
cli
    ``contrast-score`` console entry point (registered via pyproject.toml).

Quick start
-----------
>>> from entropic_contrast import contrast_quality
>>> score = contrast_quality("photo.jpg")
>>> print(f"Contrast score: {score:.4f}")

Higher values indicate better-contrasted images.  Typical range: 0–5+.
"""

from entropic_contrast.core import contrast_quality, contrast_quality_batch
from entropic_contrast.metrics import mutual_information, normalised_mutual_information

__all__ = [
    "contrast_quality",
    "contrast_quality_batch",
    "mutual_information",
    "normalised_mutual_information",
]

__version__ = "0.2.0"
__author__ = "Ashish Gupta"
