"""
core.py — Entropic Contrast Measure: per-image and batch scoring.

This module provides the two public-facing scoring functions:

:func:`contrast_quality`
    Score a single image file.  Returns a float in the range [0, ∞) where
    higher values indicate better contrast.

:func:`contrast_quality_batch`
    Score every image in a directory and write results to a CSV log.

Algorithm
---------
For a given image:

1. Convert to HSV colour space.  Only the **Value** (V) channel is used,
   as perceived contrast is primarily a luminance phenomenon.
2. Compute the V-channel intensity histogram (``num_bins`` bins, density=True).
3. Apply histogram equalisation to the V channel (producing a near-uniform
   histogram).
4. Compute the equalised histogram over the **same** bin edges.
5. Return :func:`~entropic_contrast.metrics.mutual_information` between the
   two histograms.

Mutual Information quantifies how similar the two histograms are.  A
well-exposed image already has a roughly uniform distribution, so equalisation
changes it little → high MI.  An under- or over-exposed image has a skewed
histogram that equalisation shifts substantially → low MI.

This is a **no-reference** (NR) metric: it does not require a pristine
reference image, making it suitable for in-the-wild quality assessment.

Typical output range
--------------------
- Under/over-exposed: 1.5–2.5
- Retinex-processed: 4.0–4.6
- Well-exposed: 4.5–5.0+

(Values are in nats; they scale slightly with ``num_bins``.)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from skimage import color, exposure, img_as_float, io

from entropic_contrast.metrics import mutual_information

logger = logging.getLogger(__name__)

# Image file extensions recognised by the batch scorer
_VALID_EXTENSIONS: frozenset[str] = frozenset(
    {"jpg", "jpeg", "png", "pgm", "bmp", "tif", "tiff", "webp"}
)


def _compute_contrast_score(image: np.ndarray, num_bins: int = 128) -> float:
    """Compute the Entropic Contrast Measure for a single image array.

    This is the internal implementation called by :func:`contrast_quality`.
    It is separated to allow unit-testing without disk I/O.

    Parameters
    ----------
    image : ndarray of shape (H, W) or (H, W, 3) or (H, W, 4)
        Input image loaded as a NumPy array.  Grayscale and RGBA images are
        handled automatically.
    num_bins : int, default 128
        Number of histogram bins.  Larger values give finer resolution but
        increase sensitivity to noise.  Typical range: 64–256.

    Returns
    -------
    float
        Mutual Information between the intensity histogram and the histogram
        of the contrast-equalised image.  Higher is better.

    Notes
    -----
    **Colour images**: converted to HSV; the Value channel carries luminance.
    **Grayscale images**: treated as the Value channel directly.
    **Alpha channel**: stripped before processing.
    """
    # ------------------------------------------------------------------ Pre-processing
    # Strip alpha channel if present (RGBA → RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Handle grayscale (2-D) by treating it as the V channel directly
    if image.ndim == 2:
        v_channel = img_as_float(image)
    else:
        # Convert RGB → HSV; Value (index 2) encodes perceived luminance
        image_hsv = color.rgb2hsv(image)
        v_channel = image_hsv[:, :, 2]

    v_float = img_as_float(v_channel)

    # ------------------------------------------------------------------ Histograms
    # Intensity histogram of the original V channel
    hist_orig, bin_edges = np.histogram(v_float, bins=num_bins, density=True)

    # Contrast-equalised V channel: histogram equalisation stretches the CDF
    # towards uniformity.  We use the same bin edges to ensure comparable support.
    v_equalised = exposure.equalize_hist(v_float, nbins=num_bins)
    hist_eq, _ = np.histogram(
        v_equalised,
        bins=num_bins,
        density=True,
        range=(bin_edges[0], bin_edges[-1]),
    )

    # ------------------------------------------------------------------ MI
    return mutual_information(hist_orig, hist_eq)


def contrast_quality(image_path: str | Path, num_bins: int = 128) -> float | None:
    """Compute the Entropic Contrast Measure for a single image file.

    Parameters
    ----------
    image_path : str or Path
        Absolute or relative path to an image file.  Supported formats:
        JPEG, PNG, BMP, PGM, TIFF, WebP.
    num_bins : int, default 128
        Number of histogram bins passed to the internal scorer.

    Returns
    -------
    float or None
        Contrast quality score (higher = better contrast), or ``None`` if
        the image cannot be read.

    Examples
    --------
    >>> from entropic_contrast import contrast_quality
    >>> score = contrast_quality("examples/correct1.jpg")
    >>> print(f"{score:.4f}")
    4.8087
    """
    image_path = Path(image_path)
    try:
        image = io.imread(str(image_path))
    except Exception as exc:
        logger.error("Could not read '%s': %s", image_path, exc)
        return None

    score = _compute_contrast_score(image, num_bins=num_bins)
    logger.debug("%-50s  score=%.4f", image_path.name, score)
    return score


def contrast_quality_batch(
    image_dir: str | Path,
    output_path: str | Path | None = None,
    num_bins: int = 128,
    recursive: bool = False,
) -> Path | None:
    """Score all images in a directory and write results to a CSV log.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing image files.
    output_path : str or Path or None
        Path for the output CSV file.  Defaults to
        ``<image_dir>/../contrast_quality.csv``.
    num_bins : int, default 128
        Number of histogram bins.
    recursive : bool, default False
        If True, recurse into subdirectories.

    Returns
    -------
    Path or None
        Path to the written CSV file, or ``None`` if the directory is invalid
        or no images were found.

    Output format
    -------------
    The output CSV has two columns (no header)::

        filename,score
        correct1.jpg,4.808709
        under1.jpg,2.017121

    Examples
    --------
    >>> from entropic_contrast import contrast_quality_batch
    >>> log = contrast_quality_batch("examples/", output_path="results/scores.csv")
    >>> print(log)
    results/scores.csv
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        logger.error("'%s' is not a valid directory.", image_dir)
        return None

    if output_path is None:
        output_path = image_dir.parent / "contrast_quality.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect image files
    glob_fn = image_dir.rglob if recursive else image_dir.glob
    image_files = sorted(
        p for p in glob_fn("*")
        if p.is_file() and p.suffix.lstrip(".").lower() in _VALID_EXTENSIONS
    )

    if not image_files:
        logger.warning("No image files found in '%s'.", image_dir)
        return None

    logger.info("Scoring %d images from '%s' ...", len(image_files), image_dir)

    results: list[tuple[str, float]] = []
    for img_path in image_files:
        score = contrast_quality(img_path, num_bins=num_bins)
        if score is not None:
            results.append((img_path.name, score))
            logger.info("  %-50s  %.4f", img_path.name, score)

    with output_path.open("w") as fh:
        for filename, score in results:
            fh.write(f"{filename},{score:.6f}\n")

    logger.info("Results written to '%s'", output_path)
    return output_path
