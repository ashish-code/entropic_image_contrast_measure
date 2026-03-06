"""
metrics.py — Mutual Information between discrete histogram sequences.

This module provides the core information-theoretic measures used by the
Entropic Contrast algorithm.  Both functions operate on histogram arrays
(counts or densities) — they do not accept images directly.

Algorithm context
-----------------
The ECM algorithm computes MI between the original V-channel intensity
histogram and the histogram of the contrast-equalised image.  The two
histogram arrays are treated as **paired label sequences**: position *i*
contributes a paired observation ``(hist_a[i], hist_b[i])``, and MI is
computed from the resulting K×K contingency table.

This is equivalent to the ``sklearn.metrics.mutual_info_score`` approach used
in the original 2017 implementation and produces scores that scale naturally
with the number of bins (maximum ≈ ln(K) nats for perfectly matched distinct
histograms).

Interpretation
--------------
- Two **identical** histograms with all-distinct bin values → MI ≈ ln(K) (max)
- Two **perfectly different** histograms → MI ≈ 0
- A histogram identical to itself but with all equal bin values → MI = 0
  (knowing one value tells you nothing new, since all are the same class)

For a well-exposed image the V-channel histogram is already near-uniform, so
histogram equalisation barely shifts bin densities → hist_a ≈ hist_b → high MI.
For an under/over-exposed image, equalisation dramatically redistributes the
densities → low MI.

References
----------
Cover, T.M., Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed.
"""

from __future__ import annotations

import numpy as np


def _contingency_mi(a_labels: np.ndarray, b_labels: np.ndarray) -> float:
    """MI from a contingency table of two integer label sequences.

    Computes:

    .. math::

        MI = \\sum_{i,j} \\frac{C_{ij}}{N}
             \\log \\frac{N \\cdot C_{ij}}{C_{i\\cdot} \\cdot C_{\\cdot j}}

    where :math:`C` is the contingency matrix, *N* is the total count, and
    :math:`C_{i\\cdot}`, :math:`C_{\\cdot j}` are row/column marginals.

    Parameters
    ----------
    a_labels, b_labels : ndarray of shape (N,), integer dtype
        Paired integer class labels.

    Returns
    -------
    float
        MI in nats.  Non-negative.
    """
    N = len(a_labels)
    n_a = int(a_labels.max()) + 1
    n_b = int(b_labels.max()) + 1

    # Build contingency matrix efficiently with np.add.at
    C = np.zeros((n_a, n_b), dtype=np.float64)
    np.add.at(C, (a_labels, b_labels), 1.0)

    # Marginals
    row_sum = C.sum(axis=1)  # (n_a,)
    col_sum = C.sum(axis=0)  # (n_b,)

    # MI: iterate only over non-zero cells
    rows, cols = np.nonzero(C)
    mi = 0.0
    for r, c in zip(rows, cols):
        p_joint = C[r, c] / N
        p_marg = (row_sum[r] / N) * (col_sum[c] / N)
        mi += p_joint * np.log(p_joint / p_marg)

    return float(mi)


def mutual_information(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Mutual Information between two 1-D histograms as paired label sequences.

    Treats each index *i* as a paired observation ``(hist_a[i], hist_b[i])``,
    builds a K×K contingency table from the co-occurrences of unique values,
    and computes the standard discrete MI:

    .. math::

        MI(A, B) = \\sum_{i,j} p(a_i, b_j)
                   \\log \\frac{p(a_i, b_j)}{p(a_i)\\,p(b_j)}

    This is the same computation as ``sklearn.metrics.mutual_info_score``,
    which the original ``pycontrast.py`` used.

    Score interpretation
    --------------------
    - **High score** (→ ln(K)): histograms have similar density values at the
      same bin positions — well-contrasted image.
    - **Low score** (→ 0): histograms are very different — poor contrast.

    Parameters
    ----------
    hist_a : ndarray of shape (K,)
        First histogram (counts or densities, non-negative).
    hist_b : ndarray of shape (K,)
        Second histogram over the **same** bin edges as *hist_a*.

    Returns
    -------
    float
        MI in nats.  Non-negative.

    Raises
    ------
    ValueError
        If *hist_a* and *hist_b* have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from entropic_contrast.metrics import mutual_information
    >>> p = np.array([0.1, 0.4, 0.3, 0.2])   # all distinct values
    >>> mutual_information(p, p)               # identical → MI = ln(4) ≈ 1.386
    1.386...
    >>> mutual_information(p, p[::-1])         # reversed — different pairing → MI ≈ 0
    0.0
    """
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    if hist_a.shape != hist_b.shape:
        raise ValueError(
            f"hist_a and hist_b must have the same shape; "
            f"got {hist_a.shape} vs {hist_b.shape}."
        )

    # Map float values to integer class labels via np.unique
    _, a_labels = np.unique(hist_a, return_inverse=True)
    _, b_labels = np.unique(hist_b, return_inverse=True)

    return _contingency_mi(a_labels, b_labels)


def normalised_mutual_information(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Normalised Mutual Information (NMI) bounded in [0, 1].

    NMI normalises the raw MI by the mean of the two marginal entropies:

    .. math::

        NMI(A, B) = \\frac{2 \\cdot MI(A, B)}{H(A) + H(B)}

    where :math:`H(A)` and :math:`H(B)` are the entropies of the label
    distributions induced by the unique values in each histogram.

    This removes the dependence on the number of unique values (which controls
    the scale of the raw MI score), making scores comparable across images
    with different histograms or bin counts.

    Parameters
    ----------
    hist_a : ndarray of shape (K,)
    hist_b : ndarray of shape (K,)

    Returns
    -------
    float
        NMI in [0, 1].  Returns 0.0 if both distributions have zero entropy
        (i.e. all bin values are identical).

    Raises
    ------
    ValueError
        If *hist_a* and *hist_b* have different lengths.
    """
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    if hist_a.shape != hist_b.shape:
        raise ValueError(
            f"hist_a and hist_b must have the same shape; "
            f"got {hist_a.shape} vs {hist_b.shape}."
        )

    _, a_labels = np.unique(hist_a, return_inverse=True)
    _, b_labels = np.unique(hist_b, return_inverse=True)

    # Marginal entropies from label distributions
    def _label_entropy(labels: np.ndarray) -> float:
        counts = np.bincount(labels).astype(float)
        p = counts / counts.sum()
        mask = p > 0
        return float(-np.sum(p[mask] * np.log(p[mask])))

    h_a = _label_entropy(a_labels)
    h_b = _label_entropy(b_labels)
    denom = h_a + h_b

    if denom < 1e-12:
        return 0.0

    mi = _contingency_mi(a_labels, b_labels)
    return float(2.0 * mi / denom)
