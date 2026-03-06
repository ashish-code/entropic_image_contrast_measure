"""
tests/test_metrics.py — Unit tests for the metrics module.

Key behavioral contract of the contingency-table MI:
- MI is measured between two 1-D arrays treated as paired label sequences.
- Each unique float value becomes a distinct class label.
- Two IDENTICAL arrays with K distinct values → MI = ln(K) (maximum).
- Two arrays sharing no common value at any position → MI = 0.
- All-same-value arrays → MI = 0 (no information, single class).
- MI is symmetric and non-negative.
- NMI is bounded in [0, 1]; NMI(a, a) = 1 when H(a) > 0.
"""

import numpy as np
import pytest

from entropic_contrast.metrics import mutual_information, normalised_mutual_information


class TestMutualInformation:
    def test_identical_distinct_values(self):
        """MI(a, a) with all-distinct values equals ln(K)."""
        K = 4
        a = np.array([0.1, 0.4, 0.3, 0.2])   # all distinct
        mi = mutual_information(a, a)
        assert mi == pytest.approx(np.log(K), rel=1e-6)

    def test_identical_many_bins(self):
        """MI(a, a) with 128 distinct bins ≈ ln(128) ≈ 4.85."""
        rng = np.random.default_rng(0)
        a = rng.random(128)                   # floats — all distinct with high probability
        a = np.unique(a)[:128]                # guarantee uniqueness
        mi = mutual_information(a, a)
        assert mi == pytest.approx(np.log(len(a)), rel=1e-4)

    def test_all_same_values_returns_zero(self):
        """When all elements share one value, MI = 0 (single class, no information)."""
        a = np.array([0.25, 0.25, 0.25, 0.25])
        assert mutual_information(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        """MI must be symmetric: MI(a, b) == MI(b, a)."""
        rng = np.random.default_rng(1)
        a = rng.random(32)
        b = rng.random(32)
        assert mutual_information(a, b) == pytest.approx(mutual_information(b, a), rel=1e-9)

    def test_nonnegative(self):
        """MI is always >= 0."""
        rng = np.random.default_rng(2)
        for _ in range(20):
            a = rng.random(16)
            b = rng.random(16)
            assert mutual_information(a, b) >= -1e-9

    def test_reversed_distinct_gives_max(self):
        """Any bijective pairing of K distinct values is a permutation; the
        contingency matrix is still a permutation matrix → MI = ln(K).
        This is an intrinsic property of the contingency-table MI formulation."""
        a = np.array([0.1, 0.4, 0.3, 0.2])
        b = a[::-1]
        assert mutual_information(a, b) == pytest.approx(np.log(4), rel=1e-6)

    def test_repeated_values_lower_mi(self):
        """When some bin values repeat, multiple positions share a class, which
        can reduce MI below ln(K).  Here two positions share value 0.1 in both
        arrays but with a cross-class pairing, lowering MI."""
        # a has repeated 0.1; b pairs first 0.1 with 0.5 (mismatch)
        a = np.array([0.1, 0.1, 0.3, 0.2])
        b = np.array([0.1, 0.5, 0.3, 0.2])   # position 1: class mismatch
        mi = mutual_information(a, b)
        # MI cannot reach ln(4) because of the repeated value reducing entropy
        assert mi < np.log(4)

    def test_shape_mismatch_raises(self):
        """Mismatched histogram lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            mutual_information(np.ones(8), np.ones(16))

    def test_counts_vs_density(self):
        """Scaling the histogram by a constant should not change MI (labels are invariant to scale)."""
        a = np.array([1.0, 4.0, 3.0, 2.0])
        mi_orig = mutual_information(a, a)
        mi_scaled = mutual_information(a * 100, a * 100)
        assert mi_orig == pytest.approx(mi_scaled, rel=1e-9)


class TestNormalisedMutualInformation:
    def test_identical_distinct_returns_one(self):
        """NMI(a, a) = 1 when a has all distinct values."""
        a = np.array([0.1, 0.4, 0.3, 0.2])
        assert normalised_mutual_information(a, a) == pytest.approx(1.0, rel=1e-6)

    def test_bounded_in_unit_interval(self):
        """NMI must lie in [0, 1]."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.random(16)
            b = rng.random(16)
            nmi = normalised_mutual_information(a, b)
            assert -1e-9 <= nmi <= 1 + 1e-9

    def test_all_zeros_returns_zero(self):
        """All-zero (or all-same) inputs → NMI = 0, not NaN."""
        assert normalised_mutual_information(np.zeros(8), np.zeros(8)) == pytest.approx(0.0, abs=1e-10)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            normalised_mutual_information(np.ones(4), np.ones(8))
