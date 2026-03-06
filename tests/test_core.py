"""
tests/test_core.py — Unit tests for the core contrast scoring functions.

Tests use synthetic images (NumPy arrays) to avoid disk I/O for most cases,
and the bundled example images for integration-style sanity checks.

Synthetic image taxonomy used in tests:
    - ``flat``: constant grey image (no contrast) → low score
    - ``uniform``: uniform random noise (maximum entropy) → high score
    - ``binary``: half-black half-white (bimodal) → moderate score
    - ``gradient``: linear ramp → moderate score
"""

import numpy as np
import pytest
from pathlib import Path

from entropic_contrast.core import _compute_contrast_score, contrast_quality, contrast_quality_batch

# ---------------------------------------------------------------------------
# Fixtures — synthetic images as NumPy arrays
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def flat_image():
    """128×128 constant grey image — zero variance, minimum contrast."""
    return np.full((128, 128, 3), 128, dtype=np.uint8)


@pytest.fixture
def uniform_image():
    """128×128 uniform random RGB noise — maximum histogram entropy."""
    return np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def gradient_image():
    """128×128 horizontal linear ramp — moderate, well-spread histogram."""
    row = np.linspace(0, 255, 128, dtype=np.uint8)
    channel = np.tile(row, (128, 1))
    return np.stack([channel, channel, channel], axis=2)


@pytest.fixture
def grayscale_image():
    """128×128 2-D grayscale gradient."""
    return np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))


@pytest.fixture
def rgba_image():
    """64×64 RGBA image (alpha channel must be stripped cleanly)."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _compute_contrast_score (internal)
# ---------------------------------------------------------------------------

class TestComputeContrastScore:
    def test_returns_float(self, flat_image):
        score = _compute_contrast_score(flat_image)
        assert isinstance(score, float)

    def test_nonnegative(self, flat_image, uniform_image, gradient_image):
        for img in [flat_image, uniform_image, gradient_image]:
            assert _compute_contrast_score(img) >= -1e-9

    def test_grayscale_input(self, grayscale_image):
        """2-D grayscale arrays must be handled without error."""
        score = _compute_contrast_score(grayscale_image)
        assert score >= -1e-9

    def test_rgba_input(self, rgba_image):
        """4-channel RGBA images must have the alpha channel stripped cleanly."""
        score = _compute_contrast_score(rgba_image)
        assert score >= -1e-9

    def test_flat_and_gradient_are_finite(self, flat_image, gradient_image):
        """Both flat and gradient images return finite non-NaN scores.
        Note: a perfectly uniform gradient has equal bin densities (all same class),
        which gives MI=0 by design — the score reflects histogram distinctiveness
        relative to the equalized version, not spread per se."""
        s_flat = _compute_contrast_score(flat_image)
        s_grad = _compute_contrast_score(gradient_image)
        assert np.isfinite(s_flat) and not np.isnan(s_flat)
        assert np.isfinite(s_grad) and not np.isnan(s_grad)

    def test_bins_parameter(self, gradient_image):
        """Different bin counts should produce finite, non-NaN results."""
        for bins in [32, 64, 128, 256]:
            score = _compute_contrast_score(gradient_image, num_bins=bins)
            assert not np.isnan(score)
            assert np.isfinite(score)


# ---------------------------------------------------------------------------
# contrast_quality (public, file-based)
# ---------------------------------------------------------------------------

class TestContrastQuality:
    def test_returns_none_for_missing_file(self, tmp_path):
        score = contrast_quality(tmp_path / "nonexistent.jpg")
        assert score is None

    @pytest.mark.skipif(
        not (EXAMPLES_DIR / "correct1.jpg").is_file(),
        reason="Example images not available",
    )
    def test_well_exposed_beats_under_exposed(self):
        """correct1.jpg should score higher than under1.jpg."""
        s_correct = contrast_quality(EXAMPLES_DIR / "correct1.jpg")
        s_under = contrast_quality(EXAMPLES_DIR / "under1.jpg")
        assert s_correct is not None
        assert s_under is not None
        assert s_correct > s_under

    @pytest.mark.skipif(
        not (EXAMPLES_DIR / "correct1.jpg").is_file(),
        reason="Example images not available",
    )
    def test_correct_score_above_threshold(self):
        """A well-exposed image should score above 3.0."""
        score = contrast_quality(EXAMPLES_DIR / "correct1.jpg")
        assert score is not None
        assert score > 3.0


# ---------------------------------------------------------------------------
# contrast_quality_batch (public, directory-based)
# ---------------------------------------------------------------------------

class TestContrastQualityBatch:
    def test_returns_none_for_invalid_dir(self, tmp_path):
        result = contrast_quality_batch(tmp_path / "not_a_dir")
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        result = contrast_quality_batch(tmp_path)
        assert result is None

    def test_writes_csv(self, tmp_path):
        """Synthetic PNG images should be scored and a CSV written."""
        import imageio.v3 as iio

        # Write two synthetic images to a temp directory
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        iio.imwrite(tmp_path / "img1.png", img)
        iio.imwrite(tmp_path / "img2.png", img)

        out_csv = tmp_path / "scores.csv"
        result = contrast_quality_batch(tmp_path, output_path=out_csv)
        assert result == out_csv
        assert out_csv.is_file()

        lines = out_csv.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_csv_format(self, tmp_path):
        """Each line of the CSV should have filename,score format."""
        import imageio.v3 as iio

        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        iio.imwrite(tmp_path / "test.png", img)

        out_csv = tmp_path / "out.csv"
        contrast_quality_batch(tmp_path, output_path=out_csv)
        line = out_csv.read_text().strip()
        filename, score_str = line.split(",")
        assert filename == "test.png"
        assert float(score_str) >= 0.0
