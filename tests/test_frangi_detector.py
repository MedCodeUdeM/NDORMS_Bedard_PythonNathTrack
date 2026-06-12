"""
Unit tests for FrangiDetector.

Run from project root:
    python -m pytest tests/test_frangi_detector.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ultrasound_tracker.frangi_detector as frangi_mod
from ultrasound_tracker.frangi_detector import (
    FrangiDetector,
    frangi_filter,
    _hessian_eigenvalues,
    _vesselness_at_scale,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_image():
    """200×300 uniform grey image — no structure."""
    return np.full((200, 300), 128, dtype=np.uint8)


@pytest.fixture
def horizontal_stripe():
    """200×300 image with a bright horizontal band."""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[90:110, :] = 200
    return img


@pytest.fixture
def diagonal_stripe():
    """200×300 image with a diagonal bright band."""
    img = np.zeros((200, 300), dtype=np.uint8)

    for col in range(300):
        row = int(60 + col * np.tan(np.radians(26)))
        r0 = max(0, row - 5)
        r1 = min(200, row + 5)

        if r0 < r1:
            img[r0:r1, col] = 200

    return img


@pytest.fixture
def noisy_stripe():
    """Diagonal stripe with added Gaussian noise."""
    rng = np.random.default_rng(42)

    img = np.zeros((200, 300), dtype=np.float32)

    for col in range(300):
        row = int(60 + col * np.tan(np.radians(20)))
        r0 = max(0, row - 4)
        r1 = min(200, row + 4)

        if r0 < r1:
            img[r0:r1, col] = 180

    noise = rng.normal(0, 15, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Basic smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def test_frangi_filter_runs_on_synthetic_image():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50:53, 10:90] = 255

    out = frangi_filter(
        img,
        sigmas=(1, 2),
        alpha=0.5,
        beta=15.0,
        black_ridges=False,
    )

    assert out.shape == img.shape
    assert out.max() > 0


def test_frangi_detector_runs():
    img = np.zeros((120, 120), dtype=np.uint8)

    for col in range(10, 110):
        row = int(30 + col * np.tan(np.radians(25)))
        r0 = max(0, row - 2)
        r1 = min(120, row + 3)
        img[r0:r1, col] = 255

    detector = FrangiDetector(
        sigmas=(1, 2),
        threshold=0.05,
        angle_min=10,
        angle_max=40,
        hough_threshold=5,
        min_line_length=15,
        max_line_gap=10,
    )

    lines, angles, lengths = detector.detect(img)

    assert lines is not None
    assert len(lines) > 0
    assert angles is not None
    assert lengths is not None


# ─────────────────────────────────────────────────────────────────────────────
# Hessian eigenvalues
# ─────────────────────────────────────────────────────────────────────────────

class TestHessianEigenvalues:

    def test_output_shape(self, horizontal_stripe):
        img = horizontal_stripe.astype(np.float64)
        lam1, lam2 = _hessian_eigenvalues(img, sigma=2)

        assert lam1.shape == img.shape
        assert lam2.shape == img.shape

    def test_uniform_image_near_zero(self, blank_image):
        img = blank_image.astype(np.float64)
        lam1, lam2 = _hessian_eigenvalues(img, sigma=2)

        assert np.abs(lam1).max() < 1e-6
        assert np.abs(lam2).max() < 1e-6

    def test_abs_lam1_leq_abs_lam2(self, horizontal_stripe):
        """
        Frangi convention:
            |lambda1| <= |lambda2|
        """
        img = horizontal_stripe.astype(np.float64)
        lam1, lam2 = _hessian_eigenvalues(img, sigma=2)

        assert np.all(np.abs(lam1) <= np.abs(lam2) + 1e-10)

    def test_structure_detected_on_stripe(self, horizontal_stripe):
        img = horizontal_stripe.astype(np.float64)
        lam1, lam2 = _hessian_eigenvalues(img, sigma=3)

        stripe_region = np.abs(lam2[85:115, 50:250])
        assert stripe_region.max() > 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Vesselness at scale
# ─────────────────────────────────────────────────────────────────────────────

class TestVesselnessAtScale:

    def test_output_range(self, horizontal_stripe):
        img = horizontal_stripe.astype(np.float64)
        lam1, lam2 = _hessian_eigenvalues(img, sigma=2)

        V = _vesselness_at_scale(
            lam1,
            lam2,
            alpha=0.5,
            beta=15,
            black_ridges=False,
        )

        assert V.min() >= 0.0
        assert V.max() <= 1.0 + 1e-6

    def test_blank_image_zero_vesselness(self, blank_image):
        img = blank_image.astype(np.float64)
        lam1, lam2 = _hessian_eigenvalues(img, sigma=2)

        V = _vesselness_at_scale(
            lam1,
            lam2,
            alpha=0.5,
            beta=15,
            black_ridges=False,
        )

        assert V.max() < 1e-6

    def test_bright_ridges_flag_changes_response(self, horizontal_stripe):
        img = horizontal_stripe.astype(np.float64)
        lam1, lam2 = _hessian_eigenvalues(img, sigma=2)

        V_bright = _vesselness_at_scale(lam1, lam2, 0.5, 15, black_ridges=False)
        V_dark = _vesselness_at_scale(lam1, lam2, 0.5, 15, black_ridges=True)

        assert not np.allclose(V_bright, V_dark)

    def test_output_dtype(self, horizontal_stripe):
        img = horizontal_stripe.astype(np.float64)
        lam1, lam2 = _hessian_eigenvalues(img, sigma=2)

        V = _vesselness_at_scale(lam1, lam2, 0.5, 15, False)

        assert V.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# frangi_filter
# ─────────────────────────────────────────────────────────────────────────────

class TestFrangiFilter:

    def test_output_shape(self, horizontal_stripe):
        V = frangi_filter(horizontal_stripe, sigmas=(1, 2))
        assert V.shape == horizontal_stripe.shape

    def test_output_in_0_1(self, horizontal_stripe):
        V = frangi_filter(horizontal_stripe, sigmas=(1, 2, 4))
        assert V.min() >= 0.0
        assert V.max() <= 1.0 + 1e-6

    def test_blank_image_near_zero(self, blank_image):
        V = frangi_filter(blank_image, sigmas=(1, 2))
        assert V.max() < 1e-3

    def test_stripe_gives_high_vesselness(self, horizontal_stripe):
        V = frangi_filter(horizontal_stripe, sigmas=(1, 2, 4), beta=5)
        assert V.max() > 0.3

    def test_multi_scale_runs_and_detects_structure(self, horizontal_stripe):
        V_multi = frangi_filter(horizontal_stripe, sigmas=(1, 2, 4))
        V_single = frangi_filter(horizontal_stripe, sigmas=(2,))

        assert V_multi.shape == V_single.shape
        assert V_multi.max() > 0.3
        assert V_single.max() > 0.3

    def test_different_sigmas_differ(self, diagonal_stripe):
        V_small = frangi_filter(diagonal_stripe, sigmas=(1,))
        V_large = frangi_filter(diagonal_stripe, sigmas=(6,))

        assert not np.allclose(V_small, V_large)

    def test_no_scipy_raises(self, horizontal_stripe, monkeypatch):
        monkeypatch.setattr(frangi_mod, "SCIPY_AVAILABLE", False)

        with pytest.raises(ImportError, match="scipy"):
            frangi_filter(horizontal_stripe)

        monkeypatch.setattr(frangi_mod, "SCIPY_AVAILABLE", True)


# ─────────────────────────────────────────────────────────────────────────────
# FrangiDetector — get_vesselness_map
# ─────────────────────────────────────────────────────────────────────────────

class TestFrangiDetectorVesselnessMap:

    def test_returns_float32(self, horizontal_stripe):
        det = FrangiDetector()
        V = det.get_vesselness_map(horizontal_stripe)

        assert V.dtype == np.float32

    def test_shape_preserved(self, horizontal_stripe):
        det = FrangiDetector()
        V = det.get_vesselness_map(horizontal_stripe)

        assert V.shape == horizontal_stripe.shape

    def test_values_in_0_1(self, horizontal_stripe):
        det = FrangiDetector()
        V = det.get_vesselness_map(horizontal_stripe)

        assert 0.0 <= V.min()
        assert V.max() <= 1.0 + 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# FrangiDetector — detect
# ─────────────────────────────────────────────────────────────────────────────

class TestFrangiDetectorDetect:

    def test_blank_returns_none(self, blank_image):
        det = FrangiDetector()
        lines, angles, lengths = det.detect(blank_image)

        assert lines is None
        assert angles is None
        assert lengths is None

    def test_detect_returns_tuple_of_three(self, diagonal_stripe):
        det = FrangiDetector(
            sigmas=(1, 2),
            threshold=0.05,
            angle_min=10,
            angle_max=40,
        )

        result = det.detect(diagonal_stripe)

        assert len(result) == 3

    def test_lines_shape_when_detected(self, diagonal_stripe):
        det = FrangiDetector(
            sigmas=(1, 2),
            threshold=0.05,
            angle_min=10,
            angle_max=40,
        )

        lines, angles, lengths = det.detect(diagonal_stripe)

        if lines is not None:
            assert lines.ndim == 2
            assert lines.shape[1] == 4
            assert angles.ndim == 1
            assert lengths.ndim == 1
            assert len(lines) == len(angles) == len(lengths)

    def test_angles_in_filter_range(self, diagonal_stripe):
        det = FrangiDetector(
            sigmas=(1, 2),
            threshold=0.05,
            angle_min=10,
            angle_max=40,
        )

        lines, angles, lengths = det.detect(diagonal_stripe)

        if angles is not None:
            assert np.all(angles >= det.angle_min - 1e-3)
            assert np.all(angles <= det.angle_max + 1e-3)

    def test_lengths_positive(self, diagonal_stripe):
        det = FrangiDetector(sigmas=(1, 2), threshold=0.05)

        lines, angles, lengths = det.detect(diagonal_stripe)

        if lengths is not None:
            assert np.all(lengths > 0)

    def test_lines_dtype_float32(self, diagonal_stripe):
        det = FrangiDetector(sigmas=(1, 2), threshold=0.05)

        lines, angles, lengths = det.detect(diagonal_stripe)

        if lines is not None:
            assert lines.dtype == np.float32

    def test_noisy_stripe_detection_does_not_crash(self, noisy_stripe):
        det = FrangiDetector(
            sigmas=(1, 2, 4),
            threshold=0.08,
            angle_min=10,
            angle_max=35,
            min_line_length=20,
        )

        lines, angles, lengths = det.detect(noisy_stripe)

        assert isinstance(lines, (np.ndarray, type(None)))


# ─────────────────────────────────────────────────────────────────────────────
# FrangiDetector — estimate
# ─────────────────────────────────────────────────────────────────────────────

class TestFrangiDetectorEstimate:

    def test_blank_returns_none_none(self, blank_image):
        det = FrangiDetector()
        angle, length = det.estimate(blank_image)

        assert angle is None
        assert length is None

    def test_estimate_types(self, diagonal_stripe):
        det = FrangiDetector(sigmas=(1, 2), threshold=0.05)
        angle, length = det.estimate(diagonal_stripe)

        if angle is not None:
            assert isinstance(angle, float)
            assert isinstance(length, float)

    def test_estimate_angle_in_range(self, diagonal_stripe):
        det = FrangiDetector(
            sigmas=(1, 2),
            threshold=0.05,
            angle_min=10,
            angle_max=40,
        )

        angle, length = det.estimate(diagonal_stripe)

        if angle is not None:
            assert det.angle_min <= angle <= det.angle_max


# ─────────────────────────────────────────────────────────────────────────────
# Parameter effects
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterEffects:

    def test_higher_threshold_detects_fewer(self, diagonal_stripe):
        det_low = FrangiDetector(sigmas=(1, 2), threshold=0.05)
        det_high = FrangiDetector(sigmas=(1, 2), threshold=0.5)

        lines_low, _, _ = det_low.detect(diagonal_stripe)
        lines_high, _, _ = det_high.detect(diagonal_stripe)

        n_low = 0 if lines_low is None else len(lines_low)
        n_high = 0 if lines_high is None else len(lines_high)

        assert n_high <= n_low

    def test_min_line_length_limits(self, diagonal_stripe):
        det_short = FrangiDetector(
            sigmas=(1, 2),
            threshold=0.05,
            min_line_length=10,
        )
        det_long = FrangiDetector(
            sigmas=(1, 2),
            threshold=0.05,
            min_line_length=100,
        )

        lines_short, _, _ = det_short.detect(diagonal_stripe)
        lines_long, _, _ = det_long.detect(diagonal_stripe)

        n_short = 0 if lines_short is None else len(lines_short)
        n_long = 0 if lines_long is None else len(lines_long)

        assert n_long <= n_short

    def test_angle_range_limits(self, diagonal_stripe):
        det_wide = FrangiDetector(
            sigmas=(1, 2),
            threshold=0.05,
            angle_min=0,
            angle_max=90,
        )
        det_narrow = FrangiDetector(
            sigmas=(1, 2),
            threshold=0.05,
            angle_min=25,
            angle_max=30,
        )

        lines_wide, _, _ = det_wide.detect(diagonal_stripe)
        lines_narrow, _, _ = det_narrow.detect(diagonal_stripe)

        n_wide = 0 if lines_wide is None else len(lines_wide)
        n_narrow = 0 if lines_narrow is None else len(lines_narrow)

        assert n_narrow <= n_wide
