import numpy as np

from ultrasound_tracker.legacy.speckle import (
    SpeckleCoherenceParams,
    robust_mad_sigma,
    speckle_pair_confidence,
)


def test_robust_mad_sigma_ignores_nan_values():
    values = np.array([1.0, 1.0, 2.0, np.nan, 100.0])

    sigma = robust_mad_sigma(values)

    assert np.isfinite(sigma)


def test_speckle_pair_confidence_is_high_for_identical_textured_images():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(96, 96), dtype=np.uint8)

    params = SpeckleCoherenceParams(
        block_size=11,
        stride=16,
        search_radius=5,
        rho_min=0.8,
        max_displacement_px=8.0,
        sigma_ref_px=2.0,
        confidence_min=0.05,
    )

    out = speckle_pair_confidence(image, image.copy(), params=params)

    assert out["confidence"] > 0.95
    assert out["c_ncc"] > 0.99
    assert out["c_valid"] > 0.95
    assert out["c_motion"] > 0.99
    assert out["median_dx_px"] == 0.0
    assert out["median_dy_px"] == 0.0
    assert out["n_valid_blocks"] > 0


def test_speckle_pair_confidence_uses_minimum_for_textureless_images():
    image = np.zeros((96, 96), dtype=np.uint8)
    params = SpeckleCoherenceParams(
        block_size=11,
        stride=16,
        search_radius=5,
        confidence_min=0.07,
    )

    out = speckle_pair_confidence(image, image.copy(), params=params)

    assert out["confidence"] == 0.07
    assert out["c_valid"] == 0.0
    assert out["c_motion"] == 0.0
    assert out["n_valid_blocks"] == 0
