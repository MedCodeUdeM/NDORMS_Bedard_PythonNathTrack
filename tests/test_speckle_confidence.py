import numpy as np

from ultrasound_tracker.speckle_confidence import (
    SpeckleConfidenceConfig,
    adapt_measurement_covariance,
    compute_motion_consistency,
    compute_speckle_coherence,
    confidence_to_r_scale,
    zncc,
)
from ultrasound_tracker.ultratimtrack_matlab_2state import (
    MatlabTwoStateKalmanConfig,
    run_matlab_2state_kalman,
)


def test_zncc_is_high_for_identical_textured_patches():
    rng = np.random.default_rng(123)
    patch = rng.normal(size=(21, 21)).astype(np.float32)

    score = zncc(patch, patch.copy(), min_texture_variance=1e-6)

    assert score > 0.999


def test_speckle_coherence_drops_for_decorrelated_images():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(96, 96), dtype=np.uint8)
    decorrelated = rng.integers(0, 255, size=(96, 96), dtype=np.uint8)
    config = SpeckleConfidenceConfig(
        block_size=11,
        stride=16,
        search_radius=4,
        min_texture_variance=1.0,
        zncc_low=0.4,
        zncc_high=0.9,
    )

    same = compute_speckle_coherence(image, image.copy(), roi=(0, 0, 96, 96), config=config)
    noisy = compute_speckle_coherence(image, decorrelated, roi=(0, 0, 96, 96), config=config)

    assert same["speckle_confidence"] > 0.95
    assert noisy["speckle_confidence"] < same["speckle_confidence"]
    assert noisy["speckle_zncc"] < same["speckle_zncc"]


def test_low_texture_patches_are_rejected():
    image = np.zeros((80, 80), dtype=np.uint8)
    config = SpeckleConfidenceConfig(block_size=11, stride=16, search_radius=4, confidence_floor=0.07)

    out = compute_speckle_coherence(image, image.copy(), roi=(0, 0, 80, 80), config=config)

    assert out["speckle_confidence"] == 0.07
    assert out["n_valid_patches"] == 0
    assert np.isnan(out["speckle_zncc"])


def test_motion_consistency_is_high_for_coherent_displacement():
    xs, ys = np.meshgrid(np.arange(0, 5), np.arange(0, 4))
    prev = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    curr = prev + np.array([2.0, -1.0], dtype=np.float32)

    out = compute_motion_consistency(prev, curr, config=SpeckleConfidenceConfig())

    assert out["motion_consistency"] > 0.99
    assert out["motion_spread_px"] == 0.0


def test_motion_consistency_is_low_for_random_displacements():
    rng = np.random.default_rng(7)
    prev = rng.uniform(0, 100, size=(60, 2)).astype(np.float32)
    curr = prev + rng.normal(0, 10, size=(60, 2)).astype(np.float32)
    config = SpeckleConfidenceConfig(motion_spread_scale_px=2.0)

    out = compute_motion_consistency(prev, curr, config=config)

    assert out["motion_consistency"] < 0.5


def test_adaptive_covariance_increases_when_confidence_decreases():
    config = SpeckleConfidenceConfig(r_min_scale=1.0, r_max_scale=20.0, r_gamma=1.5)
    R_base = np.diag([2.0, 3.0])

    high = adapt_measurement_covariance(R_base, 0.95, config)
    low = adapt_measurement_covariance(R_base, 0.10, config)

    assert high.shape == R_base.shape
    assert np.all(np.linalg.eigvalsh(high) > 0)
    assert np.all(np.linalg.eigvalsh(low) > 0)
    assert confidence_to_r_scale(0.10, config) > confidence_to_r_scale(0.95, config)
    assert np.all(np.diag(low) > np.diag(high))


def test_fixed_r_ignores_adaptive_scale_when_disabled():
    klt = np.array(
        [
            [80.0, 10.0, 30.0, 60.0],
            [81.0, 10.0, 31.0, 60.0],
            [82.0, 10.0, 32.0, 60.0],
        ],
        dtype=float,
    )
    superficial = np.tile(np.array([[1.0, 10.0, 101.0, 10.0]]), (3, 1))
    deep = np.tile(np.array([[1.0, 60.0, 101.0, 60.0]]), (3, 1))
    alpha = np.array([45.0, 44.0, 43.0])
    scale = np.array([1.0, 20.0, 20.0])
    fixed_config = MatlabTwoStateKalmanConfig(
        q_parameter=0.01,
        x_measurement_variance=100.0,
        alpha_measurement_variance=3.0,
        run_smoother=False,
        use_adaptive_R=False,
    )

    baseline = run_matlab_2state_kalman(klt, alpha, superficial, deep, config=fixed_config)
    with_unused_scale = run_matlab_2state_kalman(
        klt,
        alpha,
        superficial,
        deep,
        config=fixed_config,
        measurement_r_scale=scale,
    )

    np.testing.assert_allclose(with_unused_scale["X_plus"], baseline["X_plus"])
    np.testing.assert_allclose(with_unused_scale["measurement_R_diag"][:, 0], 100.0)
    np.testing.assert_allclose(with_unused_scale["measurement_R_diag"][:, 1], 3.0)


def test_adaptive_r_scales_measurement_diag_when_enabled():
    klt = np.array(
        [
            [80.0, 10.0, 30.0, 60.0],
            [81.0, 10.0, 31.0, 60.0],
        ],
        dtype=float,
    )
    superficial = np.tile(np.array([[1.0, 10.0, 101.0, 10.0]]), (2, 1))
    deep = np.tile(np.array([[1.0, 60.0, 101.0, 60.0]]), (2, 1))
    alpha = np.array([45.0, 44.0])
    config = MatlabTwoStateKalmanConfig(
        x_measurement_variance=100.0,
        alpha_measurement_variance=3.0,
        run_smoother=False,
        use_adaptive_R=True,
    )

    out = run_matlab_2state_kalman(
        klt,
        alpha,
        superficial,
        deep,
        config=config,
        measurement_r_scale=np.array([1.0, 5.0]),
    )

    np.testing.assert_allclose(out["measurement_R_diag"][1], [500.0, 15.0])

