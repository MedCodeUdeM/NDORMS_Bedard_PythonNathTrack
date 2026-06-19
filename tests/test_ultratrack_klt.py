import numpy as np

from ultrasound_tracker.ultratrack_klt import (
    apply_affine_1b,
    estimate_affine_matlab_coords,
    filter_points_by_mask,
    tracking_masks_from_geofeature,
)


def test_apply_affine_1b_preserves_segment_shape():
    segment = np.array([10.0, 20.0, 30.0, 40.0])
    affine = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -3.0]], dtype=np.float32)

    transformed = apply_affine_1b(segment, affine)

    np.testing.assert_allclose(transformed, [12.0, 17.0, 32.0, 37.0])
    assert transformed.shape == segment.shape


def test_estimate_affine_matlab_coords_recovers_translation():
    old = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    new = old + np.array([2.0, -3.0], dtype=np.float32)

    affine, inliers = estimate_affine_matlab_coords(old, new)

    assert affine is not None
    assert inliers == 4
    np.testing.assert_allclose(affine[:, :2], np.eye(2), atol=1e-5)
    np.testing.assert_allclose(affine[:, 2], [2.0, -3.0], atol=1e-5)


def test_tracking_masks_from_geofeature_builds_expected_keys():
    entry = {
        "x": np.array([[10.0, 50.0]]),
        "y": np.array([[40.0, 20.0]]),
        "super_pos": np.array([10.0, 12.0]),
        "deep_pos": np.array([80.0, 82.0]),
    }

    masks = tracking_masks_from_geofeature(
        entry,
        shape=(100, 80),
        super_cut=(0.0, 0.4),
        deep_cut=(0.6, 1.0),
    )

    assert {"line_mask", "fcor_mask", "fascicle_mask", "super_mask", "deep_mask", "apo_mask"} <= set(masks)
    assert masks["fascicle_mask"].shape == (100, 80)
    assert masks["fascicle_mask"].sum() > 0

    points = np.array([[[10.0, 40.0]], [[79.0, 99.0]]], dtype=np.float32)
    filtered = filter_points_by_mask(points, masks["line_mask"])
    assert filtered.shape[1:] == (1, 2)


def test_apply_affine_1b_accepts_point_matrix():
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 6.0]], dtype=np.float32)

    transformed = apply_affine_1b(points, affine)

    np.testing.assert_allclose(transformed, [[6.0, 8.0], [8.0, 10.0]])
    assert transformed.shape == points.shape
