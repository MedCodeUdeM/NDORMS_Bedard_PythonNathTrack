import numpy as np

from ultrasound_tracker.matlab_timtrack import (
    _parms_fascicle_mask,
    alpha_from_saved_peaks,
    compact_timtrack_geofeature,
    extract_saved_peak_arrays,
    fascicle_segment_from_aponeuroses_and_alpha,
    fascicle_segment_from_geofeature,
    get_fascicle_mask_matlab_like,
    line_mask_fascicle_subtraction,
    matlab_literal_fascicle_subtraction,
    reconstruct_saved_geofeature_alpha,
    saved_alpha_error,
)


def test_alpha_from_saved_peaks_uses_weighted_median():
    alphas = np.array([34.0, 32.0, 14.5, 12.5])
    weights = np.array([265.0, 262.0, 255.0, 250.0])

    assert alpha_from_saved_peaks(alphas, weights) == 32.0


def test_reconstruct_saved_geofeature_alpha_prefers_ws():
    entries = [
        {"alpha": 32.0, "alphas": [34.0, 32.0, 14.5, 12.5], "ws": [265.0, 262.0, 255.0, 250.0]},
        {"alpha": 19.5, "alphas": [21.0, 19.5, 19.5], "ws": [341.0, 329.0, 306.0]},
    ]

    reconstructed = reconstruct_saved_geofeature_alpha(entries)

    np.testing.assert_allclose(reconstructed, [32.0, 19.5])
    err = saved_alpha_error(entries)
    assert err["max_abs_error_deg"] == 0.0
    assert err["rmse_deg"] == 0.0


def test_extract_saved_peak_arrays_pads_to_requested_width():
    entries = [{"alphas": [1.0, 2.0], "ws": [3.0, 4.0]}]

    alpha, peak_alphas, peak_weights = extract_saved_peak_arrays(entries, max_peaks=4)

    np.testing.assert_allclose(alpha, [2.0])
    np.testing.assert_allclose(peak_alphas[:, :2], [[1.0, 2.0]])
    np.testing.assert_allclose(peak_weights[:, :2], [[3.0, 4.0]])
    assert np.isnan(peak_alphas[0, 2])
    assert np.isnan(peak_weights[0, 2])


def test_get_fascicle_mask_matlab_like_uses_mean_depths():
    fascicle = np.ones((12, 20), dtype=bool)
    super_vec = np.array([3.0, 3.0, 3.0])
    deep_vec = np.array([9.0, 9.0, 9.0])

    mask, radius, means = get_fascicle_mask_matlab_like(fascicle, super_vec, deep_vec)

    assert mask.shape == fascicle.shape
    assert radius == (3.0, 10.0)
    assert means == (3.0, 9.0)
    assert mask[5, 10]
    assert not mask[0, 0]


def test_parms_fascicle_mask_reuses_saved_emask_when_roi_not_redone():
    saved = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    fas_parms = {
        "Emask": saved,
        "Emask_radius": np.array([2.0, 3.0]),
        "redo_ROI": 0,
    }

    mask, radius = _parms_fascicle_mask(fas_parms, saved.shape)

    np.testing.assert_array_equal(mask, saved.astype(bool))
    np.testing.assert_allclose(radius, [2.0, 3.0])
    assert _parms_fascicle_mask({**fas_parms, "redo_ROI": 1}, saved.shape) is None


def test_matlab_literal_subtraction_matches_column_major_prefix_behavior():
    fascicle = np.ones((4, 4), dtype=bool)
    super_vec = np.array([0, 1, 0, 1], dtype=int)
    deep_vec = np.array([0, 0, 1, 0], dtype=int)

    out = matlab_literal_fascicle_subtraction(fascicle, super_vec, deep_vec)
    flat = out.ravel(order="F")

    np.testing.assert_array_equal(flat[:4], [True, False, False, False])
    assert flat[4:].all()


def test_line_mask_subtraction_removes_spatial_line_pixels():
    fascicle = np.ones((10, 10), dtype=bool)
    apox = np.array([1.0, 10.0])
    super_vec = np.array([5.0, 5.0])
    deep_vec = np.array([8.0, 8.0])

    out = line_mask_fascicle_subtraction(fascicle, apox, super_vec, deep_vec, radius=0)

    assert not out[4, :].any()
    assert not out[7, :].any()
    assert out[0, :].all()


def test_fascicle_segment_from_geofeature_intersects_coefficients():
    entry = {
        "fas_coef": np.array([-1.0, 100.0]),
        "super_coef": np.array([0.0, 20.0]),
        "deep_coef": np.array([0.0, 80.0]),
    }

    segment = fascicle_segment_from_geofeature(entry)

    np.testing.assert_allclose(segment, [80.0, 20.0, 20.0, 80.0])


def test_fascicle_segment_from_aponeuroses_and_alpha_rebuilds_extrapolated_line():
    segment = fascicle_segment_from_aponeuroses_and_alpha(
        np.array([0.0, 0.0]),
        np.array([0.0, 10.0]),
        45.0,
        100,
    )

    np.testing.assert_allclose(segment, [55.0, 0.0, 45.0, 10.0], atol=1e-5)


def test_fascicle_segment_from_geofeature_alpha_override():
    entry = {
        "fas_coef": np.array([-1.0, 100.0]),
        "super_coef": np.array([0.0, 0.0]),
        "deep_coef": np.array([0.0, 10.0]),
        "super_coef_linear": np.array([0.0, 0.0]),
        "deep_coef_linear": np.array([0.0, 10.0]),
        "image_shape": (20, 100),
    }

    segment = fascicle_segment_from_geofeature(entry, alpha_override=45.0)

    np.testing.assert_allclose(segment, [55.0, 0.0, 45.0, 10.0], atol=1e-5)


def test_compact_timtrack_geofeature_drops_heavy_fields_and_pads_peaks():
    entry = {
        "frame": 3,
        "alpha": 20.0,
        "alphas": np.array([20.0, 21.0]),
        "ws": np.array([5.0, 4.0]),
        "brightness": 12.5,
        "extrapolated_fraction": 0.2,
        "x": np.array([[1.0, 5.0], [2.0, 6.0]]),
        "y": np.array([[7.0, 8.0], [9.0, 10.0]]),
        "fascicle_masked": np.ones((5, 5), dtype=bool),
    }

    compact = compact_timtrack_geofeature(entry, max_peaks=4)

    assert compact["frame"] == 3
    assert "fascicle_masked" not in compact
    assert compact["brightness"] == 12.5
    assert compact["extrapolated_fraction"] == 0.2
    np.testing.assert_allclose(compact["alphas"][:2], [20.0, 21.0])
    assert np.isnan(compact["alphas"][2])
    np.testing.assert_allclose(compact["x"][:2], [[1.0, 5.0], [2.0, 6.0]])
