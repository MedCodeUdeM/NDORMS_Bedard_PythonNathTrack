import numpy as np

from ultrasound_tracker.matlab_timtrack import (
    alpha_from_saved_peaks,
    extract_saved_peak_arrays,
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
