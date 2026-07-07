import numpy as np

from ultrasound_tracker.timtrack_hough import (
    DoHoughParams,
    candidate_mass_below,
    dohough,
    dohough_angle_profile_localmax,
    estimate_fascicle_alpha_dohough,
    gap_to_nearest_lower,
    hough_bin_pixels,
    matlab_hough_accumulator,
    matlab_theta_from_range,
    should_use_localmax_fallback,
    weighted_median,
)


def _draw_up_right_line(shape=(100, 120), alpha_deg=25.0):
    image = np.zeros(shape, dtype=bool)
    slope = np.tan(np.deg2rad(alpha_deg))
    x0 = 15
    y0 = 75

    for x in range(15, 105):
        y = int(round(y0 - slope * (x - x0)))
        if 1 <= y < shape[0] - 1:
            image[y - 1 : y + 2, x] = True

    return image


def _draw_line_for_alpha(shape=(100, 120), alpha_deg=25.0):
    image = np.zeros(shape, dtype=bool)
    slope = -np.tan(np.deg2rad(alpha_deg))
    x0 = 15
    y0 = 25 if alpha_deg < 0 else 75

    for x in range(15, 105):
        y = int(round(y0 + slope * (x - x0)))
        if 1 <= y < shape[0] - 1:
            image[y - 1 : y + 2, x] = True

    return image


def test_weighted_median_matches_matlab_flatten_sort_rule():
    values = np.array([[30.0, 10.0], [20.0, 40.0]])
    weights = np.array([[0.10, 0.20], [0.45, 0.25]])

    assert weighted_median(values, weights) == 20.0


def test_matlab_theta_from_range_uses_90_minus_angle_range():
    theta = matlab_theta_from_range(
        houghangles="specified",
        angle_range=(8.0, 80.0),
        thetares=1.0,
    )

    assert theta[0] == 10.0
    assert theta[-1] == 82.0


def test_hough_bin_pixels_round_trips_peak_contributors():
    image = np.zeros((20, 20), dtype=bool)
    image[10, 3:17] = True
    hmat, theta, rho = matlab_hough_accumulator(image, [90.0], rho_resolution=1.0)

    peak = np.unravel_index(np.argmax(hmat), hmat.shape)
    contributing = hough_bin_pixels(image, theta, rho, peak)

    assert np.array_equal(contributing, image)


def test_dohough_recovers_synthetic_up_right_fascicle_angle():
    image = _draw_up_right_line(alpha_deg=25.0)
    params = DoHoughParams(
        angle_range=(8.0, 80.0),
        thetares=1.0,
        rhores=1.0,
        emask_radius=(image.shape[0] / 2.0, image.shape[1] / 2.0),
        npeaks=5,
        replace_diagonal_bias=False,
    )

    result = dohough(image, params)

    assert np.isfinite(result["alpha"])
    assert np.isclose(result["alpha"], 25.0, atol=2.5)
    assert result["alphas"].size > 0
    assert result["weights"].size == result["alphas"].size


def test_dohough_recovers_reversed_fascicle_with_negative_angle_range():
    image = _draw_line_for_alpha(alpha_deg=-25.0)
    params = DoHoughParams(
        angle_range=(-80.0, -8.0),
        thetares=1.0,
        rhores=1.0,
        emask_radius=(image.shape[0] / 2.0, image.shape[1] / 2.0),
        npeaks=5,
        replace_diagonal_bias=False,
    )

    result = dohough(image, params)

    assert np.isfinite(result["alpha"])
    assert np.isclose(result["alpha"], -25.0, atol=2.5)


def test_angle_profile_localmax_recovers_synthetic_fascicle_angle():
    image = _draw_up_right_line(alpha_deg=25.0)
    params = DoHoughParams(
        angle_range=(8.0, 80.0),
        thetares=1.0,
        rhores=1.0,
        emask_radius=(image.shape[0] / 2.0, image.shape[1] / 2.0),
        npeaks=5,
        replace_diagonal_bias=False,
    )

    result = dohough_angle_profile_localmax(image, params)

    assert np.isfinite(result["alpha"])
    assert np.isclose(result["alpha"], 25.0, atol=2.5)
    assert result["peak_source"] == "angle_profile_localmax"


def test_localmax_fallback_mass_gap_detector_matches_notebook90_rule():
    alpha = 30.0
    alphas = np.array([15.0, 26.0, 30.0, 34.0])
    weights = np.array([0.30, 0.10, 0.45, 0.15])

    use_fallback, mass, gap = should_use_localmax_fallback(
        alpha,
        alphas,
        weights,
        min_mass_below=0.25,
        min_gap_to_lower_degrees=4.0,
    )

    assert np.isclose(candidate_mass_below(alpha, alphas, weights, 10.0), 0.30)
    assert np.isclose(gap_to_nearest_lower(alpha, alphas), 4.0)
    assert use_fallback
    assert np.isclose(mass, 0.30)
    assert np.isclose(gap, 4.0)


def test_estimate_fascicle_alpha_dohough_recovers_known_segment():
    image = _draw_up_right_line(alpha_deg=32.0)

    result = estimate_fascicle_alpha_dohough(
        image,
        npeaks=5,
    )

    assert np.isfinite(result["alpha"])
    assert np.isclose(result["alpha"], 32.0, atol=2.5)
    assert result["peaks"].shape[0] > 0
