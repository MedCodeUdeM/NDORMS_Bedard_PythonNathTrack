import numpy as np

from ultrasound_tracker.timtrack_hough import (
    DoHoughParams,
    dohough,
    hough_bin_pixels,
    matlab_hough_accumulator,
    matlab_theta_from_range,
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
