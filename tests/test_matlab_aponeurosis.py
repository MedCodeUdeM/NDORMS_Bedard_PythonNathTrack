import numpy as np

from ultrasound_tracker.geometry import line_angle_from_array
from ultrasound_tracker.matlab_aponeurosis import (
    MatlabHoughAponeurosisConfig,
    MatlabHoughAponeurosisDetector,
    detect_matlab_hough_aponeuroses,
    get_aponeurosis_line_hough_matlab_like,
    line_segment_from_polyfit_1b,
    zero_outside_vertical_cut,
)


def _synthetic_two_aponeurosis_frame(shape=(100, 120)):
    image = np.zeros(shape, dtype=np.uint8)
    image[20:23, 5:115] = 255
    image[78:81, 5:115] = 255
    return image


def test_zero_outside_vertical_cut_keeps_requested_band():
    mask = np.ones((10, 4), dtype=bool)

    out = zero_outside_vertical_cut(mask, (0.2, 0.7))

    assert not out[:2].any()
    assert out[2:6].all()
    assert not out[6:].any()


def test_get_aponeurosis_line_hough_matlab_like_returns_one_based_vector():
    mask = np.zeros((80, 100), dtype=bool)
    mask[29:32, 10:90] = True
    apox_1b = np.array([20, 40, 60, 80], dtype=float)

    apoy_1b, debug = get_aponeurosis_line_hough_matlab_like(mask, apox_1b, "super")

    assert np.all(np.isfinite(apoy_1b))
    assert np.nanmean(apoy_1b) > 29.0
    assert np.nanmean(apoy_1b) < 35.0
    assert np.isfinite(debug["theta"])
    assert np.isfinite(debug["rho"])


def test_line_segment_from_polyfit_1b_converts_to_zero_based_segment():
    coef = np.array([0.0, 25.0])

    line = line_segment_from_polyfit_1b(coef, width=100)

    np.testing.assert_allclose(line, np.array([0.0, 24.0, 99.0, 24.0]))
    assert np.isclose(line_angle_from_array(line), 0.0)


def test_detect_matlab_hough_aponeuroses_finds_two_horizontal_bands():
    image = _synthetic_two_aponeurosis_frame()
    config = MatlabHoughAponeurosisConfig(
        apox_1b=np.array([20, 40, 60, 80, 100], dtype=float),
        super_cut=(0.0, 0.5),
        deep_cut=(0.5, 1.0),
        threshold_block_size=15,
        super_maxangle=2.0,
        deep_maxangle=2.0,
    )

    result = detect_matlab_hough_aponeuroses(image, config=config)

    assert result["method"] == "matlab_hough"
    assert np.nanmean(result["super_vec_1b"]) < 30.0
    assert np.nanmean(result["deep_vec_1b"]) > 70.0
    assert result["super_line_0b"].shape == (4,)
    assert result["deep_line_0b"].shape == (4,)
    assert abs(result["super_apo_angle_deg"]) < 2.0
    assert abs(result["deep_apo_angle_deg"]) < 2.0


def test_matlab_hough_aponeurosis_detector_single_kind_view():
    image = _synthetic_two_aponeurosis_frame()
    detector = MatlabHoughAponeurosisDetector(
        config=MatlabHoughAponeurosisConfig(
            apox_1b=np.array([20, 40, 60, 80, 100], dtype=float),
            super_cut=(0.0, 0.5),
            deep_cut=(0.5, 1.0),
            threshold_block_size=15,
        )
    )

    super_result = detector.detect(image, kind="superficial")
    pair_result = detector.detect_pair(image)

    assert super_result["kind"] == "superficial"
    assert super_result["line_local"].shape == (4,)
    np.testing.assert_allclose(super_result["line_local"], pair_result["super_line_0b"])
