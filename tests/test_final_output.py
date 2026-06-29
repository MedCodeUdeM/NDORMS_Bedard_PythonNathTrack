import numpy as np

from ultrasound_tracker.final_output import (
    aponeurosis_thickness_px,
    final_outputs_from_components,
    final_outputs_from_lines,
    image_depth_to_mm_per_pixel,
)


def test_image_depth_to_mm_per_pixel():
    assert np.isclose(image_depth_to_mm_per_pixel(50.7, 562), 50.7 / 562)


def test_final_outputs_from_components_match_formula():
    out = final_outputs_from_components(
        alpha_deg=np.array([24.0]),
        aponeurosis_angle_deg=np.array([-1.4156952]),
        thickness_px=np.array([274.03333]),
    )

    expected_pen = 25.4156952
    expected_fl = 274.03333 / np.sin(np.deg2rad(expected_pen))

    assert np.isclose(out["ANG_deg"][0], 24.0)
    assert np.isclose(out["PEN_deg"][0], expected_pen)
    assert np.isclose(out["FL_px"][0], expected_fl)


def test_final_outputs_from_components_optional_mm_conversion():
    out = final_outputs_from_components(
        alpha_deg=30.0,
        aponeurosis_angle_deg=0.0,
        thickness_px=40.0,
        mm_per_pixel=0.1,
    )

    assert np.isclose(out["FL_px"], 80.0)
    assert np.isclose(out["FL_mm"], 8.0)


def test_final_outputs_from_components_zero_pennation_returns_nan():
    out = final_outputs_from_components(
        alpha_deg=0.0,
        aponeurosis_angle_deg=0.0,
        thickness_px=40.0,
    )

    assert np.isnan(out["FL_px"])


def test_aponeurosis_thickness_px_uses_superficial_angle_correction():
    superficial = np.array([0.0, 10.0, 100.0, 10.0])
    deep = np.array([0.0, 50.0, 100.0, 50.0])

    thickness = aponeurosis_thickness_px(superficial, deep, x_eval=20.0)

    assert np.isclose(thickness[0], 40.0)


def test_final_outputs_from_lines_uses_superficial_reference_by_default():
    superficial = np.array([0.0, 10.0, 100.0, 10.0])
    deep = np.array([0.0, 50.0, 100.0, 50.0])

    out = final_outputs_from_lines(
        alpha_deg=30.0,
        superficial_apo_lines=superficial,
        deep_apo_lines=deep,
        x_eval=20.0,
        mm_per_pixel=0.1,
    )

    assert np.isclose(out["super_apo_angle_deg"][0], 0.0)
    assert np.isclose(out["deep_apo_angle_deg"][0], 0.0)
    assert np.isclose(out["muscle_thickness_px"][0], 40.0)
    assert np.isclose(out["PEN_deg"][0], 30.0)
    assert np.isclose(out["FL_px"][0], 80.0)
    assert np.isclose(out["FL_mm"][0], 8.0)


def test_final_outputs_from_lines_can_use_deep_reference():
    superficial = np.array([0.0, 10.0, 100.0, 10.0])
    deep = np.array([0.0, 50.0, 100.0, 50.0])

    out = final_outputs_from_lines(
        alpha_deg=30.0,
        superficial_apo_lines=superficial,
        deep_apo_lines=deep,
        pennation_reference="deep",
    )

    assert np.isclose(out["PEN_deg"][0], 30.0)
