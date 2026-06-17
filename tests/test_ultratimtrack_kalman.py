import numpy as np

from ultrasound_tracker.ultratimtrack_kalman import (
    run_geometric_ultratimtrack_fusion,
    segment_from_state,
    state_from_segment,
)


def test_down_left_segment_round_trips_through_state():
    segment = np.array([100.0, 50.0, 60.0, 80.0], dtype=np.float32)

    state = state_from_segment(segment)
    reconstructed = segment_from_state(state)

    np.testing.assert_allclose(reconstructed, segment, atol=1e-5)


def test_fusion_uses_final_output_formula_when_aponeuroses_available():
    segment = np.array([[80.0, 10.0, 40.0, 50.0]], dtype=np.float32)
    superficial = np.array([[0.0, 10.0, 100.0, 10.0]], dtype=np.float32)
    deep = np.array([[0.0, 60.0, 100.0, 60.0]], dtype=np.float32)

    result = run_geometric_ultratimtrack_fusion(
        timtrack_segments=segment,
        klt_segments=segment,
        timtrack_deep_apo_lines=deep,
        timtrack_superficial_apo_lines=superficial,
        x_eval=20.0,
        mm_per_pixel=0.2,
    )

    expected_state_length = np.hypot(40.0, 40.0)
    expected_final_length = 50.0 / np.sin(np.deg2rad(45.0))

    assert result["success"][0]
    assert result["used_final_output_formula"][0]
    np.testing.assert_allclose(
        result["state_fascicle_length_px"][0],
        expected_state_length,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result["fascicle_length_px"][0],
        expected_final_length,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result["final_fascicle_length_px"][0],
        expected_final_length,
        atol=1e-5,
    )
    np.testing.assert_allclose(result["ANG_deg"][0], 45.0, atol=1e-5)
    np.testing.assert_allclose(result["PEN_deg"][0], 45.0, atol=1e-5)
    np.testing.assert_allclose(
        result["FL_px"][0],
        expected_final_length,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result["FL_mm"][0],
        expected_final_length * 0.2,
        atol=1e-5,
    )


def test_fusion_keeps_legacy_state_length_without_superficial_aponeurosis():
    segment = np.array([[80.0, 10.0, 40.0, 50.0]], dtype=np.float32)
    deep = np.array([[0.0, 60.0, 100.0, 60.0]], dtype=np.float32)

    result = run_geometric_ultratimtrack_fusion(
        timtrack_segments=segment,
        klt_segments=segment,
        timtrack_deep_apo_lines=deep,
    )

    expected_state_length = np.hypot(40.0, 40.0)

    assert result["success"][0]
    assert not result["used_final_output_formula"][0]
    np.testing.assert_allclose(
        result["fascicle_length_px"][0],
        expected_state_length,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result["state_fascicle_length_px"][0],
        expected_state_length,
        atol=1e-5,
    )
    assert np.isnan(result["super_apo_angle_deg"][0])
