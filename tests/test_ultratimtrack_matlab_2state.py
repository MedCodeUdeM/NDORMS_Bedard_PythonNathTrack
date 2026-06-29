import numpy as np

from ultrasound_tracker.ultratimtrack_matlab_2state import (
    MatlabTwoStateKalmanConfig,
    matlab_scalar_kalman_update,
    reconstruct_fascicle_from_state,
    run_matlab_2state_kalman,
)


def test_scalar_update_matches_matlab_equation():
    x_plus, p_plus, p_minus, gain = matlab_scalar_kalman_update(
        x_minus=10.0,
        p_prev=2.0,
        q_value=1.0,
        measurement=4.0,
        measurement_variance=3.0,
    )

    assert gain == 0.5
    assert p_minus == 3.0
    assert x_plus == 7.0
    assert p_plus == 1.5


def test_reconstruct_fascicle_from_state_uses_fixed_superficial_y():
    superficial = np.array([1.0, 10.0, 101.0, 10.0])
    deep = np.array([1.0, 60.0, 101.0, 60.0])

    segment, end_segment = reconstruct_fascicle_from_state(
        x_sup=80.0,
        alpha_deg=45.0,
        superficial_apo_line=superficial,
        deep_apo_line=deep,
        fixed_superficial_y=10.0,
    )

    np.testing.assert_allclose(segment, [80.0, 10.0, 30.0, 60.0], atol=1e-8)
    np.testing.assert_allclose(end_segment, [80.0, 10.0, 30.0, 60.0], atol=1e-8)


def test_two_state_filter_keeps_constant_sequence_stable():
    klt = np.array(
        [
            [80.0, 10.0, 30.0, 60.0],
            [80.0, 10.0, 30.0, 60.0],
            [80.0, 10.0, 30.0, 60.0],
        ],
        dtype=float,
    )
    superficial = np.tile(np.array([[1.0, 10.0, 101.0, 10.0]]), (3, 1))
    deep = np.tile(np.array([[1.0, 60.0, 101.0, 60.0]]), (3, 1))
    alpha = np.array([45.0, 45.0, 45.0])

    result = run_matlab_2state_kalman(
        klt,
        alpha,
        superficial,
        deep,
        config=MatlabTwoStateKalmanConfig(
            q_parameter=0.01,
            x_measurement_variance=100.0,
            alpha_measurement_variance=3.0,
            run_smoother=True,
        ),
        mm_per_pixel=0.2,
    )

    np.testing.assert_allclose(result["X_plus"][:, 0], 80.0, atol=1e-8)
    np.testing.assert_allclose(result["X_plus"][:, 1], 45.0, atol=1e-8)
    np.testing.assert_allclose(result["fascicle_end_segments"], klt, atol=1e-8)
    np.testing.assert_allclose(result["FL_mm"], np.sqrt(50.0**2 + 50.0**2) * 0.2)
