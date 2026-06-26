import numpy as np

from ultrasound_tracker.ultratimtrack_aponeurosis import (
    AponeurosisGatingConfig,
    aponeurosis_state_from_lines,
    gate_aponeurosis_measurement_state,
    lines_from_aponeurosis_state,
)


def test_aponeurosis_state_round_trips_one_based_lines():
    superficial = np.array([1.0, 10.0, 100.0, 12.0])
    deep = np.array([1.0, 50.0, 100.0, 55.0])

    state = aponeurosis_state_from_lines(superficial, deep)
    rebuilt_superficial, rebuilt_deep = lines_from_aponeurosis_state(state, width=100)

    np.testing.assert_allclose(state, [10.0, 12.0, 50.0, 55.0])
    np.testing.assert_allclose(rebuilt_superficial, superficial)
    np.testing.assert_allclose(rebuilt_deep, deep)


def test_lines_from_aponeurosis_state_uses_width_as_one_based_endpoint():
    superficial, deep = lines_from_aponeurosis_state([1.5, 2.5, 9.5, 10.5], width=706)

    np.testing.assert_allclose(superficial, [1.0, 1.5, 706.0, 2.5])
    np.testing.assert_allclose(deep, [1.0, 9.5, 706.0, 10.5])


def test_deep_aponeurosis_gate_rejects_implausible_jump_near_maxangle():
    prior = np.array([10.0, 10.0, 50.0, 50.0])
    previous = prior.copy()
    measurement = np.array([10.0, 10.0, 86.0, 50.0])

    gate = gate_aponeurosis_measurement_state(
        prior,
        measurement,
        previous,
        previous,
        width=100,
        config=AponeurosisGatingConfig(
            enabled=True,
            deep_maxangle_deg=20.0,
            deep_mid_jump_px=6.0,
            mid_innovation_px=10.0,
            angle_jump_deg=2.5,
        ),
    )

    np.testing.assert_allclose(gate["measurement_state"][:2], measurement[:2])
    np.testing.assert_allclose(gate["measurement_state"][2:], prior[2:])
    assert not bool(gate["line_rejected"][0])
    assert bool(gate["line_rejected"][1])
    assert np.all(gate["rejected_endpoints"][2:])
    assert np.all(gate["r_scale"][2:] > 1.0e5)
    assert "near maxangle" in str(gate["reasons"][1])
