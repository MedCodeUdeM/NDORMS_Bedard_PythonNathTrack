import numpy as np

from ultrasound_tracker.ultratimtrack_aponeurosis import (
    aponeurosis_state_from_lines,
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
