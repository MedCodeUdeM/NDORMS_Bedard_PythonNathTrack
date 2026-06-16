import numpy as np

from ultrasound_tracker.ultratimtrack_kalman import (
    segment_from_state,
    state_from_segment,
)


def test_down_left_segment_round_trips_through_state():
    segment = np.array([100.0, 50.0, 60.0, 80.0], dtype=np.float32)

    state = state_from_segment(segment)
    reconstructed = segment_from_state(state)

    np.testing.assert_allclose(reconstructed, segment, atol=1e-5)
