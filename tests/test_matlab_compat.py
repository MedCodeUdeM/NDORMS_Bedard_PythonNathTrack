import numpy as np

from ultrasound_tracker.matlab_compat import (
    extract_fascicle_state_arrays,
    matlab_fascicle_segments,
    object_series_to_2d,
)


def test_object_series_to_2d_pads_short_rows_with_nan():
    values = np.asarray(
        [
            np.array([1.0, 2.0]),
            np.array([3.0]),
        ],
        dtype=object,
    )

    out = object_series_to_2d(values, 2)

    np.testing.assert_allclose(out[0], [1.0, 2.0])
    assert np.isnan(out[1, 1])
    np.testing.assert_allclose(out[1, 0], 3.0)


def test_matlab_fascicle_segments_reorder_deep_then_super_to_python_contract():
    x_values = np.asarray([np.array([-10.0, 50.0])], dtype=object)
    y_values = np.asarray([np.array([100.0, 20.0])], dtype=object)

    out = matlab_fascicle_segments(x_values, y_values)

    np.testing.assert_allclose(out, [[50.0, 20.0, -10.0, 100.0]])


def test_extract_fascicle_state_arrays_builds_dense_saved_state_views():
    mat = {
        "Fdat": {
            "Region": {
                "Fascicle": {
                    "fas_x_original": np.asarray([np.array([-10.0, 50.0])], dtype=object),
                    "fas_y_original": np.asarray([np.array([100.0, 20.0])], dtype=object),
                    "fas_x": np.asarray([np.array([-11.0, 51.0])], dtype=object),
                    "fas_y": np.asarray([np.array([101.0, 21.0])], dtype=object),
                    "fas_x_end": np.asarray([np.array([-12.0, 52.0])], dtype=object),
                    "fas_y_end": np.asarray([np.array([102.0, 22.0])], dtype=object),
                    "X_plus": np.asarray([np.array([51.0, 30.0])], dtype=object),
                    "X_minus": np.asarray([np.array([50.0, 29.0])], dtype=object),
                    "fas_p": np.asarray([np.array([0.5, 1.5])], dtype=object),
                    "fas_p_minus": np.asarray([np.array([0.7, 1.7])], dtype=object),
                    "alpha": np.asarray([30.0]),
                    "K": np.asarray([0.2]),
                    "A": np.asarray([1.0]),
                }
            }
        }
    }

    out = extract_fascicle_state_arrays(mat)

    np.testing.assert_allclose(out["fas_x_original_segment"], [[50.0, 20.0, -10.0, 100.0]])
    np.testing.assert_allclose(out["fas_x_segment"], [[51.0, 21.0, -11.0, 101.0]])
    np.testing.assert_allclose(out["fas_x_end_segment"], [[52.0, 22.0, -12.0, 102.0]])
    np.testing.assert_allclose(out["X_plus"], [[51.0, 30.0]])
    np.testing.assert_allclose(out["X_minus"], [[50.0, 29.0]])
    np.testing.assert_allclose(out["fas_p"], [[0.5, 1.5]])
    np.testing.assert_allclose(out["fas_p_minus"], [[0.7, 1.7]])
    np.testing.assert_allclose(out["alpha"], [30.0])
    np.testing.assert_allclose(out["K"], [0.2])
    np.testing.assert_allclose(out["A"], [1.0])
