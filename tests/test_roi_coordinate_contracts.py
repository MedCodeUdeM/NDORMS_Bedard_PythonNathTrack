import numpy as np

from ultrasound_tracker.roi import (
    extract_roi,
    line_local_to_global,
    lines_local_to_global,
    point_local_to_global,
    points_local_to_global,
)


def test_extract_roi_uses_xywh_image_indexing():
    frame = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    roi = (2, 1, 3, 4)

    cropped = extract_roi(frame, roi)

    np.testing.assert_array_equal(cropped, frame[1:5, 2:5])


def test_line_local_to_global_adds_roi_origin_to_both_endpoints():
    roi = (10, 20, 30, 40)
    local = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)

    global_line = line_local_to_global(local, roi)

    np.testing.assert_allclose(global_line, [11.5, 22.5, 13.5, 24.5])
    np.testing.assert_allclose(local, [1.5, 2.5, 3.5, 4.5])


def test_lines_local_to_global_adds_roi_origin_vectorized():
    roi = (10, 20, 30, 40)
    local = np.array(
        [
            [0.0, 0.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=np.float64,
    )

    global_lines = lines_local_to_global(local, roi)

    np.testing.assert_allclose(
        global_lines,
        [
            [10.0, 20.0, 15.0, 26.0],
            [11.0, 22.0, 13.0, 24.0],
        ],
    )


def test_point_local_to_global_adds_roi_origin():
    roi = (10, 20, 30, 40)

    point = point_local_to_global(np.array([1.5, 2.5]), roi)

    np.testing.assert_allclose(point, [11.5, 22.5])


def test_points_local_to_global_adds_roi_origin_vectorized():
    roi = (10, 20, 30, 40)
    points = points_local_to_global(np.array([[0.0, 0.0], [2.0, 3.0]]), roi)

    np.testing.assert_allclose(points, [[10.0, 20.0], [12.0, 23.0]])
