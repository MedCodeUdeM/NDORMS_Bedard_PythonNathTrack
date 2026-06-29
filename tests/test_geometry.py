"""
Unit tests for geometry module.

Run from project root:
    python -m pytest tests/test_geometry.py -v
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = PROJECT_ROOT / "ultrasound_tracker" / "geometry.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import geometry directly to avoid triggering ultrasound_tracker/__init__.py
spec = importlib.util.spec_from_file_location("geometry", GEOMETRY_PATH)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


# ─────────────────────────────────────────────────────────────────────────────
class TestLineAngle:

    def test_horizontal_line_is_zero(self):
        assert np.isclose(geometry.line_angle(0, 0, 1, 0), 0)

    def test_vertical_downward_is_negative_90(self):
        # In image coords y grows downward; atan2(-dy, dx) makes down negative.
        assert np.isclose(geometry.line_angle(0, 0, 0, 1), -90, atol=1e-5)

    def test_vertical_upward_is_positive_90(self):
        assert np.isclose(geometry.line_angle(0, 0, 0, -1), 90, atol=1e-5)

    def test_diagonal_up_right_is_45(self):
        # dy = -1 (going up in image), so -(-1)/1 → atan2(1,1) = 45°
        assert np.isclose(geometry.line_angle(0, 0, 1, -1), 45)

    def test_from_array(self):
        line = np.array([0, 0, 1, -1], dtype=np.float32)
        assert np.isclose(geometry.line_angle_from_array(line), 45)

    def test_batch_shape(self):
        lines = np.array([[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        angles = geometry.line_angles_batch(lines)
        assert angles.shape == (2,)

    def test_batch_values(self):
        lines = np.array([[0,0,1,0],[0,0,1,-1]], dtype=np.float32)
        angles = geometry.line_angles_batch(lines)
        assert np.isclose(angles[0], 0)
        assert np.isclose(angles[1], 45)

    def test_radians_flag(self):
        angle_rad = geometry.line_angle(0, 0, 1, -1, degrees=False)
        assert np.isclose(angle_rad, np.pi / 4)


# ─────────────────────────────────────────────────────────────────────────────
class TestLineLength:

    def test_3_4_5_triangle(self):
        assert np.isclose(geometry.line_length(0, 0, 3, 4), 5.0)

    def test_horizontal_unit(self):
        assert np.isclose(geometry.line_length(0, 0, 1, 0), 1.0)

    def test_from_array(self):
        line = np.array([0, 0, 3, 4], dtype=np.float32)
        assert np.isclose(geometry.line_length_from_array(line), 5.0)

    def test_batch(self):
        lines = np.array([[0,0,3,4],[0,0,1,0]], dtype=np.float32)
        lengths = geometry.line_lengths_batch(lines)
        assert np.isclose(lengths[0], 5.0)
        assert np.isclose(lengths[1], 1.0)

    def test_endpoints(self):
        line = np.array([1, 2, 3, 4], dtype=np.float32)
        p1, p2 = geometry.line_endpoints_from_array(line)
        assert p1 == (1.0, 2.0)
        assert p2 == (3.0, 4.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestNormalizeAngle:

    def test_positive_under_90(self):
        assert np.isclose(geometry.normalize_angle(45.0), 45.0)

    def test_negative_under_90(self):
        assert np.isclose(geometry.normalize_angle(-45.0), -45.0)

    def test_180_wraps(self):
        # 180° should map to either -180 or 0 depending on convention
        n = geometry.normalize_angle(180.0)
        assert -90 <= n < 90 or np.isclose(abs(n), 0) or np.isclose(abs(n), 180)

    def test_output_in_range(self):
        for a in np.linspace(-360, 360, 50):
            n = geometry.normalize_angle(a)
            assert -90 <= n < 90 or np.isclose(n, -90)


# ─────────────────────────────────────────────────────────────────────────────
class TestPennationAngle:

    def test_parallel_lines_zero_pennation(self):
        fas = np.array([0, 1, 10, 1], dtype=np.float32)  # horizontal
        apo = np.array([0, 5, 10, 5], dtype=np.float32)  # horizontal
        pen = geometry.pennation_angle_from_lines(fas, apo)
        assert np.isclose(pen, 0, atol=1e-4)

    def test_pennation_in_valid_range(self):
        fas = np.array([50, 100, 100, 130], dtype=np.float32)
        apo = np.array([0, 115, 200, 120], dtype=np.float32)
        pen = geometry.pennation_angle_from_lines(fas, apo)
        assert -90 <= pen <= 90

    def test_pennation_scalar(self):
        pen = geometry.pennation_angle(30.0, 10.0)
        assert np.isclose(pen, 20.0)

    def test_pennation_negative(self):
        pen = geometry.pennation_angle(5.0, 25.0)
        assert np.isclose(pen, -20.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestPointToLine:

    def test_point_above_horizontal_line(self):
        dist = geometry.point_to_line_distance(1, 1, 0, 0, 2, 0)
        assert np.isclose(dist, 1.0)

    def test_point_on_line_is_zero(self):
        dist = geometry.point_to_line_distance(1, 1, 0, 0, 2, 2)
        assert np.isclose(dist, 0.0, atol=1e-6)

    def test_point_to_vertical_line(self):
        dist = geometry.point_to_line_distance(3, 0, 0, 0, 0, 5)
        assert np.isclose(dist, 3.0)

    def test_degenerate_line(self):
        # Both endpoints the same → fallback to direct distance
        dist = geometry.point_to_line_distance(3, 4, 0, 0, 0, 0)
        assert np.isclose(dist, 5.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestProjectPointOnLine:

    def test_project_onto_horizontal(self):
        px, py = geometry.project_point_on_line(3, 7, 0, 0, 10, 0)
        assert np.isclose(px, 3.0)
        assert np.isclose(py, 0.0)

    def test_project_onto_diagonal(self):
        # Project (1,0) onto y=x line
        px, py = geometry.project_point_on_line(1, 0, 0, 0, 1, 1)
        assert np.isclose(px, 0.5)
        assert np.isclose(py, 0.5)

    def test_degenerate_line_returns_start(self):
        px, py = geometry.project_point_on_line(5, 3, 2, 2, 2, 2)
        assert px == 2.0 and py == 2.0


# ─────────────────────────────────────────────────────────────────────────────
class TestLineIntersection:

    def test_perpendicular_lines(self):
        result = geometry.line_intersection(0, 1, 2, 1,   # y = 1
                                            1, 0, 1, 2)   # x = 1
        assert result is not None
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 1.0)

    def test_parallel_lines_return_none(self):
        result = geometry.line_intersection(0, 0, 1, 0,   # y = 0
                                            0, 1, 1, 1)   # y = 1
        assert result is None

    def test_diagonal_intersection(self):
        # y=x and y=-x+2 intersect at (1,1)
        result = geometry.line_intersection(0, 0, 2, 2,
                                            0, 2, 2, 0)
        assert result is not None
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 1.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestFascicleGeometry:

    def setup_method(self):
        self.fas  = np.array([50, 50, 150, 100], dtype=np.float32)
        self.deep = np.array([0, 110, 200, 115], dtype=np.float32)
        self.sup  = np.array([0,  40, 200,  45], dtype=np.float32)

    def test_init_basic(self):
        g = geometry.FascicleGeometry(self.fas)
        assert g.fascicle_length > 0
        assert g.fascicle_angle is not None
        assert g.pennation_angle_val is None  # no apo given

    def test_init_with_deep_apo(self):
        g = geometry.FascicleGeometry(self.fas, deep_apo=self.deep)
        assert g.pennation_angle_val is not None
        assert -90 <= g.pennation_angle_val <= 90

    def test_endpoints(self):
        g = geometry.FascicleGeometry(self.fas)
        p1, p2 = g.get_fascicle_endpoints()
        assert p1 == (50.0, 50.0)
        assert p2 == (150.0, 100.0)

    def test_deep_attachment_is_on_apo(self):
        g = geometry.FascicleGeometry(self.fas, deep_apo=self.deep)
        dx, dy = g.get_deep_attachment()
        # The projected point must lie very close to the apo line
        dist = geometry.point_to_line_distance(
            dx, dy, *self.deep)
        assert dist < 0.5  # sub-pixel

    def test_repr(self):
        g = geometry.FascicleGeometry(self.fas, deep_apo=self.deep)
        r = repr(g)
        assert "angle=" in r
        assert "pennation=" in r

    def test_superficial_attachment(self):
        g = geometry.FascicleGeometry(self.fas, superficial_apo=self.sup)
        pt = g.get_superficial_attachment()
        assert pt is not None
        dist = geometry.point_to_line_distance(pt[0], pt[1], *self.sup)
        assert dist < 0.5


# ─────────────────────────────────────────────────────────────────────────────
class TestBatchOperations:

    def test_compute_line_features_keys(self):
        lines = np.array([[0,0,3,4],[0,0,1,0]], dtype=np.float32)
        f = geometry.compute_line_features(lines)
        assert set(f.keys()) == {'angles', 'lengths', 'midpoints'}

    def test_compute_line_features_shapes(self):
        lines = np.array([[0,0,3,4],[0,0,1,0]], dtype=np.float32)
        f = geometry.compute_line_features(lines)
        assert f['angles'].shape    == (2,)
        assert f['lengths'].shape   == (2,)
        assert f['midpoints'].shape == (2, 2)

    def test_filter_by_angle(self):
        lines = np.array([[0,0,10,0],   # 0°
                          [0,0,10,10],  # -45°
                          [0,0,10,5]],  # ~-26.6°
                         dtype=np.float32)
        angles = geometry.line_angles_batch(lines)
        # Keep roughly vertical fascicles – here we negate angles
        fl, fa = geometry.filter_lines_by_angle(
            lines, np.abs(angles), 20, 50)
        assert len(fl) == 2   # -45° and -26.6° have abs > 20

    def test_filter_by_length(self):
        lines = np.array([[0,0,3,4],[0,0,1,0],[0,0,10,0]], dtype=np.float32)
        lengths = geometry.line_lengths_batch(lines)
        fl, flen = geometry.filter_lines_by_length(lines, lengths, length_min=2)
        assert len(fl) == 2   # lengths 5 and 10 survive

    def test_filter_by_length_max(self):
        lines = np.array([[0,0,3,4],[0,0,1,0],[0,0,10,0]], dtype=np.float32)
        lengths = geometry.line_lengths_batch(lines)
        fl, _ = geometry.filter_lines_by_length(lines, lengths, 2, 8)
        assert len(fl) == 1   # only length=5 is in [2,8]


# ─────────────────────────────────────────────────────────────────────────────
class TestTransformations:

    def test_rotate_90(self):
        x, y = geometry.rotate_point(1, 0, 90, center=(0, 0))
        assert np.isclose(x, 0, atol=1e-6)
        assert np.isclose(y, 1, atol=1e-6)

    def test_rotate_around_center(self):
        x, y = geometry.rotate_point(2, 1, 90, center=(1, 1))
        assert np.isclose(x, 1, atol=1e-6)
        assert np.isclose(y, 2, atol=1e-6)

    def test_translate_line(self):
        line = np.array([0, 0, 1, 1], dtype=np.float32)
        t = geometry.translate_line(line, 3, 5)
        np.testing.assert_array_almost_equal(t, [3, 5, 4, 6])

    def test_translate_lines_batch(self):
        lines = np.array([[0,0,1,1],[2,2,3,3]], dtype=np.float32)
        t = geometry.translate_lines_batch(lines, 1, 2)
        np.testing.assert_array_almost_equal(t[0], [1, 2, 2, 3])
        np.testing.assert_array_almost_equal(t[1], [3, 4, 4, 5])


# ─────────────────────────────────────────────────────────────────────────────
class TestValidation:

    def test_valid_line(self):
        assert geometry.is_valid_line(np.array([0, 0, 10, 10], dtype=np.float32))

    def test_degenerate_line(self):
        assert not geometry.is_valid_line(np.array([1, 1, 1, 1], dtype=np.float32))

    def test_clip_within_bounds(self):
        line = np.array([-5, 5, 105, 95], dtype=np.float32)
        c = geometry.clip_line_to_image(line, 100, 100)
        assert c[0] >= 0 and c[2] < 100
        assert c[1] >= 0 and c[3] < 100

    def test_clip_nochange_inside(self):
        line = np.array([10, 20, 80, 60], dtype=np.float32)
        c = geometry.clip_line_to_image(line, 100, 100)
        np.testing.assert_array_equal(c, line)
