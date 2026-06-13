"""
Geometry module for ultrasound fascicle tracking.

This module provides core geometric calculations for:
  - Angle computation (fascicle angle, pennation, aponeurosis angle)
  - Line properties (length, endpoints, intersection)
  - Point transformations (line projection, perpendicular distance)
  - Filtering and batch operations

Ported from MATLAB UltraTimTrack by Tim van der Zee.
Convention: angle = atan2(-dy, dx)  →  matches MATLAB atan2d(-dy, dx)
"""

import numpy as np
from typing import Tuple, Optional, Dict


# ============================================================================
# LINE ANGLES AND ORIENTATION
# ============================================================================

def line_angle(x1: float, y1: float, x2: float, y2: float,
               degrees: bool = True) -> float:
    """
    Compute the angle of a line segment.

    Uses the MATLAB UltraTimTrack convention: atan2(-dy, dx)
      - Horizontal right  →   0°
      - Vertical downward →  90°
      - Diagonal up-right → -45° (negative because image y-axis is flipped)

    Parameters
    ----------
    x1, y1 : float  —  Start point
    x2, y2 : float  —  End point
    degrees : bool  —  Return degrees (default) or radians

    Returns
    -------
    angle : float
    """
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = np.arctan2(-dy, dx)   # MATLAB: atan2d(-dy, dx)
    return np.degrees(angle_rad) if degrees else angle_rad


def line_angle_from_array(line: np.ndarray, degrees: bool = True) -> float:
    """
    Compute the angle of a line given as [x1, y1, x2, y2].

    Parameters
    ----------
    line : np.ndarray  shape (4,)
    degrees : bool

    Returns
    -------
    angle : float
    """
    x1, y1, x2, y2 = line
    return line_angle(x1, y1, x2, y2, degrees=degrees)


def line_angles_batch(lines: np.ndarray, degrees: bool = True) -> np.ndarray:
    """
    Compute angles for multiple line segments — vectorized (no Python loop).

    Parameters
    ----------
    lines : np.ndarray  shape (N, 4)  — each row [x1, y1, x2, y2]
    degrees : bool

    Returns
    -------
    angles : np.ndarray  shape (N,)
    """
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles_rad = np.arctan2(-dy, dx)
    return np.degrees(angles_rad) if degrees else angles_rad


def normalize_angle(angle: float, degrees: bool = True) -> float:
    """
    Normalize an angle to [-90°, 90°) — useful for pennation angles.

    Parameters
    ----------
    angle : float
    degrees : bool

    Returns
    -------
    normalized : float
    """
    period = 180.0 if degrees else np.pi
    angle = angle % period
    if angle >= period / 2:
        angle -= period
    return angle


# ============================================================================
# LINE LENGTH AND ENDPOINTS
# ============================================================================

def line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Euclidean length of a line segment.

    Parameters
    ----------
    x1, y1, x2, y2 : float

    Returns
    -------
    length : float
    """
    return np.hypot(x2 - x1, y2 - y1)


def line_length_from_array(line: np.ndarray) -> float:
    """
    Length of a line given as [x1, y1, x2, y2].

    Parameters
    ----------
    line : np.ndarray  shape (4,)

    Returns
    -------
    length : float
    """
    x1, y1, x2, y2 = line
    return line_length(x1, y1, x2, y2)


def line_lengths_batch(lines: np.ndarray) -> np.ndarray:
    """
    Euclidean lengths for multiple line segments — vectorized.

    Parameters
    ----------
    lines : np.ndarray  shape (N, 4)

    Returns
    -------
    lengths : np.ndarray  shape (N,)
    """
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    return np.hypot(dx, dy)


def line_endpoints_from_array(line: np.ndarray) \
        -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Extract start and end points from [x1, y1, x2, y2].

    Returns
    -------
    p1, p2 : (float, float), (float, float)
    """
    return (float(line[0]), float(line[1])), (float(line[2]), float(line[3]))


# ============================================================================
# PENNATION ANGLE  (fascicle vs aponeurosis)
# ============================================================================

def pennation_angle(fascicle_angle: float, aponeurosis_angle: float,
                    degrees: bool = True) -> float:
    """
    Pennation angle = fascicle_angle − aponeurosis_angle,
    normalized to [−90°, 90°].

    This mirrors the MATLAB formula:
        handles.Region(i).fas_pen = fas_ang - gamma
    where gamma is the deep aponeurosis angle.

    Parameters
    ----------
    fascicle_angle   : float  (from line_angle / line_angle_from_array)
    aponeurosis_angle: float
    degrees          : bool

    Returns
    -------
    pennation : float
    """
    diff = fascicle_angle - aponeurosis_angle
    return normalize_angle(diff, degrees=degrees)


def pennation_angle_from_lines(fascicle: np.ndarray,
                                aponeurosis: np.ndarray,
                                degrees: bool = True) -> float:
    """
    Compute pennation angle directly from two line segments [x1,y1,x2,y2].

    Parameters
    ----------
    fascicle    : np.ndarray  shape (4,)
    aponeurosis : np.ndarray  shape (4,)
    degrees     : bool

    Returns
    -------
    pennation : float
    """
    fas_ang = line_angle_from_array(fascicle, degrees=degrees)
    apo_ang = line_angle_from_array(aponeurosis, degrees=degrees)
    return pennation_angle(fas_ang, apo_ang, degrees=degrees)


# ============================================================================
# POINT-TO-LINE GEOMETRY
# ============================================================================

def point_to_line_distance(px: float, py: float,
                           x1: float, y1: float,
                           x2: float, y2: float) -> float:
    """
    Perpendicular distance from point (px, py) to the infinite line
    passing through (x1,y1) and (x2,y2).

    Parameters
    ----------
    px, py         : float  — query point
    x1, y1, x2, y2 : float  — two points defining the line

    Returns
    -------
    distance : float  (non-negative)
    """
    dx = x2 - x1
    dy = y2 - y1
    denom = np.hypot(dx, dy)
    if denom < 1e-10:
        return np.hypot(px - x1, py - y1)
    numerator = abs(dy * px - dx * py + x2 * y1 - y2 * x1)
    return numerator / denom


def project_point_on_line(px: float, py: float,
                          x1: float, y1: float,
                          x2: float, y2: float) -> Tuple[float, float]:
    """
    Orthogonal projection of (px, py) onto the line through (x1,y1)-(x2,y2).

    Parameters
    ----------
    px, py         : float  — point to project
    x1, y1, x2, y2 : float  — line endpoints

    Returns
    -------
    proj_x, proj_y : float
    """
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return float(x1), float(y1)
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    return float(x1 + t * dx), float(y1 + t * dy)


# ============================================================================
# LINE INTERSECTION
# ============================================================================

def line_intersection(x1a: float, y1a: float, x2a: float, y2a: float,
                      x1b: float, y1b: float, x2b: float, y2b: float) \
        -> Optional[Tuple[float, float]]:
    """
    Intersection point of two infinite lines, or None if parallel.

    Parameters
    ----------
    x1a,y1a,x2a,y2a : float  — first line
    x1b,y1b,x2b,y2b : float  — second line

    Returns
    -------
    (x, y) or None
    """
    dxa = x2a - x1a;  dya = y2a - y1a
    dxb = x2b - x1b;  dyb = y2b - y1b
    det = dxa * dyb - dya * dxb
    if abs(det) < 1e-10:
        return None   # parallel
    t = ((x1b - x1a) * dyb - (y1b - y1a) * dxb) / det
    return float(x1a + t * dxa), float(y1a + t * dya)


# ============================================================================
# AFFINE TRANSFORMATIONS
# ============================================================================

def rotate_point(px: float, py: float, angle: float, degrees: bool = True,
                 center: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """
    Rotate point (px, py) by `angle` around `center`.

    Parameters
    ----------
    px, py   : float
    angle    : float
    degrees  : bool    — True = degrees, False = radians
    center   : (cx, cy)

    Returns
    -------
    (rotated_x, rotated_y) : float
    """
    a = np.radians(angle) if degrees else angle
    cx, cy = center
    x, y = px - cx, py - cy
    rx = x * np.cos(a) - y * np.sin(a) + cx
    ry = x * np.sin(a) + y * np.cos(a) + cy
    return float(rx), float(ry)


def translate_line(line: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Translate a line [x1,y1,x2,y2] by (dx, dy).

    Parameters
    ----------
    line     : np.ndarray  shape (4,)
    dx, dy   : float

    Returns
    -------
    translated : np.ndarray  shape (4,)
    """
    return line + np.array([dx, dy, dx, dy], dtype=line.dtype)


def translate_lines_batch(lines: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Translate multiple lines by (dx, dy) — vectorized.

    Parameters
    ----------
    lines  : np.ndarray  shape (N, 4)
    dx, dy : float

    Returns
    -------
    translated : np.ndarray  shape (N, 4)
    """
    return lines + np.array([dx, dy, dx, dy], dtype=lines.dtype)


# ============================================================================
# FASCICLE GEOMETRY CLASS
# ============================================================================

class FascicleGeometry:
    """
    Complete geometry of a single fascicle with its adjacent aponeuroses.

    Encapsulates the most common MATLAB UltraTimTrack outputs:
      - fas_ang  (fascicle angle)
      - fas_pen  (pennation angle vs deep aponeurosis)
      - fascicle length
      - attachment points on both aponeuroses

    Usage
    -----
    >>> geom = FascicleGeometry(fascicle, deep_apo=deep_apo)
    >>> print(geom.pennation_angle_val)   # e.g. 15.3
    >>> print(geom.fascicle_length)       # e.g. 112.7
    """

    def __init__(self, fascicle: np.ndarray,
                 superficial_apo: Optional[np.ndarray] = None,
                 deep_apo: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        fascicle        : np.ndarray  shape (4,) = [x1, y1, x2, y2]
        superficial_apo : np.ndarray  shape (4,), optional
        deep_apo        : np.ndarray  shape (4,), optional
        """
        self.fascicle       = np.asarray(fascicle, dtype=np.float32)
        self.superficial_apo = (np.asarray(superficial_apo, dtype=np.float32)
                                if superficial_apo is not None else None)
        self.deep_apo        = (np.asarray(deep_apo, dtype=np.float32)
                                if deep_apo is not None else None)
        self._compute()

    # ------------------------------------------------------------------
    def _compute(self):
        """Compute and cache all geometric properties."""
        self.fascicle_angle  = line_angle_from_array(self.fascicle)
        self.fascicle_length = line_length_from_array(self.fascicle)

        if self.deep_apo is not None:
            self.deep_apo_angle    = line_angle_from_array(self.deep_apo)
            self.pennation_angle_val = pennation_angle(
                self.fascicle_angle, self.deep_apo_angle)
        else:
            self.deep_apo_angle      = None
            self.pennation_angle_val = None

        if self.superficial_apo is not None:
            self.superficial_apo_angle = line_angle_from_array(self.superficial_apo)
        else:
            self.superficial_apo_angle = None

    # ------------------------------------------------------------------
    def get_fascicle_endpoints(self) \
            -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return (p_start, p_end) of the fascicle."""
        return line_endpoints_from_array(self.fascicle)

    def get_superficial_attachment(self) -> Optional[Tuple[float, float]]:
        """
        Project the fascicle's start point orthogonally onto the
        superficial aponeurosis.  Returns None if apo not provided.
        """
        if self.superficial_apo is None:
            return None
        return project_point_on_line(
            self.fascicle[0], self.fascicle[1],
            *self.superficial_apo
        )

    def get_deep_attachment(self) -> Optional[Tuple[float, float]]:
        """
        Project the fascicle's end point orthogonally onto the
        deep aponeurosis.  Returns None if apo not provided.
        """
        if self.deep_apo is None:
            return None
        return project_point_on_line(
            self.fascicle[2], self.fascicle[3],
            *self.deep_apo
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        pen = (f"{self.pennation_angle_val:.2f}°"
               if self.pennation_angle_val is not None else "N/A")
        return (f"FascicleGeometry("
                f"angle={self.fascicle_angle:.2f}°, "
                f"length={self.fascicle_length:.1f}px, "
                f"pennation={pen})")


# ============================================================================
# BATCH GEOMETRIC FEATURES
# ============================================================================

def compute_line_features(lines: np.ndarray) -> dict:
    """
    Compute angles, lengths, and midpoints for a set of lines in one call.

    Parameters
    ----------
    lines : np.ndarray  shape (N, 4)

    Returns
    -------
    features : dict with keys
        'angles'    : (N,)   degrees
        'lengths'   : (N,)   pixels
        'midpoints' : (N, 2) (x, y)
    """
    return {
        'angles':    line_angles_batch(lines, degrees=True),
        'lengths':   line_lengths_batch(lines),
        'midpoints': (lines[:, :2] + lines[:, 2:]) / 2.0,
    }


def filter_lines_by_angle(lines: np.ndarray, angles: np.ndarray,
                          angle_min: float, angle_max: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only lines whose angle is in [angle_min, angle_max] degrees.

    Parameters
    ----------
    lines      : np.ndarray  shape (N, 4)
    angles     : np.ndarray  shape (N,)  — precomputed with line_angles_batch
    angle_min  : float
    angle_max  : float

    Returns
    -------
    filtered_lines  : np.ndarray
    filtered_angles : np.ndarray
    """
    mask = (angles >= angle_min) & (angles <= angle_max)
    return lines[mask], angles[mask]


def filter_lines_by_length(lines: np.ndarray, lengths: np.ndarray,
                           length_min: float,
                           length_max: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only lines whose length is in [length_min, length_max].

    Parameters
    ----------
    lines      : np.ndarray  shape (N, 4)
    lengths    : np.ndarray  shape (N,)
    length_min : float
    length_max : float or None  — no upper bound if None

    Returns
    -------
    filtered_lines   : np.ndarray
    filtered_lengths : np.ndarray
    """
    mask = lengths >= length_min
    if length_max is not None:
        mask &= (lengths <= length_max)
    return lines[mask], lengths[mask]


# ============================================================================
# VALIDATION & UTILITIES
# ============================================================================

def is_valid_line(line: np.ndarray, min_length: float = 0.1) -> bool:
    """
    Return True if the line segment is non-degenerate (length ≥ min_length).

    Parameters
    ----------
    line       : np.ndarray  shape (4,)
    min_length : float

    Returns
    -------
    bool
    """
    return line_length_from_array(line) >= min_length


def clip_line_to_image(line: np.ndarray,
                       img_width: int, img_height: int) -> np.ndarray:
    """
    Clip all four coordinates of a line to image boundaries.

    Parameters
    ----------
    line       : np.ndarray  shape (4,)
    img_width  : int
    img_height : int

    Returns
    -------
    clipped : np.ndarray  shape (4,)
    """
    out = line.copy()
    out[0] = np.clip(out[0], 0, img_width  - 1)
    out[1] = np.clip(out[1], 0, img_height - 1)
    out[2] = np.clip(out[2], 0, img_width  - 1)
    out[3] = np.clip(out[3], 0, img_height - 1)
    return out

# ============================================================================
# FASCICLE / APONEUROSIS GEOMETRY
# ============================================================================

def point_inside_image(point: np.ndarray,
                       frame_shape: Tuple[int, int],
                       margin: int = 50) -> bool:
    """
    Check whether a point is inside image boundaries, allowing a margin.

    Parameters
    ----------
    point : np.ndarray shape (2,)
        Point as [x, y].
    frame_shape : tuple
        Image shape as (height, width) or (height, width, channels).
    margin : int
        Tolerance in pixels outside the image boundaries.

    Returns
    -------
    bool
    """
    if point is None:
        return False

    point = np.asarray(point, dtype=np.float32)

    if point.shape[0] < 2 or not np.all(np.isfinite(point[:2])):
        return False

    h, w = frame_shape[:2]
    x, y = point[:2]

    return (
        -margin <= x <= w - 1 + margin
        and -margin <= y <= h - 1 + margin
    )


def fascicle_segment_between_aponeuroses(fascicle_line: np.ndarray,
                                         superficial_apo_line: np.ndarray,
                                         deep_apo_line: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the true fascicle segment between the superficial and deep aponeuroses.

    This is the key UltraTimTrack-style geometry:
        fascicle line ∩ superficial aponeurosis
        fascicle line ∩ deep aponeurosis

    Parameters
    ----------
    fascicle_line : np.ndarray shape (4,)
        [x1, y1, x2, y2]
    superficial_apo_line : np.ndarray shape (4,)
        [x1, y1, x2, y2]
    deep_apo_line : np.ndarray shape (4,)
        [x1, y1, x2, y2]

    Returns
    -------
    segment : np.ndarray shape (4,) or None
        [x_sup, y_sup, x_deep, y_deep]
    """
    if fascicle_line is None or superficial_apo_line is None or deep_apo_line is None:
        return None

    p_sup = line_intersection(*fascicle_line, *superficial_apo_line)
    p_deep = line_intersection(*fascicle_line, *deep_apo_line)

    if p_sup is None or p_deep is None:
        return None

    p_sup = np.asarray(p_sup, dtype=np.float32)
    p_deep = np.asarray(p_deep, dtype=np.float32)

    return np.array([
        p_sup[0], p_sup[1],
        p_deep[0], p_deep[1],
    ], dtype=np.float32)


def fascicle_length_between_aponeuroses(fascicle_line: np.ndarray,
                                        superficial_apo_line: np.ndarray,
                                        deep_apo_line: np.ndarray) -> Optional[float]:
    """
    Compute true fascicle length between the superficial and deep aponeuroses.

    Returns
    -------
    length : float or None
    """
    segment = fascicle_segment_between_aponeuroses(
        fascicle_line,
        superficial_apo_line,
        deep_apo_line,
    )

    if segment is None:
        return None

    return float(line_length_from_array(segment))


def compute_fascicle_geometry(superficial_apo_line: np.ndarray,
                              deep_apo_line: np.ndarray,
                              fascicle_line: np.ndarray) -> Dict:
    """
    Compute final fascicle geometry from:
      - superficial aponeurosis line
      - deep aponeurosis line
      - fascicle line

    This function should be the main geometry output used by notebook 11.

    Parameters
    ----------
    superficial_apo_line : np.ndarray shape (4,)
    deep_apo_line : np.ndarray shape (4,)
    fascicle_line : np.ndarray shape (4,)

    Returns
    -------
    features : dict
        {
            "sup_attachment": np.ndarray shape (2,),
            "deep_attachment": np.ndarray shape (2,),
            "fascicle_segment_between_apos": np.ndarray shape (4,),
            "fascicle_length_px": float,
            "fascicle_angle_deg": float,
            "deep_apo_angle_deg": float,
            "pennation_angle_deg": float,
        }
    """
    if superficial_apo_line is None:
        raise ValueError("superficial_apo_line is None.")

    if deep_apo_line is None:
        raise ValueError("deep_apo_line is None.")

    if fascicle_line is None:
        raise ValueError("fascicle_line is None.")

    p_sup = line_intersection(*fascicle_line, *superficial_apo_line)
    p_deep = line_intersection(*fascicle_line, *deep_apo_line)

    if p_sup is None:
        raise ValueError("Fascicle line does not intersect superficial aponeurosis.")

    if p_deep is None:
        raise ValueError("Fascicle line does not intersect deep aponeurosis.")

    sup_attachment = np.asarray(p_sup, dtype=np.float32)
    deep_attachment = np.asarray(p_deep, dtype=np.float32)

    fascicle_segment = np.array([
        sup_attachment[0],
        sup_attachment[1],
        deep_attachment[0],
        deep_attachment[1],
    ], dtype=np.float32)

    fascicle_length_px = line_length_from_array(fascicle_segment)

    fascicle_angle_deg = normalize_angle(
        line_angle_from_array(fascicle_line),
        degrees=True,
    )

    deep_apo_angle_deg = normalize_angle(
        line_angle_from_array(deep_apo_line),
        degrees=True,
    )

    pennation_angle_deg = pennation_angle(
        fascicle_angle_deg,
        deep_apo_angle_deg,
        degrees=True,
    )

    return {
        "sup_attachment": sup_attachment,
        "deep_attachment": deep_attachment,
        "fascicle_segment_between_apos": fascicle_segment,
        "fascicle_length_px": float(fascicle_length_px),
        "fascicle_angle_deg": float(fascicle_angle_deg),
        "deep_apo_angle_deg": float(deep_apo_angle_deg),
        "pennation_angle_deg": float(pennation_angle_deg),
    }


def pick_best_fascicle_line(lines: Optional[np.ndarray],
                            lengths: Optional[np.ndarray] = None,
                            superficial_apo_line: Optional[np.ndarray] = None,
                            deep_apo_line: Optional[np.ndarray] = None,
                            frame_shape: Optional[Tuple[int, int]] = None,
                            margin: int = 50) -> Optional[np.ndarray]:
    """
    Pick the best fascicle line from candidate Hough/Frangi lines.

    Priority:
      1. If aponeuroses are provided, choose a line that intersects both
         aponeuroses and maximizes the segment length between them.
      2. Otherwise, choose the longest candidate line.

    Parameters
    ----------
    lines : np.ndarray or None
        Candidate lines, shape (N, 4), each [x1, y1, x2, y2].
    lengths : np.ndarray or None
        Candidate line lengths.
    superficial_apo_line : np.ndarray or None
        Superficial aponeurosis line [x1, y1, x2, y2].
    deep_apo_line : np.ndarray or None
        Deep aponeurosis line [x1, y1, x2, y2].
    frame_shape : tuple or None
        Full image shape as (height, width).
    margin : int
        Allowed margin when validating intersection points.

    Returns
    -------
    best_line : np.ndarray shape (4,) or None
    """
    if lines is None or len(lines) == 0:
        return None

    lines = np.asarray(lines, dtype=np.float32)

    if lengths is None:
        lengths = line_lengths_batch(lines)
    else:
        lengths = np.asarray(lengths, dtype=np.float32)

    if superficial_apo_line is not None and deep_apo_line is not None:
        best_line = None
        best_score = -np.inf

        for i, line in enumerate(lines):
            segment = fascicle_segment_between_aponeuroses(
                line,
                superficial_apo_line,
                deep_apo_line,
            )

            if segment is None:
                continue

            p_sup = segment[:2]
            p_deep = segment[2:]

            if frame_shape is not None:
                if not point_inside_image(p_sup, frame_shape, margin=margin):
                    continue
                if not point_inside_image(p_deep, frame_shape, margin=margin):
                    continue

            true_length = line_length_from_array(segment)

            if not np.isfinite(true_length) or true_length <= 0:
                continue

            # Main criterion: true length between aponeuroses.
            # Small bonus: detected candidate segment length.
            score = true_length + 0.05 * lengths[i]

            if score > best_score:
                best_score = score
                best_line = line

        if best_line is not None:
            return best_line.astype(np.float32)

    # Fallback: longest detected line.
    idx = int(np.argmax(lengths))
    return lines[idx].astype(np.float32)

# ============================================================================
# QUICK SELF-TEST
# ============================================================================
if __name__ == "__main__":
    print("Testing geometry.py …")

    assert np.isclose(line_angle(0, 0, 1,  0), 0),   "horizontal should be 0°"
    assert np.isclose(line_angle(0, 0, 1, -1), 45),  "up-right should be 45°"
    assert np.isclose(line_length(0, 0, 3, 4), 5),   "3-4-5 triangle"

    pen = pennation_angle_from_lines(
        np.array([0, 0, 10, 5], dtype=np.float32),
        np.array([0, 0, 10, 0], dtype=np.float32)
    )
    assert -90 <= pen <= 90, "pennation must be in [-90, 90]"

    dist = point_to_line_distance(1, 1, 0, 0, 2, 0)
    assert np.isclose(dist, 1.0), "distance point to horizontal line"

    g = FascicleGeometry(
        np.array([50, 50, 150, 100], dtype=np.float32),
        deep_apo=np.array([0, 110, 200, 115], dtype=np.float32)
    )
    print(g)

    print("✓ All tests passed!")

