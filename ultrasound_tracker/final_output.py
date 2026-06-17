"""
Final output helpers for UltraTimTrack-style fascicle measurements.

These functions define the clean output path used for MATLAB comparison:

    ANG = fascicle alpha
    PEN = alpha - aponeurosis angle
    FL  = thickness / sin(PEN)

The selected OpenCV/Frangi line segment is useful for visualization, but it is
not the final fascicle-length output in this compatibility path.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry import line_angles_batch


def _normalize_angle_array(angle_deg: np.ndarray) -> np.ndarray:
    """Vectorized normalization to [-90, 90) degrees."""
    angle = np.asarray(angle_deg, dtype=np.float64)
    out = np.mod(angle, 180.0)
    return np.where(out >= 90.0, out - 180.0, out)


def image_depth_to_mm_per_pixel(image_depth_mm: float, image_height_px: int) -> float:
    """
    Convert MATLAB image depth metadata to a pixel scale.

    Parameters
    ----------
    image_depth_mm:
        Physical image depth in millimeters.
    image_height_px:
        Image height in pixels.

    Returns
    -------
    float
        Millimeters per image pixel.
    """
    image_depth_mm = float(image_depth_mm)
    image_height_px = int(image_height_px)
    if not np.isfinite(image_depth_mm) or image_depth_mm <= 0:
        raise ValueError("image_depth_mm must be positive and finite.")
    if image_height_px <= 0:
        raise ValueError("image_height_px must be positive.")
    return image_depth_mm / image_height_px


def line_y_at_x(lines: np.ndarray, x_eval: float | np.ndarray) -> np.ndarray:
    """
    Evaluate one or more non-vertical lines at x.

    Parameters
    ----------
    lines:
        Array shaped ``(4,)`` or ``(N, 4)`` with ``[x1, y1, x2, y2]`` rows.
    x_eval:
        Scalar or array broadcastable to ``(N,)``.

    Returns
    -------
    np.ndarray
        y values at ``x_eval``. Vertical lines return NaN.
    """
    lines_arr = np.asarray(lines, dtype=np.float64)
    if lines_arr.ndim == 1:
        lines_arr = lines_arr.reshape(1, 4)
    if lines_arr.ndim != 2 or lines_arr.shape[1] != 4:
        raise ValueError("lines must have shape (4,) or (N, 4).")

    x1 = lines_arr[:, 0]
    y1 = lines_arr[:, 1]
    x2 = lines_arr[:, 2]
    y2 = lines_arr[:, 3]
    dx = x2 - x1
    x_eval_arr = np.asarray(x_eval, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (y2 - y1) / dx
        y = y1 + slope * (x_eval_arr - x1)

    return np.where(np.abs(dx) > 1e-12, y, np.nan)


def aponeurosis_thickness_px(
    superficial_apo_lines: np.ndarray,
    deep_apo_lines: np.ndarray,
    *,
    x_eval: float | np.ndarray = 20.0,
) -> np.ndarray:
    """
    Compute MATLAB-style muscle thickness from superficial/deep aponeuroses.

    This follows the TimTrack formula used in the validation notebooks:

        thickness = (deep_y(x) - superficial_y(x)) * cos(superficial_angle)

    Parameters
    ----------
    superficial_apo_lines, deep_apo_lines:
        Arrays shaped ``(4,)`` or ``(N, 4)``.
    x_eval:
        x position where the aponeurosis separation is evaluated.

    Returns
    -------
    np.ndarray
        Muscle thickness in pixels.
    """
    super_lines = np.asarray(superficial_apo_lines, dtype=np.float64)
    deep_lines = np.asarray(deep_apo_lines, dtype=np.float64)

    if super_lines.ndim == 1:
        super_lines = super_lines.reshape(1, 4)
    if deep_lines.ndim == 1:
        deep_lines = deep_lines.reshape(1, 4)
    if super_lines.ndim != 2 or super_lines.shape[1] != 4:
        raise ValueError("superficial_apo_lines must have shape (4,) or (N, 4).")
    if deep_lines.ndim != 2 or deep_lines.shape[1] != 4:
        raise ValueError("deep_apo_lines must have shape (4,) or (N, 4).")
    if len(super_lines) != len(deep_lines):
        if len(super_lines) == 1:
            super_lines = np.repeat(super_lines, len(deep_lines), axis=0)
        elif len(deep_lines) == 1:
            deep_lines = np.repeat(deep_lines, len(super_lines), axis=0)
        else:
            raise ValueError("superficial and deep line arrays must have matching lengths.")

    super_angle = _normalize_angle_array(line_angles_batch(super_lines, degrees=True))
    super_y = line_y_at_x(super_lines, x_eval)
    deep_y = line_y_at_x(deep_lines, x_eval)
    return (deep_y - super_y) * np.cos(np.deg2rad(super_angle))


def final_outputs_from_components(
    alpha_deg: np.ndarray | float,
    aponeurosis_angle_deg: np.ndarray | float,
    thickness_px: np.ndarray | float,
    *,
    mm_per_pixel: Optional[float] = None,
    normalize_pennation: bool = False,
    min_abs_sin: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Compute final ``ANG``, ``PEN``, and ``FL`` from saved components.

    Parameters
    ----------
    alpha_deg:
        Fascicle angle in degrees. This is the final ``ANG`` candidate.
    aponeurosis_angle_deg:
        Reference aponeurosis angle in degrees. For current MATLAB comparison,
        this is usually the superficial aponeurosis angle.
    thickness_px:
        Muscle thickness in pixels.
    mm_per_pixel:
        Optional conversion factor for ``FL_mm``.
    normalize_pennation:
        Whether to wrap ``PEN`` to ``[-90, 90)``. The default keeps the direct
        MATLAB-style difference ``alpha - aponeurosis_angle``.
    min_abs_sin:
        Denominator guard for near-zero pennation.

    Returns
    -------
    dict
        Keys include ``ANG_deg``, ``PEN_deg``, ``FL_px`` and, when requested,
        ``FL_mm``.
    """
    alpha, apo_angle, thickness = np.broadcast_arrays(
        np.asarray(alpha_deg, dtype=np.float64),
        np.asarray(aponeurosis_angle_deg, dtype=np.float64),
        np.asarray(thickness_px, dtype=np.float64),
    )

    pen = alpha - apo_angle
    if normalize_pennation:
        pen = _normalize_angle_array(pen)

    sin_pen = np.sin(np.deg2rad(pen))
    fl_px = np.full_like(sin_pen, np.nan, dtype=np.float64)
    valid = np.isfinite(thickness) & np.isfinite(sin_pen) & (np.abs(sin_pen) > min_abs_sin)
    fl_px[valid] = thickness[valid] / sin_pen[valid]

    out: Dict[str, np.ndarray] = {
        "ANG_deg": alpha.astype(np.float64, copy=False),
        "PEN_deg": pen.astype(np.float64, copy=False),
        "FL_px": fl_px,
    }

    if mm_per_pixel is not None:
        out["FL_mm"] = fl_px * float(mm_per_pixel)

    return out


def final_outputs_from_lines(
    alpha_deg: np.ndarray | float,
    superficial_apo_lines: np.ndarray,
    deep_apo_lines: np.ndarray,
    *,
    x_eval: float | np.ndarray = 20.0,
    pennation_reference: str = "superficial",
    mm_per_pixel: Optional[float] = None,
    normalize_pennation: bool = False,
    min_abs_sin: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Compute final outputs directly from fascicle alpha and aponeurosis lines.

    Parameters
    ----------
    alpha_deg:
        Fascicle angle in degrees.
    superficial_apo_lines, deep_apo_lines:
        Arrays shaped ``(4,)`` or ``(N, 4)``.
    x_eval:
        x position for MATLAB-style thickness evaluation.
    pennation_reference:
        ``"superficial"`` for the current TimTrack/Region compatibility path,
        or ``"deep"`` for a deep-aponeurosis-relative angle.
    mm_per_pixel:
        Optional conversion factor for ``FL_mm``.

    Returns
    -------
    dict
        Final outputs plus intermediate aponeurosis angles and thickness.
    """
    super_lines = np.asarray(superficial_apo_lines, dtype=np.float64)
    deep_lines = np.asarray(deep_apo_lines, dtype=np.float64)
    if super_lines.ndim == 1:
        super_lines = super_lines.reshape(1, 4)
    if deep_lines.ndim == 1:
        deep_lines = deep_lines.reshape(1, 4)

    if len(super_lines) != len(deep_lines):
        if len(super_lines) == 1:
            super_lines = np.repeat(super_lines, len(deep_lines), axis=0)
        elif len(deep_lines) == 1:
            deep_lines = np.repeat(deep_lines, len(super_lines), axis=0)
        else:
            raise ValueError("superficial and deep line arrays must have matching lengths.")

    super_angle = _normalize_angle_array(line_angles_batch(super_lines, degrees=True))
    deep_angle = _normalize_angle_array(line_angles_batch(deep_lines, degrees=True))
    thickness = aponeurosis_thickness_px(super_lines, deep_lines, x_eval=x_eval)

    ref = pennation_reference.lower()
    if ref == "superficial":
        apo_angle = super_angle
    elif ref == "deep":
        apo_angle = deep_angle
    else:
        raise ValueError("pennation_reference must be 'superficial' or 'deep'.")

    out = final_outputs_from_components(
        alpha_deg,
        apo_angle,
        thickness,
        mm_per_pixel=mm_per_pixel,
        normalize_pennation=normalize_pennation,
        min_abs_sin=min_abs_sin,
    )
    out.update(
        {
            "super_apo_angle_deg": super_angle,
            "deep_apo_angle_deg": deep_angle,
            "muscle_thickness_px": thickness,
        }
    )
    return out
