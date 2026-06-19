"""
MATLAB UltraTimTrack 2-state Kalman compatibility path.

This module is intentionally separate from ``ultratimtrack_kalman.py``.  The
existing module is an experimental 4-state geometric fusion model.  MATLAB's
state estimator is a scalar pair:

    [superficial fascicle attachment x, fascicle alpha]

The functions here port the MATLAB update/smoothing equations closely enough
for parity notebooks to validate the downstream Kalman gate independently from
KLT and TimTrack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .geometry import line_angles_batch, line_lengths_batch, normalize_angle


IDX_X_SUP = 0
IDX_ALPHA = 1
STATE_SIZE = 2


@dataclass(frozen=True)
class MatlabTwoStateKalmanConfig:
    """Parameters matching the MATLAB state estimator controls."""

    q_parameter: float = 0.01
    x_measurement_variance: float = 100.0
    alpha_measurement_variance: float = 3.05529211
    n_start_frames: int = 1
    run_smoother: bool = True
    use_adaptive_R: bool = False


def _as_segments(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape (n_frames, 4).")
    return arr


def _as_1d(values: np.ndarray, name: str, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"{name} must have length {n}, got {len(arr)}.")
    return arr


def _as_optional_scale(values: Optional[np.ndarray], name: str, n: int) -> np.ndarray:
    if values is None:
        return np.ones(n, dtype=np.float64)
    arr = _as_1d(values, name, n)
    arr = np.where(np.isfinite(arr), arr, 1.0)
    return np.clip(arr, np.finfo(float).eps, np.inf)


def _line_coefficients(line: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = np.asarray(line, dtype=np.float64)
    dx = x2 - x1
    if abs(dx) <= 1e-12:
        return np.nan, np.nan
    slope = (y2 - y1) / dx
    intercept = y1 - slope * x1
    return float(slope), float(intercept)


def _normalized_segment_angles(segments: np.ndarray) -> np.ndarray:
    angles = line_angles_batch(np.asarray(segments, dtype=np.float64), degrees=True)
    return np.asarray([normalize_angle(angle, degrees=True) for angle in angles], dtype=np.float64)


def matlab_scalar_kalman_update(
    x_minus: float,
    p_prev: float,
    q_value: float,
    measurement: float,
    measurement_variance: float,
) -> Tuple[float, float, float, float]:
    """
    Port MATLAB ``run_kalman_filter`` for one scalar state.

    Returns ``(x_plus, p_plus, p_minus, gain)``.
    """
    p_minus = float(p_prev) + float(q_value)
    denom = p_minus + float(measurement_variance)
    gain = p_minus / denom if denom != 0 else np.nan

    if np.isnan(gain):
        gain = 0.0
    if gain < 0.0 or gain > 1.0:
        gain = float(np.clip(gain, 0.0, 1.0))

    x_plus = float(x_minus) + gain * (float(measurement) - float(x_minus))
    p_plus = (1.0 - gain) * p_minus
    return float(x_plus), float(p_plus), float(p_minus), float(gain)


def reconstruct_fascicle_from_state(
    x_sup: float,
    alpha_deg: float,
    superficial_apo_line: np.ndarray,
    deep_apo_line: np.ndarray,
    fixed_superficial_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct MATLAB ``fas_x/fas_y`` and ``fas_x_end/fas_y_end``.

    Returns
    -------
    fascicle_segment:
        ``[x_sup_state, fixed_y_sup, x_deep, y_deep]`` matching MATLAB
        ``Fascicle.fas_x/fas_y`` after conversion to Python endpoint order.
    fascicle_end_segment:
        ``[x_sup_on_super_apo, y_sup_on_super_apo, x_deep, y_deep]`` matching
        MATLAB ``fas_x_end/fas_y_end`` after conversion.  MATLAB uses this
        end segment for final fascicle length.
    """
    super_slope, super_intercept = _line_coefficients(superficial_apo_line)
    deep_slope, deep_intercept = _line_coefficients(deep_apo_line)

    fascicle_slope = -np.tan(np.deg2rad(float(alpha_deg)))
    fascicle_intercept = float(fixed_superficial_y) - fascicle_slope * float(x_sup)

    x_deep = (fascicle_intercept - deep_intercept) / (deep_slope - fascicle_slope)
    y_deep = deep_intercept + x_deep * deep_slope

    x_sup_end = (fascicle_intercept - super_intercept) / (super_slope - fascicle_slope)
    y_sup_end = super_intercept + x_sup_end * super_slope

    fascicle_segment = np.asarray(
        [float(x_sup), float(fixed_superficial_y), x_deep, y_deep],
        dtype=np.float64,
    )
    fascicle_end_segment = np.asarray(
        [x_sup_end, y_sup_end, x_deep, y_deep],
        dtype=np.float64,
    )
    return fascicle_segment, fascicle_end_segment


def _compute_outputs(
    states: np.ndarray,
    superficial_apo_lines: np.ndarray,
    deep_apo_lines: np.ndarray,
    fixed_superficial_y: float,
    mm_per_pixel: Optional[float],
) -> Dict[str, np.ndarray]:
    n = len(states)
    fascicle_segments = np.full((n, 4), np.nan, dtype=np.float64)
    fascicle_end_segments = np.full((n, 4), np.nan, dtype=np.float64)

    for frame in range(n):
        fascicle_segments[frame], fascicle_end_segments[frame] = reconstruct_fascicle_from_state(
            states[frame, IDX_X_SUP],
            states[frame, IDX_ALPHA],
            superficial_apo_lines[frame],
            deep_apo_lines[frame],
            fixed_superficial_y,
        )

    alpha = _normalized_segment_angles(fascicle_segments)
    deep_angle = _normalized_segment_angles(deep_apo_lines)
    pennation = alpha - deep_angle
    length_px = line_lengths_batch(fascicle_end_segments)

    out: Dict[str, np.ndarray] = {
        "fascicle_segments": fascicle_segments,
        "fascicle_end_segments": fascicle_end_segments,
        "alpha_deg": alpha,
        "ANG_deg": alpha,
        "PEN_deg": pennation,
        "FL_px": length_px,
        "deep_apo_angle_deg": deep_angle,
    }
    if mm_per_pixel is not None:
        out["FL_mm"] = length_px * float(mm_per_pixel)
    return out


def run_matlab_2state_kalman(
    klt_segments: np.ndarray,
    timtrack_alpha_deg: np.ndarray,
    superficial_apo_lines: np.ndarray,
    deep_apo_lines: np.ndarray,
    *,
    config: MatlabTwoStateKalmanConfig | None = None,
    fixed_superficial_y: Optional[float] = None,
    mm_per_pixel: Optional[float] = None,
    measurement_r_scale: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Run a MATLAB-like 2-state forward filter plus optional RTS-style smoother.

    Parameters
    ----------
    klt_segments:
        Raw KLT prior geometry, shape ``(n_frames, 4)`` and endpoint order
        ``[x_sup, y_sup, x_deep, y_deep]``.
    timtrack_alpha_deg:
        TimTrack/Hough alpha measurements, one per frame.
    superficial_apo_lines, deep_apo_lines:
        Current aponeurosis lines used to reconstruct the final fascicle.

    Notes
    -----
    MATLAB predicts the state by applying saved affine warps to the previous
    corrected state.  Those MATLAB ``affine2d`` objects are not available from
    SciPy because they are MCOS opaque objects.  This compatibility path uses
    the raw KLT segment-to-segment delta as the prediction input, which is the
    practical quantity available to the Python pipeline and validation
    notebooks.
    """
    klt = _as_segments(klt_segments, "klt_segments")
    n = len(klt)
    tim_alpha = _as_1d(timtrack_alpha_deg, "timtrack_alpha_deg", n)
    superficial = _as_segments(superficial_apo_lines, "superficial_apo_lines")
    deep = _as_segments(deep_apo_lines, "deep_apo_lines")
    if len(superficial) != n or len(deep) != n:
        raise ValueError("aponeurosis line arrays must match klt_segments length.")

    cfg = config or MatlabTwoStateKalmanConfig()
    n_start = max(1, min(int(cfg.n_start_frames), n))
    fixed_y = float(klt[0, 1] if fixed_superficial_y is None else fixed_superficial_y)
    r_scale = _as_optional_scale(measurement_r_scale, "measurement_r_scale", n) if cfg.use_adaptive_R else np.ones(n)
    measurement_R_diag = np.column_stack(
        [
            np.full(n, float(cfg.x_measurement_variance), dtype=np.float64) * r_scale,
            np.full(n, float(cfg.alpha_measurement_variance), dtype=np.float64) * r_scale,
        ]
    )

    klt_alpha = _normalized_segment_angles(klt)
    states_plus = np.full((n, STATE_SIZE), np.nan, dtype=np.float64)
    states_minus = np.full((n, STATE_SIZE), np.nan, dtype=np.float64)
    p_plus = np.full((n, STATE_SIZE), np.nan, dtype=np.float64)
    p_minus = np.full((n, STATE_SIZE), np.nan, dtype=np.float64)
    gains = np.full((n, STATE_SIZE), np.nan, dtype=np.float64)

    alpha0 = klt_alpha[:n_start]
    states_plus[0] = [klt[0, 0], float(np.nanmean(alpha0))]
    # MATLAB's var with one start frame behaves as zero for the forward pass.
    alpha_var = float(np.nanvar(alpha0, ddof=1)) if n_start > 1 else 0.0
    p_plus[0] = [0.0, alpha_var]
    states_minus[0] = states_plus[0]
    p_minus[0] = p_plus[0]

    for frame in range(1, n):
        dx_sup = klt[frame, 0] - klt[frame - 1, 0]
        dy_sup = klt[frame, 1] - klt[frame - 1, 1]
        x_prior = states_plus[frame - 1, IDX_X_SUP] + dx_sup

        d_alpha = abs(klt_alpha[frame]) - abs(klt_alpha[frame - 1])
        alpha_prior = states_plus[frame - 1, IDX_ALPHA] + d_alpha

        dx = float(np.hypot(dx_sup, dy_sup))
        q_x = float(cfg.q_parameter) * dx * dx
        (
            states_plus[frame, IDX_X_SUP],
            p_plus[frame, IDX_X_SUP],
            p_minus[frame, IDX_X_SUP],
            gains[frame, IDX_X_SUP],
        ) = matlab_scalar_kalman_update(
            x_prior,
            p_plus[frame - 1, IDX_X_SUP],
            q_x,
            klt[0, 0],
            measurement_R_diag[frame, IDX_X_SUP],
        )
        states_minus[frame, IDX_X_SUP] = x_prior

        d_alpha_abs = abs(float(d_alpha))
        if d_alpha_abs < 0.005:
            d_alpha_abs = 0.0
        q_alpha = float(cfg.q_parameter) * d_alpha_abs * d_alpha_abs
        (
            states_plus[frame, IDX_ALPHA],
            p_plus[frame, IDX_ALPHA],
            p_minus[frame, IDX_ALPHA],
            gains[frame, IDX_ALPHA],
        ) = matlab_scalar_kalman_update(
            alpha_prior,
            p_plus[frame - 1, IDX_ALPHA],
            q_alpha,
            tim_alpha[frame],
            measurement_R_diag[frame, IDX_ALPHA],
        )
        states_minus[frame, IDX_ALPHA] = alpha_prior

    outputs_forward = _compute_outputs(
        states_plus,
        superficial,
        deep,
        fixed_y,
        mm_per_pixel,
    )

    states_smooth = states_plus.copy()
    p_smooth = p_plus.copy()
    smoother_gain = np.full((n, STATE_SIZE), np.nan, dtype=np.float64)

    if cfg.run_smoother and n > 1:
        forward_p_plus = p_plus.copy()
        for frame in range(n - 2, -1, -1):
            for idx in range(STATE_SIZE):
                denom = p_minus[frame + 1, idx]
                gain = forward_p_plus[frame, idx] / denom if denom != 0 else np.nan
                if np.isnan(gain):
                    gain = 1.0
                smoother_gain[frame, idx] = gain
                states_smooth[frame, idx] = states_plus[frame, idx] + gain * (
                    states_smooth[frame + 1, idx] - states_minus[frame + 1, idx]
                )
                # MATLAB initializes Psmooth to ones(1, 2) in each smoother call.
                p_smooth[frame, idx] = forward_p_plus[frame, idx] + gain * (
                    1.0 - p_minus[frame + 1, idx]
                ) * gain

    outputs_smooth = _compute_outputs(
        states_smooth,
        superficial,
        deep,
        fixed_y,
        mm_per_pixel,
    )

    result: Dict[str, np.ndarray] = {
        "X_plus": states_smooth,
        "X_minus": states_minus,
        "fas_p": p_smooth,
        "fas_p_minus": p_minus,
        "kalman_gain": gains,
        "smoother_gain": smoother_gain,
        "measurement_r_scale": r_scale,
        "measurement_R_diag": measurement_R_diag,
        "use_adaptive_R": np.asarray(bool(cfg.use_adaptive_R)),
        "forward_X_plus": states_plus,
        "forward_fas_p": p_plus,
        "forward_fascicle_segments": outputs_forward["fascicle_segments"],
        "forward_fascicle_end_segments": outputs_forward["fascicle_end_segments"],
        "forward_ANG_deg": outputs_forward["ANG_deg"],
        "forward_PEN_deg": outputs_forward["PEN_deg"],
        "forward_FL_px": outputs_forward["FL_px"],
        "fascicle_segments": outputs_smooth["fascicle_segments"],
        "fascicle_end_segments": outputs_smooth["fascicle_end_segments"],
        "alpha_deg": outputs_smooth["alpha_deg"],
        "ANG_deg": outputs_smooth["ANG_deg"],
        "PEN_deg": outputs_smooth["PEN_deg"],
        "FL_px": outputs_smooth["FL_px"],
        "deep_apo_angle_deg": outputs_smooth["deep_apo_angle_deg"],
        "fixed_superficial_y": np.asarray(fixed_y, dtype=np.float64),
    }
    if mm_per_pixel is not None:
        result["forward_FL_mm"] = outputs_forward["FL_mm"]
        result["FL_mm"] = outputs_smooth["FL_mm"]
    return result
