"""MATLAB-style aponeurosis KLT plus endpoint Kalman gate.

MATLAB UltraTimTrack treats aponeuroses as a separate downstream state from the
fascicle two-state filter.  Four scalar states are tracked:

    [superficial y at x=1, superficial y at x=width,
     deep y at x=1, deep y at x=width]

Each frame, an aponeurosis KLT affine predicts the four endpoint y values and
TimTrack's detected ``super_pos`` / ``deep_pos`` values update them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from .ultratrack_klt import (
    UltraTrackKLTConfig,
    apply_affine_1b,
    detect_min_eigen_like,
    estimate_affine_matlab_coords,
    filter_points_by_mask,
    track_points,
    tracking_masks_from_geofeature,
)
from .ultratimtrack_matlab_2state import matlab_scalar_kalman_update


def aponeurosis_state_from_lines(superficial_line_1b: np.ndarray, deep_line_1b: np.ndarray) -> np.ndarray:
    """Return ``[sup_y1, sup_y2, deep_y1, deep_y2]`` from one-based lines."""

    superficial = np.asarray(superficial_line_1b, dtype=np.float64).reshape(-1)
    deep = np.asarray(deep_line_1b, dtype=np.float64).reshape(-1)
    if superficial.size != 4 or deep.size != 4:
        raise ValueError("aponeurosis lines must each have shape (4,).")
    return np.asarray([superficial[1], superficial[3], deep[1], deep[3]], dtype=np.float64)


def lines_from_aponeurosis_state(state_y: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one-based superficial/deep endpoint lines from a four-y state."""

    state = np.asarray(state_y, dtype=np.float64).reshape(-1)
    if state.size != 4:
        raise ValueError("state_y must have four entries.")
    x2 = float(width)
    superficial = np.asarray([1.0, state[0], x2, state[1]], dtype=np.float64)
    deep = np.asarray([1.0, state[2], x2, state[3]], dtype=np.float64)
    return superficial, deep


def _transform_endpoint_state(state_y: np.ndarray, affine: np.ndarray, width: int) -> np.ndarray:
    superficial, deep = lines_from_aponeurosis_state(state_y, width)
    points = np.asarray(
        [
            [superficial[0], superficial[1]],
            [superficial[2], superficial[3]],
            [deep[0], deep[1]],
            [deep[2], deep[3]],
        ],
        dtype=np.float64,
    )
    transformed = apply_affine_1b(points, affine)
    return np.asarray([transformed[0, 1], transformed[1, 1], transformed[2, 1], transformed[3, 1]], dtype=np.float64)


def run_matlab_aponeurosis_state_video(
    video_path: str | Path,
    geofeatures: Sequence[Mapping],
    measurement_superficial_lines_1b: np.ndarray,
    measurement_deep_lines_1b: np.ndarray,
    *,
    super_cut: Sequence[float],
    deep_cut: Sequence[float],
    q_parameter: float = 0.01,
    measurement_variance: np.ndarray | Sequence[float] | None = None,
    config: UltraTrackKLTConfig | None = None,
    limit: Optional[int] = None,
    progress_every: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    Run the MATLAB-style aponeurosis endpoint filter from video frames.

    Parameters use MATLAB one-based coordinates for the input/output lines.
    The KLT affine is estimated on the previous frame's aponeurosis cut masks
    and updates the previous filtered endpoints before TimTrack measurements
    are applied.
    """

    cfg = config or UltraTrackKLTConfig()
    super_meas = np.asarray(measurement_superficial_lines_1b, dtype=np.float64)
    deep_meas = np.asarray(measurement_deep_lines_1b, dtype=np.float64)
    if super_meas.ndim != 2 or super_meas.shape[1] != 4:
        raise ValueError("measurement_superficial_lines_1b must have shape (n, 4).")
    if deep_meas.ndim != 2 or deep_meas.shape[1] != 4:
        raise ValueError("measurement_deep_lines_1b must have shape (n, 4).")

    n = min(len(geofeatures), len(super_meas), len(deep_meas))
    if limit is not None:
        n = min(n, int(limit))
    if n <= 0:
        raise ValueError("geofeatures and measurement lines must be non-empty.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read first frame from {video_path}")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame.copy()
    shape = prev_gray.shape[:2]
    width = int(shape[1])

    if measurement_variance is None:
        r_values = np.ones(4, dtype=np.float64)
    else:
        r_values = np.asarray(measurement_variance, dtype=np.float64).reshape(-1)
        if r_values.size != 4:
            raise ValueError("measurement_variance must have four entries.")

    states_plus = np.full((n, 4), np.nan, dtype=np.float64)
    states_minus = np.full((n, 4), np.nan, dtype=np.float64)
    p_plus = np.full((n, 4), np.nan, dtype=np.float64)
    p_minus = np.full((n, 4), np.nan, dtype=np.float64)
    gains = np.full((n, 4), np.nan, dtype=np.float64)
    super_lines = np.full((n, 4), np.nan, dtype=np.float64)
    deep_lines = np.full((n, 4), np.nan, dtype=np.float64)
    prior_super_lines = np.full((n, 4), np.nan, dtype=np.float64)
    prior_deep_lines = np.full((n, 4), np.nan, dtype=np.float64)
    points_count = np.zeros(n, dtype=np.int32)
    tracked_count = np.zeros(n, dtype=np.int32)
    inlier_count = np.zeros(n, dtype=np.int32)
    affine_ok = np.zeros(n, dtype=bool)
    affine_matrices = np.full((n, 2, 3), np.nan, dtype=np.float32)

    states_plus[0] = aponeurosis_state_from_lines(super_meas[0], deep_meas[0])
    states_minus[0] = states_plus[0]
    p_plus[0] = r_values
    p_minus[0] = r_values
    super_lines[0], deep_lines[0] = lines_from_aponeurosis_state(states_plus[0], width)
    prior_super_lines[0], prior_deep_lines[0] = super_lines[0], deep_lines[0]

    for frame in range(1, n):
        ok, current_frame = cap.read()
        if not ok:
            states_plus = states_plus[:frame]
            states_minus = states_minus[:frame]
            p_plus = p_plus[:frame]
            p_minus = p_minus[:frame]
            gains = gains[:frame]
            super_lines = super_lines[:frame]
            deep_lines = deep_lines[:frame]
            prior_super_lines = prior_super_lines[:frame]
            prior_deep_lines = prior_deep_lines[:frame]
            points_count = points_count[:frame]
            tracked_count = tracked_count[:frame]
            inlier_count = inlier_count[:frame]
            affine_ok = affine_ok[:frame]
            affine_matrices = affine_matrices[:frame]
            break

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if current_frame.ndim == 3 else current_frame.copy()
        masks = tracking_masks_from_geofeature(
            geofeatures[frame - 1],
            shape=shape,
            super_cut=super_cut,
            deep_cut=deep_cut,
        )
        points = detect_min_eigen_like(
            prev_gray,
            masks["apo_mask"],
            max_corners=cfg.max_aponeurosis_corners,
            config=cfg,
        )
        points = filter_points_by_mask(points, masks["apo_mask"])
        old_points, new_points = track_points(prev_gray, gray, points, config=cfg)
        affine, inliers = estimate_affine_matlab_coords(old_points, new_points, config=cfg)

        points_count[frame] = len(points)
        tracked_count[frame] = len(new_points)
        inlier_count[frame] = int(inliers)
        if affine is not None:
            prior_state = _transform_endpoint_state(states_plus[frame - 1], affine, width)
            affine_ok[frame] = True
            affine_matrices[frame] = affine
        else:
            prior_state = states_plus[frame - 1].copy()

        states_minus[frame] = prior_state
        prior_super_lines[frame], prior_deep_lines[frame] = lines_from_aponeurosis_state(prior_state, width)
        measurement_state = aponeurosis_state_from_lines(super_meas[frame], deep_meas[frame])
        delta = np.abs(prior_state - states_plus[frame - 1])

        for idx in range(4):
            (
                states_plus[frame, idx],
                p_plus[frame, idx],
                p_minus[frame, idx],
                gains[frame, idx],
            ) = matlab_scalar_kalman_update(
                prior_state[idx],
                p_plus[frame - 1, idx],
                float(q_parameter) * float(delta[idx]) ** 2,
                measurement_state[idx],
                r_values[idx],
            )

        super_lines[frame], deep_lines[frame] = lines_from_aponeurosis_state(states_plus[frame], width)
        prev_gray = gray
        if progress_every and (frame % int(progress_every) == 0 or frame == n - 1):
            print(f"aponeurosis state processed {frame + 1}/{n}")

    cap.release()
    return {
        "super_lines": super_lines,
        "deep_lines": deep_lines,
        "prior_super_lines": prior_super_lines,
        "prior_deep_lines": prior_deep_lines,
        "states_plus": states_plus,
        "states_minus": states_minus,
        "p_plus": p_plus,
        "p_minus": p_minus,
        "gains": gains,
        "points_count": points_count,
        "tracked_count": tracked_count,
        "inlier_count": inlier_count,
        "affine_ok": affine_ok,
        "affine_matrices": affine_matrices,
        "measurement_variance": r_values,
    }
