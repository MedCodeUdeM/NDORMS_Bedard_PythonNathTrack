"""MATLAB-style aponeurosis KLT plus endpoint Kalman gate.

MATLAB UltraTimTrack treats aponeuroses as a separate downstream state from the
fascicle two-state filter.  Four scalar states are tracked:

    [superficial y at x=1, superficial y at x=width,
     deep y at x=1, deep y at x=width]

Each frame, an aponeurosis KLT affine predicts the four endpoint y values and
TimTrack's detected ``super_pos`` / ``deep_pos`` values update them.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class AponeurosisGatingConfig:
    """Optional anatomical measurement gate for aponeurosis endpoint updates."""

    enabled: bool = False
    endpoint_innovation_px: float = 32.0
    endpoint_jump_px: float = 32.0
    mid_innovation_px: float = 10.0
    super_mid_jump_px: float = 12.0
    deep_mid_jump_px: float = 6.0
    angle_jump_deg: float = 2.5
    near_maxangle_fraction: float = 0.90
    giant_mid_jump_px: float = 16.0
    giant_mid_innovation_px: float = 24.0
    soft_fraction: float = 0.65
    soft_r_scale: float = 12.0
    hard_r_scale: float = 1.0e6
    max_consecutive_rejections: int = 3
    super_maxangle_deg: Optional[float] = None
    deep_maxangle_deg: Optional[float] = None


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


def _line_mid_angle(line_1b: np.ndarray) -> tuple[float, float]:
    line = np.asarray(line_1b, dtype=np.float64).reshape(4)
    dx = line[2] - line[0]
    if abs(dx) <= 1e-12:
        return float((line[1] + line[3]) / 2.0), float("nan")
    slope = (line[3] - line[1]) / dx
    angle = -float(np.rad2deg(np.arctan2(slope, 1.0)))
    return float((line[1] + line[3]) / 2.0), angle


def gate_aponeurosis_measurement_state(
    prior_state: np.ndarray,
    measurement_state: np.ndarray,
    previous_measurement_state: np.ndarray,
    previous_filtered_state: np.ndarray,
    *,
    width: int,
    config: AponeurosisGatingConfig,
    consecutive_rejections: np.ndarray | Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Gate one aponeurosis endpoint measurement against prior and recent motion."""

    prior = np.asarray(prior_state, dtype=np.float64).reshape(4)
    measurement = np.asarray(measurement_state, dtype=np.float64).reshape(4)
    prev_measurement = np.asarray(previous_measurement_state, dtype=np.float64).reshape(4)
    prev_filtered = np.asarray(previous_filtered_state, dtype=np.float64).reshape(4)
    consecutive = (
        np.zeros(2, dtype=np.int32)
        if consecutive_rejections is None
        else np.asarray(consecutive_rejections, dtype=np.int32).reshape(2).copy()
    )

    out_measurement = measurement.copy()
    r_scale = np.ones(4, dtype=np.float64)
    rejected = np.zeros(4, dtype=bool)
    soft_downweighted = np.zeros(4, dtype=bool)
    line_rejected = np.zeros(2, dtype=bool)
    line_soft = np.zeros(2, dtype=bool)
    reasons = np.asarray(["", ""], dtype=object)
    next_consecutive = consecutive.copy()

    if not bool(config.enabled):
        return {
            "measurement_state": out_measurement,
            "r_scale": r_scale,
            "rejected_endpoints": rejected,
            "soft_downweighted_endpoints": soft_downweighted,
            "line_rejected": line_rejected,
            "line_soft_downweighted": line_soft,
            "reasons": reasons,
            "consecutive_rejections": next_consecutive,
        }

    prior_lines = lines_from_aponeurosis_state(prior, width)
    measurement_lines = lines_from_aponeurosis_state(measurement, width)
    prev_measurement_lines = lines_from_aponeurosis_state(prev_measurement, width)
    prev_filtered_lines = lines_from_aponeurosis_state(prev_filtered, width)

    for side_idx, (name, endpoint_idx, maxangle) in enumerate(
        [
            ("superficial", np.asarray([0, 1]), config.super_maxangle_deg),
            ("deep", np.asarray([2, 3]), config.deep_maxangle_deg),
        ]
    ):
        prior_mid, _ = _line_mid_angle(prior_lines[side_idx])
        measurement_mid, measurement_angle = _line_mid_angle(measurement_lines[side_idx])
        prev_measurement_mid, prev_measurement_angle = _line_mid_angle(prev_measurement_lines[side_idx])
        prev_filtered_mid, _ = _line_mid_angle(prev_filtered_lines[side_idx])

        mid_innovation = abs(measurement_mid - prior_mid)
        mid_jump = abs(measurement_mid - prev_measurement_mid)
        filtered_jump = abs(measurement_mid - prev_filtered_mid)
        angle_jump = abs(measurement_angle - prev_measurement_angle)
        mid_jump_limit = config.deep_mid_jump_px if name == "deep" else config.super_mid_jump_px

        near_maxangle = (
            maxangle is not None
            and np.isfinite(float(maxangle))
            and float(maxangle) > 0.0
            and np.isfinite(measurement_angle)
            and abs(measurement_angle) >= config.near_maxangle_fraction * abs(float(maxangle))
        )
        huge_mid = mid_jump > config.giant_mid_jump_px or mid_innovation > config.giant_mid_innovation_px
        suspicious_mid = mid_jump > mid_jump_limit or mid_innovation > config.mid_innovation_px
        suspicious_angle = np.isfinite(angle_jump) and angle_jump > config.angle_jump_deg
        hard_line = bool(huge_mid or (near_maxangle and (suspicious_mid or suspicious_angle)))
        force_reacquire = consecutive[side_idx] >= int(config.max_consecutive_rejections)

        reason_parts: list[str] = []
        if suspicious_mid:
            reason_parts.append(
                f"mid jump/innovation {mid_jump:.1f}/{mid_innovation:.1f}px"
            )
        if filtered_jump > mid_jump_limit:
            reason_parts.append(f"filtered jump {filtered_jump:.1f}px")
        if suspicious_angle:
            reason_parts.append(f"angle jump {angle_jump:.1f}deg")
        if near_maxangle:
            reason_parts.append(f"near maxangle {measurement_angle:.1f}deg")
        if huge_mid:
            reason_parts.append("giant mid displacement")
        if force_reacquire and hard_line:
            reason_parts.append("forced reacquisition after repeated rejects")

        if hard_line and not force_reacquire:
            out_measurement[endpoint_idx] = prior[endpoint_idx]
            r_scale[endpoint_idx] = float(config.hard_r_scale)
            rejected[endpoint_idx] = True
            line_rejected[side_idx] = True
            next_consecutive[side_idx] += 1
            reasons[side_idx] = "; ".join(reason_parts)
            continue

        soft_line = bool(suspicious_mid or suspicious_angle or (hard_line and force_reacquire))
        if soft_line:
            line_soft[side_idx] = True
            next_consecutive[side_idx] = 0
            reasons[side_idx] = "; ".join(reason_parts)
        else:
            next_consecutive[side_idx] = 0

        for idx in endpoint_idx:
            endpoint_innovation = abs(measurement[idx] - prior[idx])
            endpoint_jump = abs(measurement[idx] - prev_measurement[idx])
            endpoint_soft = (
                soft_line
                or endpoint_innovation > config.endpoint_innovation_px * config.soft_fraction
                or endpoint_jump > config.endpoint_jump_px * config.soft_fraction
            )
            endpoint_hard = (
                near_maxangle
                and not force_reacquire
                and (
                    endpoint_innovation > config.endpoint_innovation_px
                    or endpoint_jump > config.endpoint_jump_px
                )
            )
            if endpoint_hard:
                out_measurement[idx] = prior[idx]
                r_scale[idx] = float(config.hard_r_scale)
                rejected[idx] = True
                line_rejected[side_idx] = True
                next_consecutive[side_idx] += 1
            elif endpoint_soft:
                r_scale[idx] = max(r_scale[idx], float(config.soft_r_scale))
                soft_downweighted[idx] = True

    return {
        "measurement_state": out_measurement,
        "r_scale": r_scale,
        "rejected_endpoints": rejected,
        "soft_downweighted_endpoints": soft_downweighted,
        "line_rejected": line_rejected,
        "line_soft_downweighted": line_soft,
        "reasons": reasons,
        "consecutive_rejections": next_consecutive,
    }


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
    gating_config: AponeurosisGatingConfig | None = None,
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

    gate_cfg = gating_config or AponeurosisGatingConfig(enabled=False)
    measurement_states = np.full((n, 4), np.nan, dtype=np.float64)
    accepted_measurement_states = np.full((n, 4), np.nan, dtype=np.float64)
    gating_r_scale = np.ones((n, 4), dtype=np.float64)
    rejected_endpoints = np.zeros((n, 4), dtype=bool)
    soft_downweighted_endpoints = np.zeros((n, 4), dtype=bool)
    line_rejected = np.zeros((n, 2), dtype=bool)
    line_soft_downweighted = np.zeros((n, 2), dtype=bool)
    gating_reasons = np.full((n, 2), "", dtype=object)
    consecutive_rejections = np.zeros((n, 2), dtype=np.int32)

    states_plus[0] = aponeurosis_state_from_lines(super_meas[0], deep_meas[0])
    measurement_states[0] = states_plus[0]
    accepted_measurement_states[0] = states_plus[0]
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
            measurement_states = measurement_states[:frame]
            accepted_measurement_states = accepted_measurement_states[:frame]
            gating_r_scale = gating_r_scale[:frame]
            rejected_endpoints = rejected_endpoints[:frame]
            soft_downweighted_endpoints = soft_downweighted_endpoints[:frame]
            line_rejected = line_rejected[:frame]
            line_soft_downweighted = line_soft_downweighted[:frame]
            gating_reasons = gating_reasons[:frame]
            consecutive_rejections = consecutive_rejections[:frame]
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
        gate = gate_aponeurosis_measurement_state(
            prior_state,
            measurement_state,
            accepted_measurement_states[frame - 1],
            states_plus[frame - 1],
            width=width,
            config=gate_cfg,
            consecutive_rejections=consecutive_rejections[frame - 1],
        )
        gated_measurement_state = np.asarray(gate["measurement_state"], dtype=np.float64)
        measurement_states[frame] = measurement_state
        accepted_measurement_states[frame] = measurement_state.copy()
        accepted_measurement_states[frame][gate["rejected_endpoints"]] = accepted_measurement_states[frame - 1][
            gate["rejected_endpoints"]
        ]
        gating_r_scale[frame] = np.asarray(gate["r_scale"], dtype=np.float64)
        rejected_endpoints[frame] = np.asarray(gate["rejected_endpoints"], dtype=bool)
        soft_downweighted_endpoints[frame] = np.asarray(gate["soft_downweighted_endpoints"], dtype=bool)
        line_rejected[frame] = np.asarray(gate["line_rejected"], dtype=bool)
        line_soft_downweighted[frame] = np.asarray(gate["line_soft_downweighted"], dtype=bool)
        gating_reasons[frame] = np.asarray(gate["reasons"], dtype=object)
        consecutive_rejections[frame] = np.asarray(gate["consecutive_rejections"], dtype=np.int32)
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
                gated_measurement_state[idx],
                r_values[idx] * gating_r_scale[frame, idx],
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
        "measurement_states": measurement_states,
        "accepted_measurement_states": accepted_measurement_states,
        "gating_r_scale": gating_r_scale,
        "rejected_endpoints": rejected_endpoints,
        "soft_downweighted_endpoints": soft_downweighted_endpoints,
        "line_rejected": line_rejected,
        "line_soft_downweighted": line_soft_downweighted,
        "gating_reasons": gating_reasons.astype(str),
        "consecutive_rejections": consecutive_rejections,
    }
