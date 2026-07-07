#!/usr/bin/env python3
"""Diagnose where the persistent-tracker MATLAB/Python offset enters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.compare_updated_matlab_python as cmp
from ultrasound_tracker.ultratrack_klt import apply_affine_1b
from ultrasound_tracker.ultratimtrack_matlab_2state import (
    _normalized_segment_angles,
    _segment_angle_deg,
    matlab_scalar_kalman_update,
    reconstruct_fascicle_from_state,
)

MATLAB_RESULT = PROJECT_ROOT / "data" / "matlab" / "slow_low_01_DOWN_tracked_Q=001.mat"
VIDEO = PROJECT_ROOT / "data" / "raw" / "UltraTimTrack_test.mp4"
UPDATED_NPZ = (
    PROJECT_ROOT
    / "results"
    / "notebook75_persistent_tracker_matlab_python"
    / "python_updated_exact_same_inputs"
    / "UltraTimTrack_test"
    / "UltraTimTrack_test_strict_results.npz"
)
OUT = PROJECT_ROOT / "results" / "notebook76_persistent_tracker_offset_diagnostics"


def metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    valid = np.isfinite(ref) & np.isfinite(est)
    ref = ref[valid]
    est = est[valid]
    if not len(ref):
        return {
            "n": 0,
            "bias": np.nan,
            "rmse": np.nan,
            "demeaned_rmse": np.nan,
            "first_frame_delta": np.nan,
            "median_delta": np.nan,
            "std_delta": np.nan,
        }
    delta = est - ref
    demeaned = delta - float(np.mean(delta))
    return {
        "n": int(len(delta)),
        "bias": float(np.mean(delta)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "demeaned_rmse": float(np.sqrt(np.mean(demeaned * demeaned))),
        "first_frame_delta": float(delta[0]),
        "median_delta": float(np.median(delta)),
        "std_delta": float(np.std(delta)),
    }


def draw_line_1b(image: np.ndarray, segment_1b: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    line = np.asarray(segment_1b, dtype=np.float64).reshape(4)
    x1, y1, x2, y2 = np.rint(line - 1.0).astype(int)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def forward_x_measurement_variant(
    klt_segments: np.ndarray,
    timtrack_alpha_deg: np.ndarray,
    superficial_apo_lines: np.ndarray,
    deep_apo_lines: np.ndarray,
    prediction_affine_matrices: np.ndarray,
    *,
    q_parameter: float,
    x_measurement_variance: float,
    alpha_measurement_variance: float,
) -> np.ndarray:
    """Forward-only diagnostic variant using the current frame's x measurement."""

    klt = np.asarray(klt_segments, dtype=np.float64)
    tim_alpha = np.asarray(timtrack_alpha_deg, dtype=np.float64).reshape(-1)
    superficial = np.asarray(superficial_apo_lines, dtype=np.float64)
    deep = np.asarray(deep_apo_lines, dtype=np.float64)
    affines = np.asarray(prediction_affine_matrices, dtype=np.float64)

    n = len(klt)
    fixed_y = float(klt[0, 1])
    klt_alpha = _normalized_segment_angles(klt)

    states_plus = np.full((n, 2), np.nan, dtype=np.float64)
    p_plus = np.full((n, 2), np.nan, dtype=np.float64)
    states_plus[0] = [klt[0, 0], float(klt_alpha[0])]
    p_plus[0] = [0.0, 0.0]

    for frame in range(1, n):
        previous_corrected, _ = reconstruct_fascicle_from_state(
            states_plus[frame - 1, 0],
            states_plus[frame - 1, 1],
            superficial[frame - 1],
            deep[frame - 1],
            fixed_y,
        )
        use_affine_prediction = np.all(np.isfinite(affines[frame])) and np.all(np.isfinite(previous_corrected))
        if use_affine_prediction:
            predicted_segment = apply_affine_1b(previous_corrected, affines[frame])
            dx_sup = float(predicted_segment[0] - previous_corrected[0])
            dy_sup = float(predicted_segment[1] - previous_corrected[1])
            x_prior = float(predicted_segment[0])
            predicted_alpha = _segment_angle_deg(predicted_segment)
            previous_alpha = _segment_angle_deg(previous_corrected)
            d_alpha = abs(predicted_alpha) - abs(previous_alpha)
            alpha_prior = states_plus[frame - 1, 1] + d_alpha
        else:
            dx_sup = float(klt[frame, 0] - klt[frame - 1, 0])
            dy_sup = float(klt[frame, 1] - klt[frame - 1, 1])
            x_prior = float(states_plus[frame - 1, 0] + dx_sup)
            d_alpha = abs(float(klt_alpha[frame])) - abs(float(klt_alpha[frame - 1]))
            alpha_prior = states_plus[frame - 1, 1] + d_alpha

        dx = float(np.hypot(dx_sup, dy_sup))
        q_x = float(q_parameter) * dx * dx
        states_plus[frame, 0], p_plus[frame, 0], _, _ = matlab_scalar_kalman_update(
            x_prior,
            p_plus[frame - 1, 0],
            q_x,
            float(klt[frame, 0]),
            x_measurement_variance,
        )

        d_alpha_abs = abs(float(d_alpha))
        if d_alpha_abs < 0.005:
            d_alpha_abs = 0.0
        q_alpha = float(q_parameter) * d_alpha_abs * d_alpha_abs
        states_plus[frame, 1], p_plus[frame, 1], _, _ = matlab_scalar_kalman_update(
            alpha_prior,
            p_plus[frame - 1, 1],
            q_alpha,
            float(tim_alpha[frame]),
            alpha_measurement_variance,
        )

    return states_plus


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    mat = loadmat(MATLAB_RESULT, simplify_cells=True)
    py = np.load(UPDATED_NPZ, allow_pickle=True)

    region = mat["Fdat"]["Region"]
    fascicle = region["Fascicle"]
    matlab_time = cmp.as_float1(region["Time"])
    python_time = cmp.as_float1(py["time_s"])
    python_offset = cmp.choose_python_offset(matlab_time, python_time)
    n = min(len(matlab_time), len(python_time) - python_offset)
    sl = slice(python_offset, python_offset + n)

    mat_raw_segment = cmp.matlab_segments(fascicle["fas_x_original"], fascicle["fas_y_original"])[:n]
    mat_final_segment = cmp.matlab_segments(fascicle["fas_x"], fascicle["fas_y"])[:n]
    mat_forward_state = cmp.object_series_to_2d(fascicle["X_plus"], 2)[:n]

    py_seed_segment = np.asarray(py["selected_seed_segment"], dtype=np.float64).reshape(4)
    py_raw_segment = np.asarray(py["klt_prior_segments"], dtype=np.float64)[sl]
    py_timtrack_alpha = np.asarray(py["timtrack_alpha_deg"], dtype=np.float64)[sl]
    py_raw_timtrack_alpha = np.asarray(py["raw_timtrack_alpha_deg"], dtype=np.float64)[sl]
    py_predicted_segment = np.asarray(py["fixed_predicted_segments"], dtype=np.float64)[sl]
    py_forward_state = np.asarray(py["fixed_forward_X_plus"], dtype=np.float64)[sl]
    py_smoothed_state = np.asarray(py["fixed_X_smooth"], dtype=np.float64)[sl]

    stage_rows: list[dict[str, float | int | str]] = []

    def add_stage(stage: str, unit: str, reference: np.ndarray, estimate: np.ndarray) -> None:
        row: dict[str, float | int | str] = {"stage": stage, "unit": unit}
        row.update(metrics(reference, estimate))
        stage_rows.append(row)

    add_stage("seed_x_sup_vs_matlab_raw_frame0", "px", mat_raw_segment[:1, 0], py_seed_segment[None, 0])
    add_stage("seed_angle_vs_matlab_raw_frame0", "deg", cmp.segment_angle_deg(mat_raw_segment[:1]), cmp.segment_angle_deg(py_seed_segment[None, :]))
    add_stage("raw_klt_x_sup_vs_matlab_raw", "px", mat_raw_segment[:, 0], py_raw_segment[:, 0])
    add_stage("raw_klt_angle_vs_matlab_raw", "deg", cmp.segment_angle_deg(mat_raw_segment), cmp.segment_angle_deg(py_raw_segment))
    add_stage("selected_timtrack_alpha_vs_matlab_forward_alpha", "deg", mat_forward_state[:, 1], py_timtrack_alpha)
    add_stage("raw_timtrack_alpha_vs_matlab_forward_alpha", "deg", mat_forward_state[:, 1], py_raw_timtrack_alpha)
    add_stage(
        "predicted_segment_angle_vs_matlab_forward_alpha",
        "deg",
        mat_forward_state[:, 1],
        cmp.segment_angle_deg(py_predicted_segment),
    )
    add_stage("forward_state_x_vs_matlab_forward_x", "px", mat_forward_state[:, 0], py_forward_state[:, 0])
    add_stage("forward_state_alpha_vs_matlab_forward_alpha", "deg", mat_forward_state[:, 1], py_forward_state[:, 1])
    add_stage(
        "smoothed_state_alpha_vs_matlab_final_alpha",
        "deg",
        cmp.segment_angle_deg(mat_final_segment),
        py_smoothed_state[:, 1],
    )

    stage_summary = pd.DataFrame(stage_rows)
    stage_summary_path = OUT / "stage_offset_summary.csv"
    stage_summary.to_csv(stage_summary_path, index=False)

    mat_seed0 = mat_raw_segment[0]
    seed_rows = [
        {"quantity": "x_sup_px", "matlab": float(mat_seed0[0]), "python": float(py_seed_segment[0]), "delta_python_minus_matlab": float(py_seed_segment[0] - mat_seed0[0])},
        {"quantity": "y_sup_px", "matlab": float(mat_seed0[1]), "python": float(py_seed_segment[1]), "delta_python_minus_matlab": float(py_seed_segment[1] - mat_seed0[1])},
        {"quantity": "x_deep_px", "matlab": float(mat_seed0[2]), "python": float(py_seed_segment[2]), "delta_python_minus_matlab": float(py_seed_segment[2] - mat_seed0[2])},
        {"quantity": "y_deep_px", "matlab": float(mat_seed0[3]), "python": float(py_seed_segment[3]), "delta_python_minus_matlab": float(py_seed_segment[3] - mat_seed0[3])},
        {
            "quantity": "angle_deg",
            "matlab": float(cmp.segment_angle_deg(mat_raw_segment[:1])[0]),
            "python": float(cmp.segment_angle_deg(py_seed_segment[None, :])[0]),
            "delta_python_minus_matlab": float(cmp.segment_angle_deg(py_seed_segment[None, :])[0] - cmp.segment_angle_deg(mat_raw_segment[:1])[0]),
        },
        {
            "quantity": "length_px",
            "matlab": float(cmp.segment_length_px(mat_raw_segment[:1])[0]),
            "python": float(cmp.segment_length_px(py_seed_segment[None, :])[0]),
            "delta_python_minus_matlab": float(cmp.segment_length_px(py_seed_segment[None, :])[0] - cmp.segment_length_px(mat_raw_segment[:1])[0]),
        },
        {
            "quantity": "seed_alpha_deg",
            "matlab": float(cmp.segment_angle_deg(mat_raw_segment[:1])[0]),
            "python": float(np.asarray(py["selected_seed_alpha_deg"], dtype=np.float64).reshape(-1)[0]),
            "delta_python_minus_matlab": float(np.asarray(py["selected_seed_alpha_deg"], dtype=np.float64).reshape(-1)[0] - cmp.segment_angle_deg(mat_raw_segment[:1])[0]),
        },
    ]
    seed_frame0 = pd.DataFrame(seed_rows)
    seed_frame0_path = OUT / "seed_frame0_offset.csv"
    seed_frame0.to_csv(seed_frame0_path, index=False)

    alpha_measurement_variance = float(np.asarray(mat["Fdat"]["R"], dtype=np.float64).reshape(-1)[0])
    x_variant_forward_state = forward_x_measurement_variant(
        np.asarray(py["klt_prior_segments"], dtype=np.float64),
        np.asarray(py["timtrack_alpha_deg"], dtype=np.float64),
        np.asarray(py["sup_apo_lines"], dtype=np.float64),
        np.asarray(py["deep_apo_lines"], dtype=np.float64),
        np.asarray(py["klt_affine_matrices"], dtype=np.float64),
        q_parameter=0.01,
        x_measurement_variance=100.0,
        alpha_measurement_variance=alpha_measurement_variance,
    )[sl]
    variant_rows = []
    for mode, estimate in [
        ("current_initial_x_measurement", py_forward_state[:, 0]),
        ("diagnostic_per_frame_x_measurement", x_variant_forward_state[:, 0]),
    ]:
        row: dict[str, float | int | str] = {"mode": mode}
        row.update(metrics(mat_forward_state[:, 0], estimate))
        variant_rows.append(row)
    x_variant_summary = pd.DataFrame(variant_rows)
    x_variant_summary_path = OUT / "x_measurement_variant_summary.csv"
    x_variant_summary.to_csv(x_variant_summary_path, index=False)

    time_s = matlab_time[:n]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].plot(time_s, py_raw_segment[:, 0] - mat_raw_segment[:, 0], label="raw KLT x_sup", color="tab:orange", linewidth=1.0)
    axes[0].plot(time_s, py_forward_state[:, 0] - mat_forward_state[:, 0], label="forward state x", color="tab:red", linewidth=1.0)
    axes[0].plot(time_s, x_variant_forward_state[:, 0] - mat_forward_state[:, 0], label="forward x (per-frame x meas)", color="tab:green", linewidth=1.0)
    axes[0].set_title("X offset by stage")
    axes[0].set_ylabel("Python - MATLAB (px)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(time_s, cmp.segment_angle_deg(py_raw_segment) - cmp.segment_angle_deg(mat_raw_segment), label="raw KLT angle", color="tab:orange", linewidth=1.0)
    axes[1].plot(time_s, py_timtrack_alpha - mat_forward_state[:, 1], label="selected TimTrack alpha", color="tab:blue", linewidth=1.0)
    axes[1].plot(time_s, py_forward_state[:, 1] - mat_forward_state[:, 1], label="forward state alpha", color="tab:red", linewidth=1.0)
    axes[1].plot(time_s, py_smoothed_state[:, 1] - cmp.segment_angle_deg(mat_final_segment), label="smoothed alpha vs final", color="tab:purple", linewidth=1.0)
    axes[1].set_title("Angle offset by stage")
    axes[1].set_ylabel("Python - MATLAB (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    for axis in axes:
        axis.set_xlabel("Time (s)")
    fig.suptitle("Notebook 76 — where the persistent-tracker offset enters")
    fig.tight_layout()
    stage_plot_path = OUT / "offset_stage_differences.png"
    fig.savefig(stage_plot_path, dpi=180)
    plt.close(fig)

    cap = cv2.VideoCapture(str(VIDEO))
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {VIDEO}")
    if frame0.ndim == 2:
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_GRAY2BGR)
    overlay = frame0.copy()
    draw_line_1b(overlay, mat_seed0, (255, 255, 255), 3)
    draw_line_1b(overlay, py_seed_segment, (0, 165, 255), 2)
    cv2.putText(overlay, "MATLAB raw frame 0", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, "Python selected seed", (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    frame0_overlay_path = OUT / "frame0_seed_overlay.png"
    cv2.imwrite(str(frame0_overlay_path), overlay)

    raw_x_row = stage_summary.loc[stage_summary["stage"] == "raw_klt_x_sup_vs_matlab_raw"].iloc[0]
    raw_angle_row = stage_summary.loc[stage_summary["stage"] == "raw_klt_angle_vs_matlab_raw"].iloc[0]
    timtrack_alpha_row = stage_summary.loc[stage_summary["stage"] == "selected_timtrack_alpha_vs_matlab_forward_alpha"].iloc[0]
    forward_x_row = stage_summary.loc[stage_summary["stage"] == "forward_state_x_vs_matlab_forward_x"].iloc[0]
    forward_alpha_row = stage_summary.loc[stage_summary["stage"] == "forward_state_alpha_vs_matlab_forward_alpha"].iloc[0]
    smoothed_alpha_row = stage_summary.loc[stage_summary["stage"] == "smoothed_state_alpha_vs_matlab_final_alpha"].iloc[0]
    current_x_variant = x_variant_summary.loc[x_variant_summary["mode"] == "current_initial_x_measurement"].iloc[0]
    new_x_variant = x_variant_summary.loc[x_variant_summary["mode"] == "diagnostic_per_frame_x_measurement"].iloc[0]

    summary_lines = [
        "# Notebook 76 — persistent-tracker offset diagnosis",
        "",
        f"Aligned {n} MATLAB/Python samples with Python offset {python_offset}.",
        "",
        "## Main findings",
        "",
        f"- The raw tracker offset is already present at frame 0 in the selected seed. Relative to MATLAB raw frame 0, the Python seed is {seed_frame0.loc[seed_frame0['quantity'] == 'x_sup_px', 'delta_python_minus_matlab'].iloc[0]:+.2f} px on superficial x, {seed_frame0.loc[seed_frame0['quantity'] == 'x_deep_px', 'delta_python_minus_matlab'].iloc[0]:+.2f} px on deep x, and {seed_frame0.loc[seed_frame0['quantity'] == 'angle_deg', 'delta_python_minus_matlab'].iloc[0]:+.2f} deg on angle.",
        f"- Over time, raw KLT shape is close after removing the mean bias: x_sup bias {raw_x_row['bias']:+.2f} px with demeaned RMSE {raw_x_row['demeaned_rmse']:.2f} px; angle bias {raw_angle_row['bias']:+.2f} deg with demeaned RMSE {raw_angle_row['demeaned_rmse']:.2f} deg.",
        f"- The forward x-state offset is consistent with the current Kalman x update anchoring to the initial x measurement. In a diagnostic variant that uses the current frame's x measurement instead, forward x bias drops from {current_x_variant['bias']:+.2f} px to {new_x_variant['bias']:+.2f} px.",
        f"- The angle offset changes sign at the TimTrack measurement stage: raw KLT angle bias is {raw_angle_row['bias']:+.2f} deg, selected TimTrack alpha bias is {timtrack_alpha_row['bias']:+.2f} deg, and forward alpha bias is {forward_alpha_row['bias']:+.2f} deg.",
        f"- The final smoothed alpha remains mostly a constant offset rather than a shape mismatch: bias {smoothed_alpha_row['bias']:+.2f} deg, demeaned RMSE {smoothed_alpha_row['demeaned_rmse']:.2f} deg.",
        "",
        "## Interpretation",
        "",
        "- There are two offset sources, and they happen at different stages.",
        "- First, the autonomous seed already starts from a different fascicle segment than MATLAB, so the persistent KLT tracker inherits a near-constant spatial shift from frame 0.",
        "- Second, the two-state Kalman x branch appears to preserve or amplify x bias because it is effectively tied to the initial x measurement, while the alpha branch inherits the positive TimTrack alpha baseline.",
        "- That means the curves are genuinely very close in shape; the remaining difference is mostly stage-wise baseline offset, not runaway tracking error.",
        "",
        f"- Stage summary CSV: `{stage_summary_path}`",
        f"- Seed/frame-0 CSV: `{seed_frame0_path}`",
        f"- X-measurement variant CSV: `{x_variant_summary_path}`",
        f"- Difference plot: `{stage_plot_path}`",
        f"- Frame-0 seed overlay: `{frame0_overlay_path}`",
    ]
    summary_path = OUT / "notebook76_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
