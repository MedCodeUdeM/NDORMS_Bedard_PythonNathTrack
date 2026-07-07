#!/usr/bin/env python3
"""Compare the updated Python tracker with a saved MATLAB UltraTimTrack run.

The comparison keeps the video, ROI JSON, frame alignment, seed settings, and
Python image-processing settings fixed.  It reports both final clinical
outputs and the newly exposed persistent-tracker / two-state Kalman internals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matlab-result",
        type=Path,
        default=PROJECT_ROOT / "data" / "matlab" / "slow_low_01_DOWN_tracked_Q=001.mat",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "UltraTimTrack_test.mp4",
    )
    parser.add_argument(
        "--roi-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "rois" / "UltraTimTrack_test_rois.json",
    )
    parser.add_argument(
        "--previous-npz",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "notebook74_revised_matlab_normal_adaptive_over_time"
            / "python_revised_exact_threshold_hough"
            / "UltraTimTrack_test"
            / "UltraTimTrack_test_strict_results.npz"
        ),
    )
    parser.add_argument(
        "--updated-npz",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "notebook75_persistent_tracker_matlab_python"
            / "python_updated_exact_same_inputs"
            / "UltraTimTrack_test"
            / "UltraTimTrack_test_strict_results.npz"
        ),
    )
    parser.add_argument(
        "--updated-metadata",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "notebook75_persistent_tracker_matlab_python"
            / "python_updated_exact_same_inputs"
            / "UltraTimTrack_test"
            / "UltraTimTrack_test_strict_metadata.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "notebook75_persistent_tracker_matlab_python",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def as_float1(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def lin_ccc(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if len(reference) < 2:
        return np.nan
    covariance = float(np.mean((reference - np.mean(reference)) * (estimate - np.mean(estimate))))
    denominator = float(
        np.var(reference) + np.var(estimate) + (np.mean(reference) - np.mean(estimate)) ** 2
    )
    return 2.0 * covariance / denominator if denominator > 0 else np.nan


def metrics(reference, estimate) -> dict[str, float | int]:
    reference = as_float1(reference)
    estimate = as_float1(estimate)
    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]
    valid = np.isfinite(reference) & np.isfinite(estimate)
    reference = reference[valid]
    estimate = estimate[valid]
    if not len(reference):
        return {
            "n": 0,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "max_abs": np.nan,
            "pearson_r": np.nan,
            "lins_ccc": np.nan,
        }
    delta = estimate - reference
    pearson = (
        float(np.corrcoef(reference, estimate)[0, 1])
        if len(reference) > 1 and np.std(reference) > 0 and np.std(estimate) > 0
        else np.nan
    )
    return {
        "n": int(len(reference)),
        "bias": float(np.mean(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "pearson_r": pearson,
        "lins_ccc": lin_ccc(reference, estimate),
    }


def object_series_to_2d(values, width: int = 2) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim == 2 and all(np.asarray(item).ndim == 0 for item in raw.reshape(-1)):
        numeric = np.asarray(raw, dtype=np.float64)
        out = np.full((numeric.shape[0], width), np.nan, dtype=np.float64)
        out[:, : min(width, numeric.shape[1])] = numeric[:, :width]
        return out
    rows = raw.reshape(-1)
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for idx, value in enumerate(rows):
        row = np.asarray(value, dtype=np.float64).reshape(-1)
        out[idx, : min(width, len(row))] = row[:width]
    return out


def matlab_segments(x_values, y_values) -> np.ndarray:
    x = object_series_to_2d(x_values, 2)
    y = object_series_to_2d(y_values, 2)
    # MATLAB saves [deep, superficial]; Python uses [superficial, deep].
    return np.column_stack([x[:, 1], y[:, 1], x[:, 0], y[:, 0]])


def segment_angle_deg(segments: np.ndarray) -> np.ndarray:
    segments = np.asarray(segments, dtype=np.float64)
    angle = np.rad2deg(
        np.arctan2(-(segments[:, 3] - segments[:, 1]), segments[:, 2] - segments[:, 0])
    )
    return (angle + 90.0) % 180.0 - 90.0


def segment_length_px(segments: np.ndarray) -> np.ndarray:
    segments = np.asarray(segments, dtype=np.float64)
    return np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])


def choose_python_offset(matlab_time: np.ndarray, python_time: np.ndarray) -> int:
    scores: list[tuple[float, int]] = []
    for offset in range(min(5, len(python_time))):
        n = min(len(matlab_time), len(python_time) - offset, 250)
        if n:
            scores.append((float(np.nanmedian(np.abs(python_time[offset : offset + n] - matlab_time[:n]))), offset))
    if not scores:
        raise ValueError("No overlapping MATLAB/Python time samples.")
    return min(scores)[1]


def aligned(values, offset: int, n: int) -> np.ndarray:
    arr = np.asarray(values)
    return arr[offset : offset + n]


def method_arrays(data: Mapping[str, np.ndarray], method: str) -> dict[str, np.ndarray]:
    prefix = "fixed_" if method == "normal" else ""
    return {
        "FL_mm": as_float1(data.get(f"{prefix}FL_mm", data["FL_mm"])),
        "ANG_deg": as_float1(data.get(f"{prefix}ANG_deg", data["ANG_deg"])),
        "PEN_deg": as_float1(data.get(f"{prefix}PEN_deg", data["PEN_deg"])),
    }


def save_roi_check(video: Path, roi_path: Path, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first video frame: {video}")
    rois = json.loads(roi_path.read_text())
    colors = {"superficial": (0, 220, 255), "deep": (255, 170, 0), "fascicle": (80, 220, 80)}
    for name, box in rois.items():
        x, y, w, h = [int(round(float(value))) for value in box]
        color = colors.get(name, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x + 4, max(18, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(str(output_path), frame)


def main() -> None:
    args = parse_args()
    for path in [args.matlab_result, args.video, args.roi_path, args.previous_npz, args.updated_npz]:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from scipy.io import loadmat

    mat = loadmat(args.matlab_result, simplify_cells=True)
    region = mat["Fdat"]["Region"]
    fascicle = region["Fascicle"]
    matlab = {
        "time_s": as_float1(region["Time"]),
        "FL_mm": as_float1(region["FL"]),
        "ANG_deg": as_float1(region["ANG"]),
        "PEN_deg": as_float1(region["PEN"]),
    }
    mat_raw_segment = matlab_segments(fascicle["fas_x_original"], fascicle["fas_y_original"])
    mat_final_segment = matlab_segments(fascicle["fas_x"], fascicle["fas_y"])
    mat_forward_state = object_series_to_2d(fascicle["X_plus"], 2)
    mat_smoothed_state = np.column_stack([mat_final_segment[:, 0], segment_angle_deg(mat_final_segment)])

    previous = load_npz(args.previous_npz)
    updated = load_npz(args.updated_npz)
    previous_offset = choose_python_offset(matlab["time_s"], as_float1(previous["time_s"]))
    updated_offset = choose_python_offset(matlab["time_s"], as_float1(updated["time_s"]))
    n = min(
        len(matlab["time_s"]),
        len(previous["time_s"]) - previous_offset,
        len(updated["time_s"]) - updated_offset,
    )

    rows: list[dict[str, float | int | str]] = []
    units = {"FL_mm": "mm", "ANG_deg": "deg", "PEN_deg": "deg"}
    for run_name, data, offset in [
        ("previous", previous, previous_offset),
        ("updated", updated, updated_offset),
    ]:
        for method in ["normal", "adaptive"]:
            values = method_arrays(data, method)
            for variable in ["FL_mm", "ANG_deg", "PEN_deg"]:
                row: dict[str, float | int | str] = {
                    "run": run_name,
                    "method": method,
                    "variable": variable,
                    "unit": units[variable],
                    "python_offset": offset,
                }
                row.update(metrics(matlab[variable][:n], aligned(values[variable], offset, n)))
                rows.append(row)
    agreement = pd.DataFrame(rows)
    agreement_path = args.output_dir / "agreement_metrics.csv"
    agreement.to_csv(agreement_path, index=False)

    previous_metrics = agreement[agreement["run"] == "previous"].set_index(["method", "variable"])
    updated_metrics = agreement[agreement["run"] == "updated"].set_index(["method", "variable"])
    improvement_rows = []
    for key in previous_metrics.index:
        before = previous_metrics.loc[key]
        after = updated_metrics.loc[key]
        improvement_rows.append(
            {
                "method": key[0],
                "variable": key[1],
                "unit": before["unit"],
                "before_bias": before["bias"],
                "after_bias": after["bias"],
                "before_mae": before["mae"],
                "after_mae": after["mae"],
                "mae_change_after_minus_before": after["mae"] - before["mae"],
                "before_rmse": before["rmse"],
                "after_rmse": after["rmse"],
                "rmse_change_after_minus_before": after["rmse"] - before["rmse"],
                "rmse_percent_change": 100.0 * (after["rmse"] - before["rmse"]) / before["rmse"],
                "before_lins_ccc": before["lins_ccc"],
                "after_lins_ccc": after["lins_ccc"],
                "ccc_change_after_minus_before": after["lins_ccc"] - before["lins_ccc"],
            }
        )
    improvement = pd.DataFrame(improvement_rows)
    improvement_path = args.output_dir / "before_after_improvement.csv"
    improvement.to_csv(improvement_path, index=False)

    sl = slice(updated_offset, updated_offset + n)
    py_raw_segment = np.asarray(updated["klt_prior_segments"], dtype=np.float64)[sl]
    py_forward_state = np.asarray(updated["fixed_forward_X_plus"], dtype=np.float64)[sl]
    py_smooth_state = np.asarray(updated["fixed_X_smooth"], dtype=np.float64)[sl]
    state_rows: list[dict[str, float | int | str]] = []
    state_comparisons = {
        "raw_klt_x_superficial": (mat_raw_segment[:n, 0], py_raw_segment[:, 0], "px"),
        "raw_klt_y_superficial": (mat_raw_segment[:n, 1], py_raw_segment[:, 1], "px"),
        "raw_klt_x_deep": (mat_raw_segment[:n, 2], py_raw_segment[:, 2], "px"),
        "raw_klt_y_deep": (mat_raw_segment[:n, 3], py_raw_segment[:, 3], "px"),
        "raw_klt_angle": (segment_angle_deg(mat_raw_segment[:n]), segment_angle_deg(py_raw_segment), "deg"),
        "raw_klt_segment_length": (segment_length_px(mat_raw_segment[:n]), segment_length_px(py_raw_segment), "px"),
        "forward_state_x": (mat_forward_state[:n, 0], py_forward_state[:, 0], "px"),
        "forward_state_alpha": (mat_forward_state[:n, 1], py_forward_state[:, 1], "deg"),
        "smoothed_state_x": (mat_smoothed_state[:n, 0], py_smooth_state[:, 0], "px"),
        "smoothed_state_alpha": (mat_smoothed_state[:n, 1], py_smooth_state[:, 1], "deg"),
    }
    for comparison, (reference, estimate, unit) in state_comparisons.items():
        row: dict[str, float | int | str] = {"comparison": comparison, "unit": unit}
        row.update(metrics(reference, estimate))
        state_rows.append(row)
    state_metrics = pd.DataFrame(state_rows)
    state_metrics_path = args.output_dir / "tracker_kalman_state_metrics.csv"
    state_metrics.to_csv(state_metrics_path, index=False)

    tracker_summary = {
        "n_python_frames": int(len(updated["frame"])),
        "n_matlab_rows": int(len(matlab["time_s"])),
        "python_offset": int(updated_offset),
        "matched_rows": int(n),
        "affine_success_fraction": float(np.mean(np.asarray(updated["klt_affine_ok"], dtype=bool))),
        "tracker_redetection_count_including_initialization": int(
            np.sum(np.asarray(updated["klt_tracker_redetected"], dtype=bool))
        ),
        "mean_tracker_found_fraction": float(
            np.nanmean(np.asarray(updated["klt_tracker_found_fraction"], dtype=np.float64))
        ),
        "median_tracker_point_count": float(
            np.nanmedian(np.asarray(updated["klt_points_count"], dtype=np.float64))
        ),
        "median_affine_inlier_count": float(
            np.nanmedian(np.asarray(updated["klt_inlier_count"], dtype=np.float64))
        ),
        "fixed_prediction_affine_fraction": float(
            np.mean(np.asarray(updated["fixed_prediction_used_affine"], dtype=bool))
        ),
    }
    tracker_summary_path = args.output_dir / "tracker_state_summary.json"
    tracker_summary_path.write_text(json.dumps(tracker_summary, indent=2), encoding="utf-8")

    time_s = matlab["time_s"][:n]
    old_normal = method_arrays(previous, "normal")
    new_normal = method_arrays(updated, "normal")
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex="col", gridspec_kw={"width_ratios": [2.2, 1.0]})
    labels = {"FL_mm": "Fascicle length (mm)", "ANG_deg": "Fascicle angle (deg)", "PEN_deg": "Pennation angle (deg)"}
    for row_idx, variable in enumerate(["FL_mm", "ANG_deg", "PEN_deg"]):
        reference = matlab[variable][:n]
        old_values = aligned(old_normal[variable], previous_offset, n)
        new_values = aligned(new_normal[variable], updated_offset, n)
        axes[row_idx, 0].plot(time_s, reference, color="black", linewidth=1.1, label="MATLAB")
        axes[row_idx, 0].plot(time_s, old_values, color="0.6", linewidth=0.9, label="previous Python")
        axes[row_idx, 0].plot(time_s, new_values, color="tab:orange", linewidth=1.0, label="updated Python")
        axes[row_idx, 0].set_ylabel(labels[variable])
        axes[row_idx, 1].axhline(0.0, color="black", linewidth=0.7)
        axes[row_idx, 1].plot(time_s, old_values - reference, color="0.6", linewidth=0.8, label="previous - MATLAB")
        axes[row_idx, 1].plot(time_s, new_values - reference, color="tab:orange", linewidth=0.9, label="updated - MATLAB")
        axes[row_idx, 1].set_ylabel("Difference")
        for axis in axes[row_idx]:
            axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", ncol=3, fontsize=8)
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Same video and ROI: MATLAB vs previous and updated fixed-R Python")
    fig.tight_layout()
    output_plot = args.output_dir / "matlab_python_before_after.png"
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    internal_plots = [
        (mat_raw_segment[:n, 0], py_raw_segment[:, 0], "Raw KLT superficial x (px)"),
        (segment_angle_deg(mat_raw_segment[:n]), segment_angle_deg(py_raw_segment), "Raw KLT angle (deg)"),
        (mat_forward_state[:n, 0], py_forward_state[:, 0], "Forward state x (px)"),
        (mat_forward_state[:n, 1], py_forward_state[:, 1], "Forward state alpha (deg)"),
    ]
    for axis, (reference, estimate, title) in zip(axes.flat, internal_plots):
        axis.plot(time_s, reference, color="black", linewidth=1.0, label="MATLAB")
        axis.plot(time_s, estimate, color="tab:orange", linewidth=0.9, label="updated Python")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Persistent tracker and fixed two-state Kalman parity")
    fig.tight_layout()
    state_plot = args.output_dir / "tracker_kalman_state_comparison.png"
    fig.savefig(state_plot, dpi=180)
    plt.close(fig)

    roi_plot = args.output_dir / "same_input_roi_check.png"
    save_roi_check(args.video, args.roi_path, roi_plot)

    primary = improvement[(improvement["method"] == "normal") & improvement["variable"].isin(["FL_mm", "ANG_deg"])]
    closer = bool((primary["rmse_change_after_minus_before"] < 0).all())
    conclusion = (
        "The updated fixed-R Python run is closer to MATLAB on both primary outputs."
        if closer
        else "The updated fixed-R Python run is not uniformly closer to MATLAB on both primary outputs."
    )
    normal = improvement[improvement["method"] == "normal"].set_index("variable")
    lines = [
        "# Updated MATLAB-vs-Python comparison",
        "",
        conclusion,
        "",
        f"- Same video: `{args.video}`",
        f"- Same ROI JSON: `{args.roi_path}` with `{json.loads(args.roi_path.read_text())}`",
        f"- MATLAB reference: `{args.matlab_result}`",
        f"- Matched rows: {n}; Python frame offset: {updated_offset}",
        "",
        "## Fixed-R (MATLAB-like) before/after",
        "",
    ]
    for variable, label in [("FL_mm", "Fascicle length"), ("ANG_deg", "Fascicle angle"), ("PEN_deg", "Pennation angle")]:
        row = normal.loc[variable]
        lines.append(
            f"- {label}: RMSE {row['before_rmse']:.4f} -> {row['after_rmse']:.4f} {row['unit']} "
            f"({row['rmse_percent_change']:+.1f}%); CCC {row['before_lins_ccc']:.4f} -> {row['after_lins_ccc']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Updated tracker state",
            "",
            f"- Affine success: {100.0 * tracker_summary['affine_success_fraction']:.1f}% of frames.",
            f"- Tracker redetections (including initialization): {tracker_summary['tracker_redetection_count_including_initialization']}.",
            f"- Mean tracker found fraction: {tracker_summary['mean_tracker_found_fraction']:.3f}.",
            f"- Fixed Kalman predictions using the saved affine: {100.0 * tracker_summary['fixed_prediction_affine_fraction']:.1f}% of frames.",
            "",
            f"- Final agreement table: `{agreement_path}`",
            f"- Before/after table: `{improvement_path}`",
            f"- Tracker/Kalman table: `{state_metrics_path}`",
            f"- Final output plot: `{output_plot}`",
            f"- Internal-state plot: `{state_plot}`",
            f"- ROI verification image: `{roi_plot}`",
        ]
    )
    summary_path = args.output_dir / "notebook75_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(conclusion)
    print(improvement.to_string(index=False))
    print("\nTracker state:")
    print(json.dumps(tracker_summary, indent=2))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
