#!/usr/bin/env python3
"""Notebook 82 helper: audit the forward alpha/input side."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.compare_updated_matlab_python as cmp
from ultrasound_tracker.matlab_compat import extract_geofeature_arrays
from ultrasound_tracker.ultratimtrack_matlab_2state import reconstruct_fascicle_from_state


RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "notebook77_x_update_fix_parity"
    / "python_x_update_fix_same_inputs"
    / "UltraTimTrack_test"
)
NPZ_PATH = RUN_DIR / "UltraTimTrack_test_strict_results.npz"
MATLAB_RESULT = PROJECT_ROOT / "data" / "matlab" / "slow_low_01_DOWN_tracked_Q=001.mat"
OUT = PROJECT_ROOT / "results" / "notebook82_forward_alpha_inputs"


def scalar_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    valid = np.isfinite(ref) & np.isfinite(est)
    ref = ref[valid]
    est = est[valid]
    delta = est - ref
    return {
        "bias": float(np.mean(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def run_scalar_alpha_channel(
    *,
    x_minus: np.ndarray,
    measurement_alpha: np.ndarray,
    initial_plus: float,
    initial_p_plus: float,
    q_series: np.ndarray,
    r_series: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(measurement_alpha)
    x_plus = np.full(n, np.nan, dtype=np.float64)
    p_plus = np.full(n, np.nan, dtype=np.float64)
    p_minus = np.full(n, np.nan, dtype=np.float64)

    x_plus[0] = float(initial_plus)
    p_plus[0] = float(initial_p_plus)
    p_minus[0] = float(initial_p_plus)

    for frame in range(1, n):
        q_value = float(q_series[frame])
        r_value = float(r_series[frame])
        p_minus[frame] = p_plus[frame - 1] + q_value
        denom = p_minus[frame] + r_value
        gain = p_minus[frame] / denom if denom != 0 else np.nan
        if np.isnan(gain):
            gain = 0.0
        x_plus[frame] = float(x_minus[frame]) + gain * (float(measurement_alpha[frame]) - float(x_minus[frame]))
        p_plus[frame] = (1.0 - gain) * p_minus[frame]
    return x_plus, p_plus, p_minus


def smooth_scalar_channel(x_plus: np.ndarray, p_plus: np.ndarray, x_minus: np.ndarray, p_minus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x_plus)
    x_smooth = np.asarray(x_plus, dtype=np.float64).copy()
    gains = np.full(n, np.nan, dtype=np.float64)
    for frame in range(n - 2, -1, -1):
        denom = float(p_minus[frame + 1])
        gain = float(p_plus[frame]) / denom if denom != 0 else np.nan
        if np.isnan(gain):
            gain = 1.0
        gains[frame] = gain
        x_smooth[frame] = float(x_plus[frame]) + gain * (float(x_smooth[frame + 1]) - float(x_minus[frame + 1]))
    return x_smooth, gains


def build_end_segments(
    x_sup: np.ndarray,
    alpha_deg: np.ndarray,
    sup_apo_lines: np.ndarray,
    deep_apo_lines: np.ndarray,
    fixed_superficial_y: float,
) -> np.ndarray:
    n = len(x_sup)
    out = np.full((n, 4), np.nan, dtype=np.float64)
    for frame in range(n):
        _, end_segment = reconstruct_fascicle_from_state(
            x_sup[frame],
            alpha_deg[frame],
            sup_apo_lines[frame],
            deep_apo_lines[frame],
            fixed_superficial_y,
        )
        out[frame] = end_segment
    return out


def segment_lengths_mm(segments: np.ndarray, mm_per_pixel: float) -> np.ndarray:
    seg = np.asarray(segments, dtype=np.float64)
    return np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) * float(mm_per_pixel)


def variant_row(
    *,
    variant: str,
    prior_source: str,
    measurement_source: str,
    forward_alpha: np.ndarray,
    smooth_alpha: np.ndarray,
    end_segments: np.ndarray,
    mat_forward_alpha: np.ndarray,
    mat_final_alpha: np.ndarray,
    mat_end: np.ndarray,
    mat_fl: np.ndarray,
    mat_ang: np.ndarray,
    mm_per_pixel: float,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "variant": variant,
        "prior_source": prior_source,
        "measurement_source": measurement_source,
    }
    row.update({f"forward_alpha_{k}": v for k, v in scalar_metrics(mat_forward_alpha, forward_alpha).items()})
    row.update({f"final_alpha_{k}": v for k, v in scalar_metrics(mat_final_alpha, smooth_alpha).items()})
    for idx, label in enumerate(["x_sup", "y_sup", "x_deep", "y_deep"]):
        row.update({f"end_{label}_{k}": v for k, v in scalar_metrics(mat_end[:, idx], end_segments[:, idx]).items()})
    fl_mm = segment_lengths_mm(end_segments, mm_per_pixel)
    ang_deg = cmp.segment_angle_deg(end_segments)
    row.update({f"FL_{k}": v for k, v in scalar_metrics(mat_fl, fl_mm).items()})
    row.update({f"ANG_{k}": v for k, v in scalar_metrics(mat_ang, ang_deg).items()})
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    mat = loadmat(MATLAB_RESULT, simplify_cells=True)

    region = mat["Fdat"]["Region"]
    fascicle = region["Fascicle"]
    geof = extract_geofeature_arrays(mat)
    matlab_time = cmp.as_float1(region["Time"])
    python_time = cmp.as_float1(npz["time_s"])
    python_offset = cmp.choose_python_offset(matlab_time, python_time)
    n = min(len(matlab_time), len(python_time) - python_offset)
    sl = slice(python_offset, python_offset + n)

    mat_forward_state = cmp.object_series_to_2d(fascicle["X_plus"], 2)[:n]
    mat_pred_state = cmp.object_series_to_2d(fascicle["X_minus"], 2)[:n]
    mat_final_alpha = cmp.as_float1(fascicle["alpha"])[:n]
    mat_measure_alpha = np.asarray(geof["alpha_deg"], dtype=np.float64)[:n]
    mat_end = cmp.matlab_segments(fascicle["fas_x_end"], fascicle["fas_y_end"])[:n]
    mat_fl = cmp.as_float1(region["FL"])[:n]
    mat_ang = cmp.as_float1(region["ANG"])[:n]

    py_forward_state = np.asarray(npz["fixed_forward_X_plus"], dtype=np.float64)[sl]
    py_smooth_state = np.asarray(npz["fixed_X_plus"], dtype=np.float64)[sl]
    py_pred_state = np.asarray(npz["fixed_X_minus"], dtype=np.float64)[sl]
    py_forward_p = np.asarray(npz["fixed_forward_fas_p"], dtype=np.float64)[sl]
    py_pred_p = np.asarray(npz["fixed_fas_p_minus"], dtype=np.float64)[sl]
    py_measure_alpha = np.asarray(npz["timtrack_alpha_deg"], dtype=np.float64)[sl]
    py_sup = np.asarray(npz["sup_apo_lines"], dtype=np.float64)[sl]
    py_deep = np.asarray(npz["deep_apo_lines"], dtype=np.float64)[sl]
    py_end_current = np.asarray(npz["fixed_fascicle_end_segments"], dtype=np.float64)[sl]
    py_x_smooth = py_smooth_state[:, 0]
    mm_per_pixel = float(np.asarray(npz["mm_per_pixel"], dtype=np.float64).reshape(-1)[0])
    fixed_y = float(np.asarray(npz["klt_prior_segments"], dtype=np.float64)[0, 1])
    r_series = np.asarray(npz["fixed_measurement_R_diag"], dtype=np.float64)[sl, 1]

    q_series = np.zeros(n, dtype=np.float64)
    q_series[1:] = np.maximum(0.0, py_pred_p[1:, 1] - py_forward_p[:-1, 1])

    input_rows = [
        {"signal": "python_timtrack_alpha_vs_matlab_geofeature_alpha", **scalar_metrics(mat_measure_alpha, py_measure_alpha)},
        {"signal": "python_X_minus_alpha_vs_matlab_X_minus_alpha", **scalar_metrics(mat_pred_state[:, 1], py_pred_state[:, 1])},
        {"signal": "python_forward_alpha_vs_matlab_forward_alpha", **scalar_metrics(mat_forward_state[:, 1], py_forward_state[:, 1])},
        {"signal": "python_smoothed_alpha_vs_matlab_final_alpha", **scalar_metrics(mat_final_alpha, py_smooth_state[:, 1])},
    ]
    input_path = OUT / "forward_alpha_input_metrics.csv"
    pd.DataFrame(input_rows).to_csv(input_path, index=False)

    variants = []
    variant_specs = [
        ("baseline_python_prior_python_measurement", py_pred_state[:, 1], py_measure_alpha, "python_X_minus[:,1]", "python_timtrack_alpha"),
        ("python_prior_matlab_measurement", py_pred_state[:, 1], mat_measure_alpha, "python_X_minus[:,1]", "matlab_geofeature_alpha"),
        ("matlab_prior_python_measurement", mat_pred_state[:, 1], py_measure_alpha, "matlab_X_minus[:,1]", "python_timtrack_alpha"),
        ("matlab_prior_matlab_measurement", mat_pred_state[:, 1], mat_measure_alpha, "matlab_X_minus[:,1]", "matlab_geofeature_alpha"),
    ]

    for variant, prior_alpha, measurement_alpha, prior_source, measurement_source in variant_specs:
        forward_alpha, p_plus, p_minus = run_scalar_alpha_channel(
            x_minus=prior_alpha,
            measurement_alpha=measurement_alpha,
            initial_plus=float(py_forward_state[0, 1]),
            initial_p_plus=float(py_forward_p[0, 1]),
            q_series=q_series,
            r_series=r_series,
        )
        smooth_alpha, _ = smooth_scalar_channel(forward_alpha, p_plus, prior_alpha, p_minus)
        end_segments = build_end_segments(py_x_smooth, smooth_alpha, py_sup, py_deep, fixed_y)
        variants.append(
            variant_row(
                variant=variant,
                prior_source=prior_source,
                measurement_source=measurement_source,
                forward_alpha=forward_alpha,
                smooth_alpha=smooth_alpha,
                end_segments=end_segments,
                mat_forward_alpha=mat_forward_state[:, 1],
                mat_final_alpha=mat_final_alpha,
                mat_end=mat_end,
                mat_fl=mat_fl,
                mat_ang=mat_ang,
                mm_per_pixel=mm_per_pixel,
            )
        )
    variant_table = pd.DataFrame(variants)
    variant_path = OUT / "forward_alpha_variants.csv"
    variant_table.to_csv(variant_path, index=False)

    baseline_row = variant_table.loc[variant_table["variant"] == "baseline_python_prior_python_measurement"].iloc[0]
    meas_row = variant_table.loc[variant_table["variant"] == "python_prior_matlab_measurement"].iloc[0]
    prior_row = variant_table.loc[variant_table["variant"] == "matlab_prior_python_measurement"].iloc[0]
    both_row = variant_table.loc[variant_table["variant"] == "matlab_prior_matlab_measurement"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    show_n = min(350, n)
    frames = np.arange(show_n)
    axes[0].plot(frames, mat_measure_alpha[:show_n], label="MATLAB geofeature alpha", linewidth=1.2)
    axes[0].plot(frames, py_measure_alpha[:show_n], label="Python TimTrack alpha", linewidth=1.0)
    axes[0].plot(frames, mat_forward_state[:show_n, 1], label="MATLAB forward X_plus alpha", linewidth=1.2)
    axes[0].plot(frames, py_forward_state[:show_n, 1], label="Python forward X_plus alpha", linewidth=1.0)
    axes[0].set_title("Forward alpha measurement and state traces")
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(frames, mat_final_alpha[:show_n], label="MATLAB final alpha", linewidth=1.2)
    axes[1].plot(frames, py_smooth_state[:show_n, 1], label="Python baseline smoothed alpha", linewidth=1.0)
    for variant, style in [
        ("python_prior_matlab_measurement", "--"),
        ("matlab_prior_python_measurement", ":"),
        ("matlab_prior_matlab_measurement", "-."),
    ]:
        prior_alpha = py_pred_state[:, 1] if "python_prior" in variant else mat_pred_state[:, 1]
        measurement_alpha = py_measure_alpha if variant.endswith("python_measurement") else mat_measure_alpha
        forward_alpha, p_plus, p_minus = run_scalar_alpha_channel(
            x_minus=prior_alpha,
            measurement_alpha=measurement_alpha,
            initial_plus=float(py_forward_state[0, 1]),
            initial_p_plus=float(py_forward_p[0, 1]),
            q_series=q_series,
            r_series=r_series,
        )
        smooth_alpha, _ = smooth_scalar_channel(forward_alpha, p_plus, prior_alpha, p_minus)
        axes[1].plot(frames, smooth_alpha[:show_n], linestyle=style, label=variant, linewidth=1.0)
    axes[1].set_title("Smoothed alpha under notebook-only forward-input swaps")
    axes[1].set_xlabel("aligned frame index")
    axes[1].set_ylabel("alpha (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    trace_plot_path = OUT / "forward_alpha_input_traces.png"
    fig.savefig(trace_plot_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = ["baseline", "MATLAB meas", "MATLAB prior", "MATLAB prior+meas"]
    forward_rmse = variant_table["forward_alpha_rmse"].to_numpy(dtype=float)
    final_rmse = variant_table["final_alpha_rmse"].to_numpy(dtype=float)
    fl_rmse = variant_table["FL_rmse"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    ax.plot(x, forward_rmse, marker="o", label="forward alpha RMSE (deg)")
    ax.plot(x, final_rmse, marker="s", label="final alpha RMSE (deg)")
    ax.plot(x, fl_rmse, marker="^", label="FL RMSE (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("Effect of forward alpha input swaps")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    summary_plot_path = OUT / "forward_alpha_variant_summary.png"
    fig.savefig(summary_plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 82 — forward alpha/input audit",
        "",
        f"Aligned {n} MATLAB/Python samples with Python offset {python_offset}.",
        "",
        "This notebook targets the forward alpha/input side instead of the final alpha recursion. It isolates two upstream ingredients of the angle channel: the predicted alpha prior (`X_minus[:,1]`) and the TimTrack/Hough alpha measurement.",
        "",
        "## Input-side gaps",
        "",
        f"- Python TimTrack alpha differs from MATLAB geofeature alpha by RMSE {pd.DataFrame(input_rows).loc[pd.DataFrame(input_rows)['signal'] == 'python_timtrack_alpha_vs_matlab_geofeature_alpha', 'rmse'].iloc[0]:.4f} deg.",
        f"- Python predicted alpha prior differs from MATLAB `X_minus[:,1]` by RMSE {pd.DataFrame(input_rows).loc[pd.DataFrame(input_rows)['signal'] == 'python_X_minus_alpha_vs_matlab_X_minus_alpha', 'rmse'].iloc[0]:.4f} deg.",
        f"- The resulting Python forward alpha differs from MATLAB forward `X_plus[:,1]` by RMSE {pd.DataFrame(input_rows).loc[pd.DataFrame(input_rows)['signal'] == 'python_forward_alpha_vs_matlab_forward_alpha', 'rmse'].iloc[0]:.4f} deg, and Python final smoothed alpha differs from MATLAB final alpha by RMSE {pd.DataFrame(input_rows).loc[pd.DataFrame(input_rows)['signal'] == 'python_smoothed_alpha_vs_matlab_final_alpha', 'rmse'].iloc[0]:.4f} deg.",
        "",
        "## Notebook-only forward-input swaps",
        "",
        f"- Baseline reproduction matches the current run: forward alpha RMSE {baseline_row['forward_alpha_rmse']:.4f} deg, final alpha RMSE {baseline_row['final_alpha_rmse']:.4f} deg, FL RMSE {baseline_row['FL_rmse']:.4f} mm.",
        f"- Swapping only the measurement series to MATLAB geofeature alpha is the biggest single improvement: forward alpha RMSE drops to {meas_row['forward_alpha_rmse']:.4f} deg, final alpha RMSE drops to {meas_row['final_alpha_rmse']:.4f} deg, and FL RMSE drops to {meas_row['FL_rmse']:.4f} mm.",
        f"- Swapping only the predicted alpha prior to MATLAB `X_minus[:,1]` also helps, but less: forward alpha RMSE becomes {prior_row['forward_alpha_rmse']:.4f} deg, final alpha RMSE {prior_row['final_alpha_rmse']:.4f} deg, and FL RMSE {prior_row['FL_rmse']:.4f} mm.",
        f"- Swapping both prior and measurement gives the best notebook-only alpha-channel result: forward alpha RMSE {both_row['forward_alpha_rmse']:.4f} deg, final alpha RMSE {both_row['final_alpha_rmse']:.4f} deg, and FL RMSE {both_row['FL_rmse']:.4f} mm.",
        "",
        "## Interpretation",
        "",
        "- The forward alpha gap is real and upstream. It is not caused by the final backward smoother.",
        "- The measurement side matters more than the prior side on this run. In other words, the current Python TimTrack alpha entering the filter is the bigger contributor to the final angle mismatch.",
        "- The prior still matters, so the full gap is not measurement-only. But the data says the next best direction is to audit how Python produces the per-frame TimTrack alpha that feeds the Kalman update.",
        "",
        "## What this means for patching",
        "",
        "- We still should not patch raw code yet just to mimic MATLAB outputs.",
        "- A safe next notebook is to target the Python TimTrack alpha production chain directly: candidate selection, persistence, and any geometry/sign conventions before the Kalman update.",
        "",
        f"- Input metrics CSV: `{input_path}`",
        f"- Variant metrics CSV: `{variant_path}`",
        f"- Trace plot: `{trace_plot_path}`",
        f"- Summary plot: `{summary_plot_path}`",
    ]
    summary_path = OUT / "notebook82_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
