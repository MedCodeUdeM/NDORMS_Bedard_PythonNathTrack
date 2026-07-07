#!/usr/bin/env python3
"""Notebook 83 helper: audit the TimTrack alpha measurement chain."""

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
OUT = PROJECT_ROOT / "results" / "notebook83_timtrack_alpha_measurement_chain"


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
        p_minus[frame] = p_plus[frame - 1] + float(q_series[frame])
        denom = p_minus[frame] + float(r_series[frame])
        gain = p_minus[frame] / denom if denom != 0 else np.nan
        if np.isnan(gain):
            gain = 0.0
        x_plus[frame] = float(x_minus[frame]) + gain * (float(measurement_alpha[frame]) - float(x_minus[frame]))
        p_plus[frame] = (1.0 - gain) * p_minus[frame]
    return x_plus, p_plus, p_minus


def smooth_scalar_channel(x_plus: np.ndarray, p_plus: np.ndarray, x_minus: np.ndarray, p_minus: np.ndarray) -> np.ndarray:
    n = len(x_plus)
    x_smooth = np.asarray(x_plus, dtype=np.float64).copy()
    for frame in range(n - 2, -1, -1):
        denom = float(p_minus[frame + 1])
        gain = float(p_plus[frame]) / denom if denom != 0 else np.nan
        if np.isnan(gain):
            gain = 1.0
        x_smooth[frame] = float(x_plus[frame]) + gain * (float(x_smooth[frame + 1]) - float(x_minus[frame + 1]))
    return x_smooth


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

    mat_measure_alpha = np.asarray(geof["alpha_deg"], dtype=np.float64)[:n]
    mat_final_alpha = cmp.as_float1(fascicle["alpha"])[:n]
    mat_end = cmp.matlab_segments(fascicle["fas_x_end"], fascicle["fas_y_end"])[:n]
    mat_fl = cmp.as_float1(region["FL"])[:n]
    mat_ang = cmp.as_float1(region["ANG"])[:n]

    py_prior_alpha = np.asarray(npz["fixed_X_minus"], dtype=np.float64)[sl, 1]
    py_forward_alpha = np.asarray(npz["fixed_forward_X_plus"], dtype=np.float64)[sl, 1]
    py_final_alpha = np.asarray(npz["fixed_X_plus"], dtype=np.float64)[sl, 1]
    py_forward_p = np.asarray(npz["fixed_forward_fas_p"], dtype=np.float64)[sl, 1]
    py_pred_p = np.asarray(npz["fixed_fas_p_minus"], dtype=np.float64)[sl, 1]
    py_selected_alpha = np.asarray(npz["timtrack_alpha_deg"], dtype=np.float64)[sl]
    py_raw_alpha = np.asarray(npz["raw_timtrack_alpha_deg"], dtype=np.float64)[sl]
    py_x_smooth = np.asarray(npz["fixed_X_plus"], dtype=np.float64)[sl, 0]
    py_sup = np.asarray(npz["sup_apo_lines"], dtype=np.float64)[sl]
    py_deep = np.asarray(npz["deep_apo_lines"], dtype=np.float64)[sl]
    mm_per_pixel = float(np.asarray(npz["mm_per_pixel"], dtype=np.float64).reshape(-1)[0])
    fixed_y = float(np.asarray(npz["klt_prior_segments"], dtype=np.float64)[0, 1])
    r_series = np.asarray(npz["fixed_measurement_R_diag"], dtype=np.float64)[sl, 1]

    q_series = np.zeros(n, dtype=np.float64)
    q_series[1:] = np.maximum(0.0, py_pred_p[1:] - py_forward_p[:-1])

    changed = np.abs(py_selected_alpha - py_raw_alpha) > 1e-9
    change_table = pd.DataFrame(
        {
            "aligned_frame": np.arange(n, dtype=int),
            "python_frame": np.arange(python_offset, python_offset + n, dtype=int),
            "matlab_alpha_deg": mat_measure_alpha,
            "python_raw_alpha_deg": py_raw_alpha,
            "python_selected_alpha_deg": py_selected_alpha,
            "selected_minus_raw_deg": py_selected_alpha - py_raw_alpha,
            "selected_minus_matlab_deg": py_selected_alpha - mat_measure_alpha,
            "raw_minus_matlab_deg": py_raw_alpha - mat_measure_alpha,
            "changed_from_raw": changed,
        }
    )
    change_path = OUT / "timtrack_alpha_changes.csv"
    change_table.to_csv(change_path, index=False)

    signal_rows = [
        {"signal": "python_raw_alpha_vs_matlab_geofeature_alpha", **scalar_metrics(mat_measure_alpha, py_raw_alpha)},
        {"signal": "python_selected_alpha_vs_matlab_geofeature_alpha", **scalar_metrics(mat_measure_alpha, py_selected_alpha)},
        {"signal": "python_selected_vs_raw_alpha", **scalar_metrics(py_raw_alpha, py_selected_alpha)},
    ]
    signal_path = OUT / "measurement_chain_metrics.csv"
    pd.DataFrame(signal_rows).to_csv(signal_path, index=False)

    variant_rows = []
    for variant, measurement_alpha in [
        ("selected_python_measurement", py_selected_alpha),
        ("raw_python_measurement", py_raw_alpha),
        ("matlab_measurement", mat_measure_alpha),
    ]:
        forward_alpha, p_plus, p_minus = run_scalar_alpha_channel(
            x_minus=py_prior_alpha,
            measurement_alpha=measurement_alpha,
            initial_plus=float(py_forward_alpha[0]),
            initial_p_plus=float(py_forward_p[0]),
            q_series=q_series,
            r_series=r_series,
        )
        smooth_alpha = smooth_scalar_channel(forward_alpha, p_plus, py_prior_alpha, p_minus)
        end_segments = build_end_segments(py_x_smooth, smooth_alpha, py_sup, py_deep, fixed_y)
        fl_mm = segment_lengths_mm(end_segments, mm_per_pixel)
        ang_deg = cmp.segment_angle_deg(end_segments)
        variant_rows.append(
            {
                "variant": variant,
                **{f"forward_alpha_{k}": v for k, v in scalar_metrics(py_forward_alpha if variant == 'selected_python_measurement' else cmp.object_series_to_2d(fascicle['X_plus'], 2)[:n, 1], forward_alpha).items()},
                **{f"final_alpha_{k}": v for k, v in scalar_metrics(mat_final_alpha, smooth_alpha).items()},
                **{f"FL_{k}": v for k, v in scalar_metrics(mat_fl, fl_mm).items()},
                **{f"ANG_{k}": v for k, v in scalar_metrics(mat_ang, ang_deg).items()},
                **{f"end_x_deep_{k}": v for k, v in scalar_metrics(mat_end[:, 2], end_segments[:, 2]).items()},
            }
        )
    variant_table = pd.DataFrame(variant_rows)
    variant_path = OUT / "measurement_chain_variants.csv"
    variant_table.to_csv(variant_path, index=False)

    selected_row = variant_table.loc[variant_table["variant"] == "selected_python_measurement"].iloc[0]
    raw_row = variant_table.loc[variant_table["variant"] == "raw_python_measurement"].iloc[0]
    matlab_row = variant_table.loc[variant_table["variant"] == "matlab_measurement"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    show_n = min(350, n)
    frames = np.arange(show_n)
    axes[0].plot(frames, mat_measure_alpha[:show_n], label="MATLAB geofeature alpha", linewidth=1.2)
    axes[0].plot(frames, py_raw_alpha[:show_n], label="Python raw TimTrack alpha", linewidth=1.0)
    axes[0].plot(frames, py_selected_alpha[:show_n], label="Python selected/persisted alpha", linewidth=1.0)
    axes[0].set_title("Raw vs selected Python TimTrack alpha")
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    changed_idx = np.flatnonzero(changed)
    axes[1].plot(frames, (py_selected_alpha - py_raw_alpha)[:show_n], label="selected - raw", linewidth=1.0)
    if len(changed_idx):
        idx_show = changed_idx[changed_idx < show_n]
        axes[1].scatter(idx_show, (py_selected_alpha - py_raw_alpha)[idx_show], s=18, color="tab:red", label="changed frames")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Frames where selection/persistence changed raw alpha")
    axes[1].set_xlabel("aligned frame index")
    axes[1].set_ylabel("delta alpha (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    trace_plot_path = OUT / "measurement_chain_traces.png"
    fig.savefig(trace_plot_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["selected", "raw", "MATLAB meas"]
    x = np.arange(len(labels))
    ax.plot(x, variant_table["final_alpha_rmse"].to_numpy(dtype=float), marker="o", label="final alpha RMSE (deg)")
    ax.plot(x, variant_table["FL_rmse"].to_numpy(dtype=float), marker="s", label="FL RMSE (mm)")
    ax.plot(x, variant_table["end_x_deep_rmse"].to_numpy(dtype=float), marker="^", label="end x_deep RMSE (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("Effect of raw vs selected TimTrack alpha on final parity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    summary_plot_path = OUT / "measurement_chain_summary.png"
    fig.savefig(summary_plot_path, dpi=180)
    plt.close(fig)

    changed_count = int(np.sum(changed))
    summary_lines = [
        "# Notebook 83 — TimTrack alpha measurement-chain audit",
        "",
        f"Aligned {n} MATLAB/Python samples with Python offset {python_offset}.",
        "",
        "This notebook follows the measurement side deeper: it compares Python raw TimTrack alpha, Python selected/persisted alpha, and MATLAB geofeature alpha, then tests whether the persistence/selection stage is the main cause of the forward alpha gap.",
        "",
        "## Measurement-chain findings",
        "",
        f"- Python raw TimTrack alpha differs from MATLAB geofeature alpha by RMSE {pd.DataFrame(signal_rows).loc[pd.DataFrame(signal_rows)['signal'] == 'python_raw_alpha_vs_matlab_geofeature_alpha', 'rmse'].iloc[0]:.4f} deg.",
        f"- Python selected/persisted alpha differs from MATLAB geofeature alpha by RMSE {pd.DataFrame(signal_rows).loc[pd.DataFrame(signal_rows)['signal'] == 'python_selected_alpha_vs_matlab_geofeature_alpha', 'rmse'].iloc[0]:.4f} deg.",
        f"- The persistence/selection stage changes the raw alpha on only {changed_count} of {n} aligned frames, with selected-vs-raw RMSE {pd.DataFrame(signal_rows).loc[pd.DataFrame(signal_rows)['signal'] == 'python_selected_vs_raw_alpha', 'rmse'].iloc[0]:.4f} deg.",
        "",
        "## Notebook-only Kalman measurement swaps",
        "",
        f"- With the current Python prior, using selected/persisted alpha gives final alpha RMSE {selected_row['final_alpha_rmse']:.4f} deg and FL RMSE {selected_row['FL_rmse']:.4f} mm.",
        f"- Replacing selected alpha with raw TimTrack alpha barely changes the result: final alpha RMSE becomes {raw_row['final_alpha_rmse']:.4f} deg and FL RMSE {raw_row['FL_rmse']:.4f} mm.",
        f"- Replacing the measurement with MATLAB geofeature alpha is the meaningful jump: final alpha RMSE becomes {matlab_row['final_alpha_rmse']:.4f} deg and FL RMSE {matlab_row['FL_rmse']:.4f} mm.",
        "",
        "## Interpretation",
        "",
        "- The Python persistence/selection layer is not the dominant problem in this run.",
        "- The main mismatch is already present in the raw TimTrack alpha entering that layer.",
        "- That means the next useful audit target is earlier in the TimTrack/Hough alpha-production chain itself: image/mask inputs, candidate generation, or angle convention before persistence even has a chance to help.",
        "",
        f"- Measurement metrics CSV: `{signal_path}`",
        f"- Per-frame change CSV: `{change_path}`",
        f"- Variant metrics CSV: `{variant_path}`",
        f"- Trace plot: `{trace_plot_path}`",
        f"- Summary plot: `{summary_plot_path}`",
    ]
    summary_path = OUT / "notebook83_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
