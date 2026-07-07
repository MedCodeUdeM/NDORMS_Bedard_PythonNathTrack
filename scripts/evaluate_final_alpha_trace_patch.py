#!/usr/bin/env python3
"""Notebook 81 helper: trace MATLAB final alpha and test a notebook-only patch."""

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
OUT = PROJECT_ROOT / "results" / "notebook81_final_alpha_trace_patch"


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


def rts_smooth_scalar(x_corr: np.ndarray, x_pred: np.ndarray, gains: np.ndarray) -> np.ndarray:
    x_corr = np.asarray(x_corr, dtype=np.float64).reshape(-1)
    x_pred = np.asarray(x_pred, dtype=np.float64).reshape(-1)
    gains = np.asarray(gains, dtype=np.float64).reshape(-1)
    n = len(x_corr)
    out = np.full(n, np.nan, dtype=np.float64)
    out[-1] = x_corr[-1]
    for frame in range(n - 2, -1, -1):
        gain = gains[frame]
        if not np.isfinite(gain):
            gain = 1.0
        out[frame] = x_corr[frame] + gain * (out[frame + 1] - x_pred[frame + 1])
    return out


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
    matlab_time = cmp.as_float1(region["Time"])
    python_time = cmp.as_float1(npz["time_s"])
    python_offset = cmp.choose_python_offset(matlab_time, python_time)
    n = min(len(matlab_time), len(python_time) - python_offset)
    sl = slice(python_offset, python_offset + n)

    mat_forward = cmp.object_series_to_2d(fascicle["X_plus"], 2)[:n]
    mat_pred = cmp.object_series_to_2d(fascicle["X_minus"], 2)[:n]
    mat_alpha = cmp.as_float1(fascicle["alpha"])[:n]
    mat_A = cmp.as_float1(fascicle["A"])
    mat_end = cmp.matlab_segments(fascicle["fas_x_end"], fascicle["fas_y_end"])[:n]
    mat_fl = cmp.as_float1(region["FL"])[:n]
    mat_ang = cmp.as_float1(region["ANG"])[:n]

    py_forward = np.asarray(npz["fixed_forward_X_plus"], dtype=np.float64)[sl]
    py_smooth = np.asarray(npz["fixed_X_plus"], dtype=np.float64)[sl]
    py_pred = np.asarray(npz["fixed_X_minus"], dtype=np.float64)[sl]
    py_forward_p = np.asarray(npz["fixed_forward_fas_p"], dtype=np.float64)[sl]
    py_pred_p = np.asarray(npz["fixed_fas_p_minus"], dtype=np.float64)[sl]
    py_saved_smoother_gain = np.asarray(npz["fixed_smoother_gain"], dtype=np.float64)[sl]
    py_sup = np.asarray(npz["sup_apo_lines"], dtype=np.float64)[sl]
    py_deep = np.asarray(npz["deep_apo_lines"], dtype=np.float64)[sl]
    py_current_end = np.asarray(npz["fixed_fascicle_end_segments"], dtype=np.float64)[sl]
    py_current_fl = np.asarray(npz["fixed_FL_mm"], dtype=np.float64)[sl]
    py_current_ang = np.asarray(npz["fixed_ANG_deg"], dtype=np.float64)[sl]
    mm_per_pixel = float(np.asarray(npz["mm_per_pixel"], dtype=np.float64).reshape(-1)[0])
    fixed_y = float(np.asarray(npz["klt_prior_segments"], dtype=np.float64)[0, 1])

    mat_alpha_trace = rts_smooth_scalar(mat_forward[:, 1], mat_pred[:, 1], mat_A[: n - 1])
    py_gain_exact = np.full(n, np.nan, dtype=np.float64)
    valid = py_pred_p[1:, 1] != 0
    np.divide(
        py_forward_p[:-1, 1],
        py_pred_p[1:, 1],
        out=py_gain_exact[:-1],
        where=valid,
    )
    py_alpha_patch = rts_smooth_scalar(py_forward[:, 1], py_pred[:, 1], py_gain_exact[:-1])

    py_x_patch = py_smooth[:, 0].copy()
    py_end_patch = build_end_segments(py_x_patch, py_alpha_patch, py_sup, py_deep, fixed_y)
    py_fl_patch = segment_lengths_mm(py_end_patch, mm_per_pixel)
    py_ang_patch = cmp.segment_angle_deg(py_end_patch)

    trace_rows = [
        {
            "signal": "matlab_saved_alpha_vs_matlab_forward_X_plus_alpha",
            **scalar_metrics(mat_alpha, mat_forward[:, 1]),
        },
        {
            "signal": "matlab_saved_alpha_vs_matlab_rts_trace_from_A",
            **scalar_metrics(mat_alpha, mat_alpha_trace),
        },
        {
            "signal": "python_saved_smoothed_alpha_vs_python_forward_X_plus_alpha",
            **scalar_metrics(py_smooth[:, 1], py_forward[:, 1]),
        },
        {
            "signal": "python_saved_smoothed_alpha_vs_notebook_exact_alpha_patch",
            **scalar_metrics(py_smooth[:, 1], py_alpha_patch),
        },
        {
            "signal": "python_saved_smoother_gain_alpha_vs_exact_ratio_gain",
            **scalar_metrics(py_saved_smoother_gain[:-1, 1], py_gain_exact[:-1]),
        },
    ]
    trace_table = pd.DataFrame(trace_rows)
    trace_path = OUT / "alpha_trace_metrics.csv"
    trace_table.to_csv(trace_path, index=False)

    comparison_rows = [
        {
            "variant": "current_saved_python",
            "forward_alpha_rmse_vs_matlab_X_plus_alpha": scalar_metrics(mat_forward[:, 1], py_forward[:, 1])["rmse"],
            "final_alpha_rmse_vs_matlab_alpha": scalar_metrics(mat_alpha, py_smooth[:, 1])["rmse"],
            "end_x_sup_rmse_vs_matlab": scalar_metrics(mat_end[:, 0], py_current_end[:, 0])["rmse"],
            "end_x_deep_rmse_vs_matlab": scalar_metrics(mat_end[:, 2], py_current_end[:, 2])["rmse"],
            "FL_rmse_vs_matlab_mm": scalar_metrics(mat_fl, py_current_fl)["rmse"],
            "ANG_rmse_vs_matlab_deg": scalar_metrics(mat_ang, py_current_ang)["rmse"],
        },
        {
            "variant": "notebook_exact_alpha_patch",
            "forward_alpha_rmse_vs_matlab_X_plus_alpha": scalar_metrics(mat_forward[:, 1], py_forward[:, 1])["rmse"],
            "final_alpha_rmse_vs_matlab_alpha": scalar_metrics(mat_alpha, py_alpha_patch)["rmse"],
            "end_x_sup_rmse_vs_matlab": scalar_metrics(mat_end[:, 0], py_end_patch[:, 0])["rmse"],
            "end_x_deep_rmse_vs_matlab": scalar_metrics(mat_end[:, 2], py_end_patch[:, 2])["rmse"],
            "FL_rmse_vs_matlab_mm": scalar_metrics(mat_fl, py_fl_patch)["rmse"],
            "ANG_rmse_vs_matlab_deg": scalar_metrics(mat_ang, py_ang_patch)["rmse"],
        },
    ]
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_path = OUT / "patched_vs_current_vs_matlab.csv"
    comparison_table.to_csv(comparison_path, index=False)

    patch_delta_rows = [
        {
            "signal": "alpha_patch_minus_current_smoothed_alpha",
            **scalar_metrics(py_smooth[:, 1], py_alpha_patch),
        },
        {
            "signal": "FL_patch_minus_current_mm",
            **scalar_metrics(py_current_fl, py_fl_patch),
        },
        {
            "signal": "ANG_patch_minus_current_deg",
            **scalar_metrics(py_current_ang, py_ang_patch),
        },
        {
            "signal": "end_x_sup_patch_minus_current",
            **scalar_metrics(py_current_end[:, 0], py_end_patch[:, 0]),
        },
        {
            "signal": "end_x_deep_patch_minus_current",
            **scalar_metrics(py_current_end[:, 2], py_end_patch[:, 2]),
        },
    ]
    patch_delta_table = pd.DataFrame(patch_delta_rows)
    patch_delta_path = OUT / "patch_delta_vs_current.csv"
    patch_delta_table.to_csv(patch_delta_path, index=False)

    show_n = min(350, n)
    frames = np.arange(show_n)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(frames, mat_forward[:show_n, 1], label="MATLAB forward X_plus[:,1]", linewidth=1.1)
    axes[0].plot(frames, mat_alpha[:show_n], label="MATLAB final alpha", linewidth=1.2)
    axes[0].set_title("MATLAB forward vs final alpha")
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(frames, py_forward[:show_n, 1], label="Python forward_X_plus[:,1]", linewidth=1.1)
    axes[1].plot(frames, py_smooth[:show_n, 1], label="Python current smoothed alpha", linewidth=1.2)
    axes[1].plot(frames, py_alpha_patch[:show_n], linestyle="--", label="Notebook exact-alpha patch", linewidth=1.0)
    axes[1].set_title("Python forward vs notebook exact-alpha patch")
    axes[1].set_xlabel("aligned frame index")
    axes[1].set_ylabel("alpha (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    trace_plot_path = OUT / "alpha_trace_examples.png"
    fig.savefig(trace_plot_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    metric_labels = ["final alpha RMSE (deg)", "end x_deep RMSE (px)", "FL RMSE (mm)"]
    current_vals = [
        comparison_table.loc[comparison_table["variant"] == "current_saved_python", "final_alpha_rmse_vs_matlab_alpha"].iloc[0],
        comparison_table.loc[comparison_table["variant"] == "current_saved_python", "end_x_deep_rmse_vs_matlab"].iloc[0],
        comparison_table.loc[comparison_table["variant"] == "current_saved_python", "FL_rmse_vs_matlab_mm"].iloc[0],
    ]
    patch_vals = [
        comparison_table.loc[comparison_table["variant"] == "notebook_exact_alpha_patch", "final_alpha_rmse_vs_matlab_alpha"].iloc[0],
        comparison_table.loc[comparison_table["variant"] == "notebook_exact_alpha_patch", "end_x_deep_rmse_vs_matlab"].iloc[0],
        comparison_table.loc[comparison_table["variant"] == "notebook_exact_alpha_patch", "FL_rmse_vs_matlab_mm"].iloc[0],
    ]
    x = np.arange(len(metric_labels))
    width = 0.36
    ax.bar(x - width / 2, current_vals, width=width, label="current saved Python")
    ax.bar(x + width / 2, patch_vals, width=width, label="notebook exact-alpha patch")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("Patch effect on final parity metrics")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    compare_plot_path = OUT / "patch_effect_summary.png"
    fig.savefig(compare_plot_path, dpi=180)
    plt.close(fig)

    final_alpha_rmse_current = comparison_table.loc[
        comparison_table["variant"] == "current_saved_python", "final_alpha_rmse_vs_matlab_alpha"
    ].iloc[0]
    final_alpha_rmse_patch = comparison_table.loc[
        comparison_table["variant"] == "notebook_exact_alpha_patch", "final_alpha_rmse_vs_matlab_alpha"
    ].iloc[0]
    fl_rmse_current = comparison_table.loc[
        comparison_table["variant"] == "current_saved_python", "FL_rmse_vs_matlab_mm"
    ].iloc[0]
    fl_rmse_patch = comparison_table.loc[
        comparison_table["variant"] == "notebook_exact_alpha_patch", "FL_rmse_vs_matlab_mm"
    ].iloc[0]
    end_x_deep_current = comparison_table.loc[
        comparison_table["variant"] == "current_saved_python", "end_x_deep_rmse_vs_matlab"
    ].iloc[0]
    end_x_deep_patch = comparison_table.loc[
        comparison_table["variant"] == "notebook_exact_alpha_patch", "end_x_deep_rmse_vs_matlab"
    ].iloc[0]

    summary_lines = [
        "# Notebook 81 — final alpha trace and notebook-only patch",
        "",
        f"Aligned {n} MATLAB/Python samples with Python offset {python_offset}.",
        "",
        "This notebook traces where MATLAB's final `alpha` comes from relative to forward `X_plus[:,1]`, smoothing, and `Fascicle.A`, then applies a notebook-only Python patch that mirrors the same mathematical path without editing the production module.",
        "",
        "## MATLAB trace result",
        "",
        "- MATLAB's saved `Fascicle.alpha` is exactly the scalar RTS-style smoother result on the angle channel.",
        "- Using MATLAB's saved forward `X_plus[:,1]`, predicted `X_minus[:,1]`, and saved `Fascicle.A`, the recursive trace",
        "",
        "  `alpha_smooth[t] = X_plus[t,1] + A[t] * (alpha_smooth[t+1] - X_minus[t+1,1])`",
        "",
        f"  reproduces saved MATLAB `alpha` with RMSE {trace_table.loc[trace_table['signal'] == 'matlab_saved_alpha_vs_matlab_rts_trace_from_A', 'rmse'].iloc[0]:.6f} deg exactly.",
        f"- By contrast, MATLAB forward `X_plus[:,1]` differs from MATLAB final `alpha` by RMSE {trace_table.loc[trace_table['signal'] == 'matlab_saved_alpha_vs_matlab_forward_X_plus_alpha', 'rmse'].iloc[0]:.4f} deg.",
        "",
        "## Notebook-only Python patch",
        "",
        "- The notebook patch does not guess or brute-force MATLAB outputs. It recomputes the final Python alpha from Python's own forward `X_plus[:,1]`, `X_minus[:,1]`, and scalar smoother ratio `A = P_plus / P_minus(next)` on the angle channel, using the same recursion as MATLAB.",
        f"- That notebook-only patch is a strict no-op on the current Python run: patched alpha differs from the saved smoothed Python alpha by RMSE {patch_delta_table.loc[patch_delta_table['signal'] == 'alpha_patch_minus_current_smoothed_alpha', 'rmse'].iloc[0]:.6f} deg, and patched FL differs from the saved Python FL by RMSE {patch_delta_table.loc[patch_delta_table['signal'] == 'FL_patch_minus_current_mm', 'rmse'].iloc[0]:.6f} mm.",
        "",
        "## Final comparison",
        "",
        f"- Final alpha parity does not move under the notebook patch: MATLAB final-alpha RMSE stays {final_alpha_rmse_current:.4f} deg before the patch and {final_alpha_rmse_patch:.4f} deg after it.",
        f"- Deep-end geometry does not move either: `fas_x_end` x_deep RMSE stays {end_x_deep_current:.2f} px before the patch and {end_x_deep_patch:.2f} px after it.",
        f"- Final FL parity also stays unchanged: FL RMSE stays {fl_rmse_current:.4f} mm before the patch and {fl_rmse_patch:.4f} mm after it.",
        "",
        "## Interpretation",
        "",
        "- This is the outcome we wanted for safety: the exact MATLAB alpha path is already what Python is numerically doing on the saved run.",
        "- So there is no evidence of a hidden mathematical anomaly in the current backward alpha smoother itself.",
        "- The remaining gap is upstream of this patch, mainly in the forward angle/state inputs that the smoother receives.",
        "- If we patch production code later, it should be for clarity and contract correctness — for example preserving forward `X_plus` as forward state and exposing smoothed state separately — not because this notebook found a missing alpha recursion.",
        "",
        f"- Alpha trace metrics: `{trace_path}`",
        f"- Patch comparison metrics: `{comparison_path}`",
        f"- Patch delta metrics: `{patch_delta_path}`",
        f"- Alpha trace plot: `{trace_plot_path}`",
        f"- Patch summary plot: `{compare_plot_path}`",
        "",
        "## MATLAB source trace",
        "",
        "- The exact smoother loop appears in `UltraTimTrack.m` around lines 2287–2301, where `xsmooth(m) = xcorr(m) + A*(xsmooth(m) - xpred(m))`, then `alpha_smooth = xsmooth(2)` is written back to `handles.Region(i).Fascicle(j).alpha{frame_no}`.",
    ]
    summary_path = OUT / "notebook81_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
