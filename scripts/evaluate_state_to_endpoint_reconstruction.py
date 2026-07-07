#!/usr/bin/env python3
"""Notebook 80 helper: isolate state-to-endpoint reconstruction."""

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
OUT = PROJECT_ROOT / "results" / "notebook80_state_to_endpoint_reconstruction"


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
    x_source: str,
    alpha_source: str,
    apo_source: str,
    fixed_y_source: str,
    segments: np.ndarray,
    reference_end: np.ndarray,
    reference_fl_mm: np.ndarray,
    reference_ang_deg: np.ndarray,
    mm_per_pixel: float,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "variant": variant,
        "x_source": x_source,
        "alpha_source": alpha_source,
        "apo_source": apo_source,
        "fixed_y_source": fixed_y_source,
    }
    for idx, label in enumerate(["x_sup", "y_sup", "x_deep", "y_deep"]):
        row.update(
            {
                f"end_{label}_{key}": value
                for key, value in scalar_metrics(reference_end[:, idx], segments[:, idx]).items()
            }
        )
    fl_mm = segment_lengths_mm(segments, mm_per_pixel)
    ang_deg = cmp.segment_angle_deg(segments)
    row.update({f"FL_{key}": value for key, value in scalar_metrics(reference_fl_mm, fl_mm).items()})
    row.update({f"ANG_{key}": value for key, value in scalar_metrics(reference_ang_deg, ang_deg).items()})
    return row


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

    mat_end = cmp.matlab_segments(fascicle["fas_x_end"], fascicle["fas_y_end"])[:n]
    mat_fas = cmp.matlab_segments(fascicle["fas_x"], fascicle["fas_y"])[:n]
    mat_state = cmp.object_series_to_2d(fascicle["X_plus"], 2)[:n]
    mat_x = mat_state[:, 0]
    mat_state_alpha = mat_state[:, 1]
    mat_alpha = cmp.as_float1(fascicle["alpha"])[:n]
    mat_sup_x = cmp.object_series_to_2d(region["sup_x"], 2)[:n]
    mat_sup_y = cmp.object_series_to_2d(region["sup_y"], 2)[:n]
    mat_deep_x = cmp.object_series_to_2d(region["deep_x"], 2)[:n]
    mat_deep_y = cmp.object_series_to_2d(region["deep_y"], 2)[:n]
    mat_sup = np.column_stack([mat_sup_x[:, 0], mat_sup_y[:, 0], mat_sup_x[:, 1], mat_sup_y[:, 1]])
    mat_deep = np.column_stack([mat_deep_x[:, 0], mat_deep_y[:, 0], mat_deep_x[:, 1], mat_deep_y[:, 1]])

    py_state = np.asarray(npz["fixed_X_plus"], dtype=np.float64)[sl]
    py_x = py_state[:, 0]
    py_alpha = py_state[:, 1]
    py_sup = np.asarray(npz["sup_apo_lines"], dtype=np.float64)[sl]
    py_deep = np.asarray(npz["deep_apo_lines"], dtype=np.float64)[sl]

    mm_per_pixel = float(np.asarray(npz["mm_per_pixel"], dtype=np.float64).reshape(-1)[0])
    py_fixed_y = float(np.asarray(npz["klt_prior_segments"], dtype=np.float64)[0, 1])
    mat_fixed_y = float(mat_fas[0, 1])

    refs = {
        "FL_mm": cmp.as_float1(region["FL"])[:n],
        "ANG_deg": cmp.as_float1(region["ANG"])[:n],
    }

    variants = [
        {
            "variant": "baseline_py_x_py_alpha_py_apo",
            "x_source": "python_smoothed_X_plus[:,0]",
            "alpha_source": "python_smoothed_X_plus[:,1]",
            "apo_source": "python_apo_lines",
            "fixed_y_source": "python_klt_frame0_y",
            "segments": build_end_segments(py_x, py_alpha, py_sup, py_deep, py_fixed_y),
        },
        {
            "variant": "matlab_x_py_alpha_py_apo",
            "x_source": "matlab_X_plus[:,0]",
            "alpha_source": "python_smoothed_X_plus[:,1]",
            "apo_source": "python_apo_lines",
            "fixed_y_source": "python_klt_frame0_y",
            "segments": build_end_segments(mat_x, py_alpha, py_sup, py_deep, py_fixed_y),
        },
        {
            "variant": "py_x_matlab_alpha_py_apo",
            "x_source": "python_smoothed_X_plus[:,0]",
            "alpha_source": "matlab_saved_alpha",
            "apo_source": "python_apo_lines",
            "fixed_y_source": "python_klt_frame0_y",
            "segments": build_end_segments(py_x, mat_alpha, py_sup, py_deep, py_fixed_y),
        },
        {
            "variant": "matlab_x_matlab_alpha_py_apo",
            "x_source": "matlab_X_plus[:,0]",
            "alpha_source": "matlab_saved_alpha",
            "apo_source": "python_apo_lines",
            "fixed_y_source": "python_klt_frame0_y",
            "segments": build_end_segments(mat_x, mat_alpha, py_sup, py_deep, py_fixed_y),
        },
        {
            "variant": "matlab_x_matlab_state_alpha_py_apo",
            "x_source": "matlab_X_plus[:,0]",
            "alpha_source": "matlab_X_plus[:,1]",
            "apo_source": "python_apo_lines",
            "fixed_y_source": "python_klt_frame0_y",
            "segments": build_end_segments(mat_x, mat_state_alpha, py_sup, py_deep, py_fixed_y),
        },
        {
            "variant": "matlab_x_matlab_alpha_matlab_apo_py_fixed_y",
            "x_source": "matlab_X_plus[:,0]",
            "alpha_source": "matlab_saved_alpha",
            "apo_source": "matlab_region_sup_deep_lines",
            "fixed_y_source": "python_klt_frame0_y",
            "segments": build_end_segments(mat_x, mat_alpha, mat_sup, mat_deep, py_fixed_y),
        },
        {
            "variant": "matlab_x_matlab_alpha_matlab_apo_mat_fixed_y",
            "x_source": "matlab_X_plus[:,0]",
            "alpha_source": "matlab_saved_alpha",
            "apo_source": "matlab_region_sup_deep_lines",
            "fixed_y_source": "matlab_fas_frame0_y",
            "segments": build_end_segments(mat_x, mat_alpha, mat_sup, mat_deep, mat_fixed_y),
        },
    ]

    rows = [
        variant_row(
            variant=item["variant"],
            x_source=item["x_source"],
            alpha_source=item["alpha_source"],
            apo_source=item["apo_source"],
            fixed_y_source=item["fixed_y_source"],
            segments=item["segments"],
            reference_end=mat_end,
            reference_fl_mm=refs["FL_mm"],
            reference_ang_deg=refs["ANG_deg"],
            mm_per_pixel=mm_per_pixel,
        )
        for item in variants
    ]
    variant_table = pd.DataFrame(rows)
    variant_path = OUT / "endpoint_reconstruction_variants.csv"
    variant_table.to_csv(variant_path, index=False)

    input_table = pd.DataFrame(
        [
            {
                "signal": "python_x_vs_matlab_x_state",
                **scalar_metrics(mat_x, py_x),
            },
            {
                "signal": "python_alpha_vs_matlab_saved_alpha",
                **scalar_metrics(mat_alpha, py_alpha),
            },
            {
                "signal": "matlab_X_plus_alpha_vs_matlab_saved_alpha",
                **scalar_metrics(mat_alpha, mat_state_alpha),
            },
        ]
    )
    input_path = OUT / "input_signal_deltas.csv"
    input_table.to_csv(input_path, index=False)

    baseline = variant_table.loc[variant_table["variant"] == "baseline_py_x_py_alpha_py_apo"].iloc[0]
    x_swap = variant_table.loc[variant_table["variant"] == "matlab_x_py_alpha_py_apo"].iloc[0]
    alpha_swap = variant_table.loc[variant_table["variant"] == "py_x_matlab_alpha_py_apo"].iloc[0]
    both_swap = variant_table.loc[variant_table["variant"] == "matlab_x_matlab_alpha_py_apo"].iloc[0]
    mat_state_alpha_row = variant_table.loc[
        variant_table["variant"] == "matlab_x_matlab_state_alpha_py_apo"
    ].iloc[0]
    mat_apo_py_y = variant_table.loc[
        variant_table["variant"] == "matlab_x_matlab_alpha_matlab_apo_py_fixed_y"
    ].iloc[0]
    mat_apo_mat_y = variant_table.loc[
        variant_table["variant"] == "matlab_x_matlab_alpha_matlab_apo_mat_fixed_y"
    ].iloc[0]

    labels = [
        "py x\npy α\npy apo",
        "mat x\npy α\npy apo",
        "py x\nmat α\npy apo",
        "mat x\nmat α\npy apo",
        "mat x\nmat X+ α\npy apo",
        "mat x\nmat α\nmat apo\npy y",
        "mat x\nmat α\nmat apo\nmat y",
    ]
    x_sup_rmse = variant_table["end_x_sup_rmse"].to_numpy(dtype=float)
    x_deep_rmse = variant_table["end_x_deep_rmse"].to_numpy(dtype=float)
    fl_rmse = variant_table["FL_rmse"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, x_sup_rmse, width=width, label="end x_sup RMSE")
    axes[0].bar(x + width / 2, x_deep_rmse, width=width, label="end x_deep RMSE")
    axes[0].set_title("Endpoint RMSE by reconstruction variant")
    axes[0].set_ylabel("RMSE (px)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(x, fl_rmse, color="tab:green")
    axes[1].set_title("Final FL RMSE by reconstruction variant")
    axes[1].set_ylabel("RMSE (mm)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    plot_path = OUT / "endpoint_reconstruction_variants.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 80 — state-to-endpoint reconstruction audit",
        "",
        f"Aligned {n} MATLAB/Python samples with Python offset {python_offset}.",
        "",
        "This notebook isolates the final reconstruction layer that turns `[x_sup, alpha]` into `fas_x_end / fas_y_end`. It reuses notebook 77's saved strict Python run, then swaps MATLAB vs Python sources for x, alpha, aponeurosis lines, and fixed-y anchor while keeping the same line-intersection code.",
        "",
        "## Main findings",
        "",
        f"- The notebook 79 baseline reproduces the current final-end gap: with Python smoothed x, Python smoothed alpha, and Python apo lines, endpoint RMSE is {baseline['end_x_sup_rmse']:.2f} px at the superficial x end and {baseline['end_x_deep_rmse']:.2f} px at the deep x end, with FL RMSE {baseline['FL_rmse']:.4f} mm.",
        f"- Swapping only MATLAB x mostly fixes the superficial endpoint but not final FL: `end_x_sup` RMSE drops from {baseline['end_x_sup_rmse']:.2f} px to {x_swap['end_x_sup_rmse']:.2f} px, while `end_x_deep` stays large at {x_swap['end_x_deep_rmse']:.2f} px and FL RMSE stays {x_swap['FL_rmse']:.4f} mm.",
        f"- Swapping only MATLAB saved alpha does the opposite: `end_x_sup` stays almost unchanged at {alpha_swap['end_x_sup_rmse']:.2f} px, but `end_x_deep` collapses from {baseline['end_x_deep_rmse']:.2f} px to {alpha_swap['end_x_deep_rmse']:.2f} px and FL RMSE improves from {baseline['FL_rmse']:.4f} mm to {alpha_swap['FL_rmse']:.4f} mm.",
        f"- Swapping both MATLAB x and MATLAB saved alpha on the Python apo lines brings the entire end segment close to MATLAB: `end_x_sup` RMSE {both_swap['end_x_sup_rmse']:.2f} px, `end_x_deep` RMSE {both_swap['end_x_deep_rmse']:.2f} px, FL RMSE {both_swap['FL_rmse']:.4f} mm.",
        f"- Replacing the Python apo lines with MATLAB's saved apo lines changes little at this layer. With MATLAB x + MATLAB saved alpha, moving from Python apo lines to MATLAB apo lines shifts `end_x_sup` RMSE only from {both_swap['end_x_sup_rmse']:.2f} px to {mat_apo_mat_y['end_x_sup_rmse']:.2f} px and `end_x_deep` RMSE only from {both_swap['end_x_deep_rmse']:.2f} px to {mat_apo_mat_y['end_x_deep_rmse']:.2f} px.",
        f"- The constant fixed-y anchor matters, but only mildly here. Using MATLAB apo lines with MATLAB x + alpha, switching the anchor from the Python frame-0 y to MATLAB's own frame-0 y changes `end_x_sup` RMSE from {mat_apo_py_y['end_x_sup_rmse']:.2f} px to {mat_apo_mat_y['end_x_sup_rmse']:.2f} px.",
        f"- A crucial subtlety: MATLAB's saved `alpha` is not identical to `Fascicle.X_plus[:,1]`. Their direct difference has RMSE {float(input_table.loc[input_table['signal'] == 'matlab_X_plus_alpha_vs_matlab_saved_alpha', 'rmse'].iloc[0]):.4f} deg. That small angle delta matters geometrically: with the same MATLAB x and Python apo lines, using MATLAB `X_plus[:,1]` instead of MATLAB saved `alpha` worsens deep-end x RMSE from {both_swap['end_x_deep_rmse']:.2f} px to {mat_state_alpha_row['end_x_deep_rmse']:.2f} px.",
        "",
        "## Interpretation",
        "",
        "- The line-intersection math itself is mostly behaving. When fed MATLAB x and MATLAB saved alpha, it reproduces MATLAB final endpoints closely even on the Python apo lines.",
        "- In this reconstruction layer, x mainly drives the superficial endpoint, while alpha drives the deep endpoint and almost the entire remaining FL gap.",
        "- The dominant unresolved difference is therefore not generic apo-line geometry. It is the angle that is fed into the final endpoint reconstruction.",
        "",
        "## Audit cross-check",
        "",
        "- I do not see evidence of a more important missing geometric function than this endpoint-angle issue.",
        "- The next thing that needs direct attention is MATLAB's exact final-angle path into `fas_x_end / fas_y_end`: whether that angle is a post-filter/post-smoother output convention, or logic adjacent to fields like `Fascicle.A`.",
        "- `Fascicle.A` stays on the checklist, but notebook 80 moves it from a vague possibility to a targeted question about how MATLAB derives the final angle used for endpoint reconstruction.",
        "",
        f"- Variant CSV: `{variant_path}`",
        f"- Input delta CSV: `{input_path}`",
        f"- Plot: `{plot_path}`",
    ]
    summary_path = OUT / "notebook80_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
