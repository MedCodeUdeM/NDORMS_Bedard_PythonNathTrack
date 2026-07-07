#!/usr/bin/env python3
"""Notebook 79 helper: evaluate x/length anchoring model variants."""

from __future__ import annotations

import json
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
from ultrasound_tracker.ultratimtrack_matlab_2state import MatlabTwoStateKalmanConfig, run_matlab_2state_kalman


RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "notebook77_x_update_fix_parity"
    / "python_x_update_fix_same_inputs"
    / "UltraTimTrack_test"
)
NPZ_PATH = RUN_DIR / "UltraTimTrack_test_strict_results.npz"
MATLAB_RESULT = PROJECT_ROOT / "data" / "matlab" / "slow_low_01_DOWN_tracked_Q=001.mat"
OUT = PROJECT_ROOT / "results" / "notebook79_x_length_anchoring"


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


def run_variant(
    *,
    klt_prior_segments: np.ndarray,
    timtrack_alpha_deg: np.ndarray,
    sup_apo_lines: np.ndarray,
    deep_apo_lines: np.ndarray,
    affines: np.ndarray,
    mm_per_pixel: float,
    x_measurement_variance: float,
    alpha_measurement_variance: float,
    fixed_superficial_y: float,
) -> dict[str, np.ndarray]:
    cfg = MatlabTwoStateKalmanConfig(
        q_parameter=0.01,
        x_measurement_variance=float(x_measurement_variance),
        alpha_measurement_variance=float(alpha_measurement_variance),
        n_start_frames=1,
        run_smoother=True,
    )
    return run_matlab_2state_kalman(
        klt_prior_segments,
        timtrack_alpha_deg,
        sup_apo_lines,
        deep_apo_lines,
        config=cfg,
        fixed_superficial_y=float(fixed_superficial_y),
        mm_per_pixel=mm_per_pixel,
        prediction_affine_matrices=affines,
    )


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

    klt = np.asarray(npz["klt_prior_segments"], dtype=np.float64)
    affines = np.asarray(npz["klt_affine_matrices"], dtype=np.float64)
    timtrack_alpha = np.asarray(npz["timtrack_alpha_deg"], dtype=np.float64)
    sup_lines = np.asarray(npz["sup_apo_lines"], dtype=np.float64)
    deep_lines = np.asarray(npz["deep_apo_lines"], dtype=np.float64)
    mm_per_pixel = float(np.asarray(npz["mm_per_pixel"], dtype=np.float64).reshape(-1)[0])
    baseline_fixed_y = float(klt[0, 1])
    matlab_fixed_y = float(cmp.matlab_segments(fascicle["fas_x"], fascicle["fas_y"])[0, 1])
    alpha_measurement_variance = float(np.asarray(mat["Fdat"]["R"], dtype=np.float64).reshape(-1)[0])
    mat_forward_state = cmp.object_series_to_2d(fascicle["X_plus"], 2)[:n]
    mat_fas_end = cmp.matlab_segments(fascicle["fas_x_end"], fascicle["fas_y_end"])[:n]
    refs = {
        "FL": cmp.as_float1(region["FL"])[:n],
        "ANG": cmp.as_float1(region["ANG"])[:n],
        "PEN": cmp.as_float1(region["PEN"])[:n],
    }

    x_var_rows: list[dict[str, float | str]] = []
    for x_var in [10, 25, 50, 75, 100, 150, 200, 300, 500, 1000]:
        out = run_variant(
            klt_prior_segments=klt,
            timtrack_alpha_deg=timtrack_alpha,
            sup_apo_lines=sup_lines,
            deep_apo_lines=deep_lines,
            affines=affines,
            mm_per_pixel=mm_per_pixel,
            x_measurement_variance=float(x_var),
            alpha_measurement_variance=alpha_measurement_variance,
            fixed_superficial_y=baseline_fixed_y,
        )
        row: dict[str, float | str] = {"x_measurement_variance": float(x_var)}
        row.update({f"FL_{k}": v for k, v in scalar_metrics(refs["FL"], out["FL_mm"][sl]).items()})
        row.update({f"ANG_{k}": v for k, v in scalar_metrics(refs["ANG"], out["ANG_deg"][sl]).items()})
        row.update({f"PEN_{k}": v for k, v in scalar_metrics(refs["PEN"], out["PEN_deg"][sl]).items()})
        row.update({f"forward_x_{k}": v for k, v in scalar_metrics(mat_forward_state[:, 0], out["forward_X_plus"][sl, 0]).items()})
        x_var_rows.append(row)
    x_var_table = pd.DataFrame(x_var_rows)
    x_var_path = OUT / "x_measurement_variance_sweep.csv"
    x_var_table.to_csv(x_var_path, index=False)

    y_anchor_rows: list[dict[str, float | str]] = []
    for dy in [-20, -10, -5, 0, 5, 10, 20, 30, 40, 50, 60]:
        fixed_y = baseline_fixed_y + float(dy)
        out = run_variant(
            klt_prior_segments=klt,
            timtrack_alpha_deg=timtrack_alpha,
            sup_apo_lines=sup_lines,
            deep_apo_lines=deep_lines,
            affines=affines,
            mm_per_pixel=mm_per_pixel,
            x_measurement_variance=100.0,
            alpha_measurement_variance=alpha_measurement_variance,
            fixed_superficial_y=fixed_y,
        )
        row = {
            "fixed_y_offset_px": float(dy),
            "fixed_superficial_y_px": float(fixed_y),
        }
        row.update({f"FL_{k}": v for k, v in scalar_metrics(refs["FL"], out["FL_mm"][sl]).items()})
        row.update({f"ANG_{k}": v for k, v in scalar_metrics(refs["ANG"], out["ANG_deg"][sl]).items()})
        row.update({f"PEN_{k}": v for k, v in scalar_metrics(refs["PEN"], out["PEN_deg"][sl]).items()})
        end_seg = np.asarray(out["fascicle_end_segments"], dtype=np.float64)[sl]
        for idx, label in enumerate(["x_sup", "y_sup", "x_deep", "y_deep"]):
            row.update({f"end_{label}_{k}": v for k, v in scalar_metrics(mat_fas_end[:, idx], end_seg[:, idx]).items()})
        y_anchor_rows.append(row)
    y_anchor_table = pd.DataFrame(y_anchor_rows)
    y_anchor_path = OUT / "fixed_y_anchor_sweep.csv"
    y_anchor_table.to_csv(y_anchor_path, index=False)

    named_rows: list[dict[str, float | str]] = []
    named_variants = [
        ("baseline", 100.0, baseline_fixed_y),
        ("best_x_variance_for_FL", float(x_var_table.loc[x_var_table["FL_rmse"].idxmin(), "x_measurement_variance"]), baseline_fixed_y),
        ("matlab_frame0_fixed_y", 100.0, matlab_fixed_y),
        ("best_fixed_y_for_FL_in_tested_range", 100.0, float(y_anchor_table.loc[y_anchor_table["FL_rmse"].idxmin(), "fixed_superficial_y_px"])),
    ]
    for name, x_var, fixed_y in named_variants:
        out = run_variant(
            klt_prior_segments=klt,
            timtrack_alpha_deg=timtrack_alpha,
            sup_apo_lines=sup_lines,
            deep_apo_lines=deep_lines,
            affines=affines,
            mm_per_pixel=mm_per_pixel,
            x_measurement_variance=float(x_var),
            alpha_measurement_variance=alpha_measurement_variance,
            fixed_superficial_y=float(fixed_y),
        )
        row = {
            "variant": name,
            "x_measurement_variance": float(x_var),
            "fixed_superficial_y_px": float(fixed_y),
            "fixed_y_offset_from_baseline_px": float(fixed_y - baseline_fixed_y),
        }
        row.update({f"FL_{k}": v for k, v in scalar_metrics(refs["FL"], out["FL_mm"][sl]).items()})
        row.update({f"ANG_{k}": v for k, v in scalar_metrics(refs["ANG"], out["ANG_deg"][sl]).items()})
        row.update({f"PEN_{k}": v for k, v in scalar_metrics(refs["PEN"], out["PEN_deg"][sl]).items()})
        end_seg = np.asarray(out["fascicle_end_segments"], dtype=np.float64)[sl]
        row.update({f"end_x_sup_{k}": v for k, v in scalar_metrics(mat_fas_end[:, 0], end_seg[:, 0]).items()})
        row.update({f"end_x_deep_{k}": v for k, v in scalar_metrics(mat_fas_end[:, 2], end_seg[:, 2]).items()})
        named_rows.append(row)
    named_table = pd.DataFrame(named_rows)
    named_path = OUT / "named_x_length_variants.csv"
    named_table.to_csv(named_path, index=False)

    best_x_row = x_var_table.loc[x_var_table["FL_rmse"].idxmin()]
    best_y_row = y_anchor_table.loc[y_anchor_table["FL_rmse"].idxmin()]
    baseline_y_row = y_anchor_table.loc[y_anchor_table["fixed_y_offset_px"] == 0].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(x_var_table["x_measurement_variance"], x_var_table["FL_rmse"], marker="o", label="FL RMSE")
    axes[0].plot(x_var_table["x_measurement_variance"], x_var_table["ANG_rmse"], marker="s", label="ANG RMSE")
    axes[0].set_xscale("log")
    axes[0].set_title("Effect of x-measurement variance")
    axes[0].set_xlabel("x measurement variance")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(y_anchor_table["fixed_y_offset_px"], y_anchor_table["FL_rmse"], marker="o", label="FL RMSE")
    axes[1].plot(y_anchor_table["fixed_y_offset_px"], y_anchor_table["end_x_sup_rmse"], marker="s", label="final-end x_sup RMSE")
    axes[1].plot(y_anchor_table["fixed_y_offset_px"], y_anchor_table["end_x_deep_rmse"], marker="^", label="final-end x_deep RMSE")
    axes[1].axvline(0.0, color="tab:orange", linestyle="--", linewidth=1.0, label="baseline")
    axes[1].axvline(float(best_y_row["fixed_y_offset_px"]), color="tab:green", linestyle=":", linewidth=1.0, label=f"best FL dy={int(best_y_row['fixed_y_offset_px'])}")
    axes[1].set_title("Effect of constant y/length anchor")
    axes[1].set_xlabel("fixed superficial y offset (px)")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    plot_path = OUT / "x_length_anchor_sweeps.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 79 — x/length anchoring audit",
        "",
        f"Aligned {n} MATLAB/Python samples with Python offset {python_offset}.",
        "",
        "This notebook targets the x/length anchor model directly while reusing notebook 77's saved KLT prior, affines, TimTrack alpha, and aponeurosis lines.",
        "",
        "## Main findings",
        "",
        f"- Sweeping the x measurement variance changes final FL only trivially. Across the tested range 10 to 1000, FL RMSE only moves from {float(x_var_table['FL_rmse'].min()):.4f} to {float(x_var_table['FL_rmse'].max()):.4f} mm, while forward-state x bias stays near 1.36–1.43 px.",
        f"- Sweeping the constant fixed superficial y anchor changes final FL much more. In the tested range, FL RMSE improves from {baseline_y_row['FL_rmse']:.4f} mm at dy=0 to {best_y_row['FL_rmse']:.4f} mm at dy={int(best_y_row['fixed_y_offset_px'])} px.",
        f"- That apparent FL improvement is not a faithful parity improvement: at the same dy={int(best_y_row['fixed_y_offset_px'])} px point, final-end endpoint geometry gets much worse. `fas_x_end` x_sup RMSE grows from {baseline_y_row['end_x_sup_rmse']:.2f} px to {best_y_row['end_x_sup_rmse']:.2f} px, and x_deep RMSE grows from {baseline_y_row['end_x_deep_rmse']:.2f} px to {best_y_row['end_x_deep_rmse']:.2f} px.",
        f"- Using MATLAB's own frame-0 fixed-y anchor does not help. The MATLAB-frame0-y variant gives FL RMSE {float(named_table.loc[named_table['variant'] == 'matlab_frame0_fixed_y', 'FL_rmse'].iloc[0]):.4f} mm, not better than baseline.",
        "",
        "## Audit cross-check",
        "",
        "- The crucial unresolved difference is still final-end geometry (`fas_x_end` / `fas_y_end`), especially the x coordinates. Earlier parity audits already showed those endpoint RMSEs were large, and notebook 79 confirms that simple anchor tweaks only trade FL against endpoint correctness.",
        "- I do not see another must-use MATLAB function being ignored that obviously outranks this issue. The main contract fields for the current mismatch are already in play: `X_plus`, `X_minus`, `fas_x`, `fas_y`, `fas_x_end`, `fas_y_end`, and `alpha`.",
        "- One lower-priority contract field that is still not explicitly mirrored as a named output is MATLAB `Fascicle.A`. It is worth keeping on the checklist, but nothing in these anchor sweeps suggests it is the dominant blocker for current FL parity.",
        "",
        "## Interpretation",
        "",
        "- The remaining FL gap is not primarily a scalar tuning problem in the x measurement update.",
        "- The constant-y anchor is powerful enough to improve FL, but only by compensating in a way that breaks endpoint geometry. That makes it a diagnostic clue, not a production fix.",
        "- The next useful direction is to inspect how the final fascicle end segment is reconstructed from `[x_sup, alpha]` and the aponeurosis lines, rather than continuing to tune scalar variances.",
        "",
        f"- X-variance sweep CSV: `{x_var_path}`",
        f"- Fixed-y sweep CSV: `{y_anchor_path}`",
        f"- Named variant CSV: `{named_path}`",
        f"- Sweep plot: `{plot_path}`",
    ]
    summary_path = OUT / "notebook79_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
