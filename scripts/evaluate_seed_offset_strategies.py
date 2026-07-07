#!/usr/bin/env python3
"""Notebook 78 helper: evaluate seed-offset strategies against MATLAB parity."""

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
from ultrasound_tracker.ultratrack_klt import propagate_cumulative_affines
from ultrasound_tracker.ultratimtrack_matlab_2state import MatlabTwoStateKalmanConfig, run_matlab_2state_kalman


RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "notebook77_x_update_fix_parity"
    / "python_x_update_fix_same_inputs"
    / "UltraTimTrack_test"
)
NPZ_PATH = RUN_DIR / "UltraTimTrack_test_strict_results.npz"
SEED_CANDIDATES_PATH = RUN_DIR / "UltraTimTrack_test_seed_candidates.csv"
MATLAB_RESULT = PROJECT_ROOT / "data" / "matlab" / "slow_low_01_DOWN_tracked_Q=001.mat"
OUT = PROJECT_ROOT / "results" / "notebook78_seed_offset_attack"


def metric_row(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
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


def build_variant_outputs(
    initial_seed_segment_1b: np.ndarray,
    *,
    affines: np.ndarray,
    timtrack_alpha_deg: np.ndarray,
    sup_apo_lines: np.ndarray,
    deep_apo_lines: np.ndarray,
    mm_per_pixel: float,
    kalman_config: MatlabTwoStateKalmanConfig,
) -> dict[str, np.ndarray]:
    klt_prior = propagate_cumulative_affines(initial_seed_segment_1b, affines)
    return run_matlab_2state_kalman(
        klt_prior,
        timtrack_alpha_deg,
        sup_apo_lines,
        deep_apo_lines,
        config=kalman_config,
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

    affines = np.asarray(npz["klt_affine_matrices"], dtype=np.float64)
    timtrack_alpha = np.asarray(npz["timtrack_alpha_deg"], dtype=np.float64)
    sup_lines = np.asarray(npz["sup_apo_lines"], dtype=np.float64)
    deep_lines = np.asarray(npz["deep_apo_lines"], dtype=np.float64)
    mm_per_pixel = float(np.asarray(npz["mm_per_pixel"], dtype=np.float64).reshape(-1)[0])
    baseline_seed = np.asarray(npz["selected_seed_segment"], dtype=np.float64).reshape(4)
    baseline_alpha = float(np.asarray(npz["selected_seed_alpha_deg"], dtype=np.float64).reshape(-1)[0])

    kalman_config = MatlabTwoStateKalmanConfig(
        q_parameter=0.01,
        x_measurement_variance=100.0,
        alpha_measurement_variance=float(np.asarray(mat["Fdat"]["R"], dtype=np.float64).reshape(-1)[0]),
        n_start_frames=1,
        run_smoother=True,
    )

    references = {
        "FL": cmp.as_float1(region["FL"])[:n],
        "ANG": cmp.as_float1(region["ANG"])[:n],
        "PEN": cmp.as_float1(region["PEN"])[:n],
    }
    matlab_raw_seed = cmp.matlab_segments(fascicle["fas_x_original"], fascicle["fas_y_original"])[0]

    seed_candidates = pd.read_csv(SEED_CANDIDATES_PATH)
    frame0 = seed_candidates[seed_candidates["frame"] == 0].copy().sort_values("alpha_deg").reset_index(drop=True)

    alpha_rows: list[dict[str, float | str]] = []
    for _, row in frame0[(frame0["alpha_deg"] >= 16.5) & (frame0["alpha_deg"] <= 19.1)].iterrows():
        seed = np.asarray([row["x_sup"], row["y_sup"], row["x_deep"], row["y_deep"]], dtype=np.float64)
        out = build_variant_outputs(
            seed,
            affines=affines,
            timtrack_alpha_deg=timtrack_alpha,
            sup_apo_lines=sup_lines,
            deep_apo_lines=deep_lines,
            mm_per_pixel=mm_per_pixel,
            kalman_config=kalman_config,
        )
        for variable, key in [("FL", "FL_mm"), ("ANG", "ANG_deg"), ("PEN", "PEN_deg")]:
            row_out: dict[str, float | str] = {
                "alpha_deg": float(row["alpha_deg"]),
                "variable": variable,
            }
            row_out.update(metric_row(references[variable], out[key][sl]))
            alpha_rows.append(row_out)
    alpha_sweep = pd.DataFrame(alpha_rows)
    alpha_sweep_path = OUT / "seed_alpha_sweep_metrics.csv"
    alpha_sweep.to_csv(alpha_sweep_path, index=False)

    dx_rows: list[dict[str, float | str]] = []
    for dx in range(-30, 121, 10):
        seed = baseline_seed.copy()
        seed[[0, 2]] += float(dx)
        out = build_variant_outputs(
            seed,
            affines=affines,
            timtrack_alpha_deg=timtrack_alpha,
            sup_apo_lines=sup_lines,
            deep_apo_lines=deep_lines,
            mm_per_pixel=mm_per_pixel,
            kalman_config=kalman_config,
        )
        for variable, key in [("FL", "FL_mm"), ("ANG", "ANG_deg"), ("PEN", "PEN_deg")]:
            row_out: dict[str, float | str] = {
                "dx_seed_px": int(dx),
                "variable": variable,
            }
            row_out.update(metric_row(references[variable], out[key][sl]))
            dx_rows.append(row_out)
    dx_sweep = pd.DataFrame(dx_rows)
    dx_sweep_path = OUT / "seed_x_translation_sweep_metrics.csv"
    dx_sweep.to_csv(dx_sweep_path, index=False)

    named_variants = []
    variant_specs: list[tuple[str, np.ndarray]] = [
        ("baseline_autonomous_seed", baseline_seed),
        (
            "matlab_angle_nearest_frame0_candidate",
            np.asarray(
                frame0.iloc[(frame0["alpha_deg"] - float(cmp.segment_angle_deg(matlab_raw_seed[None, :])[0])).abs().argmin()][
                    ["x_sup", "y_sup", "x_deep", "y_deep"]
                ],
                dtype=np.float64,
            ),
        ),
        ("oracle_matlab_raw_seed", np.asarray(matlab_raw_seed, dtype=np.float64)),
    ]
    matlab_seed_xsup = float(matlab_raw_seed[0])
    baseline_xsup = float(baseline_seed[0])
    variant_specs.extend(
        [
            ("left_shift_match_matlab_xsup", baseline_seed + np.asarray([matlab_seed_xsup - baseline_xsup, 0.0, matlab_seed_xsup - baseline_xsup, 0.0])),
            ("right_shift_same_magnitude", baseline_seed + np.asarray([baseline_xsup - matlab_seed_xsup, 0.0, baseline_xsup - matlab_seed_xsup, 0.0])),
            ("right_shift_plus_60px", baseline_seed + np.asarray([60.0, 0.0, 60.0, 0.0])),
        ]
    )
    for name, seed in variant_specs:
        out = build_variant_outputs(
            np.asarray(seed, dtype=np.float64),
            affines=affines,
            timtrack_alpha_deg=timtrack_alpha,
            sup_apo_lines=sup_lines,
            deep_apo_lines=deep_lines,
            mm_per_pixel=mm_per_pixel,
            kalman_config=kalman_config,
        )
        for variable, key in [("FL", "FL_mm"), ("ANG", "ANG_deg"), ("PEN", "PEN_deg")]:
            row_out: dict[str, float | str] = {"variant": name, "variable": variable}
            row_out.update(metric_row(references[variable], out[key][sl]))
            named_variants.append(row_out)
    named_variant_table = pd.DataFrame(named_variants)
    named_variant_path = OUT / "named_seed_variant_metrics.csv"
    named_variant_table.to_csv(named_variant_path, index=False)

    fl_alpha = alpha_sweep[alpha_sweep["variable"] == "FL"].sort_values("alpha_deg")
    fl_dx = dx_sweep[dx_sweep["variable"] == "FL"].sort_values("dx_seed_px")
    best_alpha = fl_alpha.loc[fl_alpha["rmse"].idxmin()]
    best_dx = fl_dx.loc[fl_dx["rmse"].idxmin()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(fl_alpha["alpha_deg"], fl_alpha["rmse"], marker="o", linewidth=1.2)
    axes[0].axvline(baseline_alpha, color="tab:orange", linestyle="--", linewidth=1.0, label=f"baseline {baseline_alpha:.1f}°")
    axes[0].axvline(float(best_alpha["alpha_deg"]), color="tab:green", linestyle=":", linewidth=1.0, label=f"best {float(best_alpha['alpha_deg']):.1f}°")
    axes[0].set_title("Final FL RMSE vs frame-0 seed alpha")
    axes[0].set_xlabel("Seed alpha (deg)")
    axes[0].set_ylabel("FL RMSE (mm)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(fl_dx["dx_seed_px"], fl_dx["rmse"], marker="o", linewidth=1.2)
    axes[1].axvline(0.0, color="tab:orange", linestyle="--", linewidth=1.0, label="baseline")
    axes[1].axvline(float(best_dx["dx_seed_px"]), color="tab:green", linestyle=":", linewidth=1.0, label=f"best {int(best_dx['dx_seed_px'])} px")
    axes[1].set_title("Final FL RMSE vs uniform seed x shift")
    axes[1].set_xlabel("Seed x shift (px)")
    axes[1].set_ylabel("FL RMSE (mm)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    sweep_plot_path = OUT / "seed_offset_sweeps_fl_rmse.png"
    fig.savefig(sweep_plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 78 — seed offset attack",
        "",
        f"Aligned {n} MATLAB/Python samples with Python offset {python_offset}.",
        "",
        "The notebook evaluates seed changes using notebook 77's saved persistent-KLT affine matrices, TimTrack alpha, and aponeurosis lines. This is valid because the tracker points/affines are independent of the chosen fascicle seed; the seed only determines the propagated segment geometry.",
        "",
        "## Main findings",
        "",
        f"- Sweeping the frame-0 seed alpha over 16.5–19.1 deg does not improve final FL over the current autonomous seed. The best FL RMSE in that sweep is {best_alpha['rmse']:.4f} mm at alpha {best_alpha['alpha_deg']:.1f} deg; the current seed is {baseline_alpha:.1f} deg with FL RMSE {float(fl_alpha.loc[fl_alpha['alpha_deg'] == baseline_alpha, 'rmse'].iloc[0]):.4f} mm.",
        f"- The oracle MATLAB raw frame-0 seed does not help final FL. It produces FL RMSE {float(named_variant_table[(named_variant_table['variant'] == 'oracle_matlab_raw_seed') & (named_variant_table['variable'] == 'FL')]['rmse'].iloc[0]):.4f} mm, worse than baseline.",
        f"- Uniform x translation of the frame-0 seed changes final FL much more than changing seed alpha. In the tested range, FL RMSE improves from {float(fl_dx.loc[fl_dx['dx_seed_px'] == 0, 'rmse'].iloc[0]):.4f} mm at dx=0 to {best_dx['rmse']:.4f} mm at dx={int(best_dx['dx_seed_px'])} px, while ANG RMSE stays essentially flat.",
        f"- Matching the MATLAB frame-0 x_sup offset directly is not beneficial: the left-shift variant that matches MATLAB x_sup has FL RMSE {float(named_variant_table[(named_variant_table['variant'] == 'left_shift_match_matlab_xsup') & (named_variant_table['variable'] == 'FL')]['rmse'].iloc[0]):.4f} mm, worse than baseline.",
        "",
        "## Interpretation",
        "",
        "- The observed raw seed-vs-MATLAB angle offset is not the lever that controls final FL parity here.",
        "- Final FL is much more sensitive to how the seed anchors x through the propagated prior than to small seed-angle changes.",
        "- Because even the oracle MATLAB raw seed does not improve FL, the remaining FL difference is unlikely to be solved by copying MATLAB's frame-0 seed geometry alone.",
        "- The next target should be the x/length anchoring model downstream of the seed, not just autonomous seed-angle selection.",
        "",
        f"- Alpha sweep CSV: `{alpha_sweep_path}`",
        f"- X-shift sweep CSV: `{dx_sweep_path}`",
        f"- Named variants CSV: `{named_variant_path}`",
        f"- Sweep plot: `{sweep_plot_path}`",
    ]
    summary_path = OUT / "notebook78_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
