#!/usr/bin/env python3
"""Notebook 93 helper: replay from frame 50 onward to test start-condition sensitivity."""

from __future__ import annotations

import json
import pickle
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
import scripts.evaluate_conditional_hough_patch as nb90
import scripts.evaluate_peakweight_precision_detector as nb91
from scripts.run_strict_ultratimtrack_video import (
    FascicleCandidatePersistenceConfig,
    select_fascicle_candidate_persistence,
)
from ultrasound_tracker.ultratimtrack_matlab_2state import MatlabTwoStateKalmanConfig, run_matlab_2state_kalman


OUT = PROJECT_ROOT / "results" / "notebook93_restart50_sensitivity"
START_FRAME = 50


def apply_rule(feature_table: pd.DataFrame, rule: list[tuple[str, str, float]]) -> pd.Series:
    flagged = pd.Series(True, index=feature_table.index, dtype=bool)
    for column, op, threshold in rule:
        if op == ">=":
            flagged &= feature_table[column] >= float(threshold)
        elif op == "<=":
            flagged &= feature_table[column] <= float(threshold)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return flagged.fillna(False)


def load_full_tail_series() -> dict[str, pd.DataFrame]:
    nb90_per = pd.read_csv(nb90.OUT / "conditional_per_frame.csv")
    nb92_per = pd.read_csv(PROJECT_ROOT / "results" / "notebook92_second_branch_detector" / "two_branch_per_frame.csv")

    tables: dict[str, pd.DataFrame] = {}
    tables["baseline_current"] = nb90_per[nb90_per["variant"] == "baseline_current"].copy()
    tables["oracle_overweight_to_localmax"] = nb90_per[nb90_per["variant"] == "oracle_overweight_to_localmax"].copy()
    tables["nb90_mass10_gap4_reference"] = nb90_per[nb90_per["variant"] == "heuristic_mass10_gap4_to_localmax"].copy()
    tables["branch1_plus_branch2_coverage"] = nb92_per[nb92_per["variant"] == "branch1_plus_branch2_coverage"].copy()

    for key, df in tables.items():
        tables[key] = df[df["frame"] >= START_FRAME].copy().reset_index(drop=True)
    return tables


def metrics_against(reference: np.ndarray, estimate: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in nb90.scalar_metrics(reference, estimate).items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    npz = np.load(nb90.NPZ_PATH, allow_pickle=True)
    mat = loadmat(nb90.MATLAB_RESULT, simplify_cells=True)
    utt = loadmat(nb90.UTT_EXPORT, simplify_cells=True)["UTT_numeric_export"]
    metadata = json.loads(nb90.METADATA_PATH.read_text())
    with nb90.NB89_CACHE.open("rb") as f:
        base_entries = pickle.load(f)["entries"]

    per_frame = pd.read_csv(nb90.NB89_PER_FRAME)
    base_pf = per_frame[per_frame["variant"] == "baseline_current"].copy().set_index("frame")
    localmax_pf = per_frame[per_frame["variant"] == "angle_profile_localmax"].copy().set_index("frame")
    feature_table = pd.read_csv(nb91.OUT / "peakweight_feature_table.csv").set_index("frame")
    rules = json.loads((PROJECT_ROOT / "results" / "notebook92_second_branch_detector" / "selected_branch_rules.json").read_text())
    branch1_rule = [tuple(x) for x in rules["branch1_sparse_rule"]]
    branch2_coverage_rule = [tuple(x) for x in rules["branch2_coverage_rule"]]

    oracle = feature_table["oracle_high_angle_overweight"].astype(bool)
    branch1 = apply_rule(feature_table, branch1_rule)
    branch2_coverage = apply_rule(feature_table, branch2_coverage_rule) & (~branch1)

    full_tail_tables = load_full_tail_series()
    base_raw = base_pf["variant_raw_alpha_deg"].to_numpy(dtype=float)
    localmax_raw = localmax_pf["variant_raw_alpha_deg"].to_numpy(dtype=float)

    detector_defs = [
        ("baseline_current", "no conditional patch", pd.Series(False, index=feature_table.index)),
        (
            "oracle_overweight_to_localmax",
            "MATLAB-aware ceiling: baseline nearest peak <=2 deg and baseline raw >5 deg too high",
            oracle,
        ),
        (
            "nb90_mass10_gap4_reference",
            "Notebook 90 broad reference: mass below alpha-10 deg >=0.25 and nearest lower-gap >=4 deg",
            (feature_table["mass_below_10deg"] >= 0.25) & (feature_table["gap_to_lower_deg"] >= 4.0),
        ),
        (
            "branch1_plus_branch2_coverage",
            f"Notebook 92 two-branch coverage union: {rules['branch1_sparse_rule_text']} OR {rules['branch2_coverage_rule_text']}",
            branch1 | branch2_coverage,
        ),
    ]

    mat_region = mat["Fdat"]["Region"]
    matlab_time = cmp.as_float1(mat_region["Time"])
    python_time = cmp.as_float1(npz["time_s"])
    python_offset = cmp.choose_python_offset(matlab_time, python_time)
    mat_final_alpha = cmp.as_float1(mat_region["Fascicle"]["alpha"])
    mat_fl = cmp.as_float1(mat_region["FL"])
    mat_ang = cmp.as_float1(mat_region["ANG"])
    mat_alpha = base_pf["matlab_alpha_deg"].to_numpy(dtype=float)

    klt = np.asarray(npz["klt_prior_segments"], dtype=np.float64)
    affines = np.asarray(npz["klt_affine_matrices"], dtype=np.float64)
    sup_lines = np.asarray(npz["sup_apo_lines"], dtype=np.float64)
    deep_lines = np.asarray(npz["deep_apo_lines"], dtype=np.float64)
    mm_per_pixel = float(np.asarray(npz["mm_per_pixel"], dtype=np.float64).reshape(-1)[0])

    kalman_cfg = MatlabTwoStateKalmanConfig(
        q_parameter=float(utt.get("Q", 0.01)),
        x_measurement_variance=float(utt.get("X", 100.0)),
        alpha_measurement_variance=float(np.asarray(utt.get("R", [3.05529211]), dtype=np.float64).reshape(-1)[0]),
        n_start_frames=int(utt.get("NS", 1)),
        run_smoother=True,
    )
    persistence_cfg = FascicleCandidatePersistenceConfig(
        enabled=bool(metadata["candidate_persistence"]),
        angle_min_deg=float(metadata["fas_angle_min_deg"]),
        angle_max_deg=float(metadata["fas_angle_max_deg"]),
        max_angle_step_deg=float(metadata["max_angle_step_deg"]),
        hough_weight_bonus_deg=float(metadata["candidate_weight_bonus_deg"]),
    )
    localmax_entries = nb90.load_or_compute_localmax_entries(base_entries, utt["parms"])

    summary_rows = []
    per_frame_rows = []

    frame_slice = slice(START_FRAME, len(base_entries))
    tail_len = len(base_entries) - START_FRAME
    mat_alpha_tail = mat_alpha[frame_slice]
    mat_final_alpha_tail = mat_final_alpha[frame_slice]
    mat_fl_tail = mat_fl[frame_slice]
    mat_ang_tail = mat_ang[frame_slice]

    for variant_name, note, flagged_series in detector_defs:
        full_tail = full_tail_tables[variant_name].copy()
        full_raw = full_tail["variant_raw_alpha_deg"].to_numpy(dtype=float)
        full_selected = full_tail["variant_selected_alpha_deg"].to_numpy(dtype=float)
        full_final = full_tail["variant_final_alpha_deg"].to_numpy(dtype=float)
        full_fl = full_tail["FL_mm"].to_numpy(dtype=float)

        summary_rows.append(
            {
                "variant": variant_name,
                "mode": "full_tail_existing",
                "note": note,
                "start_frame": START_FRAME,
                "frames_evaluated": tail_len,
                **metrics_against(mat_alpha_tail, full_raw, "raw_vs_matlab"),
                **metrics_against(mat_alpha_tail, full_selected, "selected_vs_matlab"),
                **metrics_against(mat_final_alpha_tail, full_final, "final_alpha_vs_matlab"),
                **metrics_against(mat_fl_tail, full_fl, "FL_vs_matlab"),
                **metrics_against(mat_ang_tail, full_final, "ANG_proxy_vs_matlab"),
                "raw_vs_fulltail_rmse": 0.0,
                "selected_vs_fulltail_rmse": 0.0,
                "final_alpha_vs_fulltail_rmse": 0.0,
                "FL_vs_fulltail_rmse": 0.0,
            }
        )

        flagged = flagged_series.to_numpy(dtype=bool)[frame_slice]
        raw_alpha_tail = np.where(flagged, localmax_raw[frame_slice], base_raw[frame_slice])
        trimmed_entries = [localmax_entries[i] if flagged[j] else base_entries[i] for j, i in enumerate(range(START_FRAME, len(base_entries)))]
        persistence = select_fascicle_candidate_persistence(trimmed_entries, raw_alpha_tail, config=persistence_cfg)
        selected_tail = np.asarray(persistence["selected_alpha_deg"], dtype=np.float64)
        kalman = run_matlab_2state_kalman(
            klt[python_offset + START_FRAME : python_offset + len(base_entries)],
            selected_tail,
            sup_lines[python_offset + START_FRAME : python_offset + len(base_entries)],
            deep_lines[python_offset + START_FRAME : python_offset + len(base_entries)],
            config=kalman_cfg,
            mm_per_pixel=mm_per_pixel,
            prediction_affine_matrices=affines[python_offset + START_FRAME : python_offset + len(base_entries)],
        )
        final_tail = np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1]
        fl_tail = np.asarray(kalman["FL_mm"], dtype=np.float64)

        summary_rows.append(
            {
                "variant": variant_name,
                "mode": "restart50_replayed",
                "note": note,
                "start_frame": START_FRAME,
                "frames_evaluated": tail_len,
                **metrics_against(mat_alpha_tail, raw_alpha_tail, "raw_vs_matlab"),
                **metrics_against(mat_alpha_tail, selected_tail, "selected_vs_matlab"),
                **metrics_against(mat_final_alpha_tail, final_tail, "final_alpha_vs_matlab"),
                **metrics_against(mat_fl_tail, fl_tail, "FL_vs_matlab"),
                **metrics_against(mat_ang_tail, final_tail, "ANG_proxy_vs_matlab"),
                "raw_vs_fulltail_rmse": float(np.sqrt(np.mean((raw_alpha_tail - full_raw) ** 2))),
                "selected_vs_fulltail_rmse": float(np.sqrt(np.mean((selected_tail - full_selected) ** 2))),
                "final_alpha_vs_fulltail_rmse": float(np.sqrt(np.mean((final_tail - full_final) ** 2))),
                "FL_vs_fulltail_rmse": float(np.sqrt(np.mean((fl_tail - full_fl) ** 2))),
            }
        )

        for local_idx, frame in enumerate(range(START_FRAME, len(base_entries))):
            per_frame_rows.append(
                {
                    "variant": variant_name,
                    "mode": "restart50_replayed",
                    "frame": frame,
                    "raw_alpha_deg": raw_alpha_tail[local_idx],
                    "selected_alpha_deg": selected_tail[local_idx],
                    "final_alpha_deg": final_tail[local_idx],
                    "FL_mm": fl_tail[local_idx],
                    "raw_minus_fulltail_deg": raw_alpha_tail[local_idx] - full_raw[local_idx],
                    "selected_minus_fulltail_deg": selected_tail[local_idx] - full_selected[local_idx],
                    "final_alpha_minus_fulltail_deg": final_tail[local_idx] - full_final[local_idx],
                    "FL_minus_fulltail_mm": fl_tail[local_idx] - full_fl[local_idx],
                }
            )

    summary_table = pd.DataFrame(summary_rows)
    per_frame_table = pd.DataFrame(per_frame_rows)
    summary_table.to_csv(OUT / "restart50_summary.csv", index=False)
    per_frame_table.to_csv(OUT / "restart50_per_frame.csv", index=False)

    baseline_full = summary_table[(summary_table["variant"] == "baseline_current") & (summary_table["mode"] == "full_tail_existing")].iloc[0]
    baseline_restart = summary_table[(summary_table["variant"] == "baseline_current") & (summary_table["mode"] == "restart50_replayed")].iloc[0]
    best_restart = summary_table[summary_table["mode"] == "restart50_replayed"].loc[summary_table[summary_table["mode"] == "restart50_replayed"]["FL_vs_matlab_rmse"].idxmin()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    show = summary_table.copy()
    variants = show["variant"].unique().tolist()
    x = np.arange(len(variants))
    full_rmse = [show[(show["variant"] == v) & (show["mode"] == "full_tail_existing")]["FL_vs_matlab_rmse"].iloc[0] for v in variants]
    restart_rmse = [show[(show["variant"] == v) & (show["mode"] == "restart50_replayed")]["FL_vs_matlab_rmse"].iloc[0] for v in variants]
    axes[0].bar(x - 0.17, full_rmse, width=0.34, label="full tail")
    axes[0].bar(x + 0.17, restart_rmse, width=0.34, label="restart@50")
    axes[0].set_title("FL RMSE vs MATLAB after frame 50")
    axes[0].set_ylabel("RMSE (mm)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    baseline_pf = per_frame_table[per_frame_table["variant"] == "baseline_current"].copy().reset_index(drop=True)
    frames = baseline_pf["frame"].to_numpy(dtype=int)
    axes[1].plot(frames, baseline_pf["selected_minus_fulltail_deg"], label="selected alpha shift")
    axes[1].plot(frames, baseline_pf["final_alpha_minus_fulltail_deg"], label="final alpha shift")
    axes[1].set_title("Baseline restart@50 drift from full run")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("deg")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(frames, baseline_pf["FL_minus_fulltail_mm"], label="FL shift")
    axes[2].set_title("Baseline FL shift from restart@50")
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("mm")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    plot_path = OUT / "restart50_summary.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 93 — restart-at-50 sensitivity",
        "",
        "This notebook tests whether the first 50 frames are biasing the rest of the sequence.",
        "It compares existing full-sequence outputs restricted to frames >= 50 against a fresh replay restarted at frame 50.",
        "",
        "## Key findings",
        "",
        f"- Baseline full-tail FL RMSE: {baseline_full['FL_vs_matlab_rmse']:.4f} mm",
        f"- Baseline restart@50 FL RMSE: {baseline_restart['FL_vs_matlab_rmse']:.4f} mm",
        f"- Baseline final-alpha RMSE full-tail vs restart@50: {baseline_full['final_alpha_vs_matlab_rmse']:.4f} deg -> {baseline_restart['final_alpha_vs_matlab_rmse']:.4f} deg",
        f"- Baseline raw-alpha drift from full tail after restart@50: {baseline_restart['raw_vs_fulltail_rmse']:.6f} deg",
        f"- Baseline final-alpha drift from full tail after restart@50: {baseline_restart['final_alpha_vs_fulltail_rmse']:.4f} deg",
        f"- Best restart@50 FL RMSE among replayed variants: `{best_restart['variant']}` at {best_restart['FL_vs_matlab_rmse']:.4f} mm",
        "",
        "## Interpretation",
        "",
        "- If raw alpha is unchanged but selected/final alpha move after restart@50, the early frames are affecting the stateful parts of the pipeline, not the per-frame Hough math.",
        "- If restart@50 improves MATLAB parity, that means some remaining gap is initialization/history sensitivity rather than a framewise mathematical mismatch.",
        "",
        f"- Summary CSV: `{OUT / 'restart50_summary.csv'}`",
        f"- Per-frame drift CSV: `{OUT / 'restart50_per_frame.csv'}`",
        f"- Summary plot: `{plot_path}`",
    ]
    (OUT / "notebook93_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
