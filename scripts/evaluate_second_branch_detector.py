#!/usr/bin/env python3
"""Notebook 92 helper: subclass sparse misses, learn branch 2, and test two-branch detectors."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.evaluate_conditional_hough_patch as nb90
import scripts.evaluate_peakweight_precision_detector as nb91
from scripts.run_strict_ultratimtrack_video import (
    FascicleCandidatePersistenceConfig,
    select_fascicle_candidate_persistence,
)
from ultrasound_tracker.ultratimtrack_matlab_2state import MatlabTwoStateKalmanConfig, run_matlab_2state_kalman
import scripts.compare_updated_matlab_python as cmp
from scipy.io import loadmat


OUT = PROJECT_ROOT / "results" / "notebook92_second_branch_detector"


def classify_remaining_false_negative_subclass(row: pd.Series) -> str:
    if bool(row["selected_is_top_family"]) is False and float(row["top_family_angle_offset"]) <= -4.0:
        return "displaced_topfamily_lower_dominant"
    if bool(row["selected_is_top_family"]) is False:
        return "displaced_topfamily_nonlower"
    return "selected_topfamily_residual"


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


def render_rule(rule: list[tuple[str, str, float]]) -> str:
    return nb91.render_rule(rule)


def detector_metrics(flagged: pd.Series, oracle: pd.Series) -> dict[str, float]:
    return nb91.detector_metrics(flagged, oracle)


def search_branch2_rules(remaining: pd.DataFrame) -> pd.DataFrame:
    atoms = [
        ("selected_is_top_family", "<=", 0.0),
        ("top_family_angle_offset", "<=", -6.0),
        ("top_family_angle_offset", "<=", -4.0),
        ("best_upper_share", "<=", 0.16),
        ("best_upper_share", "<=", 0.18),
        ("best_upper_share", ">=", 0.12),
        ("mass_below_10deg", ">=", 0.18),
        ("mass_below_10deg", ">=", 0.22),
        ("mass_below_10deg", ">=", 0.26),
        ("gap_to_lower_deg", ">=", 4.0),
        ("gap_to_lower_deg", ">=", 6.0),
        ("delta_localmax_deg", ">=", 4.0),
        ("delta_localmax_deg", ">=", 6.0),
        ("cum_before_selected", ">=", 0.46),
        ("cum_before_selected", ">=", 0.48),
        ("selected_family_share", "<=", 0.12),
        ("selected_family_share", "<=", 0.16),
        ("duplicate_fraction", "<=", 0.2),
    ]
    oracle = remaining["oracle_high_angle_overweight"].astype(bool)
    masks = [apply_rule(remaining, [atom]) for atom in atoms]
    records = []
    for r in (1, 2, 3, 4):
        for idxs in combinations(range(len(atoms)), r):
            flagged = pd.Series(True, index=remaining.index, dtype=bool)
            rule = [atoms[i] for i in idxs]
            for idx in idxs:
                flagged &= masks[idx]
            metrics = detector_metrics(flagged, oracle)
            if metrics["flagged_frames"] < 8 or metrics["flagged_frames"] > 220:
                continue
            if metrics["oracle_precision"] < 0.22 or metrics["oracle_tp"] < 10:
                continue
            records.append(
                {
                    "rule_text": render_rule(rule),
                    "rule_json": json.dumps(rule),
                    **metrics,
                }
            )
    table = pd.DataFrame(records).drop_duplicates(subset=["flagged_frames", "oracle_tp", "oracle_fp", "rule_text"])
    table = table.sort_values(["oracle_precision", "oracle_recall", "flagged_frames"], ascending=[False, False, True]).reset_index(drop=True)
    return table


def choose_rule(rule_table: pd.DataFrame, *, min_flagged: int, max_flagged: int, sort_by_recall: bool) -> dict:
    subset = rule_table[(rule_table["flagged_frames"] >= min_flagged) & (rule_table["flagged_frames"] <= max_flagged)].copy()
    if subset.empty:
        subset = rule_table.copy()
    if sort_by_recall:
        subset = subset.sort_values(["oracle_recall", "oracle_precision", "flagged_frames"], ascending=[False, False, True])
    else:
        subset = subset.sort_values(["oracle_precision", "oracle_recall", "flagged_frames"], ascending=[False, False, True])
    return subset.iloc[0].to_dict()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    feature_table = pd.read_csv(nb91.OUT / "peakweight_feature_table.csv").set_index("frame")
    rules_json = json.loads((nb91.OUT / "selected_rules.json").read_text())
    branch1_rule = [tuple(x) for x in rules_json["sparse_rule"]]
    branch1_flagged = apply_rule(feature_table, branch1_rule)
    oracle = feature_table["oracle_high_angle_overweight"].astype(bool)

    remaining = feature_table.loc[~branch1_flagged].copy()
    remaining["remaining_oracle_fn"] = oracle.loc[remaining.index]
    remaining_true = remaining[remaining["remaining_oracle_fn"]].copy()
    remaining_true["subclass"] = remaining_true.apply(classify_remaining_false_negative_subclass, axis=1)
    subclass_summary = (
        remaining_true.groupby("subclass")
        .agg(
            frames=("subclass", "size"),
            mass_below_10deg_mean=("mass_below_10deg", "mean"),
            gap_to_lower_deg_mean=("gap_to_lower_deg", "mean"),
            best_upper_share_mean=("best_upper_share", "mean"),
            top_family_angle_offset_mean=("top_family_angle_offset", "mean"),
        )
        .reset_index()
        .sort_values("frames", ascending=False)
    )

    branch2_frontier = search_branch2_rules(remaining)
    branch2_precision_row = choose_rule(branch2_frontier, min_flagged=30, max_flagged=90, sort_by_recall=False)
    branch2_coverage_row = choose_rule(branch2_frontier, min_flagged=60, max_flagged=130, sort_by_recall=True)
    branch2_precision_rule = [tuple(x) for x in json.loads(branch2_precision_row["rule_json"])]
    branch2_coverage_rule = [tuple(x) for x in json.loads(branch2_coverage_row["rule_json"])]

    npz = np.load(nb90.NPZ_PATH, allow_pickle=True)
    mat = loadmat(nb90.MATLAB_RESULT, simplify_cells=True)
    utt = loadmat(nb90.UTT_EXPORT, simplify_cells=True)["UTT_numeric_export"]
    metadata = json.loads(nb90.METADATA_PATH.read_text())

    with nb90.NB89_CACHE.open("rb") as f:
        base_entries = __import__("pickle").load(f)["entries"]
    per_frame = pd.read_csv(nb90.NB89_PER_FRAME)
    base_pf = per_frame[per_frame["variant"] == "baseline_current"].copy().set_index("frame")
    localmax_pf = per_frame[per_frame["variant"] == "angle_profile_localmax"].copy().set_index("frame")

    mat_region = mat["Fdat"]["Region"]
    matlab_time = cmp.as_float1(mat_region["Time"])
    python_time = cmp.as_float1(npz["time_s"])
    python_offset = cmp.choose_python_offset(matlab_time, python_time)
    mat_final_alpha = cmp.as_float1(mat_region["Fascicle"]["alpha"])[: len(base_entries)]
    mat_fl = cmp.as_float1(mat_region["FL"])[: len(base_entries)]
    mat_ang = cmp.as_float1(mat_region["ANG"])[: len(base_entries)]
    mat_alpha = base_pf["matlab_alpha_deg"].to_numpy(dtype=float)
    klt = np.asarray(npz["klt_prior_segments"], dtype=np.float64)[python_offset : python_offset + len(base_entries)]
    affines = np.asarray(npz["klt_affine_matrices"], dtype=np.float64)[python_offset : python_offset + len(base_entries)]
    sup_lines = np.asarray(npz["sup_apo_lines"], dtype=np.float64)[python_offset : python_offset + len(base_entries)]
    deep_lines = np.asarray(npz["deep_apo_lines"], dtype=np.float64)[python_offset : python_offset + len(base_entries)]
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

    branch1_only = branch1_flagged
    branch2_precision = apply_rule(feature_table, branch2_precision_rule) & (~branch1_only)
    branch2_coverage = apply_rule(feature_table, branch2_coverage_rule) & (~branch1_only)

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
        ("branch1_sparse_only", f"Notebook 91 sparse branch: {render_rule(branch1_rule)}", branch1_only),
        (
            "branch1_plus_branch2_precision",
            f"Two-branch detector: branch1 `{render_rule(branch1_rule)}` OR branch2 `{render_rule(branch2_precision_rule)}`",
            branch1_only | branch2_precision,
        ),
        (
            "branch1_plus_branch2_coverage",
            f"Two-branch detector: branch1 `{render_rule(branch1_rule)}` OR branch2 `{render_rule(branch2_coverage_rule)}`",
            branch1_only | branch2_coverage,
        ),
    ]

    raw_rows = []
    persistence_rows = []
    kalman_rows = []
    detector_rows = []
    per_frame_tables: list[pd.DataFrame] = []
    base_raw = base_pf["variant_raw_alpha_deg"].to_numpy(dtype=float)
    localmax_raw = localmax_pf["variant_raw_alpha_deg"].to_numpy(dtype=float)

    for variant_name, note, flagged_series in detector_defs:
        print(f"Evaluating notebook 92 variant: {variant_name}")
        flagged = flagged_series.reindex(feature_table.index).fillna(False).to_numpy(dtype=bool)
        raw_alpha = np.where(flagged, localmax_raw, base_raw)
        mixed_entries = [localmax_entries[i] if flagged[i] else base_entries[i] for i in range(len(base_entries))]
        persistence = select_fascicle_candidate_persistence(mixed_entries, raw_alpha, config=persistence_cfg)
        selected_alpha = np.asarray(persistence["selected_alpha_deg"], dtype=np.float64)
        kalman = run_matlab_2state_kalman(
            klt,
            selected_alpha,
            sup_lines,
            deep_lines,
            config=kalman_cfg,
            mm_per_pixel=mm_per_pixel,
            prediction_affine_matrices=affines,
        )
        nearest_errors = np.asarray([nb90.nearest_peak_error(np.asarray(e["alphas"], dtype=float), mat_alpha[i]) for i, e in enumerate(mixed_entries)], dtype=float)
        hit2 = np.isfinite(nearest_errors) & (nearest_errors <= 2.0)
        raw_wrong = np.abs(raw_alpha - mat_alpha) > 5.0

        raw_rows.append(
            {
                "variant": variant_name,
                "note": note,
                "raw_changed_fraction_vs_baseline": float(np.mean(flagged)),
                "nearest_peak_rmse_deg": nb90.scalar_metrics(np.zeros(len(nearest_errors)), nearest_errors)["rmse"],
                "candidate_hit_rate_2deg": float(np.mean(hit2)),
                "candidate_present_but_raw_wrong_rate": float(np.mean(hit2 & raw_wrong)),
                **{f"raw_vs_matlab_{k}": v for k, v in nb90.scalar_metrics(mat_alpha, raw_alpha).items()},
            }
        )
        persistence_rows.append(
            {
                "variant": variant_name,
                **{f"selected_vs_matlab_{k}": v for k, v in nb90.scalar_metrics(mat_alpha, selected_alpha).items()},
                "raw_rejected_fraction": float(np.mean(np.asarray(persistence["raw_alpha_rejected"], dtype=bool))),
            }
        )
        kalman_rows.append(
            {
                "variant": variant_name,
                **{
                    f"final_alpha_vs_matlab_{k}": v
                    for k, v in nb90.scalar_metrics(mat_final_alpha, np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1]).items()
                },
                **{f"FL_vs_matlab_{k}": v for k, v in nb90.scalar_metrics(mat_fl, np.asarray(kalman["FL_mm"], dtype=np.float64)).items()},
                **{f"ANG_vs_matlab_{k}": v for k, v in nb90.scalar_metrics(mat_ang, np.asarray(kalman["ANG_deg"], dtype=np.float64)).items()},
            }
        )
        detector_rows.append({"variant": variant_name, **detector_metrics(pd.Series(flagged, index=feature_table.index), oracle)})
        per_frame_tables.append(
            pd.DataFrame(
                {
                    "frame": np.arange(len(base_entries), dtype=int),
                    "variant": variant_name,
                    "flagged_for_patch": flagged,
                    "variant_raw_alpha_deg": raw_alpha,
                    "variant_selected_alpha_deg": selected_alpha,
                    "variant_final_alpha_deg": np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1],
                    "FL_mm": np.asarray(kalman["FL_mm"], dtype=np.float64),
                }
            )
        )

    raw_table = pd.DataFrame(raw_rows)
    persistence_table = pd.DataFrame(persistence_rows)
    kalman_table = pd.DataFrame(kalman_rows)
    detector_table = pd.DataFrame(detector_rows)
    summary_table = raw_table.merge(persistence_table, on="variant").merge(kalman_table, on="variant").merge(detector_table, on="variant")
    per_frame_table = pd.concat(per_frame_tables, ignore_index=True)

    (OUT / "remaining_false_negative_subclasses.csv").write_text(subclass_summary.to_csv(index=False), encoding="utf-8")
    branch2_frontier.to_csv(OUT / "branch2_rule_frontier.csv", index=False)
    summary_table.to_csv(OUT / "two_branch_variant_summary.csv", index=False)
    detector_table.to_csv(OUT / "two_branch_detector_stats.csv", index=False)
    per_frame_table.to_csv(OUT / "two_branch_per_frame.csv", index=False)
    feature_table.reset_index().to_csv(OUT / "peakweight_feature_table.csv", index=False)
    (OUT / "selected_branch_rules.json").write_text(
        json.dumps(
            {
                "branch1_sparse_rule": branch1_rule,
                "branch1_sparse_rule_text": render_rule(branch1_rule),
                "branch2_precision_rule": branch2_precision_rule,
                "branch2_precision_rule_text": render_rule(branch2_precision_rule),
                "branch2_coverage_rule": branch2_coverage_rule,
                "branch2_coverage_rule_text": render_rule(branch2_coverage_rule),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].bar(subclass_summary["subclass"], subclass_summary["frames"])
    axes[0].set_title("Remaining sparse-miss oracle subclasses")
    axes[0].set_ylabel("frame count")
    axes[0].tick_params(axis="x", rotation=20, labelsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].scatter(branch2_frontier["flagged_frames"], branch2_frontier["oracle_precision"], s=18, alpha=0.75)
    axes[1].set_title("Branch-2 frontier on remaining sparse misses")
    axes[1].set_xlabel("flagged frames")
    axes[1].set_ylabel("oracle precision")
    axes[1].grid(True, alpha=0.25)

    x = np.arange(len(summary_table))
    axes[2].bar(x, summary_table["FL_vs_matlab_rmse"])
    axes[2].set_title("Final FL RMSE after two-branch patching")
    axes[2].set_ylabel("RMSE (mm)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(summary_table["variant"], rotation=20, ha="right", fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    figure_path = OUT / "two_branch_detector_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    baseline = summary_table.loc[summary_table["variant"] == "baseline_current"].iloc[0]
    best_variant = summary_table.loc[summary_table["FL_vs_matlab_rmse"].idxmin()]
    precision_union = summary_table.loc[summary_table["variant"] == "branch1_plus_branch2_precision"].iloc[0]
    coverage_union = summary_table.loc[summary_table["variant"] == "branch1_plus_branch2_coverage"].iloc[0]

    summary_lines = [
        "# Notebook 92 — second detector branch from sparse false negatives",
        "",
        "This notebook takes the 121 oracle false negatives left by notebook 91's sparse detector,",
        "splits them into subclasses, learns a second Python-only branch on the dominant subclass structure,",
        "and replays sparse-branch unions through persistence and downstream parity.",
        "",
        "## Remaining sparse-miss subclasses",
        "",
        *(f"- {row.subclass}: {int(row.frames)} frames" for row in subclass_summary.itertuples()),
        "",
        "## Learned branch rules",
        "",
        f"- Branch 1 (fixed sparse rule): `{render_rule(branch1_rule)}`",
        f"- Branch 2 precision-oriented rule: `{render_rule(branch2_precision_rule)}`",
        f"- Branch 2 coverage-oriented rule: `{render_rule(branch2_coverage_rule)}`",
        "",
        "## Key findings",
        "",
        f"- Two-branch precision union: precision {precision_union['oracle_precision']:.3f}, recall {precision_union['oracle_recall']:.3f}, FL RMSE {precision_union['FL_vs_matlab_rmse']:.4f} mm.",
        f"- Two-branch coverage union: precision {coverage_union['oracle_precision']:.3f}, recall {coverage_union['oracle_recall']:.3f}, FL RMSE {coverage_union['FL_vs_matlab_rmse']:.4f} mm.",
        f"- Best downstream FL among notebook 92 variants is `{best_variant['variant']}` at {best_variant['FL_vs_matlab_rmse']:.4f} mm versus baseline {baseline['FL_vs_matlab_rmse']:.4f} mm.",
        "",
        "## Interpretation",
        "",
        "- If the two-branch unions improve over sparse-only without collapsing precision, then the sparse misses are at least partly structured and not just detector noise.",
        "- Comparing precision-oriented and coverage-oriented branch 2 rules tells us whether the second branch should stay conservative or whether broader displaced-top-family coverage helps more downstream.",
        "",
        f"- Subclass CSV: `{OUT / 'remaining_false_negative_subclasses.csv'}`",
        f"- Branch-2 frontier CSV: `{OUT / 'branch2_rule_frontier.csv'}`",
        f"- Selected rules JSON: `{OUT / 'selected_branch_rules.json'}`",
        f"- Combined summary CSV: `{OUT / 'two_branch_variant_summary.csv'}`",
        f"- Detector stats CSV: `{OUT / 'two_branch_detector_stats.csv'}`",
        f"- Per-frame CSV: `{OUT / 'two_branch_per_frame.csv'}`",
        f"- Summary figure: `{figure_path}`",
    ]
    (OUT / "notebook92_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
