#!/usr/bin/env python3
"""Notebook 91 helper: learn high-precision conditional detectors from exact peak/weight structure."""

from __future__ import annotations

import json
import pickle
import sys
from itertools import combinations
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
from scripts.run_strict_ultratimtrack_video import (
    FascicleCandidatePersistenceConfig,
    select_fascicle_candidate_persistence,
)
from ultrasound_tracker.ultratimtrack_matlab_2state import MatlabTwoStateKalmanConfig, run_matlab_2state_kalman


OUT = PROJECT_ROOT / "results" / "notebook91_peakweight_precision_detector"


def family_structure_features(entry: dict, selected_alpha: float) -> dict[str, float]:
    alphas, weights = nb90.finite_peaks(entry["alphas"], entry["weights"])
    if len(alphas) == 0 or not np.isfinite(selected_alpha):
        return {
            "n_peaks": 0.0,
            "n_unique_angles": 0.0,
            "selected_family_share": np.nan,
            "selected_family_count": np.nan,
            "cum_before_selected": np.nan,
            "cum_after_selected": np.nan,
            "crossing_margin_under": np.nan,
            "crossing_margin_over": np.nan,
            "nearest_lower_share": np.nan,
            "nearest_lower_gap": np.nan,
            "best_lower_share": np.nan,
            "best_lower_gap": np.nan,
            "best_lower_count": np.nan,
            "nearest_upper_share": np.nan,
            "nearest_upper_gap": np.nan,
            "best_upper_share": np.nan,
            "best_upper_gap": np.nan,
            "top_family_share": np.nan,
            "top_family_angle_offset": np.nan,
            "selected_is_top_family": np.nan,
            "duplicate_fraction": np.nan,
        }

    norm_w = weights / np.sum(weights)
    unique_angles = np.unique(alphas)
    family_weights = np.asarray([np.sum(norm_w[alphas == ang]) for ang in unique_angles], dtype=np.float64)
    family_counts = np.asarray([np.sum(alphas == ang) for ang in unique_angles], dtype=np.float64)
    order = np.argsort(unique_angles)
    unique_angles = unique_angles[order]
    family_weights = family_weights[order]
    family_counts = family_counts[order]

    family_idx = np.where(np.isclose(unique_angles, selected_alpha))[0]
    if len(family_idx) == 0:
        family_idx = np.asarray([int(np.argmin(np.abs(unique_angles - float(selected_alpha))))], dtype=int)
    idx = int(family_idx[0])
    top_idx = int(np.argmax(family_weights))

    lower_slice = slice(0, idx)
    upper_slice = slice(idx + 1, len(unique_angles))
    lower_weights = family_weights[lower_slice]
    lower_angles = unique_angles[lower_slice]
    lower_counts = family_counts[lower_slice]
    upper_weights = family_weights[upper_slice]
    upper_angles = unique_angles[upper_slice]

    if len(lower_weights):
        nearest_lower_share = float(family_weights[idx - 1])
        nearest_lower_gap = float(selected_alpha - unique_angles[idx - 1])
        best_lower_idx = int(np.argmax(lower_weights))
        best_lower_share = float(lower_weights[best_lower_idx])
        best_lower_gap = float(selected_alpha - lower_angles[best_lower_idx])
        best_lower_count = float(lower_counts[best_lower_idx])
    else:
        nearest_lower_share = 0.0
        nearest_lower_gap = np.nan
        best_lower_share = 0.0
        best_lower_gap = np.nan
        best_lower_count = 0.0

    if len(upper_weights):
        nearest_upper_share = float(family_weights[idx + 1])
        nearest_upper_gap = float(unique_angles[idx + 1] - selected_alpha)
        best_upper_idx = int(np.argmax(upper_weights))
        best_upper_share = float(upper_weights[best_upper_idx])
        best_upper_gap = float(upper_angles[best_upper_idx] - selected_alpha)
    else:
        nearest_upper_share = 0.0
        nearest_upper_gap = np.nan
        best_upper_share = 0.0
        best_upper_gap = np.nan

    cum_before = float(np.sum(lower_weights))
    selected_share = float(family_weights[idx])
    cum_after = float(np.sum(upper_weights))
    return {
        "n_peaks": float(len(alphas)),
        "n_unique_angles": float(len(unique_angles)),
        "selected_family_share": selected_share,
        "selected_family_count": float(family_counts[idx]),
        "cum_before_selected": cum_before,
        "cum_after_selected": cum_after,
        "crossing_margin_under": float(0.5 - cum_before),
        "crossing_margin_over": float(cum_before + selected_share - 0.5),
        "nearest_lower_share": nearest_lower_share,
        "nearest_lower_gap": nearest_lower_gap,
        "best_lower_share": best_lower_share,
        "best_lower_gap": best_lower_gap,
        "best_lower_count": best_lower_count,
        "nearest_upper_share": nearest_upper_share,
        "nearest_upper_gap": nearest_upper_gap,
        "best_upper_share": best_upper_share,
        "best_upper_gap": best_upper_gap,
        "top_family_share": float(family_weights[top_idx]),
        "top_family_angle_offset": float(unique_angles[top_idx] - selected_alpha),
        "selected_is_top_family": float(idx == top_idx),
        "duplicate_fraction": float(1.0 - len(unique_angles) / max(len(alphas), 1)),
    }


def build_feature_table(base_entries: list[dict], base_pf: pd.DataFrame, localmax_pf: pd.DataFrame) -> pd.DataFrame:
    detector_table = nb90.build_detector_table(base_entries, base_pf, localmax_pf)
    rows = []
    for frame, entry in enumerate(base_entries):
        rows.append(
            {
                "frame": int(frame),
                **family_structure_features(entry, float(base_pf.loc[frame, "variant_raw_alpha_deg"])),
            }
        )
    return detector_table.join(pd.DataFrame(rows).set_index("frame"))


def apply_rule(feature_table: pd.DataFrame, rule: list[tuple[str, str, float]]) -> pd.Series:
    flagged = pd.Series(True, index=feature_table.index, dtype=bool)
    for column, op, threshold in rule:
        series = feature_table[column]
        if op == ">=":
            flagged &= series >= float(threshold)
        elif op == "<=":
            flagged &= series <= float(threshold)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return flagged.fillna(False)


def detector_metrics(flagged: pd.Series, oracle: pd.Series) -> dict[str, float]:
    flagged_bool = flagged.astype(bool)
    oracle_bool = oracle.astype(bool)
    tp = int((flagged_bool & oracle_bool).sum())
    fp = int((flagged_bool & ~oracle_bool).sum())
    fn = int((~flagged_bool & oracle_bool).sum())
    return {
        "flagged_frames": int(flagged_bool.sum()),
        "oracle_tp": tp,
        "oracle_fp": fp,
        "oracle_fn": fn,
        "oracle_precision": float(tp / max(tp + fp, 1)),
        "oracle_recall": float(tp / max(tp + fn, 1)),
    }


def render_rule(rule: list[tuple[str, str, float]]) -> str:
    pieces = []
    for column, op, threshold in rule:
        if float(threshold).is_integer():
            t = str(int(threshold))
        else:
            t = f"{threshold:.2f}".rstrip("0").rstrip(".")
        pieces.append(f"{column} {op} {t}")
    return " and ".join(pieces)


def search_candidate_rules(feature_table: pd.DataFrame) -> pd.DataFrame:
    atoms = [
        ("mass_below_10deg", ">=", 0.18),
        ("mass_below_10deg", ">=", 0.22),
        ("mass_below_10deg", ">=", 0.26),
        ("mass_below_10deg", ">=", 0.30),
        ("gap_to_lower_deg", ">=", 4.0),
        ("gap_to_lower_deg", ">=", 6.0),
        ("gap_to_lower_deg", ">=", 8.0),
        ("best_lower_gap", ">=", 5.0),
        ("best_lower_gap", ">=", 9.0),
        ("best_upper_share", "<=", 0.10),
        ("best_upper_share", "<=", 0.12),
        ("best_upper_share", "<=", 0.14),
        ("best_upper_share", "<=", 0.16),
        ("top_family_angle_offset", "<=", -6.0),
        ("top_family_angle_offset", "<=", -4.0),
        ("top_family_angle_offset", "<=", -2.0),
        ("delta_localmax_deg", ">=", 6.0),
        ("delta_localmax_deg", ">=", 8.0),
        ("selected_is_top_family", "<=", 0.0),
        ("selected_family_share", "<=", 0.18),
        ("cum_before_selected", ">=", 0.46),
    ]

    oracle = feature_table["oracle_high_angle_overweight"].astype(bool)
    masks: list[pd.Series] = [apply_rule(feature_table, [atom]) for atom in atoms]
    seen_masks: dict[bytes, dict] = {}

    for r in (1, 2, 3):
        for idxs in combinations(range(len(atoms)), r):
            flagged = pd.Series(True, index=feature_table.index, dtype=bool)
            rule = [atoms[i] for i in idxs]
            for idx in idxs:
                flagged &= masks[idx]
            flagged = flagged.fillna(False)
            n_flagged = int(flagged.sum())
            if n_flagged < 5 or n_flagged > 250:
                continue
            signature = flagged.to_numpy(dtype=np.uint8).tobytes()
            metrics = detector_metrics(flagged, oracle)
            record = {
                "rule_text": render_rule(rule),
                "rule_json": json.dumps(rule),
                **metrics,
            }
            previous = seen_masks.get(signature)
            if previous is None or (metrics["oracle_precision"], metrics["oracle_recall"], -len(rule)) > (
                previous["oracle_precision"],
                previous["oracle_recall"],
                -len(json.loads(previous["rule_json"])),
            ):
                seen_masks[signature] = record

    frontier = pd.DataFrame(seen_masks.values()).sort_values(
        ["oracle_precision", "oracle_recall", "flagged_frames"], ascending=[False, False, True]
    )
    return frontier.reset_index(drop=True)


def choose_rule(rule_table: pd.DataFrame, *, min_flagged: int, max_flagged: int) -> dict:
    subset = rule_table[(rule_table["flagged_frames"] >= min_flagged) & (rule_table["flagged_frames"] <= max_flagged)].copy()
    if subset.empty:
        subset = rule_table.copy()
    subset = subset.sort_values(["oracle_precision", "oracle_recall", "flagged_frames"], ascending=[False, False, True])
    return subset.iloc[0].to_dict()


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

    feature_table = build_feature_table(base_entries, base_pf, localmax_pf)
    rule_table = search_candidate_rules(feature_table)

    sparse_rule_row = choose_rule(rule_table, min_flagged=5, max_flagged=15)
    balanced_rule_row = choose_rule(rule_table, min_flagged=16, max_flagged=60)
    sparse_rule = json.loads(sparse_rule_row["rule_json"])
    balanced_rule = json.loads(balanced_rule_row["rule_json"])

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

    detector_defs = [
        ("baseline_current", "no conditional patch", pd.Series(False, index=feature_table.index)),
        (
            "oracle_overweight_to_localmax",
            "MATLAB-aware ceiling: baseline nearest peak <=2 deg and baseline raw >5 deg too high",
            feature_table["oracle_high_angle_overweight"],
        ),
        (
            "nb90_mass10_gap4_reference",
            "Notebook 90 broad reference: mass below alpha-10 deg >=0.25 and nearest lower-gap >=4 deg",
            (feature_table["mass_below_10deg"] >= 0.25) & (feature_table["gap_to_lower_deg"] >= 4.0),
        ),
        (
            "exact_sparse_peakshape",
            f"Auto-selected sparse high-precision rule: {render_rule(sparse_rule)}",
            apply_rule(feature_table, sparse_rule),
        ),
        (
            "exact_balanced_peakshape",
            f"Auto-selected broader exact-structure rule: {render_rule(balanced_rule)}",
            apply_rule(feature_table, balanced_rule),
        ),
    ]

    oracle_series = feature_table["oracle_high_angle_overweight"].astype(bool)
    base_raw = base_pf["variant_raw_alpha_deg"].to_numpy(dtype=float)
    localmax_raw = localmax_pf["variant_raw_alpha_deg"].to_numpy(dtype=float)

    raw_rows = []
    persistence_rows = []
    kalman_rows = []
    detector_rows = []
    per_frame_tables: list[pd.DataFrame] = []

    for variant_name, note, flagged_series in detector_defs:
        print(f"Evaluating notebook 91 detector variant: {variant_name}")
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
        detector_rows.append({"variant": variant_name, **detector_metrics(pd.Series(flagged, index=feature_table.index), oracle_series)})
        per_frame_tables.append(
            pd.DataFrame(
                {
                    "frame": np.arange(len(base_entries), dtype=int),
                    "variant": variant_name,
                    "flagged_for_patch": flagged,
                    "matlab_alpha_deg": mat_alpha,
                    "baseline_raw_alpha_deg": base_raw,
                    "localmax_raw_alpha_deg": localmax_raw,
                    "variant_raw_alpha_deg": raw_alpha,
                    "variant_selected_alpha_deg": selected_alpha,
                    "variant_final_alpha_deg": np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1],
                    "raw_minus_matlab_deg": raw_alpha - mat_alpha,
                    "selected_minus_matlab_deg": selected_alpha - mat_alpha,
                    "final_alpha_minus_matlab_deg": np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1] - mat_final_alpha,
                    "FL_mm": np.asarray(kalman["FL_mm"], dtype=np.float64),
                    "FL_minus_matlab_mm": np.asarray(kalman["FL_mm"], dtype=np.float64) - mat_fl,
                }
            )
        )

    raw_table = pd.DataFrame(raw_rows)
    persistence_table = pd.DataFrame(persistence_rows)
    kalman_table = pd.DataFrame(kalman_rows)
    detector_table = pd.DataFrame(detector_rows)
    full_table = raw_table.merge(persistence_table, on="variant").merge(kalman_table, on="variant").merge(detector_table, on="variant")
    per_frame_table = pd.concat(per_frame_tables, ignore_index=True)

    raw_path = OUT / "detector_raw_metrics.csv"
    persistence_path = OUT / "detector_persistence_metrics.csv"
    kalman_path = OUT / "detector_downstream_metrics.csv"
    detector_path = OUT / "detector_stats.csv"
    summary_path = OUT / "detector_variant_summary.csv"
    per_frame_path = OUT / "detector_per_frame.csv"
    feature_path = OUT / "peakweight_feature_table.csv"
    rule_path = OUT / "candidate_rule_frontier.csv"
    selected_rule_path = OUT / "selected_rules.json"

    raw_table.to_csv(raw_path, index=False)
    persistence_table.to_csv(persistence_path, index=False)
    kalman_table.to_csv(kalman_path, index=False)
    detector_table.to_csv(detector_path, index=False)
    full_table.to_csv(summary_path, index=False)
    per_frame_table.to_csv(per_frame_path, index=False)
    feature_table.reset_index().to_csv(feature_path, index=False)
    rule_table.to_csv(rule_path, index=False)
    selected_rule_path.write_text(
        json.dumps(
            {
                "sparse_rule": sparse_rule,
                "sparse_rule_text": render_rule(sparse_rule),
                "balanced_rule": balanced_rule,
                "balanced_rule_text": render_rule(balanced_rule),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    baseline = full_table.loc[full_table["variant"] == "baseline_current"].iloc[0]
    sparse_variant = full_table.loc[full_table["variant"] == "exact_sparse_peakshape"].iloc[0]
    balanced_variant = full_table.loc[full_table["variant"] == "exact_balanced_peakshape"].iloc[0]
    best_fl = full_table.loc[full_table["FL_vs_matlab_rmse"].idxmin()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    frontier_plot = rule_table.head(80).copy()
    axes[0].scatter(frontier_plot["flagged_frames"], frontier_plot["oracle_precision"], s=18, alpha=0.75)
    axes[0].set_title("Exact-structure rule frontier")
    axes[0].set_xlabel("flagged frames")
    axes[0].set_ylabel("oracle precision")
    axes[0].grid(True, alpha=0.25)

    x = np.arange(len(full_table))
    axes[1].bar(x, full_table["oracle_precision"])
    axes[1].set_title("Detector precision vs oracle subset")
    axes[1].set_ylabel("precision")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(full_table["variant"], rotation=20, ha="right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, full_table["FL_vs_matlab_rmse"])
    axes[2].set_title("Final FL RMSE after conditional patch")
    axes[2].set_ylabel("RMSE (mm)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(full_table["variant"], rotation=20, ha="right", fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    frontier_plot_path = OUT / "peakweight_detector_frontier.png"
    fig.savefig(frontier_plot_path, dpi=180)
    plt.close(fig)

    show_n = min(350, len(base_entries))
    frames = np.arange(show_n)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(frames, mat_alpha[:show_n], label="MATLAB geofeature alpha", linewidth=1.2)
    for variant_name in ["baseline_current", "nb90_mass10_gap4_reference", "exact_sparse_peakshape", "exact_balanced_peakshape"]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "variant_raw_alpha_deg"].to_numpy(dtype=float)
        axes[0].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[0].set_title("Raw alpha under notebook 91 exact-structure detectors")
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(frames, mat_fl[:show_n], label="MATLAB FL", linewidth=1.2)
    for variant_name in ["baseline_current", "nb90_mass10_gap4_reference", "exact_sparse_peakshape", "exact_balanced_peakshape"]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "FL_mm"].to_numpy(dtype=float)
        axes[1].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[1].set_title("Downstream FL under notebook 91 exact-structure detectors")
    axes[1].set_xlabel("aligned frame index")
    axes[1].set_ylabel("FL (mm)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    trace_plot_path = OUT / "peakweight_detector_traces.png"
    fig.savefig(trace_plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 91 — exact peak/weight structure detector search",
        "",
        "This notebook stays in notebook-only territory: it uses the oracle overweight subset from notebook 90 only as a label,",
        "then searches for small Python-only detectors built from exact Hough peak-family structure around the weighted-median crossing.",
        "",
        "## Auto-selected rules",
        "",
        f"- Sparse high-precision rule: `{render_rule(sparse_rule)}`",
        f"- Broader exact-structure rule: `{render_rule(balanced_rule)}`",
        "",
        "## Key findings",
        "",
        f"- Sparse rule precision/recall: {sparse_variant['oracle_precision']:.3f} / {sparse_variant['oracle_recall']:.3f} on {int(sparse_variant['flagged_frames'])} flagged frames.",
        f"- Broader rule precision/recall: {balanced_variant['oracle_precision']:.3f} / {balanced_variant['oracle_recall']:.3f} on {int(balanced_variant['flagged_frames'])} flagged frames.",
        f"- Best downstream FL RMSE among replayed variants is `{best_fl['variant']}` at {best_fl['FL_vs_matlab_rmse']:.4f} mm versus baseline {baseline['FL_vs_matlab_rmse']:.4f} mm.",
        "",
        "## Interpretation",
        "",
        "- If the sparse rule helps, then a very pure but low-recall detector may already be enough to improve a few pathological segments.",
        "- If the broader exact-structure rule still underperforms the notebook 90 broad heuristic, then detector purity alone is not the full story and we still need a better coverage mechanism.",
        "- The exact-family features tell us whether the overweight frames are defined by weak upper competitors, large lower-side gaps, or a lower family that remains structurally dominant even when the weighted median lands higher.",
        "",
        f"- Feature table CSV: `{feature_path}`",
        f"- Candidate rule frontier CSV: `{rule_path}`",
        f"- Selected rules JSON: `{selected_rule_path}`",
        f"- Raw metrics CSV: `{raw_path}`",
        f"- Persistence metrics CSV: `{persistence_path}`",
        f"- Downstream metrics CSV: `{kalman_path}`",
        f"- Detector stats CSV: `{detector_path}`",
        f"- Combined summary CSV: `{summary_path}`",
        f"- Per-frame CSV: `{per_frame_path}`",
        f"- Frontier plot: `{frontier_plot_path}`",
        f"- Trace plot: `{trace_plot_path}`",
    ]
    notebook_summary_path = OUT / "notebook91_summary.md"
    notebook_summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
