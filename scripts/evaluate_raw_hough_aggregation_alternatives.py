#!/usr/bin/env python3
"""Notebook 85 helper: raw Hough aggregation experiments on fixed candidate sets."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Callable

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
from scripts.run_strict_ultratimtrack_video import (
    FascicleCandidatePersistenceConfig,
    select_fascicle_candidate_persistence,
)
from ultrasound_tracker.matlab_timtrack import run_timtrack_geofeatures_from_video
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
UTT_EXPORT = Path("/Users/grosbedou/Documents/GitHub/UltraTimTrack/UTT_numeric_export.mat")
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "UltraTimTrack_test.mp4"
ROI_PATH = PROJECT_ROOT / "data" / "rois" / "UltraTimTrack_test_rois.json"
METADATA_PATH = RUN_DIR / "UltraTimTrack_test_strict_metadata.json"
OUT = PROJECT_ROOT / "results" / "notebook85_raw_hough_aggregation_alternatives"


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


def ellipse_mask_from_roi(shape: tuple[int, int], fascicle_roi: list[float]) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, shape)
    x, y, w, h = map(float, fascicle_roi)
    rx = max(w / 2.0, 1.0)
    ry = max(h / 2.0, 1.0)
    cx = x + rx + 1.0
    cy = y + ry + 1.0
    yy, xx = np.mgrid[1 : height + 1, 1 : width + 1]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return mask.astype(bool), np.asarray([ry, rx], dtype=np.float64)


def make_matlab_apox(width: int) -> np.ndarray:
    return np.arange(1, int(width) + 1, dtype=np.float64)


def update_parms_from_rois(parms: dict, rois: dict[str, list[float]], frame_shape: tuple[int, int]) -> dict:
    out = copy.deepcopy(parms)
    height, width = map(int, frame_shape)
    if "apo" not in out:
        out["apo"] = {}
    if "fas" not in out:
        out["fas"] = {}
    out["apo"]["apox"] = make_matlab_apox(width)
    for name, key in [("superficial", "super"), ("deep", "deep")]:
        if name in rois:
            _, y, _, h = rois[name]
            out["apo"].setdefault(key, {})
            out["apo"][key]["cut"] = np.asarray([y / height, (y + h) / height], dtype=np.float64)
    if "fascicle" in rois:
        emask, radius = ellipse_mask_from_roi((height, width), rois["fascicle"])
        out["fas"]["Emask"] = emask
        out["fas"]["Emask_radius"] = radius
        out["fas"]["redo_ROI"] = 0
    return out


def _valid_candidate_arrays(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    alphas = np.asarray(entry.get("alphas", []), dtype=np.float64).reshape(-1)
    weights = np.asarray(entry.get("weights", entry.get("ws", [])), dtype=np.float64).reshape(-1)
    n = min(len(alphas), len(weights))
    alphas = alphas[:n]
    weights = weights[:n]
    valid = np.isfinite(alphas) & np.isfinite(weights) & (weights > 0)
    return alphas[valid], weights[valid]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        return float("nan")
    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    w = w[order] / np.sum(w)
    cumulative = np.cumsum(w)
    return float(vals[int(np.argmax(cumulative >= float(q)))])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        return float("nan")
    return float(np.average(vals, weights=w))


def top_peak(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        return float("nan")
    return float(vals[int(np.argmax(w))])


def apply_aggregation(geofeatures: list[dict], aggregator: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    out = np.full(len(geofeatures), np.nan, dtype=np.float64)
    for idx, entry in enumerate(geofeatures):
        vals, weights = _valid_candidate_arrays(entry)
        out[idx] = aggregator(vals, weights)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    mat = loadmat(MATLAB_RESULT, simplify_cells=True)
    utt = loadmat(UTT_EXPORT, simplify_cells=True)["UTT_numeric_export"]
    metadata = json.loads(METADATA_PATH.read_text())
    rois = json.loads(ROI_PATH.read_text())

    mat_region = mat["Fdat"]["Region"]
    matlab_time = cmp.as_float1(mat_region["Time"])
    python_time = cmp.as_float1(npz["time_s"])
    python_offset = cmp.choose_python_offset(matlab_time, python_time)
    n = min(len(matlab_time), len(python_time) - python_offset)
    sl = slice(python_offset, python_offset + n)

    video_probe = loadmat(UTT_EXPORT, simplify_cells=True)["UTT_numeric_export"]
    frame_shape = (int(video_probe["vidHeight"]), int(video_probe["vidWidth"]))
    parms = update_parms_from_rois(utt["parms"], rois, frame_shape)

    print("Recomputing full-sequence TimTrack candidate sets with the baseline raw chain...")
    geofeatures = run_timtrack_geofeatures_from_video(
        str(VIDEO_PATH),
        parms,
        limit=n + python_offset,
        subtraction_mode="matlab_literal",
        emask_mode="matlab",
        keep_debug=False,
        progress_every=300,
    )
    geofeatures = geofeatures[python_offset : python_offset + n]

    mat_geof = utt["geofeatures"][:n]
    mat_alpha = np.asarray([float(np.asarray(entry["alpha"], dtype=np.float64).reshape(-1)[0]) for entry in mat_geof], dtype=np.float64)
    py_saved_raw = np.asarray(npz["raw_timtrack_alpha_deg"], dtype=np.float64)[sl]

    aggregation_rules: list[tuple[str, Callable[[np.ndarray, np.ndarray], float], str]] = [
        ("weighted_median_current", lambda v, w: weighted_quantile(v, w, 0.5), "current raw rule"),
        ("weighted_mean", weighted_mean, "same candidates, weighted mean"),
        ("weighted_q25", lambda v, w: weighted_quantile(v, w, 0.25), "same candidates, weighted 25th percentile"),
        ("weighted_q30", lambda v, w: weighted_quantile(v, w, 0.30), "same candidates, weighted 30th percentile"),
        ("top_peak", top_peak, "same candidates, strongest single peak"),
    ]

    raw_variant_rows = []
    persistence_variant_rows = []
    kalman_variant_rows = []
    per_frame_tables: list[pd.DataFrame] = []

    klt = np.asarray(npz["klt_prior_segments"], dtype=np.float64)[sl]
    affines = np.asarray(npz["klt_affine_matrices"], dtype=np.float64)[sl]
    sup_lines = np.asarray(npz["sup_apo_lines"], dtype=np.float64)[sl]
    deep_lines = np.asarray(npz["deep_apo_lines"], dtype=np.float64)[sl]
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

    mat_final_alpha = cmp.as_float1(mat_region["Fascicle"]["alpha"])[:n]
    mat_fl = cmp.as_float1(mat_region["FL"])[:n]
    mat_ang = cmp.as_float1(mat_region["ANG"])[:n]

    for variant_name, aggregator, note in aggregation_rules:
        print(f"Evaluating aggregation variant: {variant_name}")
        raw_alpha = apply_aggregation(geofeatures, aggregator)
        persistence = select_fascicle_candidate_persistence(geofeatures, raw_alpha, config=persistence_cfg)
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

        raw_variant_rows.append(
            {
                "variant": variant_name,
                "note": note,
                "recomputed_vs_saved_python_raw_rmse_deg": scalar_metrics(py_saved_raw, raw_alpha)["rmse"],
                **{f"raw_vs_matlab_{k}": v for k, v in scalar_metrics(mat_alpha, raw_alpha).items()},
            }
        )
        persistence_variant_rows.append(
            {
                "variant": variant_name,
                **{f"selected_vs_matlab_{k}": v for k, v in scalar_metrics(mat_alpha, selected_alpha).items()},
                "raw_rejected_fraction": float(np.mean(np.asarray(persistence["raw_alpha_rejected"], dtype=bool))),
            }
        )
        kalman_variant_rows.append(
            {
                "variant": variant_name,
                **{f"final_alpha_vs_matlab_{k}": v for k, v in scalar_metrics(mat_final_alpha, np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1]).items()},
                **{f"FL_vs_matlab_{k}": v for k, v in scalar_metrics(mat_fl, np.asarray(kalman["FL_mm"], dtype=np.float64)).items()},
                **{f"ANG_vs_matlab_{k}": v for k, v in scalar_metrics(mat_ang, np.asarray(kalman["ANG_deg"], dtype=np.float64)).items()},
            }
        )
        per_frame = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=int),
                "variant": variant_name,
                "matlab_alpha_deg": mat_alpha,
                "saved_python_raw_alpha_deg": py_saved_raw,
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
        per_frame_tables.append(per_frame)

    raw_table = pd.DataFrame(raw_variant_rows)
    persistence_table = pd.DataFrame(persistence_variant_rows)
    kalman_table = pd.DataFrame(kalman_variant_rows)
    full_table = raw_table.merge(persistence_table, on="variant").merge(kalman_table, on="variant")
    per_frame_table = pd.concat(per_frame_tables, ignore_index=True)

    raw_path = OUT / "aggregation_raw_metrics.csv"
    persistence_path = OUT / "aggregation_persistence_metrics.csv"
    kalman_path = OUT / "aggregation_downstream_metrics.csv"
    full_path = OUT / "aggregation_full_summary.csv"
    per_frame_path = OUT / "aggregation_per_frame.csv"
    raw_table.to_csv(raw_path, index=False)
    persistence_table.to_csv(persistence_path, index=False)
    kalman_table.to_csv(kalman_path, index=False)
    full_table.to_csv(full_path, index=False)
    per_frame_table.to_csv(per_frame_path, index=False)

    best_raw = raw_table.loc[raw_table["raw_vs_matlab_rmse"].idxmin()]
    best_fl = kalman_table.loc[kalman_table["FL_vs_matlab_rmse"].idxmin()]
    baseline = full_table.loc[full_table["variant"] == "weighted_median_current"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    labels = full_table["variant"].tolist()
    x = np.arange(len(labels))
    axes[0].bar(x, raw_table["raw_vs_matlab_rmse"])
    axes[0].set_title("Raw alpha RMSE vs MATLAB geofeature")
    axes[0].set_ylabel("RMSE (deg)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, persistence_table["selected_vs_matlab_rmse"])
    axes[1].set_title("Selected alpha RMSE after persistence")
    axes[1].set_ylabel("RMSE (deg)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, kalman_table["FL_vs_matlab_rmse"])
    axes[2].set_title("Final FL RMSE after Kalman")
    axes[2].set_ylabel("RMSE (mm)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    summary_plot_path = OUT / "aggregation_alternative_summary.png"
    fig.savefig(summary_plot_path, dpi=180)
    plt.close(fig)

    show_n = min(350, n)
    frames = np.arange(show_n)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(frames, mat_alpha[:show_n], label="MATLAB geofeature alpha", linewidth=1.2)
    for variant_name in ["weighted_median_current", "weighted_mean", "weighted_q25"]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "variant_raw_alpha_deg"].to_numpy(dtype=float)
        axes[0].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[0].set_title("Raw alpha alternatives on the same candidate sets")
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(frames, mat_fl[:show_n], label="MATLAB FL", linewidth=1.2)
    for variant_name in ["weighted_median_current", "weighted_mean", "weighted_q25"]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "FL_mm"].to_numpy(dtype=float)
        axes[1].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[1].set_title("Downstream FL effect of raw aggregation alternatives")
    axes[1].set_xlabel("aligned frame index")
    axes[1].set_ylabel("FL (mm)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    trace_plot_path = OUT / "aggregation_alternative_traces.png"
    fig.savefig(trace_plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 85 — raw Hough aggregation alternatives",
        "",
        f"Recomputed the full baseline TimTrack candidate stream for {n} aligned frames, then held those candidate sets fixed while swapping only the raw Hough aggregation rule.",
        "",
        "This notebook does not patch production code. It asks whether a different raw aggregation rule improves parity on the same candidate sets, or merely compensates for upstream candidate bias.",
        "",
        "## Raw aggregation findings",
        "",
        f"- The current weighted-median rule reproduces the saved Python raw alpha exactly (raw-vs-saved RMSE {baseline['recomputed_vs_saved_python_raw_rmse_deg']:.6f} deg), so it is the correct baseline.",
        f"- On the same candidate sets, the best raw-alpha RMSE against MATLAB geofeature alpha is `{best_raw['variant']}` at {best_raw['raw_vs_matlab_rmse']:.4f} deg, versus {baseline['raw_vs_matlab_rmse']:.4f} deg for the current weighted median.",
        f"- The best downstream FL RMSE after the same persistence and Kalman path is `{best_fl['variant']}` at {best_fl['FL_vs_matlab_rmse']:.4f} mm, versus {baseline['FL_vs_matlab_rmse']:.4f} mm for the current weighted median.",
        "",
        "## Interpretation",
        "",
        "- If a non-MATLAB aggregation rule improves raw alpha and downstream FL on the same candidate sets, that does not prove MATLAB's aggregation is wrong.",
        "- It means the current Python candidate distribution is biased enough that a different aggregator can partially compensate for upstream errors.",
        "- So any future code patch here would be a modeling change, not a pure parity fix, unless we also show the candidate-generation side is already correct.",
        "- This notebook therefore helps rank ideas, but it is not by itself a green light to change the production aggregation rule.",
        "",
        f"- Raw metrics CSV: `{raw_path}`",
        f"- Persistence metrics CSV: `{persistence_path}`",
        f"- Downstream metrics CSV: `{kalman_path}`",
        f"- Combined summary CSV: `{full_path}`",
        f"- Per-frame CSV: `{per_frame_path}`",
        f"- Summary plot: `{summary_plot_path}`",
        f"- Trace plot: `{trace_plot_path}`",
    ]
    summary_path = OUT / "notebook85_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
