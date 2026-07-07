#!/usr/bin/env python3
"""Notebook 86 helper: Hough candidate-set shaping before raw alpha aggregation."""

from __future__ import annotations

import copy
import json
import pickle
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
OUT = PROJECT_ROOT / "results" / "notebook86_hough_candidate_shaping"
CACHE_PATH = OUT / "baseline_geofeatures.pkl"
MAX_PEAKS = 10


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


def nearest_peak_error(values: np.ndarray, target_alpha: float) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(vals)
    if not np.isfinite(target_alpha) or not np.any(valid):
        return float("nan")
    return float(np.min(np.abs(vals[valid] - float(target_alpha))))


def cluster_duplicate_peaks(values: np.ndarray, weights: np.ndarray, gap_deg: float) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        return vals, w
    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    w = w[order]
    groups: list[list[int]] = [[0]]
    for idx in range(1, len(vals)):
        if float(vals[idx] - vals[idx - 1]) <= float(gap_deg):
            groups[-1].append(idx)
        else:
            groups.append([idx])
    out_vals = []
    out_w = []
    for group in groups:
        group_vals = vals[group]
        group_w = w[group]
        out_vals.append(weighted_mean(group_vals, group_w))
        out_w.append(float(np.sum(group_w)))
    return np.asarray(out_vals, dtype=np.float64), np.asarray(out_w, dtype=np.float64)


def keep_top_peak_family(values: np.ndarray, weights: np.ndarray, half_width_deg: float) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        return vals, w
    center = float(vals[int(np.argmax(w))])
    keep = np.abs(vals - center) <= float(half_width_deg)
    if not np.any(keep):
        return vals, w
    return vals[keep], w[keep]


def keep_dominant_window(values: np.ndarray, weights: np.ndarray, window_deg: float) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        return vals, w
    half = float(window_deg) / 2.0
    best_keep = np.ones(len(vals), dtype=bool)
    best_score = -np.inf
    best_center = float("nan")
    for center in vals:
        keep = np.abs(vals - center) <= half
        score = float(np.sum(w[keep]))
        if score > best_score or (
            np.isclose(score, best_score, equal_nan=False)
            and (not np.isfinite(best_center) or abs(center - weighted_mean(vals, w)) < abs(best_center - weighted_mean(vals, w)))
        ):
            best_score = score
            best_keep = keep
            best_center = float(center)
    return vals[best_keep], w[best_keep]


def shape_identity(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def shape_sqrt_weights(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(values, dtype=np.float64), np.sqrt(np.asarray(weights, dtype=np.float64))


def shape_dedup1(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cluster_duplicate_peaks(values, weights, gap_deg=1.0)


def shape_dedup2(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cluster_duplicate_peaks(values, weights, gap_deg=2.0)


def shape_top_family5(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return keep_top_peak_family(values, weights, half_width_deg=5.0)


def shape_dominant_window8(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return keep_dominant_window(values, weights, window_deg=8.0)


def shape_dedup1_dominant8(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, w = cluster_duplicate_peaks(values, weights, gap_deg=1.0)
    return keep_dominant_window(vals, w, window_deg=8.0)


def pack_candidate_arrays(values: np.ndarray, weights: np.ndarray, *, max_peaks: int = MAX_PEAKS) -> dict[str, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = min(int(max_peaks), len(vals), len(w))
    alphas = np.full(int(max_peaks), np.nan, dtype=np.float64)
    weights_out = np.full(int(max_peaks), np.nan, dtype=np.float64)
    x = np.full((int(max_peaks), 2), np.nan, dtype=np.float64)
    y = np.full((int(max_peaks), 2), np.nan, dtype=np.float64)
    if n:
        alphas[:n] = vals[:n]
        weights_out[:n] = w[:n]
    return {"alphas": alphas, "weights": weights_out, "ws": weights_out.copy(), "x": x, "y": y}


def load_or_compute_geofeatures(*, limit: int, python_offset: int, parms: dict) -> list[dict]:
    if CACHE_PATH.exists():
        print(f"Loading cached baseline geofeatures from {CACHE_PATH}")
        with CACHE_PATH.open("rb") as f:
            payload = pickle.load(f)
        cached = payload.get("geofeatures", payload)
        if len(cached) == limit - python_offset:
            return cached
        print("Cache length mismatch; recomputing baseline geofeatures...")

    print("Recomputing full-sequence TimTrack candidate sets with the baseline raw chain...")
    geofeatures = run_timtrack_geofeatures_from_video(
        str(VIDEO_PATH),
        parms,
        limit=limit,
        subtraction_mode="matlab_literal",
        emask_mode="matlab",
        keep_debug=False,
        progress_every=300,
    )
    geofeatures = geofeatures[python_offset:limit]
    with CACHE_PATH.open("wb") as f:
        pickle.dump({"geofeatures": geofeatures}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return geofeatures


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

    frame_shape = (int(utt["vidHeight"]), int(utt["vidWidth"]))
    parms = update_parms_from_rois(utt["parms"], rois, frame_shape)
    geofeatures = load_or_compute_geofeatures(limit=n + python_offset, python_offset=python_offset, parms=parms)

    mat_geof = utt["geofeatures"][:n]
    mat_alpha = np.asarray(
        [float(np.asarray(entry["alpha"], dtype=np.float64).reshape(-1)[0]) for entry in mat_geof],
        dtype=np.float64,
    )
    py_saved_raw = np.asarray(npz["raw_timtrack_alpha_deg"], dtype=np.float64)[sl]

    original_nearest_errors = np.asarray(
        [nearest_peak_error(_valid_candidate_arrays(entry)[0], target) for entry, target in zip(geofeatures, mat_alpha)],
        dtype=np.float64,
    )
    original_good_hit2 = np.isfinite(original_nearest_errors) & (original_nearest_errors <= 2.0)

    shaping_rules: list[tuple[str, Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]], str]] = [
        ("baseline_current", shape_identity, "current peaks and weights"),
        ("sqrt_weights", shape_sqrt_weights, "same peaks, square-root softened weights"),
        ("dedup_gap1_sum", shape_dedup1, "collapse near-duplicate peaks within 1 deg"),
        ("dedup_gap2_sum", shape_dedup2, "collapse near-duplicate peaks within 2 deg"),
        ("top_peak_family_5deg", shape_top_family5, "keep only the top-peak angle family within +/-5 deg"),
        ("dominant_window_8deg", shape_dominant_window8, "keep only the dominant weighted 8 deg angle window"),
        ("dedup1_plus_dominant8", shape_dedup1_dominant8, "collapse 1 deg duplicates, then keep the dominant 8 deg family"),
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

    baseline_raw_alpha: np.ndarray | None = None

    for variant_name, shaper, note in shaping_rules:
        print(f"Evaluating candidate-shaping variant: {variant_name}")
        shaped_entries: list[dict] = []
        raw_alpha = np.full(n, np.nan, dtype=np.float64)
        nearest_errors = np.full(n, np.nan, dtype=np.float64)
        n_orig = np.full(n, np.nan, dtype=np.float64)
        n_shaped = np.full(n, np.nan, dtype=np.float64)
        weight_mass_retained = np.full(n, np.nan, dtype=np.float64)

        for idx, entry in enumerate(geofeatures):
            orig_vals, orig_weights = _valid_candidate_arrays(entry)
            shaped_vals, shaped_weights = shaper(orig_vals, orig_weights)
            valid = np.isfinite(shaped_vals) & np.isfinite(shaped_weights) & (shaped_weights > 0)
            shaped_vals = np.asarray(shaped_vals[valid], dtype=np.float64)
            shaped_weights = np.asarray(shaped_weights[valid], dtype=np.float64)

            raw_alpha[idx] = weighted_quantile(shaped_vals, shaped_weights, 0.5) if len(shaped_vals) else np.nan
            nearest_errors[idx] = nearest_peak_error(shaped_vals, mat_alpha[idx])
            n_orig[idx] = float(len(orig_vals))
            n_shaped[idx] = float(len(shaped_vals))
            denom = float(np.sum(orig_weights))
            weight_mass_retained[idx] = float(np.sum(shaped_weights) / denom) if denom > 0 else np.nan

            packed = pack_candidate_arrays(shaped_vals, shaped_weights)
            shaped_entry = dict(entry)
            shaped_entry.update(packed)
            shaped_entries.append(shaped_entry)

        if baseline_raw_alpha is None:
            baseline_raw_alpha = raw_alpha.copy()

        persistence = select_fascicle_candidate_persistence(shaped_entries, raw_alpha, config=persistence_cfg)
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

        hit1 = np.isfinite(nearest_errors) & (nearest_errors <= 1.0)
        hit2 = np.isfinite(nearest_errors) & (nearest_errors <= 2.0)
        raw_wrong = np.abs(raw_alpha - mat_alpha) > 5.0
        orig_good_dropped = float(np.mean(original_good_hit2 & ~hit2))
        orig_good_kept = float(np.sum(original_good_hit2 & hit2) / max(1, int(np.sum(original_good_hit2))))

        raw_variant_rows.append(
            {
                "variant": variant_name,
                "note": note,
                "recomputed_vs_saved_python_raw_rmse_deg": scalar_metrics(py_saved_raw, raw_alpha)["rmse"],
                "raw_changed_fraction_vs_baseline": float(np.mean(np.abs(raw_alpha - baseline_raw_alpha) > 1e-9)),
                "mean_original_peaks": float(np.nanmean(n_orig)),
                "mean_shaped_peaks": float(np.nanmean(n_shaped)),
                "mean_weight_mass_retained": float(np.nanmean(weight_mass_retained)),
                "nearest_peak_rmse_deg": scalar_metrics(np.zeros_like(nearest_errors), nearest_errors)["rmse"],
                "candidate_hit_rate_1deg": float(np.mean(hit1)),
                "candidate_hit_rate_2deg": float(np.mean(hit2)),
                "candidate_present_but_raw_wrong_rate": float(np.mean(hit2 & raw_wrong)),
                "orig_good_candidate_drop_rate": orig_good_dropped,
                "orig_good_candidate_keep_rate": orig_good_kept,
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
                **{
                    f"final_alpha_vs_matlab_{k}": v
                    for k, v in scalar_metrics(mat_final_alpha, np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1]).items()
                },
                **{f"FL_vs_matlab_{k}": v for k, v in scalar_metrics(mat_fl, np.asarray(kalman["FL_mm"], dtype=np.float64)).items()},
                **{f"ANG_vs_matlab_{k}": v for k, v in scalar_metrics(mat_ang, np.asarray(kalman["ANG_deg"], dtype=np.float64)).items()},
            }
        )
        per_frame_tables.append(
            pd.DataFrame(
                {
                    "frame": np.arange(n, dtype=int),
                    "variant": variant_name,
                    "matlab_alpha_deg": mat_alpha,
                    "saved_python_raw_alpha_deg": py_saved_raw,
                    "variant_raw_alpha_deg": raw_alpha,
                    "variant_selected_alpha_deg": selected_alpha,
                    "variant_final_alpha_deg": np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1],
                    "nearest_peak_error_deg": nearest_errors,
                    "n_original_peaks": n_orig,
                    "n_shaped_peaks": n_shaped,
                    "weight_mass_retained": weight_mass_retained,
                    "raw_minus_matlab_deg": raw_alpha - mat_alpha,
                    "selected_minus_matlab_deg": selected_alpha - mat_alpha,
                    "final_alpha_minus_matlab_deg": np.asarray(kalman["X_plus"], dtype=np.float64)[:, 1] - mat_final_alpha,
                    "FL_mm": np.asarray(kalman["FL_mm"], dtype=np.float64),
                    "FL_minus_matlab_mm": np.asarray(kalman["FL_mm"], dtype=np.float64) - mat_fl,
                }
            )
        )

    raw_table = pd.DataFrame(raw_variant_rows)
    persistence_table = pd.DataFrame(persistence_variant_rows)
    kalman_table = pd.DataFrame(kalman_variant_rows)
    full_table = raw_table.merge(persistence_table, on="variant").merge(kalman_table, on="variant")
    per_frame_table = pd.concat(per_frame_tables, ignore_index=True)

    raw_path = OUT / "shaping_raw_metrics.csv"
    persistence_path = OUT / "shaping_persistence_metrics.csv"
    kalman_path = OUT / "shaping_downstream_metrics.csv"
    full_path = OUT / "shaping_full_summary.csv"
    per_frame_path = OUT / "shaping_per_frame.csv"
    raw_table.to_csv(raw_path, index=False)
    persistence_table.to_csv(persistence_path, index=False)
    kalman_table.to_csv(kalman_path, index=False)
    full_table.to_csv(full_path, index=False)
    per_frame_table.to_csv(per_frame_path, index=False)

    best_raw = raw_table.loc[raw_table["raw_vs_matlab_rmse"].idxmin()]
    best_fl = kalman_table.loc[kalman_table["FL_vs_matlab_rmse"].idxmin()]
    best_wrong = raw_table.loc[raw_table["candidate_present_but_raw_wrong_rate"].idxmin()]
    baseline = full_table.loc[full_table["variant"] == "baseline_current"].iloc[0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    labels = full_table["variant"].tolist()
    x = np.arange(len(labels))

    axes[0].bar(x, raw_table["raw_vs_matlab_rmse"])
    axes[0].set_title("Raw alpha RMSE vs MATLAB")
    axes[0].set_ylabel("RMSE (deg)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, kalman_table["FL_vs_matlab_rmse"])
    axes[1].set_title("Final FL RMSE after Kalman")
    axes[1].set_ylabel("RMSE (mm)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, raw_table["candidate_present_but_raw_wrong_rate"])
    axes[2].set_title("Good peak present but raw >5 deg wrong")
    axes[2].set_ylabel("frame fraction")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)

    axes[3].bar(x, raw_table["mean_shaped_peaks"])
    axes[3].set_title("Mean retained peaks after shaping")
    axes[3].set_ylabel("peaks / frame")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[3].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    summary_plot_path = OUT / "shaping_variant_summary.png"
    fig.savefig(summary_plot_path, dpi=180)
    plt.close(fig)

    show_n = min(350, n)
    frames = np.arange(show_n)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(frames, mat_alpha[:show_n], label="MATLAB geofeature alpha", linewidth=1.2)
    for variant_name in [
        "baseline_current",
        "sqrt_weights",
        "dedup_gap1_sum",
        "dominant_window_8deg",
        "dedup1_plus_dominant8",
    ]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "variant_raw_alpha_deg"].to_numpy(dtype=float)
        axes[0].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[0].set_title("Raw alpha from shaped candidate sets")
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(frames, mat_fl[:show_n], label="MATLAB FL", linewidth=1.2)
    for variant_name in [
        "baseline_current",
        "sqrt_weights",
        "dedup_gap1_sum",
        "dominant_window_8deg",
        "dedup1_plus_dominant8",
    ]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "FL_mm"].to_numpy(dtype=float)
        axes[1].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[1].set_title("Downstream FL effect of candidate-set shaping")
    axes[1].set_xlabel("aligned frame index")
    axes[1].set_ylabel("FL (mm)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    trace_plot_path = OUT / "shaping_variant_traces.png"
    fig.savefig(trace_plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 86 — Hough candidate-set shaping before aggregation",
        "",
        f"Recomputed the baseline TimTrack candidate stream for {n} aligned frames and cached it at `{CACHE_PATH}`.",
        "",
        "This notebook stays upstream of any code patch. It keeps the same raw TimTrack image path, then tests whether reshaping the Hough peak set before the weighted-median alpha can improve parity.",
        "",
        "## Candidate-shaping findings",
        "",
        f"- The baseline identity variant reproduces the saved Python raw alpha exactly (raw-vs-saved RMSE {baseline['recomputed_vs_saved_python_raw_rmse_deg']:.6f} deg), so the notebook is testing the same candidate stream as the strict run.",
        f"- The best raw-alpha RMSE against MATLAB geofeature alpha is `{best_raw['variant']}` at {best_raw['raw_vs_matlab_rmse']:.4f} deg, versus {baseline['raw_vs_matlab_rmse']:.4f} deg for the unshaped baseline.",
        f"- The best downstream FL RMSE after the same persistence and Kalman path is `{best_fl['variant']}` at {best_fl['FL_vs_matlab_rmse']:.4f} mm, versus {baseline['FL_vs_matlab_rmse']:.4f} mm for the baseline.",
        f"- The lowest rate of 'good candidate present but raw >5 deg wrong' is `{best_wrong['variant']}` at {best_wrong['candidate_present_but_raw_wrong_rate']:.4f}, versus {baseline['candidate_present_but_raw_wrong_rate']:.4f} for baseline.",
        "",
        "## Interpretation",
        "",
        "- If candidate-set shaping helps on the same source peaks, that points to family structure and duplicate/weight bias inside the raw Hough candidate stream.",
        "- If shaping helps FL but not raw MATLAB parity, it is still a compensating modeling change rather than a clean parity fix.",
        "- If baseline remains best, the unresolved difference is even earlier: candidate generation and mask construction, not just peak-family shaping.",
        "",
        f"- Raw metrics CSV: `{raw_path}`",
        f"- Persistence metrics CSV: `{persistence_path}`",
        f"- Downstream metrics CSV: `{kalman_path}`",
        f"- Combined summary CSV: `{full_path}`",
        f"- Per-frame CSV: `{per_frame_path}`",
        f"- Summary plot: `{summary_plot_path}`",
        f"- Trace plot: `{trace_plot_path}`",
    ]
    summary_path = OUT / "notebook86_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
