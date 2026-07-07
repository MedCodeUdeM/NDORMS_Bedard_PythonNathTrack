#!/usr/bin/env python3
"""Notebook 90 helper: conditional Hough patch only on detected overweight frames."""

from __future__ import annotations

import copy
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
from scripts.run_strict_ultratimtrack_video import (
    FascicleCandidatePersistenceConfig,
    select_fascicle_candidate_persistence,
)
from ultrasound_tracker.timtrack_hough import (
    DoHoughParams,
    ellipse_radius_correction,
    hough_bin_pixels,
    hough_peaks,
    matlab_hough_accumulator,
    matlab_theta_from_range,
    rotate_binary_nearest,
    weighted_median,
)
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
METADATA_PATH = RUN_DIR / "UltraTimTrack_test_strict_metadata.json"
NB89_PER_FRAME = PROJECT_ROOT / "results" / "notebook89_fullsequence_hough_patches" / "fullsequence_per_frame.csv"
NB89_CACHE = PROJECT_ROOT / "results" / "notebook89_fullsequence_hough_patches" / "baseline_minimal_geofeatures.pkl"
OUT = PROJECT_ROOT / "results" / "notebook90_conditional_hough_patch"
LOCALMAX_CACHE = OUT / "angle_profile_localmax_entries.pkl"
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


def finite_peaks(alphas, weights):
    a = np.asarray(alphas, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = min(len(a), len(w))
    a = a[:n]
    w = w[:n]
    keep = np.isfinite(a) & np.isfinite(w) & (w > 0)
    return a[keep], w[keep]


def nearest_peak_error(alphas: np.ndarray, target_alpha: float) -> float:
    vals = np.asarray(alphas, dtype=float).reshape(-1)
    keep = np.isfinite(vals)
    if not np.isfinite(target_alpha) or not np.any(keep):
        return float("nan")
    return float(np.min(np.abs(vals[keep] - float(target_alpha))))


def local_maxima_indices(values: np.ndarray, *, npeaks: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        return np.asarray([], dtype=int)
    candidates: list[int] = []
    for idx in range(len(vals)):
        left = vals[idx - 1] if idx > 0 else -np.inf
        right = vals[idx + 1] if idx < len(vals) - 1 else -np.inf
        if vals[idx] >= left and vals[idx] >= right:
            candidates.append(idx)
    if not candidates:
        return np.asarray([], dtype=int)
    candidates = sorted(candidates, key=lambda i: (-vals[i], i))
    return np.asarray(candidates[: int(npeaks)], dtype=int)


def extract_line_for_peak(binary: np.ndarray, theta: np.ndarray, rho: np.ndarray, *, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
    contributing = hough_bin_pixels(binary, theta, rho, (int(row), int(col)))
    yy, xx = np.nonzero(contributing)
    if len(xx) == 0:
        return np.full(2, np.nan), np.full(2, np.nan)
    order = np.lexsort((yy, xx))
    xx = xx[order] + 1
    yy = yy[order] + 1
    return np.asarray([xx[0], xx[-1]], dtype=np.float64), np.asarray([yy[0], yy[-1]], dtype=np.float64)


def dohough_angle_profile_localmax(binary: np.ndarray, params: DoHoughParams) -> dict:
    bw = np.asarray(binary).astype(bool)
    theta = matlab_theta_from_range(params.houghangles, params.angle_range, params.thetares)
    hmat, theta, rho = matlab_hough_accumulator(bw, theta, params.rhores)

    if params.angle_range[0] < 45 < params.angle_range[1] and np.any(theta == 45):
        rot_angle = 20.0
        rotated = rotate_binary_nearest(bw, rot_angle)
        replacement_theta = np.asarray([90.0 - (45.0 + rot_angle)])
        hmat_rot, _, _ = matlab_hough_accumulator(rotated, replacement_theta, params.rhores)
        hmat[:, theta == 45] = hmat_rot

    gamma = 90.0 - theta
    radius_correction = ellipse_radius_correction(gamma, params.emask_radius)
    hmat_eff = np.rint(hmat / radius_correction[np.newaxis, :])
    h_by_angle = np.max(hmat_eff, axis=0) if hmat_eff.size else np.asarray([])

    peak_cols = local_maxima_indices(h_by_angle, npeaks=params.npeaks)
    if len(peak_cols):
        peak_cols = peak_cols[np.argsort(h_by_angle[peak_cols])[::-1]]
    peak_rows = np.asarray([int(np.nanargmax(hmat_eff[:, col])) for col in peak_cols], dtype=int) if len(peak_cols) else np.asarray([], dtype=int)
    peaks = np.column_stack([peak_rows, peak_cols]) if len(peak_cols) else np.empty((0, 2), dtype=int)
    weights = np.asarray([h_by_angle[col] for col in peak_cols], dtype=np.float64)
    alphas = np.asarray([gamma[col] for col in peak_cols], dtype=np.float64)

    x_lines = np.full((len(peaks), 2), np.nan, dtype=np.float64)
    y_lines = np.full((len(peaks), 2), np.nan, dtype=np.float64)
    for i, (row, col) in enumerate(peaks):
        x_lines[i], y_lines[i] = extract_line_for_peak(bw, theta, rho, row=int(row), col=int(col))

    alpha = weighted_median(alphas, weights) if len(alphas) else float("nan")
    return {
        "alpha": float(alpha),
        "alphas": np.asarray(alphas, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "X": x_lines,
        "Y": y_lines,
        "h_by_angle": np.asarray(h_by_angle, dtype=np.float64),
    }


def pack_candidate_arrays(values: np.ndarray, weights: np.ndarray, x_lines: np.ndarray, y_lines: np.ndarray, *, max_peaks: int = MAX_PEAKS) -> dict[str, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    x_arr = np.asarray(x_lines, dtype=np.float64)
    y_arr = np.asarray(y_lines, dtype=np.float64)
    n = min(int(max_peaks), len(vals), len(w), len(x_arr), len(y_arr))
    alphas = np.full(int(max_peaks), np.nan, dtype=np.float64)
    weights_out = np.full(int(max_peaks), np.nan, dtype=np.float64)
    x = np.full((int(max_peaks), 2), np.nan, dtype=np.float64)
    y = np.full((int(max_peaks), 2), np.nan, dtype=np.float64)
    if n:
        alphas[:n] = vals[:n]
        weights_out[:n] = w[:n]
        x[:n, :2] = np.asarray(x_arr[:n, :2], dtype=np.float64)
        y[:n, :2] = np.asarray(y_arr[:n, :2], dtype=np.float64)
    return {"alphas": alphas, "weights": weights_out, "ws": weights_out.copy(), "x": x, "y": y}


def mass_below(alpha: float, alphas: np.ndarray, weights: np.ndarray, margin_deg: float) -> float:
    a, w = finite_peaks(alphas, weights)
    if len(a) == 0:
        return float("nan")
    wn = w / np.sum(w)
    return float(np.sum(wn[a <= float(alpha) - float(margin_deg)]))


def gap_to_nearest_lower(alpha: float, alphas: np.ndarray) -> float:
    a = np.asarray(alphas, dtype=float).reshape(-1)
    keep = np.isfinite(a) & (a < float(alpha))
    if not np.any(keep):
        return float("nan")
    return float(float(alpha) - np.max(a[keep]))


def build_detector_table(base_entries: list[dict], base_per_frame: pd.DataFrame, localmax_per_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    oracle = (base_per_frame["nearest_peak_error_deg"] <= 2.0) & ((base_per_frame["variant_raw_alpha_deg"] - base_per_frame["matlab_alpha_deg"]) > 5.0)
    for idx, entry in enumerate(base_entries):
        alpha = float(base_per_frame.loc[idx, "variant_raw_alpha_deg"])
        alphas = np.asarray(entry["alphas"], dtype=np.float64)
        weights = np.asarray(entry["weights"], dtype=np.float64)
        rows.append(
            {
                "frame": int(idx),
                "baseline_alpha_deg": alpha,
                "matlab_alpha_deg": float(base_per_frame.loc[idx, "matlab_alpha_deg"]),
                "baseline_raw_error_deg": float(base_per_frame.loc[idx, "variant_raw_alpha_deg"] - base_per_frame.loc[idx, "matlab_alpha_deg"]),
                "localmax_alpha_deg": float(localmax_per_frame.loc[idx, "variant_raw_alpha_deg"]),
                "delta_localmax_deg": float(alpha - localmax_per_frame.loc[idx, "variant_raw_alpha_deg"]),
                "mass_below_8deg": float(mass_below(alpha, alphas, weights, 8.0)),
                "mass_below_10deg": float(mass_below(alpha, alphas, weights, 10.0)),
                "gap_to_lower_deg": float(gap_to_nearest_lower(alpha, alphas)),
                "oracle_high_angle_overweight": bool(oracle.loc[idx]),
            }
        )
    return pd.DataFrame(rows).set_index("frame")


def load_or_compute_localmax_entries(base_entries: list[dict], parms: dict) -> list[dict]:
    if LOCALMAX_CACHE.exists():
        print(f"Loading cached localmax entries from {LOCALMAX_CACHE}")
        with LOCALMAX_CACHE.open("rb") as f:
            payload = pickle.load(f)
        cached = payload.get("entries", payload)
        if len(cached) == len(base_entries):
            return cached
        print("Localmax cache length mismatch; recomputing...")

    print("Recomputing angle-profile localmax entries on cached masks...")
    entries: list[dict] = []
    for idx, entry in enumerate(base_entries):
        params = DoHoughParams(
            houghangles=str(parms["fas"]["houghangles"]),
            angle_range=tuple(np.asarray(parms["fas"]["range"], dtype=np.float64).reshape(-1)),
            thetares=float(parms["fas"]["thetares"]),
            rhores=float(parms["fas"]["rhores"]),
            emask_radius=tuple(np.asarray(entry["Emask_radius"], dtype=np.float64).reshape(-1)),
            npeaks=int(parms["fas"]["npeaks"]),
            replace_diagonal_bias=True,
        )
        h = dohough_angle_profile_localmax(entry["fascicle_masked"], params)
        shaped = dict(entry)
        shaped.update(pack_candidate_arrays(h["alphas"], h["weights"], h["X"], h["Y"]))
        shaped["alpha"] = float(h["alpha"])
        entries.append(shaped)
        if (idx + 1) % 500 == 0:
            print(f"Localmax entries processed {idx + 1}")
    with LOCALMAX_CACHE.open("wb") as f:
        pickle.dump({"entries": entries}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return entries


def detector_metrics(flagged: pd.Series, oracle: pd.Series) -> dict[str, float]:
    flagged = flagged.astype(bool)
    oracle = oracle.astype(bool)
    tp = int((flagged & oracle).sum())
    fp = int((flagged & ~oracle).sum())
    fn = int((~flagged & oracle).sum())
    return {
        "flagged_frames": int(flagged.sum()),
        "oracle_tp": tp,
        "oracle_fp": fp,
        "oracle_fn": fn,
        "oracle_precision": float(tp / max(tp + fp, 1)),
        "oracle_recall": float(tp / max(tp + fn, 1)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    mat = loadmat(MATLAB_RESULT, simplify_cells=True)
    utt = loadmat(UTT_EXPORT, simplify_cells=True)["UTT_numeric_export"]
    metadata = json.loads(METADATA_PATH.read_text())

    with NB89_CACHE.open("rb") as f:
        base_entries = pickle.load(f)["entries"]
    per_frame = pd.read_csv(NB89_PER_FRAME)
    base_pf = per_frame[per_frame["variant"] == "baseline_current"].copy().set_index("frame")
    localmax_pf = per_frame[per_frame["variant"] == "angle_profile_localmax"].copy().set_index("frame")

    detector_table = build_detector_table(base_entries, base_pf, localmax_pf)

    detector_defs = [
        (
            "baseline_current",
            "no conditional patch",
            pd.Series(False, index=detector_table.index),
        ),
        (
            "oracle_overweight_to_localmax",
            "MATLAB-aware ceiling: baseline nearest peak <=2 deg and baseline raw >5 deg too high",
            detector_table["oracle_high_angle_overweight"],
        ),
        (
            "heuristic_mass10_gap6_to_localmax",
            "Python-only conservative rule: mass below alpha-10 deg >=0.30 and nearest lower-gap >=6 deg",
            (detector_table["mass_below_10deg"] >= 0.30) & (detector_table["gap_to_lower_deg"] >= 6.0),
        ),
        (
            "heuristic_mass10_gap4_to_localmax",
            "Python-only broader rule: mass below alpha-10 deg >=0.25 and nearest lower-gap >=4 deg",
            (detector_table["mass_below_10deg"] >= 0.25) & (detector_table["gap_to_lower_deg"] >= 4.0),
        ),
        (
            "heuristic_localmax_gap10_to_localmax",
            "Python-only dual-solver rule: baseline alpha exceeds localmax alpha by >=10 deg",
            detector_table["delta_localmax_deg"] >= 10.0,
        ),
    ]

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

    localmax_entries = load_or_compute_localmax_entries(base_entries, utt["parms"])

    raw_rows = []
    persistence_rows = []
    kalman_rows = []
    detector_rows = []
    per_frame_tables: list[pd.DataFrame] = []

    base_raw = base_pf["variant_raw_alpha_deg"].to_numpy(dtype=float)
    localmax_raw = localmax_pf["variant_raw_alpha_deg"].to_numpy(dtype=float)
    oracle_series = detector_table["oracle_high_angle_overweight"]

    for variant_name, note, flagged_series in detector_defs:
        print(f"Evaluating conditional patch variant: {variant_name}")
        flagged = flagged_series.reindex(detector_table.index).fillna(False).to_numpy(dtype=bool)
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

        nearest_errors = np.asarray([nearest_peak_error(np.asarray(e["alphas"], dtype=float), mat_alpha[i]) for i, e in enumerate(mixed_entries)], dtype=float)
        hit2 = np.isfinite(nearest_errors) & (nearest_errors <= 2.0)
        raw_wrong = np.abs(raw_alpha - mat_alpha) > 5.0

        raw_rows.append(
            {
                "variant": variant_name,
                "note": note,
                "raw_changed_fraction_vs_baseline": float(np.mean(flagged)),
                "nearest_peak_rmse_deg": scalar_metrics(np.zeros(len(nearest_errors)), nearest_errors)["rmse"],
                "candidate_hit_rate_2deg": float(np.mean(hit2)),
                "candidate_present_but_raw_wrong_rate": float(np.mean(hit2 & raw_wrong)),
                **{f"raw_vs_matlab_{k}": v for k, v in scalar_metrics(mat_alpha, raw_alpha).items()},
            }
        )
        persistence_rows.append(
            {
                "variant": variant_name,
                **{f"selected_vs_matlab_{k}": v for k, v in scalar_metrics(mat_alpha, selected_alpha).items()},
                "raw_rejected_fraction": float(np.mean(np.asarray(persistence["raw_alpha_rejected"], dtype=bool))),
            }
        )
        kalman_rows.append(
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
        detector_rows.append({"variant": variant_name, **detector_metrics(pd.Series(flagged, index=detector_table.index), oracle_series)})
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
    detector_stats = pd.DataFrame(detector_rows)
    full_table = raw_table.merge(persistence_table, on="variant").merge(kalman_table, on="variant").merge(detector_stats, on="variant")
    per_frame_table = pd.concat(per_frame_tables, ignore_index=True)

    raw_path = OUT / "conditional_raw_metrics.csv"
    persistence_path = OUT / "conditional_persistence_metrics.csv"
    kalman_path = OUT / "conditional_downstream_metrics.csv"
    detector_path = OUT / "conditional_detector_stats.csv"
    full_path = OUT / "conditional_variant_summary.csv"
    per_frame_path = OUT / "conditional_per_frame.csv"
    feature_path = OUT / "detector_feature_table.csv"
    raw_table.to_csv(raw_path, index=False)
    persistence_table.to_csv(persistence_path, index=False)
    kalman_table.to_csv(kalman_path, index=False)
    detector_stats.to_csv(detector_path, index=False)
    full_table.to_csv(full_path, index=False)
    per_frame_table.to_csv(per_frame_path, index=False)
    detector_table.reset_index().to_csv(feature_path, index=False)

    baseline = full_table.loc[full_table["variant"] == "baseline_current"].iloc[0]
    best_raw = raw_table.loc[raw_table["raw_vs_matlab_rmse"].idxmin()]
    best_fl = kalman_table.loc[kalman_table["FL_vs_matlab_rmse"].idxmin()]

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

    axes[2].bar(x, detector_stats["flagged_frames"])
    axes[2].set_title("Frames patched conditionally")
    axes[2].set_ylabel("frame count")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)

    axes[3].bar(x, detector_stats["oracle_precision"], label="precision")
    axes[3].bar(x, detector_stats["oracle_recall"], alpha=0.75, label="recall")
    axes[3].set_title("Detector agreement with oracle overweight subset")
    axes[3].set_ylabel("fraction")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[3].grid(True, axis="y", alpha=0.25)
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    summary_plot_path = OUT / "conditional_patch_summary.png"
    fig.savefig(summary_plot_path, dpi=180)
    plt.close(fig)

    show_n = min(350, len(base_entries))
    frames = np.arange(show_n)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(frames, mat_alpha[:show_n], label="MATLAB geofeature alpha", linewidth=1.2)
    for variant_name in ["baseline_current", "oracle_overweight_to_localmax", "heuristic_mass10_gap6_to_localmax"]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "variant_raw_alpha_deg"].to_numpy(dtype=float)
        axes[0].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[0].set_title("Conditional raw alpha patching on flagged frames only")
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(frames, mat_fl[:show_n], label="MATLAB FL", linewidth=1.2)
    for variant_name in ["baseline_current", "oracle_overweight_to_localmax", "heuristic_mass10_gap6_to_localmax"]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "FL_mm"].to_numpy(dtype=float)
        axes[1].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[1].set_title("Downstream FL under conditional patching")
    axes[1].set_xlabel("aligned frame index")
    axes[1].set_ylabel("FL (mm)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    trace_plot_path = OUT / "conditional_patch_traces.png"
    fig.savefig(trace_plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 90 — conditional Hough patch only on detected overweight frames",
        "",
        f"Started from the cached full-sequence baseline masks/candidates, then patched only selected frames by swapping baseline Hough output to the notebook-only `angle_profile_localmax` variant.",
        "",
        "This notebook separates two questions:",
        "",
        "1. Is there an upper bound if we patch only the true high-angle-overweight frames?",
        "2. Can Python-only detector rules find enough of those frames to improve full-sequence parity without hurting the rest?",
        "",
        "## Key findings",
        "",
        f"- The best raw-alpha RMSE is `{best_raw['variant']}` at {best_raw['raw_vs_matlab_rmse']:.4f} deg, versus {baseline['raw_vs_matlab_rmse']:.4f} deg for baseline.",
        f"- The best downstream FL RMSE is `{best_fl['variant']}` at {best_fl['FL_vs_matlab_rmse']:.4f} mm, versus {baseline['FL_vs_matlab_rmse']:.4f} mm for baseline.",
        f"- Oracle overweight subset size is {int(detector_stats.loc[detector_stats['variant'] == 'oracle_overweight_to_localmax', 'flagged_frames'].iloc[0])} frames.",
        "",
        "## Interpretation",
        "",
        "- If the oracle subset helps but the Python-only detector rules do not, then the conditional patch idea is directionally right but the detector is not yet good enough.",
        "- If a Python-only detector also helps full-sequence FL/parity, that gives us a more credible code-facing path than a global Hough patch.",
        "- Comparing conservative and broader mass-gap detectors tells us whether the high-angle-overweight pattern is sparse and sharp or diffuse and easy to over-trigger.",
        "",
        f"- Raw metrics CSV: `{raw_path}`",
        f"- Persistence metrics CSV: `{persistence_path}`",
        f"- Downstream metrics CSV: `{kalman_path}`",
        f"- Detector stats CSV: `{detector_path}`",
        f"- Detector feature table CSV: `{feature_path}`",
        f"- Combined summary CSV: `{full_path}`",
        f"- Per-frame CSV: `{per_frame_path}`",
        f"- Summary plot: `{summary_plot_path}`",
        f"- Trace plot: `{trace_plot_path}`",
    ]
    summary_path = OUT / "notebook90_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
