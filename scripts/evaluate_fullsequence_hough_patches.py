#!/usr/bin/env python3
"""Notebook 89 helper: full-sequence replay of notebook-only Hough patches."""

from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
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
from ultrasound_tracker.matlab_timtrack import detect_timtrack_geofeature_from_image
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
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "UltraTimTrack_test.mp4"
ROI_PATH = PROJECT_ROOT / "data" / "rois" / "UltraTimTrack_test_rois.json"
METADATA_PATH = RUN_DIR / "UltraTimTrack_test_strict_metadata.json"
OUT = PROJECT_ROOT / "results" / "notebook89_fullsequence_hough_patches"
CACHE_PATH = OUT / "baseline_minimal_geofeatures.pkl"
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


def dohough_variant(
    binary: np.ndarray,
    params: DoHoughParams,
    *,
    replace_diagonal_bias: bool,
    apply_radius_correction: bool,
    peak_source: Literal["2d_houghpeaks", "angle_profile_localmax"],
) -> dict:
    bw = np.asarray(binary).astype(bool)
    theta = matlab_theta_from_range(params.houghangles, params.angle_range, params.thetares)
    hmat, theta, rho = matlab_hough_accumulator(bw, theta, params.rhores)

    if replace_diagonal_bias and params.angle_range[0] < 45 < params.angle_range[1] and np.any(theta == 45):
        rot_angle = 20.0
        rotated = rotate_binary_nearest(bw, rot_angle)
        replacement_theta = np.asarray([90.0 - (45.0 + rot_angle)])
        hmat_rot, _, _ = matlab_hough_accumulator(rotated, replacement_theta, params.rhores)
        hmat[:, theta == 45] = hmat_rot

    gamma = 90.0 - theta
    if apply_radius_correction:
        radius_correction = ellipse_radius_correction(gamma, params.emask_radius)
        hmat_eff = np.rint(hmat / radius_correction[np.newaxis, :])
    else:
        hmat_eff = hmat.copy()

    h_by_angle = np.max(hmat_eff, axis=0) if hmat_eff.size else np.asarray([])

    if peak_source == "2d_houghpeaks":
        peaks = hough_peaks(hmat_eff, params.npeaks, threshold=0.0, theta_degrees=theta)
        weights = np.asarray([hmat_eff[row, col] for row, col in peaks], dtype=np.float64)
        alphas = np.asarray([gamma[col] for _, col in peaks], dtype=np.float64)
        x_lines = np.full((len(peaks), 2), np.nan, dtype=np.float64)
        y_lines = np.full((len(peaks), 2), np.nan, dtype=np.float64)
        for i, (row, col) in enumerate(peaks):
            x_lines[i], y_lines[i] = extract_line_for_peak(bw, theta, rho, row=int(row), col=int(col))
    elif peak_source == "angle_profile_localmax":
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
    else:
        raise ValueError(f"Unknown peak_source: {peak_source}")

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


def read_gray_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise IndexError(f"Could not read frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()


def compute_minimal_entry(gray: np.ndarray, parms: dict) -> dict:
    out = detect_timtrack_geofeature_from_image(gray, parms, subtraction_mode="matlab_literal", emask_mode="matlab")
    return {
        "fascicle_masked": np.asarray(out["fascicle_masked"]).astype(bool),
        "Emask_radius": np.asarray(out["Emask_radius"], dtype=np.float64),
        "alpha": float(out["alpha"]),
        "alphas": np.asarray(out["alphas"], dtype=np.float64),
        "weights": np.asarray(out["weights"], dtype=np.float64),
        "ws": np.asarray(out["ws"], dtype=np.float64),
        "x": np.asarray(out["x"], dtype=np.float64),
        "y": np.asarray(out["y"], dtype=np.float64),
    }


def load_or_compute_minimal_entries(*, limit: int, python_offset: int, parms: dict) -> list[dict]:
    if CACHE_PATH.exists():
        print(f"Loading cached baseline minimal geofeatures from {CACHE_PATH}")
        with CACHE_PATH.open("rb") as f:
            payload = pickle.load(f)
        cached = payload.get("entries", payload)
        if len(cached) == limit - python_offset:
            return cached
        print("Cache length mismatch; recomputing baseline minimal geofeatures...")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(VIDEO_PATH)
    entries: list[dict] = []
    print("Recomputing full-sequence baseline minimal geofeatures...")
    for frame_idx in range(limit):
        gray = read_gray_frame(cap, frame_idx)
        entries.append(compute_minimal_entry(gray, parms))
        if (frame_idx + 1) % 300 == 0:
            print(f"TimTrack minimal geofeatures processed {frame_idx + 1}")
    cap.release()
    entries = entries[python_offset:limit]
    with CACHE_PATH.open("wb") as f:
        pickle.dump({"entries": entries}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return entries


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
    base_entries = load_or_compute_minimal_entries(limit=n + python_offset, python_offset=python_offset, parms=parms)

    mat_entries = list(np.asarray(utt["geofeatures"], dtype=object).reshape(-1))[:n]
    mat_alpha = np.asarray([float(np.asarray(entry["alpha"], dtype=np.float64).reshape(-1)[0]) for entry in mat_entries], dtype=np.float64)
    py_saved_raw = np.asarray(npz["raw_timtrack_alpha_deg"], dtype=np.float64)[sl]

    variants = [
        ("baseline_current", dict(replace_diagonal_bias=True, apply_radius_correction=True, peak_source="2d_houghpeaks"), "current Python Hough path"),
        ("no_radius_correction", dict(replace_diagonal_bias=True, apply_radius_correction=False, peak_source="2d_houghpeaks"), "disable ellipse/radius correction"),
        ("angle_profile_localmax", dict(replace_diagonal_bias=True, apply_radius_correction=True, peak_source="angle_profile_localmax"), "replace 2D houghpeaks with 1D angle-profile local maxima"),
        ("angle_profile_localmax_no_radius", dict(replace_diagonal_bias=True, apply_radius_correction=False, peak_source="angle_profile_localmax"), "1D angle-profile local maxima without radius correction"),
    ]

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

    raw_rows = []
    persistence_rows = []
    kalman_rows = []
    per_frame_tables: list[pd.DataFrame] = []

    for variant_name, knobs, note in variants:
        print(f"Evaluating full-sequence Hough patch variant: {variant_name}")
        raw_alpha = np.full(n, np.nan, dtype=np.float64)
        nearest_errors = np.full(n, np.nan, dtype=np.float64)
        shaped_entries: list[dict] = []

        for idx, entry in enumerate(base_entries):
            local_params = DoHoughParams(
                houghangles=str(parms["fas"]["houghangles"]),
                angle_range=tuple(np.asarray(parms["fas"]["range"], dtype=np.float64).reshape(-1)),
                thetares=float(parms["fas"]["thetares"]),
                rhores=float(parms["fas"]["rhores"]),
                emask_radius=tuple(np.asarray(entry["Emask_radius"], dtype=np.float64).reshape(-1)),
                npeaks=int(parms["fas"]["npeaks"]),
                replace_diagonal_bias=True,
            )
            h = dohough_variant(
                entry["fascicle_masked"],
                local_params,
                replace_diagonal_bias=bool(knobs["replace_diagonal_bias"]),
                apply_radius_correction=bool(knobs["apply_radius_correction"]),
                peak_source=str(knobs["peak_source"]),
            )
            raw_alpha[idx] = float(h["alpha"])
            nearest_errors[idx] = nearest_peak_error(h["alphas"], mat_alpha[idx])
            shaped_entry = dict(entry)
            shaped_entry.update(pack_candidate_arrays(h["alphas"], h["weights"], h["X"], h["Y"]))
            shaped_entry["alpha"] = float(h["alpha"])
            shaped_entries.append(shaped_entry)

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

        hit2 = np.isfinite(nearest_errors) & (nearest_errors <= 2.0)
        raw_wrong = np.abs(raw_alpha - mat_alpha) > 5.0
        raw_rows.append(
            {
                "variant": variant_name,
                "note": note,
                "recomputed_vs_saved_python_raw_rmse_deg": scalar_metrics(py_saved_raw, raw_alpha)["rmse"],
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
    full_table = raw_table.merge(persistence_table, on="variant").merge(kalman_table, on="variant")
    per_frame_table = pd.concat(per_frame_tables, ignore_index=True)

    raw_path = OUT / "fullsequence_raw_metrics.csv"
    persistence_path = OUT / "fullsequence_persistence_metrics.csv"
    kalman_path = OUT / "fullsequence_downstream_metrics.csv"
    full_path = OUT / "fullsequence_variant_summary.csv"
    per_frame_path = OUT / "fullsequence_per_frame.csv"
    raw_table.to_csv(raw_path, index=False)
    persistence_table.to_csv(persistence_path, index=False)
    kalman_table.to_csv(kalman_path, index=False)
    full_table.to_csv(full_path, index=False)
    per_frame_table.to_csv(per_frame_path, index=False)

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

    axes[1].bar(x, persistence_table["selected_vs_matlab_rmse"])
    axes[1].set_title("Selected alpha RMSE after persistence")
    axes[1].set_ylabel("RMSE (deg)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, kalman_table["final_alpha_vs_matlab_rmse"])
    axes[2].set_title("Final alpha RMSE after Kalman")
    axes[2].set_ylabel("RMSE (deg)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)

    axes[3].bar(x, kalman_table["FL_vs_matlab_rmse"])
    axes[3].set_title("Final FL RMSE after Kalman")
    axes[3].set_ylabel("RMSE (mm)")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[3].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    summary_plot_path = OUT / "fullsequence_hough_patch_summary.png"
    fig.savefig(summary_plot_path, dpi=180)
    plt.close(fig)

    show_n = min(350, n)
    frames = np.arange(show_n)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(frames, mat_alpha[:show_n], label="MATLAB geofeature alpha", linewidth=1.2)
    for variant_name in ["baseline_current", "no_radius_correction", "angle_profile_localmax_no_radius"]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "variant_raw_alpha_deg"].to_numpy(dtype=float)
        axes[0].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[0].set_title("Full-sequence raw alpha under notebook-only Hough patches")
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(frames, mat_fl[:show_n], label="MATLAB FL", linewidth=1.2)
    for variant_name in ["baseline_current", "no_radius_correction", "angle_profile_localmax_no_radius"]:
        vals = per_frame_table.loc[per_frame_table["variant"] == variant_name, "FL_mm"].to_numpy(dtype=float)
        axes[1].plot(frames, vals[:show_n], label=variant_name, linewidth=1.0)
    axes[1].set_title("Downstream FL under notebook-only Hough patches")
    axes[1].set_xlabel("aligned frame index")
    axes[1].set_ylabel("FL (mm)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    trace_plot_path = OUT / "fullsequence_hough_patch_traces.png"
    fig.savefig(trace_plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 89 — full-sequence replay of notebook-only Hough patches",
        "",
        f"Recomputed or loaded the full baseline minimal geofeature stream for {n} aligned frames, then replayed selected notebook-only Hough patches on the same per-frame binary fascicle masks.",
        "",
        "This notebook answers the next practical question after notebook 88: do the mathematically suspicious Hough changes improve final FL/parity on the full sequence, not just on selected worst frames?",
        "",
        "## Full-sequence findings",
        "",
        f"- The baseline variant reproduces the saved Python raw alpha closely (raw-vs-saved RMSE {baseline['recomputed_vs_saved_python_raw_rmse_deg']:.6f} deg), so the replay is anchored to the strict run.",
        f"- The best raw-alpha RMSE against MATLAB geofeature alpha is `{best_raw['variant']}` at {best_raw['raw_vs_matlab_rmse']:.4f} deg, versus {baseline['raw_vs_matlab_rmse']:.4f} deg for baseline.",
        f"- The best downstream FL RMSE after the same persistence and Kalman path is `{best_fl['variant']}` at {best_fl['FL_vs_matlab_rmse']:.4f} mm, versus {baseline['FL_vs_matlab_rmse']:.4f} mm for baseline.",
        "",
        "## Interpretation",
        "",
        "- If the no-radius Hough patches improve full-sequence raw alpha and FL together, that is strong evidence the ellipse/radius correction is the main mathematical anomaly.",
        "- If they help worst-frame raw alpha but hurt or fail to improve full-sequence FL, then the knob is still compensatory and needs a narrower patch or a sequence-aware follow-up.",
        "- Comparing `no_radius_correction` against `angle_profile_localmax_no_radius` also tells us whether the remaining gain comes mostly from removing the correction or from changing peak extraction across rho.",
        "",
        f"- Raw metrics CSV: `{raw_path}`",
        f"- Persistence metrics CSV: `{persistence_path}`",
        f"- Downstream metrics CSV: `{kalman_path}`",
        f"- Combined summary CSV: `{full_path}`",
        f"- Per-frame CSV: `{per_frame_path}`",
        f"- Summary plot: `{summary_plot_path}`",
        f"- Trace plot: `{trace_plot_path}`",
        f"- Baseline cache: `{CACHE_PATH}`",
    ]
    summary_path = OUT / "notebook89_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
