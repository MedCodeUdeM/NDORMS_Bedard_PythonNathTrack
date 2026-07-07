#!/usr/bin/env python3
"""Notebook 88 helper: isolate Hough-internal knobs on representative worst frames."""

from __future__ import annotations

import copy
import json
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

from ultrasound_tracker.matlab_timtrack import alpha_from_saved_peaks, detect_timtrack_geofeature_from_image
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


UTT_EXPORT = Path("/Users/grosbedou/Documents/GitHub/UltraTimTrack/UTT_numeric_export.mat")
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "UltraTimTrack_test.mp4"
ROI_PATH = PROJECT_ROOT / "data" / "rois" / "UltraTimTrack_test_rois.json"
WORST_FRAME_SUMMARY = PROJECT_ROOT / "results" / "notebook87_hough_internals_worst_frames" / "selected_worst_frame_summary.csv"
OUT = PROJECT_ROOT / "results" / "notebook88_hough_knob_audit"


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


def cum_weight_at_alpha(alphas, weights, alpha: float) -> float:
    a, w = finite_peaks(alphas, weights)
    if len(a) == 0 or not np.isfinite(alpha):
        return float("nan")
    order = np.argsort(a, kind="mergesort")
    a_sorted = a[order]
    w_sorted = w[order] / np.sum(w)
    return float(np.sum(w_sorted[a_sorted <= float(alpha)]))


def weight_mass_within(alphas, weights, center: float, half_width_deg: float) -> float:
    a, w = finite_peaks(alphas, weights)
    if len(a) == 0 or not np.isfinite(center):
        return float("nan")
    wn = w / np.sum(w)
    keep = np.abs(a - float(center)) <= float(half_width_deg)
    return float(np.sum(wn[keep]))


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


def classify_bias_mode(py_alpha: float, mat_alpha: float, nearest_py_to_mat: float, py_cum_at_mat: float) -> str:
    if not np.isfinite(nearest_py_to_mat) or nearest_py_to_mat > 2.0:
        return "matlab candidate absent in python peaks"
    if not np.isfinite(py_alpha) or not np.isfinite(mat_alpha):
        return "invalid"
    if abs(py_alpha - mat_alpha) <= 2.0:
        return "near parity"
    if py_alpha > mat_alpha and py_cum_at_mat < 0.5:
        return "higher-angle overweight before matlab"
    if py_alpha < mat_alpha and py_cum_at_mat >= 0.5:
        return "lower-angle overweight before matlab"
    return "candidate present but mismatch not explained by crossing"


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
        radius_correction = np.ones_like(gamma, dtype=np.float64)
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
        "hmat": np.asarray(hmat, dtype=np.float64),
        "hmat_effective": np.asarray(hmat_eff, dtype=np.float64),
        "theta": np.asarray(theta, dtype=np.float64),
        "rho": np.asarray(rho, dtype=np.float64),
        "gamma": np.asarray(gamma, dtype=np.float64),
        "peaks": np.asarray(peaks),
        "radius_correction": np.asarray(radius_correction, dtype=np.float64),
    }


def read_gray_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise IndexError(f"Could not read frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    worst_summary = pd.read_csv(WORST_FRAME_SUMMARY)
    aligned_frames = worst_summary["aligned_frame"].astype(int).tolist()
    python_frames = worst_summary["python_frame"].astype(int).tolist()

    utt = loadmat(UTT_EXPORT, simplify_cells=True)["UTT_numeric_export"]
    rois = json.loads(ROI_PATH.read_text())
    mat_entries = list(np.asarray(utt["geofeatures"], dtype=object).reshape(-1))

    frame_shape = (int(utt["vidHeight"]), int(utt["vidWidth"]))
    parms = update_parms_from_rois(utt["parms"], rois, frame_shape)
    fas_parms = parms["fas"]
    do_params = DoHoughParams(
        houghangles=str(fas_parms["houghangles"]),
        angle_range=tuple(np.asarray(fas_parms["range"], dtype=np.float64).reshape(-1)),
        thetares=float(fas_parms["thetares"]),
        rhores=float(fas_parms["rhores"]),
        emask_radius=tuple(np.asarray(fas_parms["Emask_radius"], dtype=np.float64).reshape(-1)),
        npeaks=int(fas_parms["npeaks"]),
        replace_diagonal_bias=True,
    )

    gamma_grid = 90.0 - matlab_theta_from_range(do_params.houghangles, do_params.angle_range, do_params.thetares)

    variants = [
        ("baseline_current", dict(replace_diagonal_bias=True, apply_radius_correction=True, peak_source="2d_houghpeaks"), "current Python Hough path"),
        ("no_diagonal_replacement", dict(replace_diagonal_bias=False, apply_radius_correction=True, peak_source="2d_houghpeaks"), "disable 45 deg diagonal replacement"),
        ("no_radius_correction", dict(replace_diagonal_bias=True, apply_radius_correction=False, peak_source="2d_houghpeaks"), "disable ellipse/radius correction"),
        ("no_diag_no_radius", dict(replace_diagonal_bias=False, apply_radius_correction=False, peak_source="2d_houghpeaks"), "disable both diagonal replacement and radius correction"),
        ("angle_profile_localmax", dict(replace_diagonal_bias=True, apply_radius_correction=True, peak_source="angle_profile_localmax"), "replace 2D houghpeaks with 1D angle-profile local maxima"),
        ("angle_profile_localmax_no_radius", dict(replace_diagonal_bias=True, apply_radius_correction=False, peak_source="angle_profile_localmax"), "1D angle-profile local maxima without radius correction"),
    ]

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

    frame_rows: list[dict] = []
    baseline_masks: dict[int, np.ndarray] = {}

    for aligned_frame, python_frame in zip(aligned_frames, python_frames):
        print(f"Auditing Hough knobs on aligned frame {aligned_frame} (python frame {python_frame})")
        gray = read_gray_frame(cap, python_frame)
        baseline_out = detect_timtrack_geofeature_from_image(gray, parms, subtraction_mode="matlab_literal", emask_mode="matlab")
        fascicle_masked = np.asarray(baseline_out["fascicle_masked"]).astype(bool)
        baseline_masks[int(aligned_frame)] = fascicle_masked
        emask_radius = tuple(np.asarray(baseline_out["Emask_radius"], dtype=np.float64).reshape(-1))

        mat_entry = mat_entries[int(aligned_frame)]
        matlab_alpha = float(np.asarray(mat_entry["alpha"], dtype=float).reshape(-1)[0])
        matlab_peaks = np.asarray(mat_entry["alphas"], dtype=float).reshape(-1)
        matlab_weights = np.asarray(mat_entry.get("ws", mat_entry.get("weights", [])), dtype=float).reshape(-1)
        matlab_alpha_recon = alpha_from_saved_peaks(matlab_peaks, matlab_weights)
        matlab_grid_nearest = float(np.min(np.abs(gamma_grid - matlab_alpha))) if len(gamma_grid) else np.nan

        local_params = DoHoughParams(
            houghangles=do_params.houghangles,
            angle_range=do_params.angle_range,
            thetares=do_params.thetares,
            rhores=do_params.rhores,
            emask_radius=emask_radius,
            npeaks=do_params.npeaks,
            show=False,
            replace_diagonal_bias=True,
        )

        for variant_name, knobs, note in variants:
            out = dohough_variant(
                fascicle_masked,
                local_params,
                replace_diagonal_bias=bool(knobs["replace_diagonal_bias"]),
                apply_radius_correction=bool(knobs["apply_radius_correction"]),
                peak_source=str(knobs["peak_source"]),
            )
            nearest_to_mat = nearest_peak_error(out["alphas"], matlab_alpha)
            cum_at_mat = cum_weight_at_alpha(out["alphas"], out["weights"], matlab_alpha)
            mass_2 = weight_mass_within(out["alphas"], out["weights"], matlab_alpha, 2.0)
            frame_rows.append(
                {
                    "variant": variant_name,
                    "note": note,
                    "aligned_frame": int(aligned_frame),
                    "python_frame": int(python_frame),
                    "matlab_alpha_deg": float(matlab_alpha),
                    "matlab_alpha_reconstructed_deg": float(matlab_alpha_recon),
                    "python_alpha_deg": float(out["alpha"]),
                    "abs_raw_error_deg": float(abs(out["alpha"] - matlab_alpha)),
                    "nearest_python_peak_to_matlab_deg": float(nearest_to_mat),
                    "python_peak_hit_1deg": bool(np.isfinite(nearest_to_mat) and nearest_to_mat <= 1.0),
                    "python_peak_hit_2deg": bool(np.isfinite(nearest_to_mat) and nearest_to_mat <= 2.0),
                    "python_cum_weight_at_matlab_alpha": float(cum_at_mat),
                    "python_mass_within_2deg_of_matlab": float(mass_2),
                    "python_peak_count": int(len(finite_peaks(out["alphas"], out["weights"])[0])),
                    "python_profile_max": float(np.nanmax(out["h_by_angle"])) if len(out["h_by_angle"]) else np.nan,
                    "python_profile_at_matlab_alpha": float(np.interp(matlab_alpha, out["gamma"], out["h_by_angle"])) if len(out["gamma"]) else np.nan,
                    "matlab_alpha_nearest_gamma_grid_deg": float(matlab_grid_nearest),
                    "bias_mode": classify_bias_mode(out["alpha"], matlab_alpha, nearest_to_mat, cum_at_mat),
                }
            )

    cap.release()

    frame_table = pd.DataFrame(frame_rows)
    variant_rows = []
    mode_rows = []

    for variant_name, _, note in variants:
        sub = frame_table[frame_table["variant"] == variant_name].copy()
        variant_rows.append(
            {
                "variant": variant_name,
                "note": note,
                "n_frames": int(len(sub)),
                **{f"raw_vs_matlab_{k}": v for k, v in scalar_metrics(sub["matlab_alpha_deg"], sub["python_alpha_deg"]).items()},
                "nearest_peak_rmse_deg": scalar_metrics(np.zeros(len(sub)), sub["nearest_python_peak_to_matlab_deg"])["rmse"],
                "candidate_hit_rate_1deg": float(sub["python_peak_hit_1deg"].mean()),
                "candidate_hit_rate_2deg": float(sub["python_peak_hit_2deg"].mean()),
                "mean_cum_weight_at_matlab_alpha": float(sub["python_cum_weight_at_matlab_alpha"].mean()),
                "mean_mass_within_2deg_of_matlab": float(sub["python_mass_within_2deg_of_matlab"].mean()),
                "high_angle_overweight_rate": float((sub["bias_mode"] == "higher-angle overweight before matlab").mean()),
                "candidate_absent_rate": float((sub["bias_mode"] == "matlab candidate absent in python peaks").mean()),
                "matlab_alpha_nearest_gamma_grid_mean_deg": float(sub["matlab_alpha_nearest_gamma_grid_deg"].mean()),
                "matlab_alpha_nearest_gamma_grid_max_deg": float(sub["matlab_alpha_nearest_gamma_grid_deg"].max()),
            }
        )
        counts = sub["bias_mode"].value_counts(dropna=False).to_dict()
        mode_rows.append({"variant": variant_name, **counts})

    variant_table = pd.DataFrame(variant_rows).sort_values("raw_vs_matlab_rmse").reset_index(drop=True)
    mode_table = pd.DataFrame(mode_rows).fillna(0)
    best_variant = variant_table.iloc[0]
    baseline = variant_table.loc[variant_table["variant"] == "baseline_current"].iloc[0]

    frame_path = OUT / "hough_knob_frame_metrics.csv"
    variant_path = OUT / "hough_knob_variant_summary.csv"
    mode_path = OUT / "hough_knob_bias_mode_counts.csv"
    frame_table.to_csv(frame_path, index=False)
    variant_table.to_csv(variant_path, index=False)
    mode_table.to_csv(mode_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    labels = variant_table["variant"].tolist()
    x = np.arange(len(labels))

    axes[0, 0].bar(x, variant_table["raw_vs_matlab_rmse"])
    axes[0, 0].set_title("Raw alpha RMSE vs MATLAB on worst frames")
    axes[0, 0].set_ylabel("RMSE (deg)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0, 0].grid(True, axis="y", alpha=0.25)

    axes[0, 1].bar(x, variant_table["mean_cum_weight_at_matlab_alpha"], color="#ff7043")
    axes[0, 1].axhline(0.5, color="white", linestyle=":", linewidth=1.2)
    axes[0, 1].set_title("Mean cumulative weight at MATLAB alpha")
    axes[0, 1].set_ylabel("normalized cumulative weight")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    axes[1, 0].bar(x, variant_table["high_angle_overweight_rate"], label="higher-angle overweight")
    axes[1, 0].bar(x, variant_table["candidate_absent_rate"], bottom=variant_table["high_angle_overweight_rate"], label="candidate absent")
    axes[1, 0].set_title("Bias-mode rates on worst frames")
    axes[1, 0].set_ylabel("frame fraction")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(x, variant_table["candidate_hit_rate_2deg"], label="peak within 2 deg")
    axes[1, 1].bar(x, variant_table["mean_mass_within_2deg_of_matlab"], alpha=0.75, label="weight mass within ±2 deg")
    axes[1, 1].set_title("Candidate presence near MATLAB alpha")
    axes[1, 1].set_ylabel("fraction / normalized weight")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    summary_plot_path = OUT / "hough_knob_summary.png"
    fig.savefig(summary_plot_path, dpi=180)
    plt.close(fig)

    show_frames = aligned_frames[: min(6, len(aligned_frames))]
    compare_variants = ["baseline_current", best_variant["variant"]]
    fig, axes = plt.subplots(len(show_frames), len(compare_variants), figsize=(10, 3.2 * len(show_frames)), squeeze=False)
    for row_idx, frame in enumerate(show_frames):
        for col_idx, variant_name in enumerate(compare_variants):
            sub = frame_table[(frame_table["aligned_frame"] == frame) & (frame_table["variant"] == variant_name)].iloc[0]
            ax = axes[row_idx, col_idx]
            ax.bar(["MATLAB", "Python"], [sub["matlab_alpha_deg"], sub["python_alpha_deg"]], color=["#00e5ff", "#ff7043"])
            ax.set_ylim(
                min(frame_table["matlab_alpha_deg"].min(), frame_table["python_alpha_deg"].min()) - 2,
                max(frame_table["matlab_alpha_deg"].max(), frame_table["python_alpha_deg"].max()) + 2,
            )
            ax.set_title(
                f"frame {frame} | {variant_name}\nerr {sub['abs_raw_error_deg']:.1f} deg | cum@MAT {sub['python_cum_weight_at_matlab_alpha']:.3f}",
                fontsize=9,
            )
            ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    exemplar_plot_path = OUT / "hough_knob_exemplars.png"
    fig.savefig(exemplar_plot_path, dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Notebook 88 — Hough internal knob audit on representative worst frames",
        "",
        f"Audited {len(aligned_frames)} representative worst frames from notebook 87 while keeping the same per-frame binary fascicle mask fixed and changing only Hough-side internals.",
        "",
        "This notebook isolates four suspect areas inside the raw Hough path:",
        "",
        "1. 45 deg diagonal-bias replacement;",
        "2. ellipse/radius correction;",
        "3. 2D `houghpeaks` suppression versus 1D angle-profile peak extraction;",
        "4. angle-bin / gamma-grid alignment diagnostics.",
        "",
        "## Key findings",
        "",
        f"- The best raw-alpha RMSE on these worst frames is `{best_variant['variant']}` at {best_variant['raw_vs_matlab_rmse']:.4f} deg, versus {baseline['raw_vs_matlab_rmse']:.4f} deg for the current baseline.",
        f"- The baseline mean cumulative weight at MATLAB alpha is {baseline['mean_cum_weight_at_matlab_alpha']:.4f}. The best variant reaches {best_variant['mean_cum_weight_at_matlab_alpha']:.4f}.",
        f"- Baseline higher-angle-overweight rate is {baseline['high_angle_overweight_rate']:.4f}; the best variant changes it to {best_variant['high_angle_overweight_rate']:.4f}.",
        f"- MATLAB alpha is already aligned to the Python gamma grid very closely (mean nearest-grid distance {baseline['matlab_alpha_nearest_gamma_grid_mean_deg']:.4f} deg, max {baseline['matlab_alpha_nearest_gamma_grid_max_deg']:.4f} deg), so angle-bin / gamma-grid mismatch does not look like the dominant issue on these frames.",
        "",
        "## Interpretation",
        "",
        "- If disabling radius correction helps most, the higher-angle bias is being introduced mainly by the correction weighting rather than by peak extraction.",
        "- If switching from 2D `houghpeaks` to 1D profile peaks helps most, the problem is duplicate/suppressed peak structure across rho rather than the accumulator itself.",
        "- If diagonal replacement matters, the 45 deg workaround is leaking weight into the wrong family.",
        "- If none of these materially improves raw parity, the remaining problem is earlier still: the mask and underlying accumulator population before any of these knobs act.",
        "",
        f"- Per-frame metrics CSV: `{frame_path}`",
        f"- Variant summary CSV: `{variant_path}`",
        f"- Bias mode CSV: `{mode_path}`",
        f"- Summary plot: `{summary_plot_path}`",
        f"- Exemplar plot: `{exemplar_plot_path}`",
    ]
    summary_path = OUT / "notebook88_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
