#!/usr/bin/env python3
"""Notebook 87 helper: per-frame Hough internals audit on representative worst frames."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

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
from ultrasound_tracker.matlab_timtrack import (
    alpha_from_saved_peaks,
    detect_timtrack_geofeature_from_image,
)


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
MATLAB_MASK_EXPORT_N36 = PROJECT_ROOT / "results" / "notebook36_mask_parity" / "matlab_intermediate_masks_notebook36.mat"
OUT = PROJECT_ROOT / "results" / "notebook87_hough_internals_worst_frames"
FRAME_DIR = OUT / "frame_debug"
SAMPLE_N = 16
MIN_FRAME_GAP = 20


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


def weighted_median_trace(alphas, weights):
    a, w = finite_peaks(alphas, weights)
    if len(a) == 0 or np.sum(w) <= 0:
        return np.nan, a, w, np.full_like(a, np.nan), -1
    order = np.argsort(a, kind="mergesort")
    a_sorted = a[order]
    w_sorted = w[order] / np.sum(w)
    cumulative = np.cumsum(w_sorted)
    selected = int(np.argmax(cumulative >= 0.5))
    return float(a_sorted[selected]), a_sorted, w_sorted, cumulative, selected


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


def profile_value_at_alpha(gamma: np.ndarray, profile: np.ndarray, alpha: float) -> float:
    g = np.asarray(gamma, dtype=float).reshape(-1)
    p = np.asarray(profile, dtype=float).reshape(-1)
    keep = np.isfinite(g) & np.isfinite(p)
    g = g[keep]
    p = p[keep]
    if len(g) == 0 or not np.isfinite(alpha):
        return float("nan")
    order = np.argsort(g, kind="mergesort")
    return float(np.interp(float(alpha), g[order], p[order]))


def select_representative_worst_frames(errors_abs: np.ndarray, *, n: int, min_gap: int) -> list[int]:
    order = np.argsort(np.asarray(errors_abs, dtype=float))[::-1]
    selected: list[int] = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - prev) >= int(min_gap) for prev in selected):
            selected.append(idx)
        if len(selected) >= int(n):
            break
    return selected


def read_gray_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise IndexError(f"Could not read frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()


def matlab_hough_for_aligned_frame(aligned_idx: int, python_frame_idx: int, mat_entry: dict, matlab_mask_by_frame: dict[int, dict]) -> dict:
    mask_entry = matlab_mask_by_frame.get(int(python_frame_idx))
    return {
        "source": "saved UTT geofeatures",
        "mask_source": "Notebook36 mask export" if mask_entry is not None else "none",
        "mask": np.asarray(mask_entry["fascicle_masked"]).astype(bool) if mask_entry is not None else None,
        "alphas": np.asarray(mat_entry.get("alphas", []), dtype=float).reshape(-1),
        "weights": np.asarray(mat_entry.get("ws", mat_entry.get("weights", [])), dtype=float).reshape(-1),
        "x": np.asarray(mat_entry.get("x", []), dtype=float),
        "y": np.asarray(mat_entry.get("y", []), dtype=float),
        "alpha": float(np.asarray(mat_entry.get("alpha", np.nan), dtype=float).reshape(-1)[0]),
        "aligned_frame": int(aligned_idx),
        "python_frame": int(python_frame_idx),
    }


def python_hough_for_frame(gray: np.ndarray, parms: dict) -> dict:
    out = detect_timtrack_geofeature_from_image(
        gray,
        parms,
        subtraction_mode="matlab_literal",
        emask_mode="matlab",
    )
    h = out["hough_result"]
    return {
        "mask": np.asarray(out["fascicle_masked"]).astype(bool),
        "alphas": np.asarray(h["alphas"], dtype=float),
        "weights": np.asarray(h["weights"], dtype=float),
        "x": np.asarray(h["X"], dtype=float),
        "y": np.asarray(h["Y"], dtype=float),
        "alpha": float(h["alpha"]),
        "gamma": np.asarray(h["gamma"], dtype=float),
        "h_by_angle": np.asarray(h["h_by_angle"], dtype=float),
        "hmat_corrected": np.asarray(h["hmat_corrected"], dtype=float),
        "theta": np.asarray(h["theta"], dtype=float),
        "rho": np.asarray(h["rho"], dtype=float),
        "peaks": np.asarray(h["peaks"]),
        "out": out,
    }


def selected_peak_index(hdata: dict) -> int:
    original_alphas = np.asarray(hdata.get("alphas", []), dtype=float).reshape(-1)
    if len(original_alphas) == 0 or not np.isfinite(hdata.get("alpha", np.nan)):
        return -1
    valid = np.isfinite(original_alphas)
    if not np.any(valid):
        return -1
    valid_idx = np.flatnonzero(valid)
    best_local = int(np.nanargmin(np.abs(original_alphas[valid] - float(hdata["alpha"]))))
    return int(valid_idx[best_local])


def plot_peak_lines(ax, x_lines, y_lines, color, label):
    x_lines = np.asarray(x_lines, dtype=float)
    y_lines = np.asarray(y_lines, dtype=float)
    if x_lines.ndim != 2 or y_lines.ndim != 2:
        return
    labeled = False
    for i in range(min(len(x_lines), len(y_lines))):
        xs = np.asarray(x_lines[i], dtype=float).reshape(-1)
        ys = np.asarray(y_lines[i], dtype=float).reshape(-1)
        if len(xs) < 2 or len(ys) < 2:
            continue
        if not (np.all(np.isfinite(xs[:2])) and np.all(np.isfinite(ys[:2]))):
            continue
        ax.plot(xs[:2] - 1, ys[:2] - 1, color=color, linewidth=1.0, alpha=0.34, label=label if not labeled else None)
        labeled = True


def plot_selected_peak_line(ax, hdata, color, label):
    idx = selected_peak_index(hdata)
    if idx < 0:
        return
    x_lines = np.asarray(hdata.get("x", []), dtype=float)
    y_lines = np.asarray(hdata.get("y", []), dtype=float)
    if x_lines.ndim != 2 or y_lines.ndim != 2 or idx >= len(x_lines) or idx >= len(y_lines):
        return
    xs = np.asarray(x_lines[idx], dtype=float).reshape(-1)
    ys = np.asarray(y_lines[idx], dtype=float).reshape(-1)
    if len(xs) >= 2 and len(ys) >= 2 and np.all(np.isfinite(xs[:2])) and np.all(np.isfinite(ys[:2])):
        ax.plot(xs[:2] - 1, ys[:2] - 1, color=color, linewidth=3.0, alpha=0.96, label=label)


def classify_bias_mode(py_alpha: float, mat_alpha: float, nearest_py_to_mat: float, py_cum_at_mat: float) -> str:
    if not np.isfinite(nearest_py_to_mat) or nearest_py_to_mat > 2.0:
        return "matlab candidate absent in python peaks"
    if not np.isfinite(py_alpha) or not np.isfinite(mat_alpha):
        return "invalid"
    if abs(py_alpha - mat_alpha) <= 2.0:
        return "near parity"
    if py_alpha < mat_alpha and py_cum_at_mat >= 0.5:
        return "lower-angle overweight before matlab"
    if py_alpha > mat_alpha and py_cum_at_mat < 0.5:
        return "higher-angle overweight before matlab"
    return "candidate present but mismatch not explained by crossing"


def plot_debug_frame(path: Path, frame_label: str, matlab_data: dict, python_data: dict, record: dict) -> None:
    mat_med_calc, mat_a_sorted, mat_w_sorted, mat_cum, mat_sel = weighted_median_trace(matlab_data["alphas"], matlab_data["weights"])
    py_med_calc, py_a_sorted, py_w_sorted, py_cum, py_sel = weighted_median_trace(python_data["alphas"], python_data["weights"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{frame_label} | MATLAB alpha {matlab_data['alpha']:.2f} deg, Python alpha {python_data['alpha']:.2f} deg",
        y=0.98,
    )

    ax = axes[0, 0]
    if matlab_data["mask"] is not None:
        diff_rgb = np.zeros((*python_data["mask"].shape, 3), dtype=float)
        mat_mask = np.asarray(matlab_data["mask"], dtype=bool)
        py_mask = np.asarray(python_data["mask"], dtype=bool)
        diff_rgb[..., 1] = mat_mask.astype(float) * 0.85
        diff_rgb[..., 0] = py_mask.astype(float) * 0.85
        diff_rgb[..., 2] = (mat_mask & py_mask).astype(float) * 0.85
        ax.imshow(diff_rgb)
        ax.set_title("fascicle_masked: red Python, green MATLAB, white overlap")
    else:
        ax.imshow(python_data["mask"], cmap="gray")
        ax.set_title("Python fascicle_masked; MATLAB mask unavailable for this frame")
    plot_peak_lines(ax, matlab_data["x"], matlab_data["y"], "#00e5ff", "MATLAB peak lines")
    plot_peak_lines(ax, python_data["x"], python_data["y"], "#ff7043", "Python peak lines")
    plot_selected_peak_line(ax, matlab_data, "#00e5ff", "MATLAB selected median line")
    plot_selected_peak_line(ax, python_data, "#ff7043", "Python selected median line")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.55)

    ax = axes[0, 1]
    ax.scatter(matlab_data["alphas"], matlab_data["weights"], s=58, color="#00e5ff", alpha=0.75, label="MATLAB peaks")
    ax.scatter(python_data["alphas"], python_data["weights"], s=46, color="#ff7043", alpha=0.75, marker="x", label="Python peaks")
    ax.axvline(matlab_data["alpha"], color="#00e5ff", linestyle="--", linewidth=1.6, label=f"MATLAB median {matlab_data['alpha']:.1f}")
    ax.axvline(python_data["alpha"], color="#ff7043", linestyle="--", linewidth=1.6, label=f"Python median {python_data['alpha']:.1f}")
    ax.set_xlabel("Peak alpha (deg)")
    ax.set_ylabel("Peak weight")
    ax.set_title("Exact saved MATLAB peaks vs Python peaks")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    if len(mat_a_sorted):
        ax.step(mat_a_sorted, mat_cum, where="post", color="#00e5ff", linewidth=2, label=f"MATLAB cumulative, median {matlab_data['alpha']:.1f}")
        ax.scatter([mat_a_sorted[mat_sel]], [mat_cum[mat_sel]], color="#00e5ff", s=80)
    if len(py_a_sorted):
        ax.step(py_a_sorted, py_cum, where="post", color="#ff7043", linewidth=2, label=f"Python cumulative, median {python_data['alpha']:.1f}")
        ax.scatter([py_a_sorted[py_sel]], [py_cum[py_sel]], color="#ff7043", s=80)
    ax.axhline(0.5, color="white", linestyle=":", linewidth=1.2)
    ax.axvline(matlab_data["alpha"], color="#00e5ff", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Sorted alpha (deg)")
    ax.set_ylabel("Cumulative normalized weight")
    ax.set_title("Weighted-median selection")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(python_data["gamma"], python_data["h_by_angle"], color="#ff7043", linewidth=1.5, label="Python Hough profile")
    ax.scatter(python_data["alphas"], python_data["weights"], color="#ff7043", s=34, marker="x", label="Python peak weights")
    ax.scatter(matlab_data["alphas"], matlab_data["weights"], color="#00e5ff", s=26, alpha=0.75, label="MATLAB peak weights")
    ax.axvline(matlab_data["alpha"], color="#00e5ff", linestyle="--", linewidth=1.3)
    ax.axvline(python_data["alpha"], color="#ff7043", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Alpha / gamma (deg)")
    ax.set_ylabel("Corrected accumulator max by angle")
    ax.set_title("Python accumulator profile with MATLAB peaks overlaid")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    hmat = np.asarray(python_data["hmat_corrected"], dtype=float)
    peaks = np.asarray(python_data["peaks"])
    im = ax.imshow(hmat, aspect="auto", origin="lower", cmap="magma")
    if peaks.size:
        ax.scatter(peaks[:, 1], peaks[:, 0], s=30, facecolors="none", edgecolors="#00e5ff", linewidths=1.2, label="Python peaks")
    ax.set_title("Python corrected Hough accumulator (exact)")
    ax.set_xlabel("theta / angle bin")
    ax.set_ylabel("rho bin")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.55)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    ax.axis("off")
    text_lines = [
        f"abs raw error: {record['abs_raw_error_deg']:.2f} deg",
        f"nearest Python peak to MATLAB: {record['nearest_python_peak_to_matlab_deg']:.2f} deg",
        f"MATLAB reconstructed alpha from saved peaks: {record['matlab_alpha_reconstructed_deg']:.2f} deg",
        f"Python reconstructed alpha from current peaks: {record['python_alpha_reconstructed_deg']:.2f} deg",
        f"Python cumulative weight at MATLAB alpha: {record['python_cum_weight_at_matlab_alpha']:.3f}",
        f"Python weight mass within ±1 deg of MATLAB: {record['python_mass_within_1deg_of_matlab']:.3f}",
        f"Python weight mass within ±2 deg of MATLAB: {record['python_mass_within_2deg_of_matlab']:.3f}",
        f"MATLAB peak count: {int(record['matlab_peak_count'])}",
        f"Python peak count: {int(record['python_peak_count'])}",
        f"bias mode: {record['bias_mode']}",
        "",
        "Note: exact saved MATLAB peaks/weights/selected median are available.",
        "Exact MATLAB full accumulator/profile is not in the local exports,",
        "so the accumulator heatmap/profile panel is Python-only.",
    ]
    ax.text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    mat_result = loadmat(MATLAB_RESULT, simplify_cells=True)
    utt = loadmat(UTT_EXPORT, simplify_cells=True)["UTT_numeric_export"]
    rois = json.loads(ROI_PATH.read_text())

    matlab_mask_by_frame = {}
    if MATLAB_MASK_EXPORT_N36.exists():
        mask_export = loadmat(MATLAB_MASK_EXPORT_N36, simplify_cells=True)
        matlab_mask_by_frame = {int(entry["frame0"]): entry for entry in mask_export["M"]}

    mat_region = mat_result["Fdat"]["Region"]
    matlab_time = cmp.as_float1(mat_region["Time"])
    python_time = cmp.as_float1(npz["time_s"])
    python_offset = cmp.choose_python_offset(matlab_time, python_time)
    n = min(len(matlab_time), len(python_time) - python_offset)

    py_saved_raw = np.asarray(npz["raw_timtrack_alpha_deg"], dtype=np.float64)[python_offset : python_offset + n]
    mat_entries = list(np.asarray(utt["geofeatures"], dtype=object).reshape(-1))[:n]
    mat_alpha = np.asarray(
        [float(np.asarray(entry["alpha"], dtype=np.float64).reshape(-1)[0]) for entry in mat_entries],
        dtype=np.float64,
    )

    abs_error = np.abs(py_saved_raw - mat_alpha)
    sample_frames = select_representative_worst_frames(abs_error, n=SAMPLE_N, min_gap=MIN_FRAME_GAP)
    sample_frames = sorted(sample_frames)

    frame_shape = (int(utt["vidHeight"]), int(utt["vidWidth"]))
    parms = update_parms_from_rois(utt["parms"], rois, frame_shape)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

    records: list[dict] = []
    profile_rows: list[dict] = []

    for aligned_idx in sample_frames:
        python_frame_idx = int(aligned_idx + python_offset)
        print(f"Auditing aligned frame {aligned_idx} (python frame {python_frame_idx})")
        gray = read_gray_frame(cap, python_frame_idx)
        py = python_hough_for_frame(gray, parms)
        ma = matlab_hough_for_aligned_frame(aligned_idx, python_frame_idx, mat_entries[aligned_idx], matlab_mask_by_frame)

        mat_alpha_recon = alpha_from_saved_peaks(ma["alphas"], ma["weights"])
        py_alpha_recon = alpha_from_saved_peaks(py["alphas"], py["weights"])
        nearest_py_to_mat = nearest_peak_error(py["alphas"], ma["alpha"])
        py_cum_at_mat = cum_weight_at_alpha(py["alphas"], py["weights"], ma["alpha"])
        mass_1 = weight_mass_within(py["alphas"], py["weights"], ma["alpha"], 1.0)
        mass_2 = weight_mass_within(py["alphas"], py["weights"], ma["alpha"], 2.0)
        bias_mode = classify_bias_mode(py["alpha"], ma["alpha"], nearest_py_to_mat, py_cum_at_mat)

        py_top_idx = int(np.nanargmax(py["weights"])) if len(py["weights"]) else -1
        ma_top_idx = int(np.nanargmax(ma["weights"])) if len(ma["weights"]) else -1
        record = {
            "aligned_frame": int(aligned_idx),
            "python_frame": int(python_frame_idx),
            "abs_raw_error_deg": float(abs_error[aligned_idx]),
            "matlab_alpha_deg": float(ma["alpha"]),
            "matlab_alpha_reconstructed_deg": float(mat_alpha_recon),
            "python_alpha_deg": float(py["alpha"]),
            "python_alpha_reconstructed_deg": float(py_alpha_recon),
            "nearest_python_peak_to_matlab_deg": float(nearest_py_to_mat),
            "python_peak_hit_1deg": bool(np.isfinite(nearest_py_to_mat) and nearest_py_to_mat <= 1.0),
            "python_peak_hit_2deg": bool(np.isfinite(nearest_py_to_mat) and nearest_py_to_mat <= 2.0),
            "matlab_peak_count": int(len(finite_peaks(ma["alphas"], ma["weights"])[0])),
            "python_peak_count": int(len(finite_peaks(py["alphas"], py["weights"])[0])),
            "matlab_top_peak_alpha_deg": float(ma["alphas"][ma_top_idx]) if ma_top_idx >= 0 else np.nan,
            "matlab_top_peak_weight": float(ma["weights"][ma_top_idx]) if ma_top_idx >= 0 else np.nan,
            "python_top_peak_alpha_deg": float(py["alphas"][py_top_idx]) if py_top_idx >= 0 else np.nan,
            "python_top_peak_weight": float(py["weights"][py_top_idx]) if py_top_idx >= 0 else np.nan,
            "python_cum_weight_at_matlab_alpha": float(py_cum_at_mat),
            "python_mass_within_1deg_of_matlab": float(mass_1),
            "python_mass_within_2deg_of_matlab": float(mass_2),
            "python_profile_at_matlab_alpha": float(profile_value_at_alpha(py["gamma"], py["h_by_angle"], ma["alpha"])),
            "python_profile_max": float(np.nanmax(py["h_by_angle"])) if len(py["h_by_angle"]) else np.nan,
            "bias_mode": bias_mode,
        }
        records.append(record)

        for gamma, value in zip(np.asarray(py["gamma"], dtype=float).reshape(-1), np.asarray(py["h_by_angle"], dtype=float).reshape(-1)):
            profile_rows.append(
                {
                    "aligned_frame": int(aligned_idx),
                    "python_frame": int(python_frame_idx),
                    "alpha_deg": float(gamma),
                    "python_hough_profile": float(value),
                }
            )

        plot_debug_frame(
            FRAME_DIR / f"frame_{aligned_idx:04d}.png",
            f"Aligned frame {aligned_idx} / python frame {python_frame_idx}",
            ma,
            py,
            record,
        )

    cap.release()

    summary_table = pd.DataFrame(records).sort_values("abs_raw_error_deg", ascending=False).reset_index(drop=True)
    profile_table = pd.DataFrame(profile_rows)
    mode_counts = summary_table["bias_mode"].value_counts(dropna=False).rename_axis("bias_mode").reset_index(name="count")

    summary_path = OUT / "selected_worst_frame_summary.csv"
    profile_path = OUT / "python_hough_profiles_long.csv"
    mode_path = OUT / "bias_mode_counts.csv"
    frames_path = OUT / "representative_frames.json"
    summary_table.to_csv(summary_path, index=False)
    profile_table.to_csv(profile_path, index=False)
    mode_counts.to_csv(mode_path, index=False)
    frames_path.write_text(
        json.dumps(
            {
                "python_offset": int(python_offset),
                "sample_n": int(SAMPLE_N),
                "min_frame_gap": int(MIN_FRAME_GAP),
                "aligned_frames": [int(x) for x in sample_frames],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.bar(np.arange(len(summary_table)), summary_table["abs_raw_error_deg"], label="|Python raw - MATLAB alpha|")
    ax.bar(np.arange(len(summary_table)), summary_table["nearest_python_peak_to_matlab_deg"], alpha=0.75, label="nearest Python peak to MATLAB")
    ax.set_xticks(np.arange(len(summary_table)))
    ax.set_xticklabels(summary_table["aligned_frame"].astype(int), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("deg")
    ax.set_title("Representative worst-frame errors")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(summary_table["aligned_frame"], summary_table["python_cum_weight_at_matlab_alpha"], c="#ff7043", s=55)
    ax.axhline(0.5, color="white", linestyle=":", linewidth=1.2)
    ax.set_xlabel("aligned frame")
    ax.set_ylabel("Python cumulative normalized weight at MATLAB alpha")
    ax.set_title("Where MATLAB alpha lands on Python weighted-median curve")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.bar(mode_counts["bias_mode"], mode_counts["count"], color="#66bb6a")
    ax.set_ylabel("frame count")
    ax.set_title("Bias mode counts on representative worst frames")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.scatter(summary_table["python_mass_within_2deg_of_matlab"], summary_table["abs_raw_error_deg"], s=55, color="#29b6f6")
    ax.set_xlabel("Python weight mass within ±2° of MATLAB alpha")
    ax.set_ylabel("abs raw error (deg)")
    ax.set_title("Candidate present vs final raw error")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    overview_plot_path = OUT / "hough_internals_overview.png"
    fig.savefig(overview_plot_path, dpi=180)
    plt.close(fig)

    peak_hit_2 = int(np.sum(summary_table["python_peak_hit_2deg"]))
    large_error_present = int(np.sum(summary_table["python_peak_hit_2deg"] & (summary_table["abs_raw_error_deg"] > 5.0)))
    lower_over = int(np.sum(summary_table["bias_mode"] == "lower-angle overweight before matlab"))
    higher_over = int(np.sum(summary_table["bias_mode"] == "higher-angle overweight before matlab"))
    absent = int(np.sum(summary_table["bias_mode"] == "matlab candidate absent in python peaks"))
    mat_recon_max_err = float(np.nanmax(np.abs(summary_table["matlab_alpha_reconstructed_deg"] - summary_table["matlab_alpha_deg"])))

    summary_lines = [
        "# Notebook 87 — per-frame Hough internals on representative worst frames",
        "",
        f"Audited {len(summary_table)} representative worst aligned frames chosen from the full-sequence raw-alpha error ranking (minimum spacing {MIN_FRAME_GAP} frames).",
        "",
        "This notebook compares exact saved MATLAB Hough peaks/weights/selected median against the current Python per-frame Hough output on the same frames.",
        "",
        "## Ground-truth boundary",
        "",
        "- Exact MATLAB saved peak angles, weights, peak lines, and selected weighted-median alpha are available in the local UTT export.",
        "- Exact MATLAB full fascicle Hough accumulator/profile is not present in the saved exports, and no local MATLAB/Octave runtime is available here to regenerate it.",
        "- So the per-frame accumulator heatmap/profile panels are exact Python-side internals with MATLAB peaks overlaid, not a full MATLAB-vs-Python accumulator matrix diff.",
        "",
        "## Key findings",
        "",
        f"- On these representative worst frames, MATLAB alpha is reconstructed from the saved MATLAB peaks with max absolute error {mat_recon_max_err:.6f} deg, so the MATLAB peak/weight reference is self-consistent.",
        f"- Python still contains a peak within 2 deg of MATLAB on {peak_hit_2}/{len(summary_table)} representative worst frames.",
        f"- Yet the raw Python alpha is still >5 deg wrong on {large_error_present}/{len(summary_table)} frames even though such a close candidate exists.",
        f"- Bias-mode counts: lower-angle overweight {lower_over}, higher-angle overweight {higher_over}, candidate absent {absent}.",
        "",
        "## Interpretation",
        "",
        "- When a MATLAB-like Python peak is present but the Python weighted median still misses badly, the implementation of weighted median itself is not the main issue.",
        "- The failure is that the Python candidate-family weight distribution crosses 0.5 on the wrong side before the MATLAB-like peak can dominate.",
        "- Frames where the MATLAB-like peak is absent point further upstream to candidate generation / mask construction rather than aggregation logic.",
        "",
        f"- Selected-frame summary CSV: `{summary_path}`",
        f"- Bias mode counts CSV: `{mode_path}`",
        f"- Python Hough profile CSV: `{profile_path}`",
        f"- Representative frame list: `{frames_path}`",
        f"- Overview plot: `{overview_plot_path}`",
        f"- Per-frame debug panels: `{FRAME_DIR}`",
    ]
    markdown_path = OUT / "notebook87_summary.md"
    markdown_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
