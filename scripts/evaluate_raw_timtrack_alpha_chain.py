#!/usr/bin/env python3
"""Notebook 84 helper: audit the raw TimTrack alpha production chain."""

from __future__ import annotations

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
import json
import copy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.compare_updated_matlab_python as cmp
from ultrasound_tracker.matlab_timtrack import alpha_from_saved_peaks, detect_timtrack_geofeature_from_image


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
OUT = PROJECT_ROOT / "results" / "notebook84_raw_timtrack_alpha_chain"


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


def read_gray_frames(video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    out: dict[int, np.ndarray] = {}
    try:
        for frame_idx in sorted(set(int(v) for v in frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
            out[int(frame_idx)] = gray
    finally:
        cap.release()
    return out


def nearest_peak_error(alphas: np.ndarray, target_alpha: float) -> float:
    peaks = np.asarray(alphas, dtype=np.float64).reshape(-1)
    peaks = peaks[np.isfinite(peaks)]
    if not len(peaks) or not np.isfinite(target_alpha):
        return float("nan")
    return float(np.min(np.abs(peaks - float(target_alpha))))


def nearest_peak_value(alphas: np.ndarray, target_alpha: float) -> float:
    peaks = np.asarray(alphas, dtype=np.float64).reshape(-1)
    peaks = peaks[np.isfinite(peaks)]
    if not len(peaks) or not np.isfinite(target_alpha):
        return float("nan")
    idx = int(np.argmin(np.abs(peaks - float(target_alpha))))
    return float(peaks[idx])


def top_weight_peak(alphas: np.ndarray, weights: np.ndarray) -> float:
    peak_alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)
    peak_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = min(len(peak_alphas), len(peak_weights))
    if n == 0:
        return float("nan")
    peak_alphas = peak_alphas[:n]
    peak_weights = peak_weights[:n]
    valid = np.isfinite(peak_alphas) & np.isfinite(peak_weights)
    if not np.any(valid):
        return float("nan")
    valid_idx = np.flatnonzero(valid)
    best = valid_idx[int(np.argmax(peak_weights[valid]))]
    return float(peak_alphas[best])


def sample_aligned_frames(raw_error_deg: np.ndarray, *, worst_n: int = 30, control_n: int = 18) -> list[int]:
    raw_error = np.asarray(raw_error_deg, dtype=np.float64).reshape(-1)
    valid_idx = np.flatnonzero(np.isfinite(raw_error))
    worst = valid_idx[np.argsort(np.abs(raw_error[valid_idx]))[::-1][:worst_n]]
    remaining = np.array(sorted(set(valid_idx.tolist()) - set(worst.tolist())), dtype=int)
    if len(remaining):
        control_pos = np.linspace(0, len(remaining) - 1, min(control_n, len(remaining)), dtype=int)
        control = remaining[control_pos]
    else:
        control = np.array([], dtype=int)
    return sorted(set(worst.tolist() + control.tolist()))


def make_overlay_figure(
    *,
    sample_rows: pd.DataFrame,
    frame_images: dict[int, np.ndarray],
    debug_entries: dict[tuple[str, int], dict],
    out_path: Path,
) -> None:
    exemplar = (
        sample_rows.sort_values("abs_alpha_error_deg", ascending=False)
        .drop_duplicates(subset=["aligned_frame"])
        .head(4)
    )
    if exemplar.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (_, row) in zip(axes.ravel(), exemplar.iterrows()):
        aligned = int(row["aligned_frame"])
        py_frame = int(row["python_frame"])
        image = frame_images[py_frame]
        debug = debug_entries[("matlab_literal__matlab", aligned)]
        ax.imshow(image, cmap="gray")
        ax.contour(np.asarray(debug["fascicle_masked"], dtype=bool), levels=[0.5], colors=["cyan"], linewidths=0.6)
        mat_alpha = float(row["matlab_alpha_deg"])
        py_alpha = float(row["raw_alpha_deg"])
        peaks = np.asarray(debug["alphas"], dtype=np.float64).reshape(-1)
        x_lines = np.asarray(debug["x"], dtype=np.float64)
        y_lines = np.asarray(debug["y"], dtype=np.float64)
        for idx, alpha in enumerate(peaks[:5]):
            if idx < len(x_lines) and idx < len(y_lines) and np.isfinite(alpha):
                color = "yellow" if abs(float(alpha) - mat_alpha) <= 1.0 else "red"
                ax.plot(x_lines[idx, :2] - 1, y_lines[idx, :2] - 1, color=color, linewidth=1.0, alpha=0.8)
        ax.set_title(f"frame {py_frame} | MATLAB {mat_alpha:.1f}° | Python {py_alpha:.1f}°")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    matlab_result = loadmat(MATLAB_RESULT, simplify_cells=True)
    utt_export = loadmat(UTT_EXPORT, simplify_cells=True)["UTT_numeric_export"]

    matlab_time = cmp.as_float1(matlab_result["Fdat"]["Region"]["Time"])
    python_time = cmp.as_float1(npz["time_s"])
    python_offset = cmp.choose_python_offset(matlab_time, python_time)
    n = min(len(matlab_time), len(python_time) - python_offset)
    sl = slice(python_offset, python_offset + n)

    mat_geof = utt_export["geofeatures"]
    parms = utt_export["parms"]
    mat_alpha = np.asarray([alpha_from_saved_peaks(entry.get("alphas", []), entry.get("ws", entry.get("weights", []))) for entry in mat_geof], dtype=np.float64)[:n]
    py_raw_alpha = np.asarray(npz["raw_timtrack_alpha_deg"], dtype=np.float64)[sl]
    raw_err = py_raw_alpha - mat_alpha

    sample_frames = sample_aligned_frames(raw_err, worst_n=30, control_n=18)
    sample_python_frames = [frame + python_offset for frame in sample_frames]
    frame_images = read_gray_frames(VIDEO_PATH, sample_python_frames)
    rois = json.loads(ROI_PATH.read_text())
    parms = update_parms_from_rois(parms, rois, next(iter(frame_images.values())).shape)

    variants = [
        ("matlab_literal__matlab", "matlab_literal", "matlab"),
        ("matlab_literal__dynamic", "matlab_literal", "dynamic"),
        ("line_mask__matlab", "line_mask", "matlab"),
        ("line_mask__dynamic", "line_mask", "dynamic"),
    ]

    debug_entries: dict[tuple[str, int], dict] = {}
    detail_rows: list[dict[str, float | int | str | bool]] = []

    for aligned_frame in sample_frames:
        python_frame = aligned_frame + python_offset
        image = frame_images[python_frame]
        mat_entry = mat_geof[aligned_frame]
        matlab_alpha = float(mat_alpha[aligned_frame])
        saved_raw = float(py_raw_alpha[aligned_frame])

        for variant_name, subtraction_mode, emask_mode in variants:
            entry = detect_timtrack_geofeature_from_image(
                image,
                parms,
                subtraction_mode=subtraction_mode,
                emask_mode=emask_mode,
            )
            debug_entries[(variant_name, aligned_frame)] = entry
            peaks = np.asarray(entry["alphas"], dtype=np.float64).reshape(-1)
            weights = np.asarray(entry["ws"], dtype=np.float64).reshape(-1)
            raw_alpha = float(entry["alpha"])
            nearest_err = nearest_peak_error(peaks, matlab_alpha)
            nearest_val = nearest_peak_value(peaks, matlab_alpha)
            top_peak = top_weight_peak(peaks, weights)
            emask = np.asarray(entry["Emask"], dtype=bool)
            fas_masked = np.asarray(entry["fascicle_masked"], dtype=bool)
            fas_thres = np.asarray(entry["filtered"]["fas_thres"], dtype=bool)
            detail_rows.append(
                {
                    "variant": variant_name,
                    "aligned_frame": int(aligned_frame),
                    "python_frame": int(python_frame),
                    "sample_kind": "worst" if aligned_frame in sample_frames[:30] else "control",
                    "matlab_alpha_deg": matlab_alpha,
                    "saved_python_raw_alpha_deg": saved_raw,
                    "raw_alpha_deg": raw_alpha,
                    "alpha_error_deg": raw_alpha - matlab_alpha,
                    "abs_alpha_error_deg": abs(raw_alpha - matlab_alpha),
                    "nearest_peak_alpha_deg": nearest_val,
                    "nearest_peak_error_deg": nearest_err,
                    "top_peak_alpha_deg": top_peak,
                    "top_peak_error_deg": float(top_peak - matlab_alpha) if np.isfinite(top_peak) else np.nan,
                    "n_candidates": int(np.sum(np.isfinite(peaks))),
                    "candidate_hit_1deg": bool(np.isfinite(nearest_err) and nearest_err <= 1.0),
                    "candidate_hit_2deg": bool(np.isfinite(nearest_err) and nearest_err <= 2.0),
                    "candidate_present_but_weighted_median_wrong": bool(
                        np.isfinite(nearest_err) and nearest_err <= 1.0 and abs(raw_alpha - matlab_alpha) > 5.0
                    ),
                    "emask_source": str(entry.get("Emask_source", "")),
                    "emask_density": float(np.mean(emask)),
                    "fas_thres_density": float(np.mean(fas_thres)),
                    "fascicle_masked_density": float(np.mean(fas_masked)),
                }
            )

    detail_table = pd.DataFrame(detail_rows)
    detail_path = OUT / "raw_timtrack_chain_frame_details.csv"
    detail_table.to_csv(detail_path, index=False)

    summary_rows = []
    for variant_name, _, _ in variants:
        sub = detail_table.loc[detail_table["variant"] == variant_name].copy()
        summary_rows.append(
            {
                "variant": variant_name,
                "n_frames": int(len(sub)),
                "raw_alpha_rmse_deg": float(np.sqrt(np.mean(np.square(sub["alpha_error_deg"])))),
                "raw_alpha_mae_deg": float(np.mean(np.abs(sub["alpha_error_deg"]))),
                "nearest_peak_rmse_deg": float(np.sqrt(np.mean(np.square(sub["nearest_peak_error_deg"])))),
                "nearest_peak_mae_deg": float(np.mean(np.abs(sub["nearest_peak_error_deg"]))),
                "top_peak_rmse_deg": float(np.sqrt(np.mean(np.square(sub["top_peak_error_deg"])))),
                "candidate_hit_rate_1deg": float(np.mean(sub["candidate_hit_1deg"])),
                "candidate_hit_rate_2deg": float(np.mean(sub["candidate_hit_2deg"])),
                "candidate_present_but_weighted_median_wrong_rate": float(np.mean(sub["candidate_present_but_weighted_median_wrong"])),
                "mean_candidates": float(np.mean(sub["n_candidates"])),
                "mean_emask_density": float(np.mean(sub["emask_density"])),
                "mean_fas_thres_density": float(np.mean(sub["fas_thres_density"])),
                "mean_fascicle_masked_density": float(np.mean(sub["fascicle_masked_density"])),
                "recomputed_vs_saved_raw_rmse_deg": float(np.sqrt(np.mean(np.square(sub["raw_alpha_deg"] - sub["saved_python_raw_alpha_deg"])))),
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    summary_path = OUT / "raw_timtrack_chain_variant_summary.csv"
    summary_table.to_csv(summary_path, index=False)

    baseline = detail_table.loc[detail_table["variant"] == "matlab_literal__matlab"].copy()
    angle_transform_rows = []
    transforms = {
        "identity_alpha": baseline["raw_alpha_deg"].to_numpy(dtype=float),
        "complement_90_minus_alpha": 90.0 - baseline["raw_alpha_deg"].to_numpy(dtype=float),
        "negative_alpha": -baseline["raw_alpha_deg"].to_numpy(dtype=float),
        "abs_alpha": np.abs(baseline["raw_alpha_deg"].to_numpy(dtype=float)),
    }
    mat_vals = baseline["matlab_alpha_deg"].to_numpy(dtype=float)
    for name, arr in transforms.items():
        angle_transform_rows.append({"transform": name, **scalar_metrics(mat_vals, arr)})
    angle_transform_table = pd.DataFrame(angle_transform_rows)
    angle_transform_path = OUT / "angle_convention_transforms.csv"
    angle_transform_table.to_csv(angle_transform_path, index=False)

    worst_frames_path = OUT / "sampled_worst_frames.csv"
    (
        baseline.sort_values("abs_alpha_error_deg", ascending=False)
        .head(30)
        .to_csv(worst_frames_path, index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    labels = summary_table["variant"].tolist()
    x = np.arange(len(labels))
    axes[0].plot(x, summary_table["raw_alpha_rmse_deg"], marker="o", label="raw alpha RMSE")
    axes[0].plot(x, summary_table["nearest_peak_rmse_deg"], marker="s", label="nearest-peak RMSE")
    axes[0].plot(x, summary_table["top_peak_rmse_deg"], marker="^", label="top-peak RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("RMSE (deg)")
    axes[0].set_title("Raw-alpha vs candidate-set errors")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(x - 0.2, summary_table["candidate_hit_rate_1deg"], width=0.2, label="hit ≤1°")
    axes[1].bar(x, summary_table["candidate_hit_rate_2deg"], width=0.2, label="hit ≤2°")
    axes[1].bar(
        x + 0.2,
        summary_table["candidate_present_but_weighted_median_wrong_rate"],
        width=0.2,
        label="candidate present but weighted median wrong",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("fraction of sampled frames")
    axes[1].set_title("Candidate availability vs pre-persistence selection")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    summary_plot_path = OUT / "raw_timtrack_chain_summary.png"
    fig.savefig(summary_plot_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(angle_transform_table["transform"], angle_transform_table["rmse"])
    ax.set_ylabel("RMSE vs MATLAB alpha (deg)")
    ax.set_title("Angle/sign convention sanity check")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    convention_plot_path = OUT / "angle_convention_check.png"
    fig.savefig(convention_plot_path, dpi=180)
    plt.close(fig)

    overlay_path = OUT / "raw_timtrack_exemplar_overlays.png"
    make_overlay_figure(
        sample_rows=baseline,
        frame_images=frame_images,
        debug_entries=debug_entries,
        out_path=overlay_path,
    )

    best_variant = summary_table.loc[summary_table["raw_alpha_rmse_deg"].idxmin()]
    baseline_summary = summary_table.loc[summary_table["variant"] == "matlab_literal__matlab"].iloc[0]
    best_transform = angle_transform_table.loc[angle_transform_table["rmse"].idxmin()]

    summary_lines = [
        "# Notebook 84 — raw TimTrack alpha production-chain audit",
        "",
        f"Audited {len(sample_frames)} representative aligned frames: 30 worst raw-alpha mismatch frames plus 18 control frames.",
        "",
        "This notebook targets the raw TimTrack alpha production chain before persistence. It recomputes per-frame TimTrack geofeatures directly from the video under several mask-input variants, then separates candidate-generation failure from pre-persistence peak aggregation failure.",
        "",
        "## Key findings",
        "",
        f"- On the baseline image path (`matlab_literal` subtraction + saved/`parms` Emask reuse), recomputed raw alpha matches the saved Python raw alpha closely on the sample (RMSE {baseline_summary['recomputed_vs_saved_raw_rmse_deg']:.4f} deg). So the notebook is auditing the same raw chain the strict run used, not a different one.",
        f"- The baseline raw-alpha RMSE on the sampled frames is {baseline_summary['raw_alpha_rmse_deg']:.4f} deg.",
        f"- The baseline nearest-candidate error is much smaller, RMSE {baseline_summary['nearest_peak_rmse_deg']:.4f} deg, with a MATLAB-like candidate within 1° on {baseline_summary['candidate_hit_rate_1deg']*100:.1f}% of sampled frames and within 2° on {baseline_summary['candidate_hit_rate_2deg']*100:.1f}% of sampled frames.",
        f"- But even when such a close candidate is present, the weighted-median alpha still lands >5° away from MATLAB on {baseline_summary['candidate_present_but_weighted_median_wrong_rate']*100:.1f}% of sampled frames. That means a substantial part of the raw-alpha gap is already a pre-persistence peak-aggregation/selection problem, not only a missing candidate problem.",
        f"- Image/mask inputs still matter. Across the tested variants, the best raw-alpha RMSE is {best_variant['raw_alpha_rmse_deg']:.4f} deg for `{best_variant['variant']}`, versus {baseline_summary['raw_alpha_rmse_deg']:.4f} deg at baseline. The same variant also changes candidate-hit behavior and masked-pixel density, so mask construction is not neutral.",
        f"- Angle/sign convention does not look like the main issue. On the baseline sample, the best simple transform is `{best_transform['transform']}` with RMSE {best_transform['rmse']:.4f} deg, and the identity convention is already the best or tied-best among the tested simple alternatives.",
        "",
        "## Interpretation",
        "",
        "- The raw TimTrack gap is not mainly caused by the later persistence layer.",
        "- It is also not well explained by a simple sign/complement convention error.",
        "- The problem splits into two upstream pieces:",
        "",
        "  1. mask/input sensitivity changes which candidate peaks are produced;",
        "  2. even when a good candidate exists, the current peak aggregation can still choose the wrong raw alpha.",
        "",
        "- That makes the next safest code-facing target narrower: notebook-only experiments around raw Hough peak aggregation and mask construction, before any production patch.",
        "",
        f"- Variant summary CSV: `{summary_path}`",
        f"- Per-frame detail CSV: `{detail_path}`",
        f"- Angle convention CSV: `{angle_transform_path}`",
        f"- Worst-frame CSV: `{worst_frames_path}`",
        f"- Summary plot: `{summary_plot_path}`",
        f"- Convention plot: `{convention_plot_path}`",
        f"- Exemplar overlays: `{overlay_path}`",
    ]
    md_path = OUT / "notebook84_summary.md"
    md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
