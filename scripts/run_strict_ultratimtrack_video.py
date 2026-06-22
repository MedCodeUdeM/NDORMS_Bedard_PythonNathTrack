#!/usr/bin/env python3
"""Run the strict Python UltraTimTrack-style pipeline on a video.

Examples
--------
Interactive:
    python scripts/run_strict_ultratimtrack_video.py --interactive

Command-line:
    python scripts/run_strict_ultratimtrack_video.py data/raw/UltraTimTrack_test.mp4 \
        --roi-path data/rois/UltraTimTrack_test_rois.json \
        --utt-export /Users/grosbedou/Documents/GitHub/UltraTimTrack/UTT_numeric_export.mat
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np
from scipy.io import loadmat


def find_project_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "ultrasound_tracker").exists():
            return candidate
    raise RuntimeError("Could not find project root containing ultrasound_tracker.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ultrasound_tracker.roi as roi
import ultrasound_tracker.utils as ut
from ultrasound_tracker.matlab_aponeurosis import make_matlab_apox
from ultrasound_tracker.matlab_timtrack import (
    detect_timtrack_geofeature_from_image,
    fascicle_segment_from_geofeature,
    run_timtrack_geofeatures_from_video,
)
from ultrasound_tracker.speckle_confidence import (
    SpeckleConfidenceConfig,
    anisotropic_confidence_to_r_scales,
    combine_anisotropic_confidence_metrics,
    combine_confidence_metrics,
    compute_feature_detection_reliability,
    compute_geometry_stability,
    compute_motion_consistency,
    compute_speckle_coherence,
    confidence_to_r_scale,
)
from ultrasound_tracker.strict_fascicle_seed import (
    FascicleSeedScoringConfig,
    cluster_seed_candidates,
    extract_fascicle_seed_candidates,
    select_autonomous_fascicle_seed,
)
from ultrasound_tracker.ultratrack_klt import (
    UltraTrackKLTConfig,
    propagate_cumulative_affines,
    run_one_step_affine_video,
)
from ultrasound_tracker.ultratimtrack_aponeurosis import run_matlab_aponeurosis_state_video
from ultrasound_tracker.ultratimtrack_matlab_2state import (
    MatlabTwoStateKalmanConfig,
    run_matlab_2state_kalman,
)


VIDEO_EXTENSIONS = [".avi", ".AVI", ".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV"]


def list_videos_in_raw(raw_dir: Path) -> list[Path]:
    videos: list[Path] = []
    if raw_dir.exists():
        for ext in VIDEO_EXTENSIONS:
            videos.extend(raw_dir.glob(f"*{ext}"))
    return sorted(set(videos), key=lambda p: p.name.lower())


def prompt_video_from_raw(raw_dir: Path) -> Path:
    videos = list_videos_in_raw(raw_dir)
    if not videos:
        raise FileNotFoundError(f"No video found in {raw_dir}.")

    print("\nVideos found in data/raw:")
    for idx, path in enumerate(videos, start=1):
        print(f"  {idx}. {path.name}")

    while True:
        answer = input("\nVideo to analyze (number or filename): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(videos):
            return videos[int(answer) - 1]
        candidate = raw_dir / answer
        if candidate.exists():
            return candidate
        matches = [p for p in videos if p.name.lower() == answer.lower() or p.stem.lower() == answer.lower()]
        if len(matches) == 1:
            return matches[0]
        print("Could not find that video.")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input(question + suffix).strip().lower()
        if answer == "":
            return default
        if answer in {"y", "yes", "o", "oui"}:
            return True
        if answer in {"n", "no", "non"}:
            return False
        print("Please answer yes/no.")


def read_first_frame(video_path: Path) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    return gray, fps, n_frames


def read_gray_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frames: list[np.ndarray] = []
    for _ in range(int(n_frames)):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return frames


def select_or_load_rois(
    frame0_gray: np.ndarray,
    roi_path: Path,
    *,
    select_roi: bool,
    overwrite_roi: bool,
) -> dict[str, roi.ROI]:
    if roi_path.exists() and not select_roi:
        print(f"Loading ROIs: {roi_path}")
        return roi.load_rois(roi_path)
    if roi_path.exists() and select_roi and not overwrite_roi:
        raise FileExistsError(f"ROI file already exists: {roi_path}. Use --overwrite-roi to replace it.")

    print("\nSelect ROIs in this order:")
    print("  1. superficial aponeurosis")
    print("  2. deep aponeurosis")
    print("  3. fascicle")
    rois = roi.select_all_rois_cv2(frame0_gray, include_fascicle_roi=True)
    roi.save_rois(rois, roi_path)
    print(f"Saved ROIs: {roi_path}")
    return rois


def ellipse_mask_from_roi(shape: tuple[int, int], fascicle_roi: roi.ROI) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, shape)
    x, y, w, h = map(float, fascicle_roi)
    rx = max(w / 2.0, 1.0)
    ry = max(h / 2.0, 1.0)
    cx = x + rx + 1.0
    cy = y + ry + 1.0
    yy, xx = np.mgrid[1 : height + 1, 1 : width + 1]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return mask.astype(bool), np.asarray([ry, rx], dtype=np.float64)


def update_parms_from_rois(parms: dict, rois: Mapping[str, roi.ROI], frame_shape: tuple[int, int]) -> dict:
    """Update MATLAB-like parameter cuts and fascicle Emask from selected ROIs."""

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


def geofeature_measurement_lines(geofeatures: list[Mapping], width: int) -> tuple[np.ndarray, np.ndarray]:
    super_lines = []
    deep_lines = []
    for entry in geofeatures:
        super_pos = np.asarray(entry["super_pos"], dtype=np.float64).reshape(-1)
        deep_pos = np.asarray(entry["deep_pos"], dtype=np.float64).reshape(-1)
        super_lines.append([1.0, super_pos[0], float(width), super_pos[1]])
        deep_lines.append([1.0, deep_pos[0], float(width), deep_pos[1]])
    return np.asarray(super_lines, dtype=np.float64), np.asarray(deep_lines, dtype=np.float64)


def geofeature_alpha(geofeatures: list[Mapping]) -> np.ndarray:
    return np.asarray([float(np.asarray(entry["alpha"]).reshape(-1)[0]) for entry in geofeatures], dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_time_series_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print("\nANG/PEN/FL over time:")
    print("Frame,Time_s,ANG_deg,PEN_deg,FL")
    for row in rows:
        print(
            f"{int(row['Frame'])},"
            f"{float(row['Time']):.6f},"
            f"{float(row['ANG']):.6f},"
            f"{float(row['PEN']):.6f},"
            f"{float(row['FL']):.6f}"
        )


def draw_line_1b(image: np.ndarray, segment_1b: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    if segment_1b is None or not np.all(np.isfinite(segment_1b)):
        return
    line = np.asarray(segment_1b, dtype=np.float64).reshape(4).copy()
    line -= np.asarray([1.0, 1.0, 1.0, 1.0])
    ut.draw_line_on_image(image, line, color=color, thickness=thickness)


def draw_overlay_frame(
    frame: np.ndarray,
    rois: Mapping[str, roi.ROI],
    arrays: Mapping[str, np.ndarray],
    result_idx: int,
) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    vis = cv2.cvtColor(roi.ensure_uint8_image(gray), cv2.COLOR_GRAY2BGR)
    roi_colors = {
        "superficial": (255, 0, 0),
        "deep": (0, 255, 0),
        "fascicle": (0, 255, 255),
    }
    for name, box in rois.items():
        x, y, w, h = map(int, box)
        cv2.rectangle(vis, (x, y), (x + w, y + h), roi_colors.get(name, (255, 255, 255)), 1)
    draw_line_1b(vis, arrays["sup_apo_lines"][result_idx], (255, 0, 0), 2)
    draw_line_1b(vis, arrays["deep_apo_lines"][result_idx], (0, 255, 0), 2)
    draw_line_1b(vis, arrays["klt_prior_segments"][result_idx], (0, 165, 255), 1)
    draw_line_1b(vis, arrays["fascicle_segments"][result_idx], (0, 0, 255), 3)

    fl = arrays["FL_mm"][result_idx] if "FL_mm" in arrays else np.nan
    text_lines = [
        f"Frame: {int(arrays['frame'][result_idx])}",
        f"ANG: {arrays['ANG_deg'][result_idx]:.2f} deg",
        f"PEN: {arrays['PEN_deg'][result_idx]:.2f} deg",
        f"FL: {fl:.2f} mm" if np.isfinite(fl) else f"FL: {arrays['FL_px'][result_idx]:.1f} px",
    ]
    if "combined_confidence" in arrays:
        conf = float(arrays["combined_confidence"][result_idx])
        scale = float(arrays.get("r_scale", np.full_like(arrays["combined_confidence"], np.nan))[result_idx])
        text_lines.append(f"Conf: {conf:.2f}  R x{scale:.1f}")
    ut.put_text_lines_on_image(
        vis,
        text_lines,
        origin=(30, 35),
        line_spacing=24,
        font_scale=0.65,
        color=(255, 255, 255),
        outline_color=(0, 0, 0),
    )
    if "combined_confidence" in arrays:
        cv2.circle(vis, (18, 35 + 24 * (len(text_lines) - 1) - 7), 7, _confidence_color_bgr(conf), -1)
    return vis


def save_annotated_video(
    video_path: Path,
    output_path: Path,
    rois: Mapping[str, roi.ROI],
    arrays: Mapping[str, np.ndarray],
    fps: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frame_indices = np.asarray(arrays["frame"], dtype=int)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame for annotated video.")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps and fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video writer: {output_path}")

    for result_idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        vis = draw_overlay_frame(frame, rois, arrays, result_idx)
        if vis.shape[:2] != (height, width):
            vis = cv2.resize(vis, (width, height))
        writer.write(vis)
        if result_idx and result_idx % 250 == 0:
            print(f"Annotated video frame {result_idx}/{len(frame_indices)}")
    writer.release()
    cap.release()
    print(f"Annotated video: {output_path}")


def save_overlay_images(
    video_path: Path,
    output_dir: Path,
    name: str,
    rois: Mapping[str, roi.ROI],
    arrays: Mapping[str, np.ndarray],
    count: int,
) -> list[Path]:
    if count <= 0:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = np.asarray(arrays["frame"], dtype=int)
    positions = np.linspace(0, len(frame_indices) - 1, min(int(count), len(frame_indices)))
    selected = [int(round(pos)) for pos in positions]
    cap = cv2.VideoCapture(str(video_path))
    paths: list[Path] = []
    for result_idx in selected:
        frame_idx = int(frame_indices[result_idx])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        vis = draw_overlay_frame(frame, rois, arrays, result_idx)
        path = output_dir / f"{name}_frame_{frame_idx:06d}.png"
        cv2.imwrite(str(path), vis)
        paths.append(path)
    cap.release()
    return paths


def save_time_series_plot(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    ang = np.asarray(arrays["ANG_deg"], dtype=np.float64)
    pen = np.asarray(arrays["PEN_deg"], dtype=np.float64)
    if "FL_mm" in arrays:
        fl = np.asarray(arrays["FL_mm"], dtype=np.float64)
        fl_label = "FL (mm)"
    else:
        fl = np.asarray(arrays["FL_px"], dtype=np.float64)
        fl_label = "FL (px)"

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, values, label, color in [
        (axes[0], ang, "ANG (deg)", "tab:red"),
        (axes[1], pen, "PEN (deg)", "tab:blue"),
        (axes[2], fl, fl_label, "tab:green"),
    ]:
        ax.plot(time_s, values, color=color, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Strict Python UltraTimTrack outputs over time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _segment_length_px(segment: np.ndarray) -> float:
    line = np.asarray(segment, dtype=np.float64).reshape(-1)
    if line.size < 4 or not np.all(np.isfinite(line[:4])):
        return float("nan")
    return float(np.hypot(line[2] - line[0], line[3] - line[1]))


def _confidence_color_bgr(confidence: float) -> tuple[int, int, int]:
    if not np.isfinite(confidence):
        return (180, 180, 180)
    if confidence >= 0.75:
        return (0, 180, 0)
    if confidence >= 0.45:
        return (0, 165, 255)
    return (0, 0, 220)


def compute_confidence_series(
    frames: list[np.ndarray],
    rois: Mapping[str, roi.ROI],
    geofeatures: list[Mapping],
    fascicle_prior: np.ndarray,
    timtrack_alpha: np.ndarray,
    *,
    config: SpeckleConfidenceConfig | None = None,
    progress_every: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-frame confidence metrics used by adaptive-R mode."""

    cfg = config or SpeckleConfidenceConfig()
    n = min(len(frames), len(geofeatures), len(fascicle_prior), len(timtrack_alpha))
    out: dict[str, np.ndarray] = {
        "speckle_zncc": np.full(n, np.nan, dtype=np.float64),
        "speckle_confidence": np.ones(n, dtype=np.float64),
        "forward_backward_error": np.full(n, np.nan, dtype=np.float64),
        "valid_patch_fraction": np.full(n, np.nan, dtype=np.float64),
        "n_valid_patches": np.zeros(n, dtype=np.int32),
        "n_total_patches": np.zeros(n, dtype=np.int32),
        "motion_consistency": np.ones(n, dtype=np.float64),
        "motion_spread_px": np.full(n, np.nan, dtype=np.float64),
        "n_motion_points": np.zeros(n, dtype=np.int32),
        "feature_reliability": np.ones(n, dtype=np.float64),
        "feature_peak_score": np.full(n, np.nan, dtype=np.float64),
        "feature_peak_count_score": np.full(n, np.nan, dtype=np.float64),
        "feature_mask_score": np.full(n, np.nan, dtype=np.float64),
        "feature_mask_density": np.full(n, np.nan, dtype=np.float64),
        "geometry_stability": np.ones(n, dtype=np.float64),
        "geometry_alpha_score": np.full(n, np.nan, dtype=np.float64),
        "geometry_pennation_score": np.full(n, np.nan, dtype=np.float64),
        "geometry_length_score": np.full(n, np.nan, dtype=np.float64),
        "geometry_angle_jump_deg": np.full(n, np.nan, dtype=np.float64),
        "geometry_angle_jump_score": np.full(n, np.nan, dtype=np.float64),
        "geometry_length_jump_px": np.full(n, np.nan, dtype=np.float64),
        "geometry_length_jump_score": np.full(n, np.nan, dtype=np.float64),
        "confidence_theta": np.ones(n, dtype=np.float64),
        "confidence_length": np.ones(n, dtype=np.float64),
        "combined_confidence": np.ones(n, dtype=np.float64),
        "r_scale": np.ones(n, dtype=np.float64),
        "r_scale_theta": np.ones(n, dtype=np.float64),
        "r_scale_length": np.ones(n, dtype=np.float64),
        "detection_success": np.ones(n, dtype=bool),
    }

    fascicle_roi = rois.get("fascicle")
    prior_lengths = np.asarray([_segment_length_px(seg) for seg in np.asarray(fascicle_prior)[:n]], dtype=np.float64)

    for idx in range(n):
        entry = geofeatures[idx]
        feature = compute_feature_detection_reliability(entry, config=cfg)
        geometry = compute_geometry_stability(
            alpha_deg=float(timtrack_alpha[idx]) if np.isfinite(timtrack_alpha[idx]) else None,
            fascicle_length_px=float(prior_lengths[idx]) if np.isfinite(prior_lengths[idx]) else None,
            segment=np.asarray(fascicle_prior[idx], dtype=np.float64),
            previous_alpha_deg=float(timtrack_alpha[idx - 1]) if idx > 0 and np.isfinite(timtrack_alpha[idx - 1]) else None,
            previous_length_px=float(prior_lengths[idx - 1]) if idx > 0 and np.isfinite(prior_lengths[idx - 1]) else None,
            previous_segment=np.asarray(fascicle_prior[idx - 1], dtype=np.float64) if idx > 0 else None,
            config=cfg,
        )

        if idx > 0:
            speckle = compute_speckle_coherence(frames[idx - 1], frames[idx], roi=fascicle_roi, config=cfg)
            motion = compute_motion_consistency(
                speckle["points_prev"],
                speckle["points_curr"],
                roi=fascicle_roi,
                config=cfg,
            )
            out["speckle_zncc"][idx] = float(speckle["speckle_zncc"])
            out["speckle_confidence"][idx] = float(speckle["speckle_confidence"])
            out["forward_backward_error"][idx] = float(speckle["forward_backward_error"])
            out["valid_patch_fraction"][idx] = float(speckle["valid_patch_fraction"])
            out["n_valid_patches"][idx] = int(speckle["n_valid_patches"])
            out["n_total_patches"][idx] = int(speckle["n_total_patches"])
            out["motion_consistency"][idx] = float(motion["motion_consistency"])
            out["motion_spread_px"][idx] = float(motion["motion_spread_px"])
            out["n_motion_points"][idx] = int(motion["n_motion_points"])

        out["feature_reliability"][idx] = float(feature["feature_reliability"])
        out["feature_peak_score"][idx] = float(feature["feature_peak_score"])
        out["feature_peak_count_score"][idx] = float(feature["feature_peak_count_score"])
        out["feature_mask_score"][idx] = float(feature["feature_mask_score"])
        out["feature_mask_density"][idx] = float(feature.get("feature_mask_density", np.nan))
        out["detection_success"][idx] = bool(feature["detection_success"])
        out["geometry_stability"][idx] = float(geometry["geometry_stability"])
        out["geometry_alpha_score"][idx] = float(geometry["geometry_alpha_score"])
        out["geometry_pennation_score"][idx] = float(geometry["geometry_pennation_score"])
        out["geometry_length_score"][idx] = float(geometry["geometry_length_score"])
        out["geometry_angle_jump_deg"][idx] = float(geometry["geometry_angle_jump_deg"])
        out["geometry_angle_jump_score"][idx] = float(geometry["geometry_angle_jump_score"])
        out["geometry_length_jump_px"][idx] = float(geometry["geometry_length_jump_px"])
        out["geometry_length_jump_score"][idx] = float(geometry["geometry_length_jump_score"])
        combined = combine_confidence_metrics(
            {
                "speckle_confidence": out["speckle_confidence"][idx],
                "motion_consistency": out["motion_consistency"][idx],
                "feature_reliability": out["feature_reliability"][idx],
                "geometry_stability": out["geometry_stability"][idx],
            },
            config=cfg,
        )
        out["combined_confidence"][idx] = combined
        out["r_scale"][idx] = confidence_to_r_scale(combined, cfg)
        anisotropic_conf = combine_anisotropic_confidence_metrics(
            {
                "speckle_confidence": out["speckle_confidence"][idx],
                "motion_consistency": out["motion_consistency"][idx],
                "feature_reliability": out["feature_reliability"][idx],
                "geometry_stability": out["geometry_stability"][idx],
                "geometry_alpha_score": out["geometry_alpha_score"][idx],
                "geometry_angle_jump_score": out["geometry_angle_jump_score"][idx],
                "geometry_length_score": out["geometry_length_score"][idx],
                "geometry_length_jump_score": out["geometry_length_jump_score"][idx],
            },
            config=cfg,
        )
        out["confidence_theta"][idx] = anisotropic_conf["confidence_theta"]
        out["confidence_length"][idx] = anisotropic_conf["confidence_length"]
        anisotropic_scales = anisotropic_confidence_to_r_scales(
            anisotropic_conf["confidence_theta"],
            anisotropic_conf["confidence_length"],
            cfg,
        )
        out["r_scale_theta"][idx] = anisotropic_scales["r_scale_theta"]
        out["r_scale_length"][idx] = anisotropic_scales["r_scale_length"]

        if progress_every and idx > 0 and (idx % int(progress_every) == 0 or idx == n - 1):
            print(f"confidence processed {idx + 1}/{n}")

    return out


def save_confidence_plot(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if "combined_confidence" not in arrays:
        return
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    confidence = np.asarray(arrays["combined_confidence"], dtype=np.float64)
    r_scale = np.asarray(arrays.get("r_scale", np.full_like(confidence, np.nan)), dtype=np.float64)
    confidence_theta = np.asarray(arrays.get("confidence_theta", confidence), dtype=np.float64)
    confidence_length = np.asarray(arrays.get("confidence_length", confidence), dtype=np.float64)
    angle = np.asarray(arrays.get("ANG_deg", np.full_like(confidence, np.nan)), dtype=np.float64)
    length = np.asarray(arrays.get("FL_mm", arrays.get("FL_px", np.full_like(confidence, np.nan))), dtype=np.float64)
    length_label = "FL (mm)" if "FL_mm" in arrays else "FL (px)"

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(11, 10.5), sharex=True)
    axes[0].plot(time_s, confidence, color="tab:purple", linewidth=1.6)
    axes[0].plot(time_s, confidence_theta, color="tab:red", linewidth=1.0, alpha=0.75, label="theta")
    axes[0].plot(time_s, confidence_length, color="tab:green", linewidth=1.0, alpha=0.75, label="length")
    axes[0].set_ylabel("confidence")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].plot(time_s, r_scale, color="tab:orange", linewidth=1.4, label="global")
    if "r_scale_theta" in arrays:
        axes[1].plot(time_s, arrays["r_scale_theta"], color="tab:red", linewidth=1.0, alpha=0.75, label="theta")
    if "r_scale_length" in arrays:
        axes[1].plot(time_s, arrays["r_scale_length"], color="tab:green", linewidth=1.0, alpha=0.75, label="length")
    axes[1].set_ylabel("R scale")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(time_s, angle, color="tab:red", linewidth=1.2)
    axes[2].scatter(time_s, angle, c=confidence_theta, cmap="RdYlGn", vmin=0, vmax=1, s=12)
    axes[2].set_ylabel("ANG (deg)")
    axes[3].plot(time_s, length, color="tab:green", linewidth=1.2)
    axes[3].scatter(time_s, length, c=confidence_length, cmap="RdYlGn", vmin=0, vmax=1, s=12)
    axes[3].set_ylabel(length_label)
    axes[4].plot(time_s, confidence_theta - confidence_length, color="tab:gray", linewidth=1.2)
    axes[4].axhline(0.0, color="black", linewidth=0.8)
    axes[4].set_ylabel("c_theta-c_L")
    axes[4].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Adaptive confidence diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_export = Path("/Users/grosbedou/Documents/GitHub/UltraTimTrack/UTT_numeric_export.mat")
    parser = argparse.ArgumentParser(description="Run strict Python UltraTimTrack and write an annotated video.")
    parser.add_argument("video", nargs="?", type=Path, default=None, help="Video path. Omit with --interactive.")
    parser.add_argument("--interactive", action="store_true", help="Choose video and ROI interactively.")
    parser.add_argument("--name", default=None, help="Output prefix. Defaults to video stem.")
    parser.add_argument("--utt-export", type=Path, default=default_export, help="UTT numeric export .mat with parameters.")
    parser.add_argument("--roi-path", type=Path, default=None, help="ROI JSON path.")
    parser.add_argument("--select-roi", action="store_true", help="Open ROI selector and save/update ROI JSON.")
    parser.add_argument("--overwrite-roi", action="store_true", help="Allow replacing an existing ROI JSON.")
    parser.add_argument(
        "--no-roi-parameter-update",
        action="store_true",
        help="Draw ROIs but do not update aponeurosis cuts or fascicle Emask from them.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N frames.")
    parser.add_argument("--seed-frames", type=int, default=11, help="Frames used for autonomous seed selection.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "strict_ultratimtrack_runs")
    parser.add_argument("--annotated-video", type=Path, default=None, help="Output MP4 path.")
    parser.add_argument("--no-annotated-video", action="store_true", help="Do not write annotated MP4.")
    parser.add_argument("--save-overlays", type=int, default=3, help="Number of overlay PNGs to save.")
    parser.add_argument("--no-time-series-plot", action="store_true", help="Do not write ANG/PEN/FL time-series PNG.")
    parser.add_argument("--print-time-series", action="store_true", help="Print ANG/PEN/FL values for every processed frame.")
    parser.add_argument("--adaptive-r", action="store_true", help="Use confidence-adaptive Kalman measurement covariance R_t.")
    parser.add_argument(
        "--confidence-debug",
        action="store_true",
        help="Compute confidence metrics but keep the Kalman filter on fixed R unless --adaptive-r is also set.",
    )
    parser.add_argument("--save-confidence-plots", action="store_true", help="Write confidence/R-scale diagnostic PNG.")
    parser.add_argument("--mm-per-pixel", type=float, default=None, help="Override pixel scale.")
    parser.add_argument("--image-depth-mm", type=float, default=None, help="Image depth; used if --mm-per-pixel is absent.")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    if args.video is None or args.interactive:
        args.video = prompt_video_from_raw(PROJECT_ROOT / "data" / "raw") if args.video is None else args.video

    args.video = (PROJECT_ROOT / args.video).resolve() if not args.video.is_absolute() else args.video.resolve()
    args.name = args.name or args.video.stem
    args.results_dir = (PROJECT_ROOT / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir.resolve()
    args.utt_export = (PROJECT_ROOT / args.utt_export).resolve() if not args.utt_export.is_absolute() else args.utt_export.resolve()
    args.roi_path = args.roi_path or (PROJECT_ROOT / "data" / "rois" / f"{args.name}_rois.json")
    args.roi_path = (PROJECT_ROOT / args.roi_path).resolve() if not args.roi_path.is_absolute() else args.roi_path.resolve()
    if args.annotated_video is not None and not args.annotated_video.is_absolute():
        args.annotated_video = (PROJECT_ROOT / args.annotated_video).resolve()

    if args.interactive:
        if args.roi_path.exists():
            args.select_roi = prompt_yes_no("ROI file exists. Select new ROIs?", default=False)
            args.overwrite_roi = args.select_roi
        else:
            args.select_roi = True
            args.overwrite_roi = True
        args.no_annotated_video = not prompt_yes_no("Write annotated video?", default=True)

    return args


def process_video(args: argparse.Namespace) -> dict[str, Path | None]:
    start_time = time.time()
    frame0, fps, n_video_frames = read_first_frame(args.video)
    n_limit = min(args.limit or n_video_frames, n_video_frames)
    if n_limit <= 1:
        raise ValueError("--limit must leave at least two frames.")

    mat_root = loadmat(args.utt_export, simplify_cells=True)["UTT_numeric_export"]
    parms = copy.deepcopy(mat_root["parms"])

    rois: dict[str, roi.ROI] = {}
    should_select_roi = args.select_roi or (not args.roi_path.exists() and not args.no_roi_parameter_update)
    if args.roi_path.exists() or should_select_roi:
        rois = select_or_load_rois(
            frame0,
            args.roi_path,
            select_roi=should_select_roi,
            overwrite_roi=args.overwrite_roi,
        )
        if not args.no_roi_parameter_update:
            parms = update_parms_from_rois(parms, rois, frame0.shape[:2])

    if args.mm_per_pixel is not None:
        mm_per_px = float(args.mm_per_pixel)
    elif args.image_depth_mm is not None:
        mm_per_px = float(args.image_depth_mm) / float(frame0.shape[0])
    elif "ID" in mat_root and np.isfinite(float(mat_root["ID"])):
        mm_per_px = float(mat_root["ID"]) / float(frame0.shape[0])
    else:
        mm_per_px = np.nan

    block_size = np.asarray(mat_root.get("BlockSize", [81, 81]), dtype=int).reshape(-1)
    win_size = (int(block_size[-1]), int(block_size[0])) if block_size.size >= 2 else (81, 81)
    klt_config = UltraTrackKLTConfig(lk_win_size=win_size)
    kalman_config = MatlabTwoStateKalmanConfig(
        q_parameter=float(mat_root.get("Q", 0.01)),
        x_measurement_variance=float(mat_root.get("X", 100.0)),
        alpha_measurement_variance=float(np.asarray(mat_root.get("R", [3.05529211]), dtype=float).reshape(-1)[0]),
        n_start_frames=int(mat_root.get("NS", 1)),
        run_smoother=True,
    )
    r_values = np.asarray(mat_root.get("R", [3.05529211, 100, 100, 100, 100]), dtype=float).reshape(-1)
    apo_measurement_variance = r_values[1:5] * 0.01 if r_values.size >= 5 else np.ones(4, dtype=np.float64)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.results_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning TimTrack image stream...")
    geofeatures = run_timtrack_geofeatures_from_video(
        str(args.video),
        parms,
        limit=n_limit,
        keep_debug=False,
        emask_mode="matlab",
        progress_every=args.progress_every,
    )
    n = len(geofeatures)
    if n <= 1:
        raise RuntimeError("TimTrack stream produced too few frames.")

    print("\nSelecting autonomous fascicle seed...")
    seed_frame_count = min(int(args.seed_frames), n)
    seed_frames = read_gray_frames(args.video, seed_frame_count)
    seed_entries = []
    for idx, gray in enumerate(seed_frames):
        entry = detect_timtrack_geofeature_from_image(gray, parms, emask_mode="matlab")
        entry["frame"] = idx
        seed_entries.append(entry)
    seed_config = FascicleSeedScoringConfig(min_cluster_frame_coverage=min(8, seed_frame_count))
    candidates = extract_fascicle_seed_candidates(seed_entries, seed_frames, mm_per_px=mm_per_px, config=seed_config)
    candidates, clusters = cluster_seed_candidates(candidates, min_frame_coverage=seed_config.min_cluster_frame_coverage)
    selection = select_autonomous_fascicle_seed(candidates, clusters, seed_entries[0], frame0.shape[1])
    selected_seed = np.asarray(selection["selected_seed_segment"], dtype=np.float64)
    selected_alpha = float(selection["selected_alpha_deg"])
    selected_cluster = str(selection["selected_cluster"]["cluster_id"])
    print(f"Selected seed alpha: {selected_alpha:.3f} deg ({selected_cluster})")

    candidates.to_csv(run_dir / f"{args.name}_seed_candidates.csv", index=False)
    clusters.to_csv(run_dir / f"{args.name}_seed_clusters.csv", index=False)
    selection["per_frame_best"].to_csv(run_dir / f"{args.name}_selected_seed_cluster.csv", index=False)

    print("\nEstimating fascicle KLT affines...")
    reference_segments = np.asarray([fascicle_segment_from_geofeature(entry) for entry in geofeatures], dtype=np.float64)
    klt = run_one_step_affine_video(
        args.video,
        geofeatures,
        reference_segments,
        super_cut=np.asarray(parms["apo"]["super"]["cut"], dtype=float).reshape(-1),
        config=klt_config,
        limit=n,
        progress_every=args.progress_every,
    )
    fascicle_prior = propagate_cumulative_affines(selected_seed, np.asarray(klt["f_affine_matrices"], dtype=float))

    print("\nRunning aponeurosis state estimator...")
    sup_meas, deep_meas = geofeature_measurement_lines(geofeatures, frame0.shape[1])
    apo_state = run_matlab_aponeurosis_state_video(
        args.video,
        geofeatures,
        sup_meas,
        deep_meas,
        super_cut=np.asarray(parms["apo"]["super"]["cut"], dtype=float).reshape(-1),
        deep_cut=np.asarray(parms["apo"]["deep"]["cut"], dtype=float).reshape(-1),
        q_parameter=float(mat_root.get("Q", 0.01)),
        measurement_variance=apo_measurement_variance,
        config=klt_config,
        limit=n,
        progress_every=args.progress_every,
    )

    timtrack_alpha = geofeature_alpha(geofeatures)
    confidence_arrays: dict[str, np.ndarray] = {}
    compute_confidence = bool(args.adaptive_r or args.confidence_debug or args.save_confidence_plots)
    if compute_confidence:
        print("\nComputing ultrasound confidence metrics...")
        confidence_frames = read_gray_frames(args.video, n)
        confidence_arrays = compute_confidence_series(
            confidence_frames,
            rois,
            geofeatures,
            fascicle_prior,
            timtrack_alpha,
            config=SpeckleConfidenceConfig(),
            progress_every=args.progress_every if args.confidence_debug else None,
        )
    kalman_run_config = replace(kalman_config, use_adaptive_R=bool(args.adaptive_r))
    print("\nRunning 2-state fascicle Kalman...")
    final = run_matlab_2state_kalman(
        fascicle_prior,
        timtrack_alpha,
        np.asarray(apo_state["super_lines"], dtype=float),
        np.asarray(apo_state["deep_lines"], dtype=float),
        config=kalman_run_config,
        mm_per_pixel=mm_per_px if np.isfinite(mm_per_px) else None,
        measurement_r_scale=confidence_arrays.get("r_scale") if args.adaptive_r else None,
        measurement_r_scale_theta=confidence_arrays.get("r_scale_theta") if args.adaptive_r else None,
        measurement_r_scale_length=confidence_arrays.get("r_scale_length") if args.adaptive_r else None,
    )

    frames = np.arange(len(final["ANG_deg"]), dtype=np.int32)
    arrays: dict[str, np.ndarray] = {
        "frame": frames,
        "time_s": frames.astype(np.float64) / fps if fps and fps > 0 else np.full(len(frames), np.nan),
        "sup_apo_lines": np.asarray(apo_state["super_lines"], dtype=np.float64),
        "deep_apo_lines": np.asarray(apo_state["deep_lines"], dtype=np.float64),
        "klt_prior_segments": fascicle_prior,
        "fascicle_segments": np.asarray(final["fascicle_segments"], dtype=np.float64),
        "fascicle_end_segments": np.asarray(final["fascicle_end_segments"], dtype=np.float64),
        "ANG_deg": np.asarray(final["ANG_deg"], dtype=np.float64),
        "PEN_deg": np.asarray(final["PEN_deg"], dtype=np.float64),
        "FL_px": np.asarray(final["FL_px"], dtype=np.float64),
        "timtrack_alpha_deg": timtrack_alpha,
        "selected_seed_segment": selected_seed,
        "selected_seed_alpha_deg": np.asarray(selected_alpha, dtype=np.float64),
        "mm_per_pixel": np.asarray(mm_per_px, dtype=np.float64),
    }
    if "FL_mm" in final:
        arrays["FL_mm"] = np.asarray(final["FL_mm"], dtype=np.float64)
    if confidence_arrays:
        arrays.update(confidence_arrays)
    if "measurement_R_diag" in final:
        arrays["R_t_x_variance"] = np.asarray(final["measurement_R_diag"], dtype=np.float64)[:, 0]
        arrays["R_t_alpha_variance"] = np.asarray(final["measurement_R_diag"], dtype=np.float64)[:, 1]
        arrays["R_t_length_variance"] = arrays["R_t_x_variance"]
        arrays["R_t_theta_variance"] = arrays["R_t_alpha_variance"]
        arrays["kalman_measurement_r_scale"] = np.asarray(final["measurement_r_scale"], dtype=np.float64)
        arrays["kalman_measurement_r_scale_theta"] = np.asarray(
            final["measurement_r_scale_theta"], dtype=np.float64
        )
        arrays["kalman_measurement_r_scale_length"] = np.asarray(
            final["measurement_r_scale_length"], dtype=np.float64
        )

    npz_path = run_dir / f"{args.name}_strict_results.npz"
    np.savez_compressed(npz_path, **arrays)

    rows = []
    confidence_csv_columns = [
        "speckle_zncc",
        "speckle_confidence",
        "forward_backward_error",
        "valid_patch_fraction",
        "n_valid_patches",
        "n_total_patches",
        "motion_consistency",
        "motion_spread_px",
        "n_motion_points",
        "feature_reliability",
        "feature_peak_score",
        "feature_peak_count_score",
        "feature_mask_score",
        "feature_mask_density",
        "geometry_stability",
        "geometry_alpha_score",
        "geometry_pennation_score",
        "geometry_length_score",
        "geometry_angle_jump_deg",
        "geometry_angle_jump_score",
        "geometry_length_jump_px",
        "geometry_length_jump_score",
        "confidence_theta",
        "confidence_length",
        "combined_confidence",
        "r_scale",
        "r_scale_theta",
        "r_scale_length",
        "R_t_x_variance",
        "R_t_alpha_variance",
        "R_t_length_variance",
        "R_t_theta_variance",
        "kalman_measurement_r_scale",
        "kalman_measurement_r_scale_theta",
        "kalman_measurement_r_scale_length",
        "detection_success",
    ]
    for idx in range(len(frames)):
        row = {
            "Frame": int(frames[idx]),
            "Time": float(arrays["time_s"][idx]),
            "FL": float(arrays.get("FL_mm", arrays["FL_px"])[idx]),
            "PEN": float(arrays["PEN_deg"][idx]),
            "ANG": float(arrays["ANG_deg"][idx]),
            "FL_px": float(arrays["FL_px"][idx]),
            "TimTrackAlpha": float(timtrack_alpha[idx]),
        }
        for column in confidence_csv_columns:
            if column not in arrays:
                continue
            value = arrays[column][idx]
            row[column] = bool(value) if np.asarray(value).dtype == np.dtype(bool) else float(value)
        rows.append(row)
    csv_path = run_dir / f"{args.name}_strict_FL_PEN_ANG.csv"
    write_csv(csv_path, rows)
    if args.print_time_series:
        print_time_series_rows(rows)

    time_series_plot_path: Optional[Path] = None
    if not args.no_time_series_plot:
        time_series_plot_path = run_dir / f"{args.name}_ANG_PEN_FL_over_time.png"
        save_time_series_plot(time_series_plot_path, arrays)
    confidence_plot_path: Optional[Path] = None
    if args.save_confidence_plots and "combined_confidence" in arrays:
        confidence_plot_path = run_dir / f"{args.name}_confidence_diagnostics.png"
        save_confidence_plot(confidence_plot_path, arrays)

    metadata = {
        "video": str(args.video),
        "utt_export": str(args.utt_export),
        "roi_path": str(args.roi_path) if args.roi_path else None,
        "roi_parameter_update": not args.no_roi_parameter_update,
        "adaptive_r": bool(args.adaptive_r),
        "confidence_debug": bool(args.confidence_debug),
        "frames": int(len(frames)),
        "fps": fps,
        "mm_per_pixel": mm_per_px,
        "selected_seed_alpha_deg": selected_alpha,
        "selected_seed_segment": selected_seed.tolist(),
        "selected_cluster_id": selected_cluster,
        "runtime_s": time.time() - start_time,
    }
    metadata_path = run_dir / f"{args.name}_strict_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    overlay_paths = save_overlay_images(args.video, run_dir / "overlays", args.name, rois, arrays, args.save_overlays)
    annotated_path: Optional[Path] = None
    if not args.no_annotated_video:
        annotated_path = args.annotated_video or (run_dir / f"{args.name}_strict_annotated.mp4")
        save_annotated_video(args.video, annotated_path, rois, arrays, fps)

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"NPZ: {npz_path}")
    print(f"Metadata: {metadata_path}")
    if time_series_plot_path:
        print(f"ANG/PEN/FL plot: {time_series_plot_path}")
    if confidence_plot_path:
        print(f"Confidence plot: {confidence_plot_path}")
    if annotated_path:
        print(f"Annotated video: {annotated_path}")
    for path in overlay_paths:
        print(f"Overlay image: {path}")

    return {
        "csv": csv_path,
        "npz": npz_path,
        "metadata": metadata_path,
        "time_series_plot": time_series_plot_path,
        "confidence_plot": confidence_plot_path,
        "annotated_video": annotated_path,
    }


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
