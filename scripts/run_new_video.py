#!/usr/bin/env python3
"""
Run the current Python UltraTimTrack-style sequence path on a new video.

This script is the command-line version of the current Notebook 23 workflow:

1. Load a video.
2. Select or reuse superficial, deep, and fascicle ROIs.
3. Detect aponeuroses and a fascicle mask frame by frame.
4. Estimate fascicle alpha with MATLAB-style dohough + weighted median.
5. Compute final FL/PEN/ANG with final_outputs_from_lines().

The selected OpenCV line segment is saved as debug/visualization only. The
final fascicle length is the formula output: thickness / sin(pennation).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def find_project_root() -> Path:
    """Find the repository root from either cwd or this script location."""
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "ultrasound_tracker").exists():
            return candidate
    raise RuntimeError("Could not find project root containing ultrasound_tracker.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import ultrasound_tracker.geometry as geom
import ultrasound_tracker.roi as roi
import ultrasound_tracker.utils as ut
from ultrasound_tracker.aponeurosis_detector import AponeurosisDetector
from ultrasound_tracker.final_output import final_outputs_from_lines
from ultrasound_tracker.frangi_detector import FrangiDetector
from ultrasound_tracker.timtrack_hough import DoHoughParams, dohough


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current Python UltraTimTrack-style final-output path on a video.",
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the ultrasound video, for example data/raw/my_video.mp4.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output name prefix. Defaults to the video filename stem.",
    )
    parser.add_argument(
        "--roi-path",
        type=Path,
        default=None,
        help="ROI JSON path. Defaults to data/rois/<name>_rois.json.",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Open the OpenCV ROI selector even if the ROI JSON already exists.",
    )
    parser.add_argument(
        "--overwrite-roi",
        action="store_true",
        help="Allow replacing an existing ROI JSON when selecting ROIs.",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for CSV, NPZ, and overlay outputs.",
    )
    parser.add_argument(
        "--x-eval",
        type=float,
        default=20.0,
        help="X coordinate used for MATLAB-style aponeurosis thickness.",
    )
    parser.add_argument(
        "--thetares",
        type=float,
        default=1.0,
        help="dohough theta resolution in degrees.",
    )
    parser.add_argument("--alpha-min", type=float, default=8.0)
    parser.add_argument("--alpha-max", type=float, default=80.0)
    parser.add_argument(
        "--image-depth-mm",
        type=float,
        default=None,
        help="Optional image depth in mm. Used to derive mm_per_pixel from frame height.",
    )
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=None,
        help="Optional direct pixel scale. Overrides --image-depth-mm.",
    )
    parser.add_argument(
        "--save-overlays",
        type=int,
        default=3,
        help="Number of successful frames to save as overlay PNGs. Use 0 to disable.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N processed frames.",
    )
    return parser.parse_args()


def read_first_frame(video_path: Path) -> Tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read first frame from: {video_path}")

    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame, fps, n_frames


def select_or_load_rois(
    frame0_gray: np.ndarray,
    roi_path: Path,
    *,
    select_roi: bool,
    overwrite_roi: bool,
) -> Dict[str, roi.ROI]:
    if roi_path.exists() and not select_roi:
        print(f"Loading ROIs: {roi_path}")
        return roi.load_rois(roi_path)

    if roi_path.exists() and select_roi and not overwrite_roi:
        raise FileExistsError(
            f"ROI file already exists: {roi_path}\n"
            "Use --overwrite-roi if you want to replace it."
        )

    print("Select ROIs in this order: superficial, deep, fascicle.")
    selected = roi.select_all_rois_cv2(frame0_gray, include_fascicle_roi=True)
    roi.save_rois(selected, roi_path)
    print(f"Saved ROIs: {roi_path}")
    return selected


def make_aponeurosis_detector() -> AponeurosisDetector:
    return AponeurosisDetector(
        method="frangi",
        sigma=10.0,
        th=0.5,
        filtfac=1.0,
        maxlengthratio=0.9,
        frangi_scale_range=(18, 20),
        frangi_scale_ratio=1,
        frangi_black_ridges=False,
        apomargin=20,
        napo=10,
        fillgap=50,
        fit_method="enforce_maxangle",
        maxangle=0.5,
        adaptive_block_size=51,
    )


def make_fascicle_detector() -> FrangiDetector:
    return FrangiDetector(
        sigmas=(1, 2, 4),
        alpha=0.5,
        beta=15.0,
        black_ridges=False,
        threshold=0.08,
        angle_min=10,
        angle_max=70,
        hough_threshold=15,
        min_line_length=25,
        max_line_gap=15,
    )


def scalar0(value: Any) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def frangi_mask_and_lines(
    fas_img: np.ndarray,
    detector: FrangiDetector,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    vesselness = detector.get_vesselness_map(fas_img)
    binary_bool = vesselness >= detector.threshold
    binary_u8 = binary_bool.astype(np.uint8) * 255

    raw_lines = cv2.HoughLinesP(
        binary_u8,
        rho=1,
        theta=np.pi / 180,
        threshold=detector.hough_threshold,
        minLineLength=detector.min_line_length,
        maxLineGap=detector.max_line_gap,
    )
    if raw_lines is None:
        return binary_bool, None, None, None

    lines = raw_lines[:, 0, :].astype(np.float32)
    signed_angles = geom.line_angles_batch(lines, degrees=True)
    abs_angles = np.abs(signed_angles)
    lengths = geom.line_lengths_batch(lines)
    keep = (abs_angles >= detector.angle_min) & (abs_angles <= detector.angle_max)

    filtered_lines = lines[keep]
    filtered_angles = abs_angles[keep]
    filtered_lengths = lengths[keep]
    if len(filtered_lines) == 0:
        return binary_bool, None, None, None

    return binary_bool, filtered_lines, filtered_angles, filtered_lengths


def frame_indices(n_frames: int, start: int, end: Optional[int], step: int) -> List[int]:
    if step <= 0:
        raise ValueError("--frame-step must be positive.")
    if start < 0:
        raise ValueError("--frame-start must be >= 0.")
    effective_end = n_frames if end is None else min(int(end), n_frames)
    indices = list(range(int(start), effective_end, int(step)))
    if not indices:
        raise ValueError("No frames selected. Check frame start/end/step.")
    return indices


def finite_line(line: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(line)))


def write_csv(path: Path, results: Dict[str, List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(len(results[fieldnames[0]])):
            writer.writerow({key: results[key][row_idx] for key in fieldnames})


def save_overlay(
    video_path: Path,
    output_path: Path,
    rois: Dict[str, roi.ROI],
    frame_idx: int,
    result_idx: int,
    arrays: Dict[str, np.ndarray],
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    if frame.ndim == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame.copy()

    vis = roi.draw_rois(frame_gray, rois)
    ut.draw_line_on_image(vis, arrays["sup_apo_lines"][result_idx], color=(255, 0, 0), thickness=3)
    ut.draw_line_on_image(vis, arrays["deep_apo_lines"][result_idx], color=(0, 255, 0), thickness=3)
    ut.draw_line_on_image(vis, arrays["fascicle_lines"][result_idx], color=(0, 255, 255), thickness=2)
    ut.draw_line_on_image(vis, arrays["fascicle_segments"][result_idx], color=(0, 0, 255), thickness=3)

    text_lines = [
        f"Frame: {frame_idx}",
        f"ANG: {arrays['ANG_deg'][result_idx]:.2f} deg",
        f"PEN: {arrays['PEN_deg'][result_idx]:.2f} deg",
        f"FL: {arrays['FL_px'][result_idx]:.1f} px",
    ]
    if "FL_mm" in arrays:
        text_lines[-1] += f" / {arrays['FL_mm'][result_idx]:.2f} mm"

    ut.put_text_lines_on_image(
        vis,
        text_lines,
        origin=(30, 35),
        line_spacing=24,
        font_scale=0.65,
        color=(255, 255, 255),
        outline_color=(0, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def process_video(args: argparse.Namespace) -> Dict[str, Path]:
    video_path = args.video.expanduser().resolve()
    name = args.name or video_path.stem
    roi_path = args.roi_path or (PROJECT_ROOT / "data" / "rois" / f"{name}_rois.json")
    roi_path = roi_path.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    overlays_dir = results_dir / f"{name}_overlays"

    frame0_gray, fps, n_frames = read_first_frame(video_path)
    rois = select_or_load_rois(
        frame0_gray,
        roi_path,
        select_roi=args.select_roi or not roi_path.exists(),
        overwrite_roi=args.overwrite_roi,
    )

    missing = {"superficial", "deep", "fascicle"} - set(rois)
    if missing:
        raise KeyError(f"ROI file is missing required entries: {sorted(missing)}")

    mm_per_pixel = args.mm_per_pixel
    if mm_per_pixel is None and args.image_depth_mm is not None:
        mm_per_pixel = float(args.image_depth_mm) / float(frame0_gray.shape[0])

    indices = frame_indices(n_frames, args.frame_start, args.frame_end, args.frame_step)
    target_frames = set(indices)

    output_csv = results_dir / f"{name}_timtrack_final_features.csv"
    output_npz = results_dir / f"{name}_timtrack_final_features_arrays.npz"

    apo_detector = make_aponeurosis_detector()
    fas_detector = make_fascicle_detector()
    dohough_params = DoHoughParams(
        angle_range=(args.alpha_min, args.alpha_max),
        thetares=args.thetares,
        rhores=1.0,
        emask_radius=(rois["fascicle"][3] / 2.0, rois["fascicle"][2] / 2.0),
        npeaks=10,
        replace_diagonal_bias=True,
    )

    results: Dict[str, List[Any]] = {
        "frame": [],
        "time_s": [],
        "success": [],
        "ANG_deg": [],
        "PEN_deg": [],
        "FL_px": [],
        "FL_mm": [],
        "fascicle_angle_deg": [],
        "pennation_angle_deg": [],
        "fascicle_length_px": [],
        "final_fascicle_length_px": [],
        "super_apo_angle_deg": [],
        "deep_apo_angle_deg": [],
        "muscle_thickness_px": [],
        "selected_line_angle_deg": [],
        "selected_line_length_px": [],
        "dohough_mask_density": [],
        "dohough_n_peaks": [],
        "n_fascicle_candidates": [],
        "error": [],
    }

    sup_apo_lines = []
    deep_apo_lines = []
    fascicle_lines = []
    fascicle_segments = []
    sup_attachments = []
    deep_attachments = []
    dohough_peak_alphas = []
    dohough_peak_weights = []

    cap = cv2.VideoCapture(str(video_path))
    processed = 0
    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx not in target_frames:
            continue

        processed += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        time_s = frame_idx / fps if fps and fps > 0 else np.nan

        success = False
        error_msg = ""
        n_candidates = 0

        sup_line_global = np.full(4, np.nan, dtype=np.float32)
        deep_line_global = np.full(4, np.nan, dtype=np.float32)
        fas_best_line_global = np.full(4, np.nan, dtype=np.float32)
        fas_segment = np.full(4, np.nan, dtype=np.float32)
        sup_attachment = np.full(2, np.nan, dtype=np.float32)
        deep_attachment = np.full(2, np.nan, dtype=np.float32)
        peak_alphas = np.full(10, np.nan, dtype=np.float32)
        peak_weights = np.full(10, np.nan, dtype=np.float32)

        fascicle_angle_deg = np.nan
        pennation_angle_deg = np.nan
        fascicle_length_px = np.nan
        final_fascicle_length_px = np.nan
        fascicle_length_mm = np.nan
        super_apo_angle_deg = np.nan
        deep_apo_angle_deg = np.nan
        muscle_thickness_px = np.nan
        selected_line_angle_deg = np.nan
        selected_line_length_px = np.nan
        dohough_mask_density = np.nan
        dohough_n_peaks = 0

        try:
            sup_img = roi.extract_roi(frame_gray, rois["superficial"])
            deep_img = roi.extract_roi(frame_gray, rois["deep"])
            fas_img = roi.extract_roi(frame_gray, rois["fascicle"])

            sup_result = apo_detector.detect(sup_img, kind="superficial")
            sup_line_tmp = roi.line_local_to_global(sup_result["line_local"], rois["superficial"])
            if sup_line_tmp is None:
                raise RuntimeError("No superficial aponeurosis detected.")

            deep_result = apo_detector.detect(deep_img, kind="deep")
            deep_line_tmp = roi.line_local_to_global(deep_result["line_local"], rois["deep"])
            if deep_line_tmp is None:
                raise RuntimeError("No deep aponeurosis detected.")

            fas_binary, fas_lines_local, _, fas_lengths = frangi_mask_and_lines(fas_img, fas_detector)
            hough_result = dohough(fas_binary, dohough_params)
            fascicle_angle_deg = float(hough_result["alpha"])
            if not np.isfinite(fascicle_angle_deg):
                raise RuntimeError("No dohough fascicle alpha detected.")

            dohough_mask_density = float(np.mean(fas_binary))
            dohough_n_peaks = int(len(hough_result["alphas"]))
            if dohough_n_peaks:
                n_fill = min(10, dohough_n_peaks)
                peak_alphas[:n_fill] = hough_result["alphas"][:n_fill]
                peak_weights[:n_fill] = hough_result["weights"][:n_fill]

            fas_lines_global = roi.lines_local_to_global(fas_lines_local, rois["fascicle"])
            n_candidates = 0 if fas_lines_global is None else len(fas_lines_global)
            fas_best_tmp = geom.pick_best_fascicle_line(
                fas_lines_global,
                lengths=fas_lengths,
                superficial_apo_line=sup_line_tmp,
                deep_apo_line=deep_line_tmp,
                frame_shape=frame_gray.shape,
                margin=50,
            )
            if fas_best_tmp is None:
                raise RuntimeError("No fascicle line selected for visualization/debug.")

            geometry_features = geom.compute_fascicle_geometry(
                superficial_apo_line=sup_line_tmp,
                deep_apo_line=deep_line_tmp,
                fascicle_line=fas_best_tmp,
            )
            final_output = final_outputs_from_lines(
                fascicle_angle_deg,
                sup_line_tmp,
                deep_line_tmp,
                x_eval=args.x_eval,
                pennation_reference="superficial",
                mm_per_pixel=mm_per_pixel,
            )

            success = True
            sup_line_global = np.asarray(sup_line_tmp, dtype=np.float32)
            deep_line_global = np.asarray(deep_line_tmp, dtype=np.float32)
            fas_best_line_global = np.asarray(fas_best_tmp, dtype=np.float32)
            fas_segment = np.asarray(geometry_features["fascicle_segment_between_apos"], dtype=np.float32)
            sup_attachment = np.asarray(geometry_features["sup_attachment"], dtype=np.float32)
            deep_attachment = np.asarray(geometry_features["deep_attachment"], dtype=np.float32)

            selected_line_angle_deg = float(geometry_features["fascicle_angle_deg"])
            selected_line_length_px = float(geometry_features["fascicle_length_px"])
            pennation_angle_deg = scalar0(final_output["PEN_deg"])
            fascicle_length_px = scalar0(final_output["FL_px"])
            final_fascicle_length_px = fascicle_length_px
            super_apo_angle_deg = scalar0(final_output["super_apo_angle_deg"])
            deep_apo_angle_deg = scalar0(final_output["deep_apo_angle_deg"])
            muscle_thickness_px = scalar0(final_output["muscle_thickness_px"])
            if "FL_mm" in final_output:
                fascicle_length_mm = scalar0(final_output["FL_mm"])

        except Exception as exc:
            error_msg = str(exc)

        results["frame"].append(frame_idx)
        results["time_s"].append(time_s)
        results["success"].append(success)
        results["ANG_deg"].append(fascicle_angle_deg)
        results["PEN_deg"].append(pennation_angle_deg)
        results["FL_px"].append(fascicle_length_px)
        results["FL_mm"].append(fascicle_length_mm)
        results["fascicle_angle_deg"].append(fascicle_angle_deg)
        results["pennation_angle_deg"].append(pennation_angle_deg)
        results["fascicle_length_px"].append(fascicle_length_px)
        results["final_fascicle_length_px"].append(final_fascicle_length_px)
        results["super_apo_angle_deg"].append(super_apo_angle_deg)
        results["deep_apo_angle_deg"].append(deep_apo_angle_deg)
        results["muscle_thickness_px"].append(muscle_thickness_px)
        results["selected_line_angle_deg"].append(selected_line_angle_deg)
        results["selected_line_length_px"].append(selected_line_length_px)
        results["dohough_mask_density"].append(dohough_mask_density)
        results["dohough_n_peaks"].append(dohough_n_peaks)
        results["n_fascicle_candidates"].append(n_candidates)
        results["error"].append(error_msg)

        sup_apo_lines.append(sup_line_global)
        deep_apo_lines.append(deep_line_global)
        fascicle_lines.append(fas_best_line_global)
        fascicle_segments.append(fas_segment)
        sup_attachments.append(sup_attachment)
        deep_attachments.append(deep_attachment)
        dohough_peak_alphas.append(peak_alphas)
        dohough_peak_weights.append(peak_weights)

        if args.progress_every > 0 and processed % args.progress_every == 0:
            n_success = int(np.sum(results["success"]))
            print(f"Processed {processed}/{len(indices)} frames - success {n_success}")

    cap.release()

    arrays: Dict[str, np.ndarray] = {
        "frame": np.asarray(results["frame"], dtype=np.int32),
        "time_s": np.asarray(results["time_s"], dtype=np.float32),
        "success": np.asarray(results["success"], dtype=bool),
        "sup_apo_lines": np.vstack(sup_apo_lines).astype(np.float32),
        "deep_apo_lines": np.vstack(deep_apo_lines).astype(np.float32),
        "fascicle_lines": np.vstack(fascicle_lines).astype(np.float32),
        "fascicle_segments": np.vstack(fascicle_segments).astype(np.float32),
        "sup_attachments": np.vstack(sup_attachments).astype(np.float32),
        "deep_attachments": np.vstack(deep_attachments).astype(np.float32),
        "ANG_deg": np.asarray(results["ANG_deg"], dtype=np.float32),
        "PEN_deg": np.asarray(results["PEN_deg"], dtype=np.float32),
        "FL_px": np.asarray(results["FL_px"], dtype=np.float32),
        "FL_mm": np.asarray(results["FL_mm"], dtype=np.float32),
        "fascicle_angle_deg": np.asarray(results["fascicle_angle_deg"], dtype=np.float32),
        "pennation_angle_deg": np.asarray(results["pennation_angle_deg"], dtype=np.float32),
        "fascicle_length_px": np.asarray(results["fascicle_length_px"], dtype=np.float32),
        "final_fascicle_length_px": np.asarray(results["final_fascicle_length_px"], dtype=np.float32),
        "super_apo_angle_deg": np.asarray(results["super_apo_angle_deg"], dtype=np.float32),
        "deep_apo_angle_deg": np.asarray(results["deep_apo_angle_deg"], dtype=np.float32),
        "muscle_thickness_px": np.asarray(results["muscle_thickness_px"], dtype=np.float32),
        "selected_line_angle_deg": np.asarray(results["selected_line_angle_deg"], dtype=np.float32),
        "selected_line_length_px": np.asarray(results["selected_line_length_px"], dtype=np.float32),
        "dohough_mask_density": np.asarray(results["dohough_mask_density"], dtype=np.float32),
        "dohough_n_peaks": np.asarray(results["dohough_n_peaks"], dtype=np.int32),
        "dohough_peak_alphas": np.vstack(dohough_peak_alphas).astype(np.float32),
        "dohough_peak_weights": np.vstack(dohough_peak_weights).astype(np.float32),
        "n_fascicle_candidates": np.asarray(results["n_fascicle_candidates"], dtype=np.int32),
    }
    if mm_per_pixel is not None:
        arrays["mm_per_pixel"] = np.asarray(float(mm_per_pixel), dtype=np.float32)

    write_csv(output_csv, results)
    np.savez(output_npz, **arrays)

    overlay_paths: List[Path] = []
    if args.save_overlays > 0:
        valid_indices = np.where(arrays["success"])[0]
        if len(valid_indices):
            positions = np.linspace(0, len(valid_indices) - 1, min(args.save_overlays, len(valid_indices)))
            selected = [int(valid_indices[int(round(pos))]) for pos in positions]
            for result_idx in selected:
                frame_idx = int(arrays["frame"][result_idx])
                output_path = overlays_dir / f"{name}_frame_{frame_idx:06d}.png"
                save_overlay(video_path, output_path, rois, frame_idx, result_idx, arrays)
                overlay_paths.append(output_path)

    success_count = int(np.sum(arrays["success"]))
    print("\nDone.")
    print(f"Video: {video_path}")
    print(f"Frames processed: {len(indices)}")
    print(f"Successful frames: {success_count} ({100 * success_count / len(indices):.1f}%)")
    print(f"CSV: {output_csv}")
    print(f"NPZ: {output_npz}")
    print(f"ROI: {roi_path}")
    for path in overlay_paths:
        print(f"Overlay: {path}")

    errors = [err for err in results["error"] if err]
    if errors:
        counts: Dict[str, int] = {}
        for err in errors:
            counts[err] = counts.get(err, 0) + 1
        print("\nTop errors:")
        for err, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"{count}x - {err}")

    return {
        "csv": output_csv,
        "npz": output_npz,
        "roi": roi_path,
        "overlays_dir": overlays_dir,
    }


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
