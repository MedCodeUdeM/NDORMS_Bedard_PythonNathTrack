#!/usr/bin/env python3
"""
Compare current Python outputs against MATLAB UltraTimTrack results.

This script intentionally compares both:
  1. final Fdat.Region.FL/PEN/ANG outputs
  2. TimTrack intermediate Fdat.geofeatures outputs

The intermediate comparison prevents Kalman tuning from hiding an upstream
TimTrack mismatch.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

from ultrasound_tracker.matlab_compat import (
    align_by_index,
    extract_final_region_arrays,
    extract_geofeature_arrays,
    format_metric_rows,
    load_matlab_result,
    metric_row,
)
from ultrasound_tracker.final_output import (
    final_outputs_from_components,
    image_depth_to_mm_per_pixel,
)


def read_first_frame_height(video_path: Path) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to infer image height from video.") from exc

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    return int(frame.shape[0])


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def add_row(
    rows: List[Dict],
    name: str,
    reference,
    estimate,
    estimate_offset: int,
) -> None:
    ref, est = align_by_index(reference, estimate, estimate_offset=estimate_offset)
    rows.append(metric_row(name, ref, est))


def build_final_output_estimate(data: Dict[str, np.ndarray], mm_per_pixel: float) -> Dict[str, np.ndarray]:
    """Return FL/PEN/ANG arrays from either legacy or package-style NPZ keys."""
    if {"FL_mm", "PEN_deg", "ANG_deg"}.issubset(data):
        return {
            "FL_mm": data["FL_mm"],
            "PEN_deg": data["PEN_deg"],
            "ANG_deg": data["ANG_deg"],
        }

    if {
        "fascicle_angle_deg",
        "super_apo_angle_deg",
        "muscle_thickness_px",
    }.issubset(data):
        out = final_outputs_from_components(
            data["fascicle_angle_deg"],
            data["super_apo_angle_deg"],
            data["muscle_thickness_px"],
            mm_per_pixel=mm_per_pixel,
        )
        return {
            "FL_mm": out["FL_mm"],
            "PEN_deg": out["PEN_deg"],
            "ANG_deg": out["ANG_deg"],
        }

    if {
        "utt_fascicle_length_px",
        "utt_pennation_angle_deg",
        "utt_fascicle_angle_deg",
    }.issubset(data):
        return {
            "FL_mm": data["utt_fascicle_length_px"] * mm_per_pixel,
            "PEN_deg": data["utt_pennation_angle_deg"],
            "ANG_deg": data["utt_fascicle_angle_deg"],
        }

    raise KeyError(
        "Could not infer final output arrays. Expected package keys "
        "(FL_mm/PEN_deg/ANG_deg), TimTrack component keys, or legacy UTT keys."
    )


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["comparison", "n", "bias", "mae", "rmse", "corr"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matlab-result",
        type=Path,
        default=Path("data/matlab/slow_low_01_DOWN_tracked_Q=001.mat"),
        help="Path to MATLAB UltraTimTrack .mat output.",
    )
    parser.add_argument(
        "--python-utt",
        type=Path,
        default=Path("results/ultratimtrack_geometric_kalman_features_arrays.npz"),
        help="Path to Python final UltraTimTrack-like NPZ output.",
    )
    parser.add_argument(
        "--python-timtrack",
        type=Path,
        default=Path("results/timtrack_sequence_dohough_alpha_features_arrays.npz"),
        help="Path to Python TimTrack-like NPZ output.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/raw/UltraTimTrack_test.mp4"),
        help="Video used to infer pixel-to-mm scale when --image-height-px is absent.",
    )
    parser.add_argument(
        "--image-height-px",
        type=int,
        default=None,
        help="Image height used for MATLAB mm conversion. Defaults to first video frame height.",
    )
    parser.add_argument(
        "--estimate-offset",
        type=int,
        default=0,
        help="Index offset applied to Python estimates before comparison.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/matlab_comparison/parity_metrics.csv"),
        help="Where to write metric rows.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    mat = load_matlab_result(args.matlab_result)
    matlab_final = extract_final_region_arrays(mat)
    matlab_geo = extract_geofeature_arrays(mat)

    python_utt = load_npz(args.python_utt)
    python_tim = load_npz(args.python_timtrack)

    image_height_px = args.image_height_px
    if image_height_px is None:
        image_height_px = read_first_frame_height(args.video)

    image_depth_mm = float(matlab_final["image_depth_mm"])
    if not np.isfinite(image_depth_mm):
        raise RuntimeError("Could not read TrackingData.res from MATLAB result.")

    mm_per_pixel = image_depth_to_mm_per_pixel(image_depth_mm, image_height_px)

    python_utt_final = build_final_output_estimate(python_utt, mm_per_pixel)
    python_tim_final = build_final_output_estimate(python_tim, mm_per_pixel)

    rows: List[Dict] = []

    add_row(
        rows,
        "final_FL_mm",
        matlab_final["length_mm"],
        python_utt_final["FL_mm"],
        args.estimate_offset,
    )
    add_row(
        rows,
        "final_PEN_deg",
        matlab_final["pennation_deg"],
        python_utt_final["PEN_deg"],
        args.estimate_offset,
    )
    add_row(
        rows,
        "final_ANG_deg",
        matlab_final["fascicle_angle_deg"],
        python_utt_final["ANG_deg"],
        args.estimate_offset,
    )

    add_row(
        rows,
        "timtrack_alpha_deg",
        matlab_geo["alpha_deg"],
        python_tim["fascicle_angle_deg"],
        args.estimate_offset,
    )
    add_row(
        rows,
        "timtrack_phi_vs_python_pen_deg",
        matlab_geo["phi_deg"],
        python_tim_final["PEN_deg"],
        args.estimate_offset,
    )
    add_row(
        rows,
        "timtrack_formula_faslen_px",
        matlab_geo["faslen_px"],
        python_tim_final["FL_mm"] / mm_per_pixel,
        args.estimate_offset,
    )
    if "selected_line_length_px" in python_tim:
        add_row(
            rows,
            "timtrack_selected_segment_length_px_debug",
            matlab_geo["faslen_px"],
            python_tim["selected_line_length_px"],
            args.estimate_offset,
        )
    add_row(
        rows,
        "timtrack_gamma_deep_apo_deg",
        matlab_geo["deep_apo_angle_deg"],
        python_tim["deep_apo_angle_deg"],
        args.estimate_offset,
    )

    write_csv(args.out_csv, rows)

    print(f"MATLAB result: {args.matlab_result}")
    print(f"Python final:  {args.python_utt}")
    print(f"Python TimTrack-like: {args.python_timtrack}")
    print(f"image_depth_mm={image_depth_mm:.6g}")
    print(f"image_height_px={image_height_px}")
    print(f"mm_per_pixel={mm_per_pixel:.8f}")
    print()
    print(format_metric_rows(rows))
    print()
    print(f"Wrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
