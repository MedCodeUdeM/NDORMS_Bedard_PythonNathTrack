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
from typing import Dict, List, Optional, Sequence

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
    line_y_at_x,
)
from ultrasound_tracker.matlab_aponeurosis import fit_apo_matlab_like, make_matlab_apox


def read_first_frame_shape(video_path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to infer image shape from video.") from exc

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    return int(frame.shape[0]), int(frame.shape[1])


def read_first_frame_height(video_path: Path) -> int:
    return read_first_frame_shape(video_path)[0]


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_exported_apo_params(path: Optional[Path]) -> Optional[dict]:
    """Load MATLAB ``parms.apo`` from an exported UTT numeric .mat file."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to load MATLAB parameter exports.") from exc

    mat = loadmat(path, simplify_cells=True)
    root = mat.get("UTT_numeric_export", mat)
    if not isinstance(root, dict):
        return None

    parms = root.get("parms")
    if not isinstance(parms, dict):
        return None
    apo = parms.get("apo")
    if not isinstance(apo, dict):
        return None
    return apo


def load_exported_apox(path: Optional[Path]) -> Optional[np.ndarray]:
    """Load MATLAB ``parms.apo.apox`` from an exported UTT numeric .mat file."""
    apo = load_exported_apo_params(path)
    if not isinstance(apo, dict) or "apox" not in apo:
        return None

    apox = np.asarray(apo["apox"], dtype=np.float64).reshape(-1)
    return apox if apox.size else None


def _entry_list(value) -> List[dict]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return [entry for entry in value if isinstance(entry, dict)]
    arr = np.asarray(value, dtype=object).reshape(-1)
    return [entry for entry in arr if isinstance(entry, dict)]


def extract_timtrack_export_geofeatures(path: Path) -> Dict[str, np.ndarray]:
    """
    Build TimTrack geofeature references from an intermediate MATLAB mask export.

    NB36-style exports capture the raw ``filter_usimage -> get_fasMask ->
    dohough`` path. That is the correct reference for the TimTrack gate when the
    later ``Fdat.geofeatures`` result file disagrees with the exported masks.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to load MATLAB TimTrack exports.") from exc

    mat = loadmat(path, simplify_cells=True)
    entries = _entry_list(mat.get("M", []))
    if not entries:
        raise ValueError(f"No M entries found in MATLAB TimTrack export: {path}")

    parms = mat.get("parms", {})
    apo = parms.get("apo", {}) if isinstance(parms, dict) else {}
    if not isinstance(apo, dict) or "apox" not in apo:
        raise ValueError(f"Missing parms.apo.apox in MATLAB TimTrack export: {path}")

    apox = np.asarray(apo["apox"], dtype=np.float64).reshape(-1)
    super_settings = apo.get("super", {}) if isinstance(apo.get("super", {}), dict) else {}
    deep_settings = apo.get("deep", {}) if isinstance(apo.get("deep", {}), dict) else {}
    x_eval = float(apo.get("x", 20.0))
    frames = np.asarray([int(np.asarray(entry["frame0"]).reshape(-1)[0]) for entry in entries], dtype=np.int64)
    n_out = int(np.max(frames)) + 1

    keys = [
        "alpha_deg",
        "phi_deg",
        "faslen_px",
        "deep_apo_angle_deg",
        "super_apo_angle_deg",
        "muscle_thickness_px",
        "super_pos_y1",
        "super_pos_y2",
        "deep_pos_y1",
        "deep_pos_y2",
    ]
    out = {key: np.full(n_out, np.nan, dtype=np.float64) for key in keys}

    def _fit(vec, settings, order=None):
        fit_order = int(settings.get("order", 1) if order is None else order)
        return fit_apo_matlab_like(
            apox,
            vec,
            fit_method=str(settings.get("fit_method", "enforce_maxangle")),
            maxangle=float(settings.get("maxangle", 0.5)),
            order=fit_order,
        )

    for entry, frame in zip(entries, frames):
        super_vec = np.asarray(entry["super_vec"], dtype=np.float64).reshape(-1)
        deep_vec = np.asarray(entry["deep_vec"], dtype=np.float64).reshape(-1)
        alpha = float(np.asarray(entry["alpha"]).reshape(-1)[0])
        width = int(np.asarray(entry["fascicle_masked"]).shape[1])

        super_coef = _fit(super_vec, super_settings)
        deep_coef = _fit(deep_vec, deep_settings)
        super_coef_lin = _fit(super_vec, super_settings, order=1)
        deep_coef_lin = _fit(deep_vec, deep_settings, order=1)
        if (
            super_coef is None
            or deep_coef is None
            or super_coef_lin is None
            or deep_coef_lin is None
        ):
            continue

        beta = -float(np.rad2deg(np.arctan2(super_coef_lin[0], 1.0)))
        gamma = -float(np.rad2deg(np.arctan2(deep_coef_lin[0], 1.0)))
        thickness = (np.polyval(deep_coef, x_eval) - np.polyval(super_coef, x_eval)) * np.cos(
            np.deg2rad(beta)
        )
        phi = alpha - beta
        faslen = thickness / np.sin(np.deg2rad(phi))
        super_pos = np.polyval(super_coef, [1.0, float(width)])
        deep_pos = np.polyval(deep_coef, [1.0, float(width)])

        out["alpha_deg"][frame] = alpha
        out["phi_deg"][frame] = phi
        out["faslen_px"][frame] = faslen
        out["deep_apo_angle_deg"][frame] = gamma
        out["super_apo_angle_deg"][frame] = beta
        out["muscle_thickness_px"][frame] = thickness
        out["super_pos_y1"][frame] = super_pos[0]
        out["super_pos_y2"][frame] = super_pos[1]
        out["deep_pos_y1"][frame] = deep_pos[0]
        out["deep_pos_y2"][frame] = deep_pos[1]

    return out


def add_row(
    rows: List[Dict],
    name: str,
    reference,
    estimate,
    estimate_offset: int,
) -> None:
    ref, est = align_by_index(reference, estimate, estimate_offset=estimate_offset)
    rows.append(metric_row(name, ref, est))


def infer_image_width(data: Dict[str, np.ndarray], image_width_px: Optional[int]) -> int:
    if image_width_px is not None:
        return int(image_width_px)
    if "image_width_px" in data:
        return int(np.asarray(data["image_width_px"]).reshape(-1)[0])
    if "sup_apo_lines" in data:
        lines = np.asarray(data["sup_apo_lines"], dtype=np.float64)
        if lines.size:
            max_x = np.nanmax(lines[:, [0, 2]])
            if np.isfinite(max_x):
                return int(round(max_x)) + 1
    if "deep_apo_lines" in data:
        lines = np.asarray(data["deep_apo_lines"], dtype=np.float64)
        if lines.size:
            max_x = np.nanmax(lines[:, [0, 2]])
            if np.isfinite(max_x):
                return int(round(max_x)) + 1
    raise ValueError(
        "Could not infer image width for MATLAB super_pos/deep_pos comparison. "
        "Pass --image-width-px."
    )


def matlab_endpoint_positions_from_vectors(
    vectors: np.ndarray,
    apox_1b: np.ndarray,
    width_px: int,
    fit_settings: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Recreate MATLAB ``polyval(apo_coef, [1 n])`` from saved aponeurosis vectors.

    The Python aponeurosis vectors are kept in MATLAB one-based y coordinates, so
    the resulting endpoint y values are one-based too, matching Fdat.geofeatures.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    apox = np.asarray(apox_1b, dtype=np.float64).reshape(-1)
    if vectors.ndim != 2:
        raise ValueError("aponeurosis vectors must be a 2D array.")
    if vectors.shape[1] != apox.size:
        raise ValueError(
            f"aponeurosis vector width ({vectors.shape[1]}) does not match apox "
            f"length ({apox.size})."
        )

    y1 = np.full(vectors.shape[0], np.nan, dtype=np.float64)
    y2 = np.full(vectors.shape[0], np.nan, dtype=np.float64)
    x_endpoints = np.asarray([1.0, float(width_px)], dtype=np.float64)
    fit_settings = fit_settings or {}
    fit_method = str(fit_settings.get("fit_method", "enforce_maxangle"))
    maxangle = float(fit_settings.get("maxangle", 0.5))
    order = int(fit_settings.get("order", 1))

    for idx, vec in enumerate(vectors):
        coef = fit_apo_matlab_like(
            apox,
            vec,
            fit_method=fit_method,
            maxangle=maxangle,
            order=order,
        )
        if coef is None:
            continue
        endpoints = np.polyval(coef, x_endpoints)
        y1[idx], y2[idx] = endpoints

    return {"pos_y1": y1, "pos_y2": y2}


def matlab_endpoint_positions_from_lines(lines: np.ndarray, width_px: int) -> Dict[str, np.ndarray]:
    """Evaluate zero-based Python line segments at MATLAB endpoints x=[1,width]."""
    lines = np.asarray(lines, dtype=np.float64)
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("line array must have shape (N, 4).")

    # Python lines use zero-based image coordinates. MATLAB endpoint positions are
    # one-based y values at one-based x=[1,width], so evaluate at zero-based
    # x=[0,width-1] and convert y back to one-based.
    y1 = line_y_at_x(lines, 0.0) + 1.0
    y2 = line_y_at_x(lines, float(width_px - 1)) + 1.0
    return {"pos_y1": y1, "pos_y2": y2}


def build_position_data(
    data: Dict[str, np.ndarray],
    *,
    prefix: str,
    vector_key: str,
    line_key: str,
    width_px: int,
    apox_1b: Optional[np.ndarray],
    fit_settings: Optional[dict] = None,
) -> Optional[Dict[str, np.ndarray]]:
    if vector_key in data:
        vectors = np.asarray(data[vector_key])
        if apox_1b is None:
            apox_1b = make_matlab_apox(width_px, napo=vectors.shape[1])
        endpoints = matlab_endpoint_positions_from_vectors(
            vectors,
            apox_1b,
            width_px,
            fit_settings=fit_settings,
        )
    elif line_key in data:
        endpoints = matlab_endpoint_positions_from_lines(np.asarray(data[line_key]), width_px)
    else:
        return None

    return {
        **data,
        f"{prefix}_pos_y1": endpoints["pos_y1"],
        f"{prefix}_pos_y2": endpoints["pos_y2"],
    }


def _normalise_frames(frames: Optional[Sequence[int]]) -> Optional[np.ndarray]:
    if not frames:
        return None
    return np.asarray([int(frame) for frame in frames], dtype=np.int64)


def select_reference_frames(values, frames: Optional[np.ndarray]):
    """Select MATLAB reference values by absolute frame number when requested."""
    arr = np.asarray(values)
    if frames is None:
        return arr
    return arr[frames]


def select_estimate_frames(data: Dict[str, np.ndarray], key: str, frames: Optional[np.ndarray]):
    """
    Select Python estimate values for a sparse frame list.

    Full-sequence NPZ files usually include a ``frame`` array, while small
    diagnostic NPZ files may already contain only the requested frames.
    """
    arr = np.asarray(data[key])
    if frames is None:
        return arr

    if "frame" in data:
        frame_values = np.asarray(data["frame"], dtype=np.int64).reshape(-1)
        index_by_frame = {int(frame): idx for idx, frame in enumerate(frame_values)}
        missing = [int(frame) for frame in frames if int(frame) not in index_by_frame]
        if missing:
            raise KeyError(f"{key}: missing requested frames in Python NPZ: {missing}")
        indices = np.asarray([index_by_frame[int(frame)] for frame in frames], dtype=np.int64)
        return arr[indices]

    if arr.shape[0] == len(frames):
        return arr
    if arr.shape[0] > int(np.max(frames)):
        return arr[frames]
    raise ValueError(
        f"{key}: cannot select sparse frames from array with shape {arr.shape}; "
        "provide a matching frame array in the NPZ."
    )


def add_frame_row(
    rows: List[Dict],
    name: str,
    reference,
    estimate_data: Dict[str, np.ndarray],
    estimate_key: str,
    frames: Optional[np.ndarray],
    estimate_offset: int,
) -> None:
    ref = select_reference_frames(reference, frames)
    est = select_estimate_frames(estimate_data, estimate_key, frames)
    add_row(rows, name, ref, est, estimate_offset)


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
        "--image-width-px",
        type=int,
        default=None,
        help=(
            "Image width used for MATLAB super_pos/deep_pos endpoint comparison. "
            "Defaults to line metadata when available."
        ),
    )
    parser.add_argument(
        "--utt-export",
        type=Path,
        default=None,
        help=(
            "Optional MATLAB UTT_numeric_export .mat file. When provided, "
            "parms.apo.apox is used to fit vector-only aponeurosis outputs."
        ),
    )
    parser.add_argument(
        "--matlab-timtrack-export",
        type=Path,
        default=None,
        help=(
            "Optional NB36-style MATLAB intermediate mask export. When provided, "
            "TimTrack geofeature rows use this raw dohough reference instead of "
            "Fdat.geofeatures from --matlab-result."
        ),
    )
    parser.add_argument(
        "--estimate-offset",
        type=int,
        default=0,
        help="Index offset applied to Python estimates before comparison.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional zero-based frame numbers to compare. Useful for sparse "
            "MATLAB intermediate-mask exports such as the 9-frame NB36 gate."
        ),
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
    frames = _normalise_frames(args.frames)

    mat = load_matlab_result(args.matlab_result)
    matlab_final = extract_final_region_arrays(mat)
    matlab_geo = extract_geofeature_arrays(mat)
    if args.matlab_timtrack_export is not None:
        matlab_geo = extract_timtrack_export_geofeatures(args.matlab_timtrack_export)

    python_utt = load_npz(args.python_utt)
    python_tim = load_npz(args.python_timtrack)
    apo_params = load_exported_apo_params(args.utt_export)
    apox_1b = (
        np.asarray(apo_params["apox"], dtype=np.float64).reshape(-1)
        if isinstance(apo_params, dict) and "apox" in apo_params
        else None
    )

    image_height_px = args.image_height_px
    image_width_px = args.image_width_px
    if image_height_px is None or image_width_px is None:
        video_height_px, video_width_px = read_first_frame_shape(args.video)
        if image_height_px is None:
            image_height_px = video_height_px
        if image_width_px is None:
            image_width_px = video_width_px

    image_depth_mm = float(matlab_final["image_depth_mm"])
    if not np.isfinite(image_depth_mm):
        raise RuntimeError("Could not read TrackingData.res from MATLAB result.")

    mm_per_pixel = image_depth_to_mm_per_pixel(image_depth_mm, image_height_px)
    image_width_px = infer_image_width(python_tim, image_width_px)

    python_utt_final = build_final_output_estimate(python_utt, mm_per_pixel)
    python_tim_final = build_final_output_estimate(python_tim, mm_per_pixel)
    if "frame" in python_utt:
        python_utt_final["frame"] = python_utt["frame"]
    if "frame" in python_tim:
        python_tim_final["frame"] = python_tim["frame"]

    rows: List[Dict] = []

    add_frame_row(
        rows,
        "final_FL_mm",
        matlab_final["length_mm"],
        python_utt_final,
        "FL_mm",
        frames,
        args.estimate_offset,
    )
    add_frame_row(
        rows,
        "final_PEN_deg",
        matlab_final["pennation_deg"],
        python_utt_final,
        "PEN_deg",
        frames,
        args.estimate_offset,
    )
    add_frame_row(
        rows,
        "final_ANG_deg",
        matlab_final["fascicle_angle_deg"],
        python_utt_final,
        "ANG_deg",
        frames,
        args.estimate_offset,
    )

    add_frame_row(
        rows,
        "timtrack_alpha_deg",
        matlab_geo["alpha_deg"],
        python_tim,
        "fascicle_angle_deg",
        frames,
        args.estimate_offset,
    )
    add_frame_row(
        rows,
        "timtrack_phi_vs_python_pen_deg",
        matlab_geo["phi_deg"],
        python_tim_final,
        "PEN_deg",
        frames,
        args.estimate_offset,
    )
    add_frame_row(
        rows,
        "timtrack_formula_faslen_px",
        matlab_geo["faslen_px"],
        {**python_tim_final, "FL_px": python_tim_final["FL_mm"] / mm_per_pixel},
        "FL_px",
        frames,
        args.estimate_offset,
    )
    if "selected_line_length_px" in python_tim:
        add_frame_row(
            rows,
            "timtrack_selected_segment_length_px_debug",
            matlab_geo["faslen_px"],
            python_tim,
            "selected_line_length_px",
            frames,
            args.estimate_offset,
        )
    add_frame_row(
        rows,
        "timtrack_gamma_deep_apo_deg",
        matlab_geo["deep_apo_angle_deg"],
        python_tim,
        "deep_apo_angle_deg",
        frames,
        args.estimate_offset,
    )
    add_frame_row(
        rows,
        "timtrack_betha_super_apo_deg",
        matlab_geo["super_apo_angle_deg"],
        python_tim,
        "super_apo_angle_deg",
        frames,
        args.estimate_offset,
    )

    position_data = build_position_data(
        python_tim,
        prefix="super",
        vector_key="super_aponeurosis_vector",
        line_key="sup_apo_lines",
        width_px=image_width_px,
        apox_1b=apox_1b,
        fit_settings=apo_params.get("super") if isinstance(apo_params, dict) else None,
    )

    if position_data is not None:
        add_frame_row(
            rows,
            "timtrack_super_pos_y1",
            matlab_geo["super_pos_y1"],
            position_data,
            "super_pos_y1",
            frames,
            args.estimate_offset,
        )
        add_frame_row(
            rows,
            "timtrack_super_pos_y2",
            matlab_geo["super_pos_y2"],
            position_data,
            "super_pos_y2",
            frames,
            args.estimate_offset,
        )

    position_data = build_position_data(
        python_tim,
        prefix="deep",
        vector_key="deep_aponeurosis_vector",
        line_key="deep_apo_lines",
        width_px=image_width_px,
        apox_1b=apox_1b,
        fit_settings=apo_params.get("deep") if isinstance(apo_params, dict) else None,
    )

    if position_data is not None:
        add_frame_row(
            rows,
            "timtrack_deep_pos_y1",
            matlab_geo["deep_pos_y1"],
            position_data,
            "deep_pos_y1",
            frames,
            args.estimate_offset,
        )
        add_frame_row(
            rows,
            "timtrack_deep_pos_y2",
            matlab_geo["deep_pos_y2"],
            position_data,
            "deep_pos_y2",
            frames,
            args.estimate_offset,
        )

    write_csv(args.out_csv, rows)

    print(f"MATLAB result: {args.matlab_result}")
    if args.matlab_timtrack_export is not None:
        print(f"MATLAB TimTrack export: {args.matlab_timtrack_export}")
    print(f"Python final:  {args.python_utt}")
    print(f"Python TimTrack-like: {args.python_timtrack}")
    print(f"image_depth_mm={image_depth_mm:.6g}")
    print(f"image_height_px={image_height_px}")
    print(f"image_width_px={image_width_px}")
    print(f"mm_per_pixel={mm_per_pixel:.8f}")
    if apox_1b is not None:
        print(f"apox_1b={apox_1b.tolist()}")
    if frames is not None:
        print(f"frames={frames.tolist()}")
    print()
    print(format_metric_rows(rows))
    print()
    print(f"Wrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
