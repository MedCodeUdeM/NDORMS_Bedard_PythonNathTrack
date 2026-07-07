#!/usr/bin/env python3
"""Validate tracked phantom displacement against imposed ground-truth motion.

This script is intentionally conservative: it does not modify or tune the
tracking algorithm. It reads already-produced strict runner outputs, converts
tracked image coordinates to displacement, loads an external phantom ground
truth table when available, and reports whether the comparison is mathematically
valid enough to support a tracking claim.

Coordinate contract used here
-----------------------------
The strict runner stores line segments as MATLAB-style one-based image
coordinates ``[x1, y1, x2, y2]`` in pixels. Displacements are formed by
subtracting the frame-0 segment midpoint, so the one-based offset cancels.
Image x increases to the right (lateral image axis) and image y increases
downward (axial/depth image axis). Angles use ``atan2(-dy, dx)``.

Unit contract used here
-----------------------
The current strict runner writes a scalar ``mm_per_pixel``. In the existing
code this is derived from image depth divided by image height, so it is an
axial pixel spacing. It is safe for axial displacement if the metadata depth is
correct. It is only safe for lateral/vector displacement if the ultrasound image
has square pixels or an independent lateral spacing is supplied separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


def find_project_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "ultrasound_tracker").exists():
            return candidate
    raise RuntimeError("Could not find project root containing ultrasound_tracker.")


PROJECT_ROOT = find_project_root()


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    opened: bool
    frame_count: int
    fps: float
    width_px: int
    height_px: int


@dataclass(frozen=True)
class EstimateBundle:
    source: str
    frame: np.ndarray
    time_s: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    magnitude_mm: np.ndarray
    failure: np.ndarray


@dataclass(frozen=True)
class GroundTruthBundle:
    path: Path
    frame: np.ndarray | None
    time_s: np.ndarray | None
    x_mm: np.ndarray | None
    y_mm: np.ndarray | None
    scalar_mm: np.ndarray | None
    axis_hint: str
    unit_source: str
    kind_source: str
    columns_used: dict[str, str | None]


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_video_metadata(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    opened = bool(cap.isOpened())
    if not opened:
        return VideoMetadata(path, False, 0, float("nan"), 0, 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoMetadata(path, opened, frame_count, fps, width, height)


def find_strict_outputs(video_stem: str, results_root: Path) -> dict[str, Path | None]:
    direct_dir = results_root / video_stem
    npz = direct_dir / f"{video_stem}_strict_results.npz"
    csv = direct_dir / f"{video_stem}_strict_FL_PEN_ANG.csv"
    metadata = direct_dir / f"{video_stem}_strict_metadata.json"
    if npz.exists():
        return {"npz": npz, "csv": csv if csv.exists() else None, "metadata": metadata if metadata.exists() else None}

    matches = sorted(results_root.glob(f"**/{video_stem}_strict_results.npz"))
    if matches:
        npz = matches[0]
        stem = npz.name.replace("_strict_results.npz", "")
        return {
            "npz": npz,
            "csv": npz.with_name(f"{stem}_strict_FL_PEN_ANG.csv") if npz.with_name(f"{stem}_strict_FL_PEN_ANG.csv").exists() else None,
            "metadata": npz.with_name(f"{stem}_strict_metadata.json") if npz.with_name(f"{stem}_strict_metadata.json").exists() else None,
        }
    return {"npz": None, "csv": None, "metadata": None}


def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(finite[0]) if finite.size else float("nan")


def midpoint_displacement_mm(segments: np.ndarray, mm_per_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg = np.asarray(segments, dtype=float)
    if seg.ndim != 2 or seg.shape[1] != 4:
        raise ValueError("segments must have shape (N, 4).")
    mid = np.column_stack([(seg[:, 0] + seg[:, 2]) / 2.0, (seg[:, 1] + seg[:, 3]) / 2.0])
    origin = mid[0].copy()
    disp_px = mid - origin
    x_mm = disp_px[:, 0] * float(mm_per_px)
    y_mm = disp_px[:, 1] * float(mm_per_px)
    mag_mm = np.hypot(x_mm, y_mm)
    return x_mm, y_mm, mag_mm


def cumulative_tracker_median_displacement_mm(data: np.lib.npyio.NpzFile, mm_per_px: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old_points = data["klt_tracked_old_points"] if "klt_tracked_old_points" in data.files else None
    new_points = data["klt_tracked_new_points"] if "klt_tracked_new_points" in data.files else None
    dx = np.full(n, np.nan, dtype=float)
    dy = np.full(n, np.nan, dtype=float)
    ok = np.zeros(n, dtype=bool)
    dx[0] = 0.0
    dy[0] = 0.0
    ok[0] = True
    if old_points is not None and new_points is not None:
        for idx in range(1, min(n, len(old_points), len(new_points))):
            old = np.asarray(old_points[idx], dtype=float).reshape(-1, 2)
            new = np.asarray(new_points[idx], dtype=float).reshape(-1, 2)
            m = min(len(old), len(new))
            if m == 0:
                continue
            delta = new[:m] - old[:m]
            finite = np.all(np.isfinite(delta), axis=1)
            if not np.any(finite):
                continue
            dx[idx] = float(np.nanmedian(delta[finite, 0]))
            dy[idx] = float(np.nanmedian(delta[finite, 1]))
            ok[idx] = True
    dx = np.where(np.isfinite(dx), dx, 0.0)
    dy = np.where(np.isfinite(dy), dy, 0.0)
    x_mm = np.cumsum(dx) * float(mm_per_px)
    y_mm = np.cumsum(dy) * float(mm_per_px)
    return x_mm, y_mm, np.hypot(x_mm, y_mm), ~ok


def estimate_bundles_from_strict(npz_path: Path, metadata_path: Path | None, video_meta: VideoMetadata) -> tuple[list[EstimateBundle], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    metadata = load_metadata(metadata_path)
    frame = np.asarray(data["frame"] if "frame" in data.files else np.arange(len(data["time_s"])), dtype=int)
    time_s = np.asarray(data["time_s"] if "time_s" in data.files else frame / video_meta.fps, dtype=float)
    mm_per_px = float(np.asarray(data["mm_per_pixel"]).reshape(-1)[0]) if "mm_per_pixel" in data.files else float(metadata.get("mm_per_pixel", np.nan))
    if not np.isfinite(mm_per_px) or mm_per_px <= 0:
        raise ValueError("Strict output does not contain a finite positive mm_per_pixel; pass --mm-per-pixel or fix metadata.")

    n = len(frame)
    detection_success = np.asarray(data["detection_success"], dtype=bool) if "detection_success" in data.files else np.ones(n, dtype=bool)
    bundles: list[EstimateBundle] = []

    segment_sources = {
        "final_kalman_midpoint": "fascicle_segments",
        "final_kalman_end_midpoint": "fascicle_end_segments",
        "klt_prior_midpoint": "klt_prior_segments",
        "fixed_kalman_midpoint": "fixed_fascicle_segments",
    }
    for source, key in segment_sources.items():
        if key not in data.files:
            continue
        x_mm, y_mm, mag_mm = midpoint_displacement_mm(np.asarray(data[key], dtype=float), mm_per_px)
        finite = np.isfinite(x_mm) & np.isfinite(y_mm)
        failure = (~finite) | (~detection_success[: len(finite)])
        if source.startswith("klt") and "klt_affine_ok" in data.files:
            klt_ok = np.asarray(data["klt_affine_ok"], dtype=bool)
            failure = failure | (~klt_ok[: len(failure)])
            failure[0] = False
        bundles.append(EstimateBundle(source, frame[: len(x_mm)], time_s[: len(x_mm)], x_mm, y_mm, mag_mm, failure))

    if "klt_tracked_old_points" in data.files and "klt_tracked_new_points" in data.files:
        x_mm, y_mm, mag_mm, failure = cumulative_tracker_median_displacement_mm(data, mm_per_px, n)
        bundles.append(EstimateBundle("klt_tracker_median_cumulative", frame, time_s, x_mm, y_mm, mag_mm, failure))

    audit = {
        "strict_npz": str(npz_path),
        "strict_metadata": str(metadata_path) if metadata_path else None,
        "mm_per_pixel": mm_per_px,
        "mm_per_pixel_interpretation": "scalar axial spacing from image depth / image height unless overridden upstream",
        "frames_in_output": int(n),
        "available_arrays": list(data.files),
        "estimate_sources": [bundle.source for bundle in bundles],
    }
    return bundles, audit


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def read_npz_or_npy_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return pd.DataFrame({"value": arr})
        return pd.DataFrame(arr)
    npz = np.load(path, allow_pickle=True)
    lengths = {}
    arrays: dict[str, np.ndarray] = {}
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.ndim == 0 or arr.dtype == object:
            continue
        if arr.ndim == 1:
            arrays[key] = arr
            lengths[key] = len(arr)
        elif arr.ndim == 2 and arr.shape[1] <= 4:
            for idx in range(arr.shape[1]):
                arrays[f"{key}_{idx}"] = arr[:, idx]
                lengths[f"{key}_{idx}"] = arr.shape[0]
    if not arrays:
        raise ValueError(f"No tabular numeric arrays found in {path}.")
    target_len = max(set(lengths.values()), key=list(lengths.values()).count)
    return pd.DataFrame({key: arr for key, arr in arrays.items() if len(arr) == target_len})


def flatten_mat_arrays(obj: Any, prefix: str, out: dict[str, np.ndarray]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten_mat_arrays(value, f"{prefix}_{key}" if prefix else str(key), out)
        return
    arr = np.asarray(obj)
    if arr.dtype == object or arr.ndim == 0:
        return
    if arr.ndim == 1 and arr.size > 1:
        out[prefix] = arr
    elif arr.ndim == 2 and min(arr.shape) <= 4 and max(arr.shape) > 1:
        shaped = arr if arr.shape[0] >= arr.shape[1] else arr.T
        for idx in range(shaped.shape[1]):
            out[f"{prefix}_{idx}"] = shaped[:, idx]


def read_mat_table(path: Path) -> pd.DataFrame:
    mat = loadmat(path, simplify_cells=True)
    arrays: dict[str, np.ndarray] = {}
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        flatten_mat_arrays(value, key, arrays)
    if not arrays:
        raise ValueError(f"No simple numeric arrays found in {path}.")
    lengths = {key: len(value) for key, value in arrays.items()}
    target_len = max(set(lengths.values()), key=list(lengths.values()).count)
    return pd.DataFrame({key: value for key, value in arrays.items() if len(value) == target_len})


def read_ground_truth_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.json_normalize(payload)
    if suffix in {".npz", ".npy"}:
        return read_npz_or_npy_table(path)
    if suffix == ".mat":
        return read_mat_table(path)
    raise ValueError(f"Unsupported ground-truth extension: {path.suffix}")


def find_column(columns: Iterable[str], include_any: Iterable[str], include_all: Iterable[str] = (), exclude_any: Iterable[str] = ()) -> str | None:
    normalized = {col: normalize_name(col) for col in columns}
    for col, norm in normalized.items():
        if any(token in norm for token in exclude_any):
            continue
        if include_all and not all(token in norm for token in include_all):
            continue
        if any(token in norm for token in include_any):
            return col
    return None


def infer_unit(column: str | None, fallback: str) -> str:
    if fallback != "auto":
        return fallback
    name = normalize_name(column or "")
    if "px" in name or "pixel" in name:
        return "px"
    if "mm" in name or "millimeter" in name or "millimetre" in name:
        return "mm"
    return "mm"


def infer_kind(column: str | None, fallback: str) -> str:
    if fallback != "auto":
        return fallback
    name = normalize_name(column or "")
    if any(token in name for token in ["increment", "incremental", "frame_to_frame", "per_frame", "delta", "step"]):
        return "incremental"
    if any(token in name for token in ["position", "pos", "absolute"]):
        return "absolute"
    return "cumulative"


def convert_displacement_to_mm(values: np.ndarray, *, unit: str, mm_per_px: float, kind: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if unit == "px":
        arr = arr * float(mm_per_px)
    elif unit != "mm":
        raise ValueError("unit must be 'mm', 'px', or 'auto'.")

    if kind == "incremental":
        out = np.cumsum(np.nan_to_num(arr, nan=0.0))
        return out - first_finite(out)
    if kind in {"cumulative", "absolute"}:
        return arr - first_finite(arr)
    raise ValueError("kind must be 'cumulative', 'incremental', 'absolute', or 'auto'.")


def load_ground_truth(args: argparse.Namespace, mm_per_px: float, video_meta: VideoMetadata) -> GroundTruthBundle | None:
    gt_path = resolve_path(args.ground_truth)
    if gt_path is None or not gt_path.exists():
        synthetic = synthetic_linear_ground_truth(args, video_meta)
        if synthetic is not None:
            return synthetic
        return None

    table = read_ground_truth_table(gt_path)
    if table.empty:
        raise ValueError(f"Ground-truth table is empty: {gt_path}")

    columns = list(table.columns)
    frame_col = args.gt_frame_col or find_column(columns, ["frame", "index"], exclude_any=["rate"])
    time_col = args.gt_time_col or find_column(columns, ["time", "seconds", "sec", "timestamp"], exclude_any=["frame"])
    x_col = args.gt_x_col or find_column(
        columns,
        ["x", "lateral", "lat"],
        exclude_any=["max", "index", "pixel_spacing"],
    )
    y_col = args.gt_y_col or find_column(
        columns,
        ["y", "axial", "depth", "vertical"],
        exclude_any=["max", "index", "pixel_spacing"],
    )
    scalar_col = args.gt_scalar_col or find_column(
        columns,
        ["displacement", "disp", "motion", "travel", "position"],
        exclude_any=["x", "y", "lateral", "axial", "depth", "frame", "time"],
    )

    frame = np.asarray(table[frame_col], dtype=int) if frame_col else None
    time_s = np.asarray(table[time_col], dtype=float) if time_col else None
    if frame is None and time_s is None:
        frame = np.arange(len(table), dtype=int)
        time_s = frame / video_meta.fps if video_meta.fps > 0 else None

    unit_source = args.gt_unit
    kind_source = args.gt_kind
    x_mm = None
    y_mm = None
    scalar_mm = None
    if x_col:
        x_mm = convert_displacement_to_mm(
            np.asarray(table[x_col], dtype=float),
            unit=infer_unit(x_col, args.gt_unit),
            mm_per_px=mm_per_px,
            kind=infer_kind(x_col, args.gt_kind),
        )
    if y_col:
        y_mm = convert_displacement_to_mm(
            np.asarray(table[y_col], dtype=float),
            unit=infer_unit(y_col, args.gt_unit),
            mm_per_px=mm_per_px,
            kind=infer_kind(y_col, args.gt_kind),
        )
        if args.gt_y_positive == "up":
            y_mm = -y_mm
    if scalar_col and x_mm is None and y_mm is None:
        scalar_mm = convert_displacement_to_mm(
            np.asarray(table[scalar_col], dtype=float),
            unit=infer_unit(scalar_col, args.gt_unit),
            mm_per_px=mm_per_px,
            kind=infer_kind(scalar_col, args.gt_kind),
        )

    if x_mm is None and y_mm is None and scalar_mm is None:
        raise ValueError(
            "Could not infer a displacement column from ground truth. "
            "Use --gt-x-col, --gt-y-col, or --gt-scalar-col."
        )

    axis_hint = "vector" if x_mm is not None and y_mm is not None else ("x" if x_mm is not None else "y" if y_mm is not None else "scalar")
    return GroundTruthBundle(
        path=gt_path,
        frame=frame,
        time_s=time_s,
        x_mm=x_mm,
        y_mm=y_mm,
        scalar_mm=scalar_mm,
        axis_hint=axis_hint,
        unit_source=unit_source,
        kind_source=kind_source,
        columns_used={
            "frame": frame_col,
            "time_s": time_col,
            "x": x_col,
            "y": y_col,
            "scalar": scalar_col,
        },
    )


def synthetic_linear_ground_truth(args: argparse.Namespace, video_meta: VideoMetadata) -> GroundTruthBundle | None:
    """Build an explicit cumulative linear-ramp ground truth from user-known travel.

    This is useful for phantom trials where the actuator/plate travel is known
    but a per-frame encoder file is not available. It is intentionally labelled
    synthetic so reports do not confuse it with an independently measured trace.
    """

    has_x = args.synthetic_total_x_mm is not None
    has_y = args.synthetic_total_y_mm is not None
    if not has_x and not has_y:
        return None
    if video_meta.frame_count <= 0:
        raise ValueError("Cannot create synthetic ground truth because the video frame count is unavailable.")

    n = int(video_meta.frame_count)
    frame = np.arange(n, dtype=int)
    time_s = frame / float(video_meta.fps) if video_meta.fps > 0 else np.full(n, np.nan)
    start = 0 if args.synthetic_start_frame is None else int(args.synthetic_start_frame)
    end = n - 1 if args.synthetic_end_frame is None else int(args.synthetic_end_frame)
    if end <= start:
        raise ValueError("--synthetic-end-frame must be greater than --synthetic-start-frame.")
    progress = np.clip((frame.astype(float) - float(start)) / float(end - start), 0.0, 1.0)
    x_mm = progress * float(args.synthetic_total_x_mm) if has_x else None
    y_mm = progress * float(args.synthetic_total_y_mm) if has_y else None

    parts = []
    if has_x:
        parts.append(f"x{float(args.synthetic_total_x_mm):g}mm")
    if has_y:
        parts.append(f"y{float(args.synthetic_total_y_mm):g}mm")
    pseudo_path = Path("synthetic_linear_cumulative_" + "_".join(parts))
    axis_hint = "vector" if has_x and has_y else "x" if has_x else "y"
    return GroundTruthBundle(
        path=pseudo_path,
        frame=frame,
        time_s=time_s,
        x_mm=x_mm,
        y_mm=y_mm,
        scalar_mm=None,
        axis_hint=axis_hint,
        unit_source="mm",
        kind_source="synthetic_linear_cumulative",
        columns_used={
            "frame": "video_frame",
            "time_s": "frame/fps",
            "x": "synthetic_total_x_mm" if has_x else None,
            "y": "synthetic_total_y_mm" if has_y else None,
            "scalar": None,
        },
    )


def choose_axis_values(bundle: EstimateBundle, gt: GroundTruthBundle, axis: str) -> tuple[np.ndarray, np.ndarray, str]:
    if axis == "auto":
        if gt.x_mm is not None and gt.y_mm is not None:
            axis = "vector"
        elif gt.y_mm is not None:
            axis = "y"
        elif gt.x_mm is not None:
            axis = "x"
        else:
            axis = "magnitude"

    if axis == "x":
        return bundle.x_mm, gt.x_mm if gt.x_mm is not None else gt.scalar_mm, "x/lateral"
    if axis == "y":
        return bundle.y_mm, gt.y_mm if gt.y_mm is not None else gt.scalar_mm, "y/axial positive-down"
    if axis == "magnitude":
        gt_mag = np.hypot(gt.x_mm, gt.y_mm) if gt.x_mm is not None and gt.y_mm is not None else gt.scalar_mm
        return bundle.magnitude_mm, gt_mag, "magnitude"
    raise ValueError("Scalar axis must be one of x, y, magnitude, or auto.")


def align_by_frame_or_time(
    bundle: EstimateBundle,
    gt: GroundTruthBundle,
    est_values: np.ndarray,
    gt_values: np.ndarray,
    *,
    lag_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if gt.frame is not None:
        gt_frame = np.asarray(gt.frame, dtype=int) + int(lag_frames)
        est_frame = np.asarray(bundle.frame, dtype=int)
        est_map = {int(frame): idx for idx, frame in enumerate(est_frame)}
        est_idx = []
        gt_idx = []
        for idx, frame in enumerate(gt_frame):
            if int(frame) in est_map:
                est_idx.append(est_map[int(frame)])
                gt_idx.append(idx)
        if not est_idx:
            return np.array([]), np.array([]), np.array([]), np.array([])
        est_idx_arr = np.asarray(est_idx, dtype=int)
        gt_idx_arr = np.asarray(gt_idx, dtype=int)
        return (
            bundle.time_s[est_idx_arr],
            np.asarray(est_values, dtype=float)[est_idx_arr],
            np.asarray(gt_values, dtype=float)[gt_idx_arr],
            bundle.failure[est_idx_arr],
        )

    if gt.time_s is not None:
        frame_dt_s = np.nanmedian(np.diff(np.asarray(bundle.time_s, dtype=float)))
        if not np.isfinite(frame_dt_s) or frame_dt_s <= 0:
            frame_dt_s = 1.0
        gt_time = np.asarray(gt.time_s, dtype=float) + float(lag_frames) * float(frame_dt_s)
        valid_gt = np.isfinite(gt_time) & np.isfinite(gt_values)
        if np.sum(valid_gt) < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])
        gt_interp = np.interp(bundle.time_s, gt_time[valid_gt], np.asarray(gt_values, dtype=float)[valid_gt], left=np.nan, right=np.nan)
        return bundle.time_s, np.asarray(est_values, dtype=float), gt_interp, bundle.failure

    n = min(len(est_values), len(gt_values))
    start_est = max(0, lag_frames)
    start_gt = max(0, -lag_frames)
    n = min(len(est_values) - start_est, len(gt_values) - start_gt)
    if n <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    est_slice = slice(start_est, start_est + n)
    gt_slice = slice(start_gt, start_gt + n)
    return bundle.time_s[est_slice], np.asarray(est_values)[est_slice], np.asarray(gt_values)[gt_slice], bundle.failure[est_slice]


def scalar_metrics(est: np.ndarray, gt: np.ndarray, failure: np.ndarray, threshold_mm: float) -> dict[str, float | int]:
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    failure = np.asarray(failure, dtype=bool)
    valid = np.isfinite(est) & np.isfinite(gt)
    error = est - gt
    valid_error = error[valid]
    if valid_error.size == 0:
        return {
            "n": 0,
            "mae_mm": np.nan,
            "rmse_mm": np.nan,
            "bias_mm": np.nan,
            "error_sd_mm": np.nan,
            "max_abs_error_mm": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "endpoint_error_mm": np.nan,
            "drift_slope_mm_per_s": np.nan,
            "failure_rate": 1.0,
        }
    mae = float(np.mean(np.abs(valid_error)))
    rmse = float(np.sqrt(np.mean(valid_error**2)))
    bias = float(np.mean(valid_error))
    sd = float(np.std(valid_error, ddof=1)) if valid_error.size > 1 else 0.0
    max_abs = float(np.max(np.abs(valid_error)))
    gt_valid = gt[valid]
    est_valid = est[valid]
    sst = float(np.sum((gt_valid - np.mean(gt_valid)) ** 2))
    r2 = float(1.0 - np.sum(valid_error**2) / sst) if sst > 1e-12 else np.nan
    pearson = float(np.corrcoef(est_valid, gt_valid)[0, 1]) if valid_error.size > 1 and np.std(est_valid) > 0 and np.std(gt_valid) > 0 else np.nan
    endpoint = float(valid_error[-1])
    threshold_fail = np.abs(error) > float(threshold_mm)
    failure_rate = float(np.mean((~valid) | failure | np.nan_to_num(threshold_fail, nan=True)))
    return {
        "n": int(valid_error.size),
        "mae_mm": mae,
        "rmse_mm": rmse,
        "bias_mm": bias,
        "error_sd_mm": sd,
        "max_abs_error_mm": max_abs,
        "r2": r2,
        "pearson_r": pearson,
        "endpoint_error_mm": endpoint,
        "drift_slope_mm_per_s": np.nan,
        "failure_rate": failure_rate,
    }


def fit_error_drift_slope(time_s: np.ndarray, error: np.ndarray) -> float:
    valid = np.isfinite(time_s) & np.isfinite(error)
    if np.sum(valid) < 3:
        return float("nan")
    centered_t = time_s[valid] - float(time_s[valid][0])
    slope = np.polyfit(centered_t, error[valid], deg=1)[0]
    return float(slope)


def compare_scalar_bundle(
    bundle: EstimateBundle,
    gt: GroundTruthBundle,
    *,
    axis: str,
    threshold_mm: float,
    max_lag_frames: int,
    allow_sign_flip: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    est_values, gt_values, axis_label = choose_axis_values(bundle, gt, axis)
    if gt_values is None:
        raise ValueError(f"Ground truth has no compatible values for axis {axis}.")
    rows = []
    best_payload = None
    best_signs = [-1.0, 1.0] if allow_sign_flip else [1.0]
    for label, lag_values in [("raw", [0]), ("best_aligned", range(-int(max_lag_frames), int(max_lag_frames) + 1))]:
        signs = [1.0] if label == "raw" else best_signs
        best_row = None
        best_series = None
        for lag in lag_values:
            for sign in signs:
                time_s, est_aligned, gt_aligned, failure = align_by_frame_or_time(
                    bundle,
                    gt,
                    est_values,
                    gt_values,
                    lag_frames=int(lag),
                )
                est_signed = sign * est_aligned
                metrics = scalar_metrics(est_signed, gt_aligned, failure, threshold_mm)
                error = est_signed - gt_aligned
                metrics["drift_slope_mm_per_s"] = fit_error_drift_slope(time_s, error)
                row = {
                    "source": bundle.source,
                    "comparison": label,
                    "axis": axis_label,
                    "lag_frames": int(lag),
                    "estimate_sign": float(sign),
                    **metrics,
                }
                if best_row is None or float(row["rmse_mm"]) < float(best_row["rmse_mm"]):
                    best_row = row
                    best_series = pd.DataFrame(
                        {
                            "source": bundle.source,
                            "comparison": label,
                            "time_s": time_s,
                            "estimate_mm": est_signed,
                            "ground_truth_mm": gt_aligned,
                            "error_mm": error,
                            "failure": failure,
                            "axis": axis_label,
                            "lag_frames": int(lag),
                            "estimate_sign": float(sign),
                        }
                    )
        if best_row is not None:
            rows.append(best_row)
            if label == "best_aligned":
                best_payload = best_series
    return pd.DataFrame(rows), best_payload if best_payload is not None else pd.DataFrame()


def vector_metrics(ex: np.ndarray, ey: np.ndarray, gx: np.ndarray, gy: np.ndarray, failure: np.ndarray, threshold_mm: float) -> dict[str, float | int]:
    valid = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(gx) & np.isfinite(gy)
    dx = ex - gx
    dy = ey - gy
    epe = np.hypot(dx, dy)
    finite_epe = epe[valid]
    if finite_epe.size == 0:
        return {"n": 0, "epe_mae_mm": np.nan, "epe_rmse_mm": np.nan, "epe_max_mm": np.nan, "endpoint_epe_mm": np.nan, "failure_rate": 1.0}
    failure_rate = float(np.mean((~valid) | failure | np.nan_to_num(epe > float(threshold_mm), nan=True)))
    return {
        "n": int(finite_epe.size),
        "epe_mae_mm": float(np.mean(finite_epe)),
        "epe_rmse_mm": float(np.sqrt(np.mean(finite_epe**2))),
        "epe_max_mm": float(np.max(finite_epe)),
        "endpoint_epe_mm": float(finite_epe[-1]),
        "failure_rate": failure_rate,
    }


def compare_vector_bundle(
    bundle: EstimateBundle,
    gt: GroundTruthBundle,
    *,
    threshold_mm: float,
    max_lag_frames: int,
    allow_sign_flip: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if gt.x_mm is None or gt.y_mm is None:
        raise ValueError("Vector comparison requires both x and y ground-truth components.")
    best_signs = [-1.0, 1.0] if allow_sign_flip else [1.0]
    rows = []
    best_series = None
    for label, lag_values in [("raw", [0]), ("best_aligned", range(-int(max_lag_frames), int(max_lag_frames) + 1))]:
        signs = [1.0] if label == "raw" else best_signs
        best_row = None
        best_df = None
        for lag in lag_values:
            tx, ex, gx, failure_x = align_by_frame_or_time(bundle, gt, bundle.x_mm, gt.x_mm, lag_frames=int(lag))
            ty, ey, gy, failure_y = align_by_frame_or_time(bundle, gt, bundle.y_mm, gt.y_mm, lag_frames=int(lag))
            n = min(len(tx), len(ty))
            if n == 0:
                continue
            for sx in signs:
                for sy in signs:
                    exs = sx * ex[:n]
                    eys = sy * ey[:n]
                    failure = failure_x[:n] | failure_y[:n]
                    metrics = vector_metrics(exs, eys, gx[:n], gy[:n], failure, threshold_mm)
                    row = {
                        "source": bundle.source,
                        "comparison": label,
                        "axis": "vector",
                        "lag_frames": int(lag),
                        "estimate_sign_x": float(sx),
                        "estimate_sign_y": float(sy),
                        **metrics,
                    }
                    if best_row is None or float(row["epe_rmse_mm"]) < float(best_row["epe_rmse_mm"]):
                        best_row = row
                        best_df = pd.DataFrame(
                            {
                                "source": bundle.source,
                                "comparison": label,
                                "time_s": tx[:n],
                                "estimate_x_mm": exs,
                                "estimate_y_mm": eys,
                                "ground_truth_x_mm": gx[:n],
                                "ground_truth_y_mm": gy[:n],
                                "error_x_mm": exs - gx[:n],
                                "error_y_mm": eys - gy[:n],
                                "epe_mm": np.hypot(exs - gx[:n], eys - gy[:n]),
                                "failure": failure,
                                "lag_frames": int(lag),
                                "estimate_sign_x": float(sx),
                                "estimate_sign_y": float(sy),
                            }
                        )
        if best_row is not None:
            rows.append(best_row)
            if label == "best_aligned":
                best_series = best_df
    return pd.DataFrame(rows), best_series if best_series is not None else pd.DataFrame()


def save_scalar_plots(series: pd.DataFrame, out_dir: Path, source: str) -> list[Path]:
    paths: list[Path] = []
    if series.empty:
        return paths
    safe = normalize_name(source)
    t = series["time_s"].to_numpy(dtype=float)
    est = series["estimate_mm"].to_numpy(dtype=float)
    gt = series["ground_truth_mm"].to_numpy(dtype=float)
    err = series["error_mm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, gt, label="ground truth", linewidth=2.0)
    ax.plot(t, est, label="estimate", linewidth=1.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.set_title(f"{source}: estimated displacement vs ground truth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out_dir / f"{safe}_estimated_vs_ground_truth.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, err, color="tab:red", linewidth=1.3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (mm)")
    ax.set_title(f"{source}: signed error over time")
    ax.grid(True, alpha=0.25)
    path = out_dir / f"{safe}_error_over_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    finite = np.isfinite(est) & np.isfinite(gt)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt[finite], est[finite], s=12, alpha=0.6)
    if np.any(finite):
        lo = float(np.nanmin([gt[finite].min(), est[finite].min()]))
        hi = float(np.nanmax([gt[finite].max(), est[finite].max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("Ground truth (mm)")
    ax.set_ylabel("Estimate (mm)")
    ax.set_title(f"{source}: estimate vs truth")
    ax.grid(True, alpha=0.25)
    path = out_dir / f"{safe}_scatter_identity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    mean_pair = (est + gt) / 2.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean_pair[finite], err[finite], s=12, alpha=0.6)
    bias = np.nanmean(err[finite]) if np.any(finite) else np.nan
    sd = np.nanstd(err[finite], ddof=1) if np.sum(finite) > 1 else np.nan
    ax.axhline(bias, color="tab:red", linewidth=1.2, label="bias")
    if np.isfinite(sd):
        ax.axhline(bias + 1.96 * sd, color="tab:gray", linestyle="--", linewidth=1.0)
        ax.axhline(bias - 1.96 * sd, color="tab:gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean of estimate and truth (mm)")
    ax.set_ylabel("Estimate - truth (mm)")
    ax.set_title(f"{source}: Bland-Altman")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out_dir / f"{safe}_bland_altman.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    drift = err - first_finite(err)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, drift, color="tab:purple", linewidth=1.3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error drift from start (mm)")
    ax.set_title(f"{source}: cumulative drift")
    ax.grid(True, alpha=0.25)
    path = out_dir / f"{safe}_cumulative_drift.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def save_vector_plots(series: pd.DataFrame, out_dir: Path, source: str) -> list[Path]:
    paths: list[Path] = []
    if series.empty:
        return paths
    safe = normalize_name(source)
    t = series["time_s"].to_numpy(dtype=float)
    epe = series["epe_mm"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, epe, color="tab:red", linewidth=1.3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Endpoint/vector error (mm)")
    ax.set_title(f"{source}: vector endpoint error")
    ax.grid(True, alpha=0.25)
    path = out_dir / f"{safe}_vector_endpoint_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(series["ground_truth_x_mm"], series["ground_truth_y_mm"], label="ground truth", linewidth=2.0)
    ax.plot(series["estimate_x_mm"], series["estimate_y_mm"], label="estimate", linewidth=1.5)
    ax.set_xlabel("x/lateral displacement (mm)")
    ax.set_ylabel("y/axial displacement (mm, positive down)")
    ax.set_title(f"{source}: displacement trajectory")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    ax.legend()
    path = out_dir / f"{safe}_vector_trajectory.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def summarize_estimates_without_ground_truth(
    estimates: list[EstimateBundle],
    *,
    expected_axis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize estimated motion when no external phantom ground truth exists."""

    rows: list[dict[str, Any]] = []
    series_rows: list[pd.DataFrame] = []
    expected = "x" if expected_axis in {"auto", "x", "vector"} else expected_axis
    orthogonal = "y" if expected == "x" else "x" if expected == "y" else "orthogonal"
    for bundle in estimates:
        x = np.asarray(bundle.x_mm, dtype=float)
        y = np.asarray(bundle.y_mm, dtype=float)
        mag = np.asarray(bundle.magnitude_mm, dtype=float)
        failure = np.asarray(bundle.failure, dtype=bool)
        finite_xy = np.isfinite(x) & np.isfinite(y)
        rows.append(
            {
                "status": "not_validated",
                "reason": "missing_ground_truth",
                "source": bundle.source,
                "n": int(np.sum(finite_xy)),
                "estimate_axis": expected,
                "orthogonal_axis": orthogonal,
                "x_end_mm": float(x[finite_xy][-1]) if np.any(finite_xy) else np.nan,
                "x_min_mm": float(np.nanmin(x)) if np.any(np.isfinite(x)) else np.nan,
                "x_max_mm": float(np.nanmax(x)) if np.any(np.isfinite(x)) else np.nan,
                "x_range_mm": float(np.nanmax(x) - np.nanmin(x)) if np.any(np.isfinite(x)) else np.nan,
                "y_end_mm": float(y[finite_xy][-1]) if np.any(finite_xy) else np.nan,
                "y_min_mm": float(np.nanmin(y)) if np.any(np.isfinite(y)) else np.nan,
                "y_max_mm": float(np.nanmax(y)) if np.any(np.isfinite(y)) else np.nan,
                "y_range_mm": float(np.nanmax(y) - np.nanmin(y)) if np.any(np.isfinite(y)) else np.nan,
                "magnitude_end_mm": float(mag[finite_xy][-1]) if np.any(finite_xy) else np.nan,
                "magnitude_range_mm": float(np.nanmax(mag) - np.nanmin(mag)) if np.any(np.isfinite(mag)) else np.nan,
                "failure_rate": float(np.mean(failure)) if len(failure) else np.nan,
            }
        )
        series_rows.append(
            pd.DataFrame(
                {
                    "source": bundle.source,
                    "frame": bundle.frame,
                    "time_s": bundle.time_s,
                    "x_lateral_mm": x,
                    "y_axial_mm_positive_down": y,
                    "magnitude_mm": mag,
                    "failure": failure,
                }
            )
        )
    summary = pd.DataFrame(rows)
    series = pd.concat(series_rows, ignore_index=True) if series_rows else pd.DataFrame()
    return summary, series


def save_no_ground_truth_motion_plots(series: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Plot stage-wise estimated motion and orthogonal bounce diagnostics."""

    if series.empty:
        return []
    paths: list[Path] = []
    for column, ylabel, filename, title in [
        (
            "x_lateral_mm",
            "Lateral x displacement (mm)",
            "estimated_lateral_x_by_stage.png",
            "Estimated lateral displacement by tracker stage",
        ),
        (
            "y_axial_mm_positive_down",
            "Axial y displacement (mm, positive down)",
            "estimated_axial_bounce_by_stage.png",
            "Estimated axial bounce by tracker stage",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for source, group in series.groupby("source"):
            ax.plot(group["time_s"], group[column], linewidth=1.4, label=str(source))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 5))
    for source, group in series.groupby("source"):
        ax.plot(group["x_lateral_mm"], group["y_axial_mm_positive_down"], linewidth=1.4, label=str(source))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Lateral x displacement (mm)")
    ax.set_ylabel("Axial y displacement (mm, positive down)")
    ax.set_title("Estimated x/y trajectory by tracker stage")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    path = out_dir / "estimated_xy_trajectory_by_stage.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_ground_truth_template(path: Path, video_meta: VideoMetadata) -> None:
    n = max(video_meta.frame_count, 1)
    frame = np.arange(n, dtype=int)
    time_s = frame / video_meta.fps if video_meta.fps > 0 else np.full(n, np.nan)
    template = pd.DataFrame(
        {
            "frame": frame,
            "time_s": time_s,
            "gt_x_lateral_mm_positive_right": np.nan,
            "gt_y_axial_mm_positive_down": np.nan,
            "notes": "",
        }
    )
    template.to_csv(path, index=False)


def format_metric_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_No metrics computed._"
    keep = [
        col
        for col in [
            "status",
            "reason",
            "source",
            "comparison",
            "axis",
            "estimate_axis",
            "orthogonal_axis",
            "lag_frames",
            "estimate_sign",
            "estimate_sign_x",
            "estimate_sign_y",
            "n",
            "x_end_mm",
            "x_range_mm",
            "y_end_mm",
            "y_range_mm",
            "magnitude_end_mm",
            "magnitude_range_mm",
            "mae_mm",
            "rmse_mm",
            "bias_mm",
            "error_sd_mm",
            "max_abs_error_mm",
            "r2",
            "pearson_r",
            "epe_mae_mm",
            "epe_rmse_mm",
            "endpoint_error_mm",
            "endpoint_epe_mm",
            "drift_slope_mm_per_s",
            "failure_rate",
        ]
        if col in summary.columns
    ]
    table = summary[keep].copy()

    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(float(value)) else f"{float(value):.4g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if value is None or (isinstance(value, str) and value == ""):
            return ""
        return str(value)

    header = "| " + " | ".join(keep) + " |"
    divider = "| " + " | ".join(["---"] * len(keep)) + " |"
    rows = ["| " + " | ".join(fmt(value) for value in row) + " |" for row in table.to_numpy(dtype=object)]
    return "\n".join([header, divider, *rows])


def write_report(
    path: Path,
    *,
    video_meta: VideoMetadata,
    strict_audit: Mapping[str, Any] | None,
    gt: GroundTruthBundle | None,
    summary: pd.DataFrame,
    motion_audit: pd.DataFrame | None,
    plots: list[Path],
    audit: Mapping[str, Any],
    threshold_mm: float,
) -> None:
    has_metric_rows = not summary.empty and "status" not in summary.columns
    has_no_gt_audit = not summary.empty and "status" in summary.columns and "source" in summary.columns
    lines: list[str] = []
    lines.append("# Phantom Ground-Truth Validation")
    lines.append("")
    lines.append("## Data Audit")
    lines.append(f"- Video: `{video_meta.path}`")
    lines.append(f"- Video opened: `{video_meta.opened}`")
    lines.append(f"- Frames/FPS/size: `{video_meta.frame_count}` frames, `{video_meta.fps:g}` fps, `{video_meta.width_px}x{video_meta.height_px}` px")
    if strict_audit:
        lines.append(f"- Strict runner NPZ: `{strict_audit.get('strict_npz')}`")
        lines.append(f"- Estimate sources: `{', '.join(strict_audit.get('estimate_sources', []))}`")
        lines.append(f"- mm_per_pixel: `{strict_audit.get('mm_per_pixel')}`")
    else:
        lines.append("- Strict runner NPZ: **not found** for this video.")
    if gt:
        lines.append(f"- Ground truth: `{gt.path}`")
        lines.append(f"- GT columns used: `{gt.columns_used}`")
        lines.append(f"- GT axis type: `{gt.axis_hint}`")
        if gt.kind_source == "synthetic_linear_cumulative":
            lines.append("- GT source note: synthetic linear cumulative ramp from the supplied total displacement; replace with an actuator/encoder trace if the plate motion was not constant-speed over the full video.")
    else:
        lines.append("- Ground truth: **not found/provided**.")
    lines.append(f"- Failure threshold: `{threshold_mm:g}` mm")
    lines.append("")

    lines.append("## Coordinate And Unit Contract")
    lines.append("- Image origin is the top-left pixel. x increases to the right; y increases downward.")
    lines.append("- Strict line segments are stored in one-based MATLAB-style pixel coordinates, but displacement is computed relative to frame 0, so the one-pixel origin offset cancels.")
    lines.append("- Axial displacement corresponds to image y positive downward. Lateral displacement corresponds to image x positive rightward.")
    lines.append("- The current scalar `mm_per_pixel` is depth/height-derived axial spacing. Vector or lateral validation needs independent lateral spacing unless square pixels are confirmed.")
    lines.append("- `fascicle_segments` are final Kalman output; `klt_prior_segments` are cumulative/persistent KLT prior segments; `fixed_fascicle_segments` is the normal fixed-R Kalman comparator when present.")
    lines.append("")

    if gt is None:
        lines.append("## Validation Status")
        lines.append("No phantom ground-truth displacement file was available in the workspace for this run. Therefore this output is an audit and scaffold, not evidence that the tracker is accurate.")
        lines.append("A template CSV was written beside this report so the imposed phantom displacement can be added with explicit frame/time and axis convention.")
        lines.append("")
        if has_no_gt_audit:
            lines.append("## Estimated Motion Audit Without Ground Truth")
            lines.append(format_metric_table(summary))
            lines.append("")
            lines.append("For this phantom trial the expected imposed motion is lateral/x. The y range is therefore an orthogonal-motion/bounce diagnostic, not a validation target unless the plate motion had an axial component.")
            lines.append("")
    elif strict_audit is None:
        lines.append("## Validation Status")
        lines.append("Ground truth was available, but no strict runner output was found for this video. Run the tracker first or pass `--strict-npz`.")
        lines.append("")
    else:
        lines.append("## Metrics")
        lines.append(format_metric_table(summary))
        lines.append("")
        lines.append("Interpret the `raw` rows as the mathematically direct comparison. The `best_aligned` rows are diagnostic: a non-zero lag or sign flip may reveal a frame offset or coordinate convention error and should not be hidden in a paper result.")
        lines.append("")
        if motion_audit is not None and not motion_audit.empty:
            lines.append("## Orthogonal Motion / Bounce Audit")
            lines.append(format_metric_table(motion_audit))
            lines.append("")
            lines.append("This table is not an additional ground-truth validation. For the lateral phantom trial, it checks whether the tracker reports axial y motion even though the imposed motion was x-only.")
            lines.append("")

    lines.append("## Reviewer Interpretation")
    if not has_metric_rows:
        lines.append("- Accuracy cannot be concluded yet. Missing ground truth and/or missing tracker output prevents a numerical validation.")
    else:
        best_rows = summary[summary["comparison"] == "raw"] if "comparison" in summary else summary
        scalar_rmse = best_rows["rmse_mm"].dropna() if "rmse_mm" in best_rows else pd.Series(dtype=float)
        vector_rmse = best_rows["epe_rmse_mm"].dropna() if "epe_rmse_mm" in best_rows else pd.Series(dtype=float)
        rmse_value = float(scalar_rmse.min()) if len(scalar_rmse) else float(vector_rmse.min()) if len(vector_rmse) else np.nan
        if np.isfinite(rmse_value) and rmse_value <= threshold_mm:
            lines.append(f"- Raw RMSE is at or below the configured `{threshold_mm:g}` mm threshold for at least one source.")
        elif np.isfinite(rmse_value):
            lines.append(f"- Raw RMSE exceeds the configured `{threshold_mm:g}` mm threshold; this is not yet strong phantom evidence.")
        lines.append("- Check whether any best-aligned result requires sign flip or lag. If yes, fix/report the coordinate or synchronization cause before claiming validation.")
        lines.append("- For strain, small displacement bias can be amplified because strain divides a length change by a baseline length; validate displacement and length first.")
    lines.append("")

    if plots:
        lines.append("## Plots")
        for plot in plots:
            lines.append(f"- `{plot}`")
        lines.append("")

    lines.append("## Paper-Strength Validation Target")
    lines.append("- Pre-register the phantom displacement waveform, units, axis convention, frame synchronization, and tolerance.")
    lines.append("- Report raw, not only best-aligned, MAE/RMSE/bias/limits-of-agreement over multiple amplitudes and speeds.")
    lines.append("- Include failure rate and show overlays at worst-error frames.")
    lines.append("- Demonstrate low drift over long sequences and compare against at least KLT-only and fixed-R Kalman baselines.")
    lines.append("- Phantom validation does not replace in vivo validation because tissue deformation, out-of-plane motion, probe pressure, anisotropic speckle decorrelation, and manual ROI variability are different failure modes.")
    lines.append("")
    lines.append("## Machine-Readable Audit")
    lines.append("```json")
    lines.append(json.dumps(json_safe(audit), indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_strict_runner(args: argparse.Namespace, out_root: Path) -> dict[str, Path | None]:
    video = resolve_path(args.video)
    if video is None:
        raise ValueError("--video is required to run strict runner.")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_strict_ultratimtrack_video.py"),
        str(video),
        "--results-dir",
        str(out_root),
        "--no-annotated-video",
        "--save-overlays",
        "0",
    ]
    if args.roi_path:
        cmd.extend(["--roi-path", str(resolve_path(args.roi_path))])
    if args.limit:
        cmd.extend(["--limit", str(int(args.limit))])
    if args.strict_extra_args:
        cmd.extend(args.strict_extra_args)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return find_strict_outputs(video.stem, out_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate phantom tracking displacement against ground truth.")
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "data" / "raw" / "june29_3.mp4")
    parser.add_argument("--ground-truth", "--gt", dest="ground_truth", type=Path, default=None)
    parser.add_argument("--strict-npz", type=Path, default=None, help="Existing strict runner NPZ. Auto-detected if omitted.")
    parser.add_argument("--strict-metadata", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "strict_ultratimtrack_runs")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "phantom_ground_truth_validation" / "june29_3")
    parser.add_argument("--roi-path", type=Path, default=None, help="ROI path only used with --run-strict-if-missing.")
    parser.add_argument("--run-strict-if-missing", action="store_true", help="Run the strict tracker if no NPZ is available.")
    parser.add_argument("--limit", type=int, default=None, help="Frame limit passed only to --run-strict-if-missing.")
    parser.add_argument("--strict-extra-args", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--mm-per-pixel", type=float, default=None, help="Override mm/px for GT px conversion and audit.")
    parser.add_argument("--gt-frame-col", default=None)
    parser.add_argument("--gt-time-col", default=None)
    parser.add_argument("--gt-x-col", default=None)
    parser.add_argument("--gt-y-col", default=None)
    parser.add_argument("--gt-scalar-col", default=None)
    parser.add_argument("--gt-unit", choices=["auto", "mm", "px"], default="auto")
    parser.add_argument("--gt-kind", choices=["auto", "cumulative", "incremental", "absolute"], default="auto")
    parser.add_argument("--gt-y-positive", choices=["down", "up"], default="down")
    parser.add_argument("--synthetic-total-x-mm", type=float, default=None, help="Use a synthetic cumulative linear x/lateral ground truth ending at this total displacement.")
    parser.add_argument("--synthetic-total-y-mm", type=float, default=None, help="Use a synthetic cumulative linear y/axial-positive-down ground truth ending at this total displacement.")
    parser.add_argument("--synthetic-start-frame", type=int, default=None, help="First frame of the synthetic ramp. Defaults to the first video frame.")
    parser.add_argument("--synthetic-end-frame", type=int, default=None, help="Last frame of the synthetic ramp. Defaults to the last video frame.")
    parser.add_argument("--axis", choices=["auto", "x", "y", "magnitude", "vector"], default="auto")
    parser.add_argument("--max-lag-frames", type=int, default=10)
    parser.add_argument("--no-auto-sign", action="store_true", help="Do not test sign flips in best-aligned diagnostics.")
    parser.add_argument("--failure-threshold-mm", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = resolve_path(args.video)
    if video is None:
        raise ValueError("--video is required.")
    out_dir = resolve_path(args.out_dir)
    results_root = resolve_path(args.results_root)
    if out_dir is None or results_root is None:
        raise ValueError("Invalid output paths.")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    video_meta = read_video_metadata(video)
    found = {
        "npz": resolve_path(args.strict_npz),
        "metadata": resolve_path(args.strict_metadata),
        "csv": None,
    }
    if found["npz"] is None:
        found = find_strict_outputs(video.stem, results_root)
    if found["npz"] is None and args.run_strict_if_missing:
        strict_root = out_dir / "strict_runner_output"
        found = run_strict_runner(args, strict_root)

    strict_audit = None
    estimates: list[EstimateBundle] = []
    if found["npz"] is not None and Path(found["npz"]).exists():
        estimates, strict_audit = estimate_bundles_from_strict(Path(found["npz"]), found.get("metadata"), video_meta)
        if args.mm_per_pixel is not None:
            strict_audit["mm_per_pixel_override_note"] = (
                "Override was supplied but existing estimates already store mm displacement from their own mm_per_pixel. "
                "Re-run strict tracker or regenerate estimates if you need a different scale."
            )

    mm_per_px = (
        float(args.mm_per_pixel)
        if args.mm_per_pixel is not None
        else float(strict_audit["mm_per_pixel"])
        if strict_audit and np.isfinite(float(strict_audit["mm_per_pixel"]))
        else float("nan")
    )
    if not np.isfinite(mm_per_px) and video_meta.height_px > 0:
        mm_per_px = float("nan")

    gt = load_ground_truth(args, mm_per_px, video_meta)

    audit = {
        "video": video_meta.__dict__,
        "strict_outputs": found,
        "strict_audit": strict_audit,
        "ground_truth_path": str(gt.path) if gt is not None else str(resolve_path(args.ground_truth)) if args.ground_truth else None,
        "ground_truth_kind": gt.kind_source if gt is not None else None,
        "synthetic_ground_truth": {
            "total_x_mm": args.synthetic_total_x_mm,
            "total_y_mm": args.synthetic_total_y_mm,
            "start_frame": args.synthetic_start_frame,
            "end_frame": args.synthetic_end_frame,
        },
        "data_phantom_files": [str(p) for p in sorted((PROJECT_ROOT / "data" / "phantom").glob("**/*")) if p.is_file()],
        "ground_truth_loaded": gt is not None,
        "axis_requested": args.axis,
        "failure_threshold_mm": args.failure_threshold_mm,
    }
    (out_dir / "source_audit.json").write_text(json.dumps(json_safe(audit), indent=2), encoding="utf-8")

    all_summary: list[pd.DataFrame] = []
    all_series: list[pd.DataFrame] = []
    plots: list[Path] = []
    motion_audit_df = pd.DataFrame()
    motion_series_df = pd.DataFrame()
    if estimates:
        motion_audit_df, motion_series_df = summarize_estimates_without_ground_truth(estimates, expected_axis=args.axis)
        motion_audit_df.to_csv(out_dir / "estimated_motion_audit.csv", index=False)
        motion_series_df.to_csv(out_dir / "estimated_motion_per_frame.csv", index=False)
        if not motion_series_df.empty:
            plots.extend(save_no_ground_truth_motion_plots(motion_series_df, plot_dir))
    if gt is not None and estimates:
        for bundle in estimates:
            if args.axis == "vector" or (args.axis == "auto" and gt.x_mm is not None and gt.y_mm is not None):
                summary, series = compare_vector_bundle(
                    bundle,
                    gt,
                    threshold_mm=float(args.failure_threshold_mm),
                    max_lag_frames=int(args.max_lag_frames),
                    allow_sign_flip=not bool(args.no_auto_sign),
                )
                plots.extend(save_vector_plots(series, plot_dir, bundle.source))
            else:
                summary, series = compare_scalar_bundle(
                    bundle,
                    gt,
                    axis=args.axis,
                    threshold_mm=float(args.failure_threshold_mm),
                    max_lag_frames=int(args.max_lag_frames),
                    allow_sign_flip=not bool(args.no_auto_sign),
                )
                plots.extend(save_scalar_plots(series, plot_dir, bundle.source))
            all_summary.append(summary)
            if not series.empty:
                all_series.append(series)
    elif gt is None and estimates:
        all_summary.append(motion_audit_df)
        if not motion_series_df.empty:
            all_series.append(motion_series_df)

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    per_frame_df = pd.concat(all_series, ignore_index=True) if all_series else pd.DataFrame()
    if summary_df.empty:
        if gt is None:
            reason = "missing_ground_truth"
        elif not estimates:
            reason = "missing_tracker_output"
        else:
            reason = "no_comparable_series"
        summary_df = pd.DataFrame(
            [
                {
                    "status": "not_validated",
                    "reason": reason,
                    "video": str(video),
                    "ground_truth_loaded": bool(gt is not None),
                    "strict_output_loaded": bool(estimates),
                }
            ]
        )
    if per_frame_df.empty:
        per_frame_df = pd.DataFrame(columns=["source", "comparison", "time_s"])
    summary_csv = out_dir / "phantom_validation_summary.csv"
    per_frame_csv = out_dir / "phantom_validation_per_frame.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_frame_df.to_csv(per_frame_csv, index=False)

    if gt is None:
        write_ground_truth_template(out_dir / "ground_truth_template.csv", video_meta)

    report_path = out_dir / "phantom_validation_report.md"
    write_report(
        report_path,
        video_meta=video_meta,
        strict_audit=strict_audit,
        gt=gt,
        summary=summary_df,
        motion_audit=motion_audit_df,
        plots=plots,
        audit=audit,
        threshold_mm=float(args.failure_threshold_mm),
    )

    print(f"Report: {report_path}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-frame CSV: {per_frame_csv}")
    if gt is None:
        print(f"Ground-truth template: {out_dir / 'ground_truth_template.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
