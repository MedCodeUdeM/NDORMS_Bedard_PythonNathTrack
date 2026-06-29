"""MATLAB-style UltraTrack KLT affine helpers.

These helpers package the OpenCV compatibility code that was validated in the
KLT parity notebooks.  The important boundary is local affine motion: applying
small frame-to-frame affines directly to a long-running raw fascicle segment
compounds drift, while handing the local transition into the downstream Kalman
gate keeps the validated one-step behavior available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class UltraTrackKLTConfig:
    """OpenCV settings matching the current MATLAB KLT parity prototype."""

    max_fascicle_corners: int = 300
    max_aponeurosis_corners: int = 0
    quality_level: float = 0.005
    min_distance: int = 1
    block_size: int = 11
    lk_win_size: Tuple[int, int] = (81, 81)
    lk_max_level: int = 3
    lk_max_iter: int = 50
    lk_epsilon: float = 0.01
    ransac_reproj_threshold: float = 50.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.99
    ransac_refine_iters: int = 10
    min_fascicle_points: int = 100
    min_aponeurosis_points: int = 500


def poly_mask_1b(x_values: Sequence[float], y_values: Sequence[float], shape: Tuple[int, int]) -> np.ndarray:
    """Approximate MATLAB ``poly2mask`` for one-based polygon coordinates."""

    x = np.asarray(x_values, dtype=np.float64).reshape(-1) - 1.0
    y = np.asarray(y_values, dtype=np.float64).reshape(-1) - 1.0
    points = np.rint(np.column_stack([x, y])).astype(np.int32)
    mask = np.zeros(tuple(map(int, shape)), dtype=np.uint8)
    if len(points):
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def tracking_masks_from_geofeature(
    entry: Mapping,
    *,
    shape: Tuple[int, int],
    super_cut: Sequence[float],
    deep_cut: Optional[Sequence[float]] = None,
) -> dict[str, np.ndarray]:
    """Build MATLAB-style UltraTrack masks from one saved geofeature entry."""

    height, width = int(shape[0]), int(shape[1])
    line_mask = np.zeros((height, width), dtype=bool)
    xs = np.asarray(entry["x"], dtype=np.float64)
    ys = np.asarray(entry["y"], dtype=np.float64)

    for (x1, x2), (y1, y2) in zip(xs, ys):
        px = [x1, x1, x2, x2]
        py = [y1 - 5.0, y1 + 5.0, y2 + 5.0, y2 - 5.0]
        if np.all(np.isfinite(px)) and np.all(np.isfinite(py)):
            line_mask |= poly_mask_1b(px, py, shape=(height, width))

    super_pos = np.asarray(entry["super_pos"], dtype=np.float64).reshape(-1)
    deep_pos = np.asarray(entry["deep_pos"], dtype=np.float64).reshape(-1)
    thickness = deep_pos - super_pos
    r = 0.1
    roix = [1.0, 1.0, float(width), float(width), 1.0]
    roiy_fcor = np.rint(
        [
            super_pos[0] + thickness[0] * r,
            deep_pos[0] - thickness[0] * r,
            deep_pos[1] - thickness[1] * r,
            super_pos[1] + thickness[1] * r,
            super_pos[0] + thickness[0] * r,
        ]
    )
    fcor_mask = poly_mask_1b(roix, roiy_fcor, shape=(height, width))

    out = {
        "line_mask": line_mask,
        "fcor_mask": fcor_mask,
        "fascicle_mask": line_mask & fcor_mask,
    }

    if deep_cut is not None:
        super_cut_arr = np.asarray(super_cut, dtype=np.float64).reshape(-1)
        deep_cut_arr = np.asarray(deep_cut, dtype=np.float64).reshape(-1)
        super_y = [
            super_cut_arr[0] * height,
            super_cut_arr[1] * height,
            super_cut_arr[1] * height,
            super_cut_arr[0] * height,
            super_cut_arr[0] * height,
        ]
        deep_y = [
            deep_cut_arr[0] * height,
            deep_cut_arr[1] * height,
            deep_cut_arr[1] * height,
            deep_cut_arr[0] * height,
            deep_cut_arr[0] * height,
        ]
        super_mask = poly_mask_1b(roix, super_y, shape=(height, width))
        deep_mask = poly_mask_1b(roix, deep_y, shape=(height, width))
        out.update({"super_mask": super_mask, "deep_mask": deep_mask, "apo_mask": super_mask | deep_mask})

    return out


def masked_image(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``gray`` with pixels outside ``mask`` set to zero."""

    out = np.asarray(gray).copy()
    out[~np.asarray(mask, dtype=bool)] = 0
    return out


def detect_min_eigen_like(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    max_corners: int,
    config: UltraTrackKLTConfig | None = None,
) -> np.ndarray:
    """OpenCV approximation of MATLAB ``detectMinEigenFeatures``."""

    cfg = config or UltraTrackKLTConfig()
    points = cv2.goodFeaturesToTrack(
        masked_image(gray, mask),
        maxCorners=int(max_corners),
        qualityLevel=float(cfg.quality_level),
        minDistance=int(cfg.min_distance),
        blockSize=int(cfg.block_size),
        useHarrisDetector=False,
    )
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return points.astype(np.float32)


def filter_points_by_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep tracked points whose rounded location remains inside ``mask``."""

    points_2d = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points_2d) == 0:
        return points_2d.reshape(0, 1, 2)
    mask_arr = np.asarray(mask, dtype=bool)
    height, width = mask_arr.shape
    xy = np.rint(points_2d).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    keep = mask_arr[xy[:, 1], xy[:, 0]]
    return points_2d[keep].reshape(-1, 1, 2).astype(np.float32)


def track_points(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    points: np.ndarray,
    *,
    config: UltraTrackKLTConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track points one frame forward with OpenCV PyrLK."""

    cfg = config or UltraTrackKLTConfig()
    points_in = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points_in) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        points_in,
        None,
        winSize=tuple(map(int, cfg.lk_win_size)),
        maxLevel=int(cfg.lk_max_level),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(cfg.lk_max_iter),
            float(cfg.lk_epsilon),
        ),
    )
    if new_points is None or status is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    keep = status.reshape(-1).astype(bool)
    return points_in.reshape(-1, 2)[keep], new_points.reshape(-1, 2)[keep]


def estimate_affine_matlab_coords(
    old_points_0b: np.ndarray,
    new_points_0b: np.ndarray,
    *,
    config: UltraTrackKLTConfig | None = None,
) -> tuple[np.ndarray | None, int]:
    """Estimate a full affine transform in MATLAB-style one-based coordinates."""

    cfg = config or UltraTrackKLTConfig()
    old_points = np.asarray(old_points_0b, dtype=np.float32).reshape(-1, 2)
    new_points = np.asarray(new_points_0b, dtype=np.float32).reshape(-1, 2)
    if len(old_points) < 3 or len(new_points) < 3:
        return None, 0

    affine, inliers = cv2.estimateAffine2D(
        old_points + 1.0,
        new_points + 1.0,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.ransac_reproj_threshold),
        maxIters=int(cfg.ransac_max_iters),
        confidence=float(cfg.ransac_confidence),
        refineIters=int(cfg.ransac_refine_iters),
    )
    if affine is None:
        return None, 0
    n_inliers = int(np.asarray(inliers).sum()) if inliers is not None else len(old_points)
    return affine.astype(np.float32), n_inliers


def apply_affine_1b(segment_or_points_1b: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply an affine matrix to one-based ``(x, y)`` points or flat segments."""

    arr = np.asarray(segment_or_points_1b, dtype=np.float32)
    original_shape = arr.shape
    points = arr.reshape(-1, 2)
    transformed = cv2.transform(points[None, :, :], np.asarray(affine, dtype=np.float32))[0]
    return transformed.reshape(original_shape).astype(np.float64)


def propagate_cumulative_affines(
    initial_segment_1b: np.ndarray,
    affine_matrices: np.ndarray,
    *,
    fallback: str = "previous",
) -> np.ndarray:
    """
    Propagate one fascicle seed through a sequence of one-based affine matrices.

    This is the compounding/raw KLT path: frame ``i`` applies affine ``i`` to
    the segment from frame ``i - 1``.  ``affine_matrices[0]`` is ignored so the
    array can be passed directly from :func:`run_one_step_affine_video`.
    """

    affines = np.asarray(affine_matrices, dtype=np.float64)
    if affines.ndim != 3 or affines.shape[1:] != (2, 3):
        raise ValueError("affine_matrices must have shape (n_frames, 2, 3).")
    if fallback not in {"previous", "nan", "raise"}:
        raise ValueError("fallback must be 'previous', 'nan', or 'raise'.")

    initial = np.asarray(initial_segment_1b, dtype=np.float64).reshape(4)
    n = len(affines)
    out = np.full((n, 4), np.nan, dtype=np.float64)
    if n == 0:
        return out
    out[0] = initial

    for frame in range(1, n):
        affine = affines[frame]
        can_apply = np.all(np.isfinite(affine)) and np.all(np.isfinite(out[frame - 1]))
        if can_apply:
            out[frame] = apply_affine_1b(out[frame - 1], affine)
        elif fallback == "previous":
            out[frame] = out[frame - 1]
        elif fallback == "raise":
            raise ValueError(f"Missing finite affine or prior segment at frame {frame}.")

    return out


def read_gray_frames(video_path: str | Path, *, limit: Optional[int] = None) -> list[np.ndarray]:
    """Read grayscale frames from a video file."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frames = []
    while limit is None or len(frames) < int(limit):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy())
    cap.release()
    return frames


def run_one_step_affine_sequence(
    frames: Sequence[np.ndarray],
    geofeatures: Sequence[Mapping],
    reference_fascicle_segments_1b: np.ndarray,
    *,
    super_cut: Sequence[float],
    config: UltraTrackKLTConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    Run the validated non-compounding fascicle KLT handoff.

    Each affine is estimated from frame ``i - 1`` to ``i`` and applied to the
    provided reference segment at ``i - 1``.  This returns the local transition
    prior used to avoid cumulative raw KLT drift in the downstream gate.
    """

    cfg = config or UltraTrackKLTConfig()
    n = min(len(frames), len(geofeatures), len(reference_fascicle_segments_1b))
    if n == 0:
        raise ValueError("frames, geofeatures, and reference segments must be non-empty.")

    shape = np.asarray(frames[0]).shape[:2]
    out = np.full((n, 4), np.nan, dtype=np.float32)
    out[0] = np.asarray(reference_fascicle_segments_1b[0], dtype=np.float32)
    f_points_count = np.zeros(n, dtype=np.int32)
    f_tracked_count = np.zeros(n, dtype=np.int32)
    f_inlier_count = np.zeros(n, dtype=np.int32)
    f_affine_ok = np.zeros(n, dtype=bool)
    f_affine_matrices = np.full((n, 2, 3), np.nan, dtype=np.float32)

    for frame in range(1, n):
        masks = tracking_masks_from_geofeature(
            geofeatures[frame - 1],
            shape=shape,
            super_cut=super_cut,
        )
        points = detect_min_eigen_like(
            frames[frame - 1],
            masks["fascicle_mask"],
            max_corners=cfg.max_fascicle_corners,
            config=cfg,
        )
        points = filter_points_by_mask(points, masks["fcor_mask"])
        old_points, new_points = track_points(frames[frame - 1], frames[frame], points, config=cfg)
        affine, inliers = estimate_affine_matlab_coords(old_points, new_points, config=cfg)

        f_points_count[frame] = len(points)
        f_tracked_count[frame] = len(new_points)
        f_inlier_count[frame] = int(inliers)
        if affine is not None:
            out[frame] = apply_affine_1b(reference_fascicle_segments_1b[frame - 1], affine)
            f_affine_ok[frame] = True
            f_affine_matrices[frame] = affine

    return {
        "fascicle_segments": out,
        "f_points_count": f_points_count,
        "f_tracked_count": f_tracked_count,
        "f_inlier_count": f_inlier_count,
        "f_affine_ok": f_affine_ok,
        "f_affine_matrices": f_affine_matrices,
    }


def run_one_step_affine_video(
    video_path: str | Path,
    geofeatures: Sequence[Mapping],
    reference_fascicle_segments_1b: np.ndarray,
    *,
    super_cut: Sequence[float],
    config: UltraTrackKLTConfig | None = None,
    limit: Optional[int] = None,
    progress_every: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Streaming version of :func:`run_one_step_affine_sequence` for videos."""

    cfg = config or UltraTrackKLTConfig()
    n = min(len(geofeatures), len(reference_fascicle_segments_1b))
    if limit is not None:
        n = min(n, int(limit))
    if n <= 0:
        raise ValueError("geofeatures and reference segments must be non-empty.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read first frame from {video_path}")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame.copy()
    shape = prev_gray.shape[:2]

    out = np.full((n, 4), np.nan, dtype=np.float32)
    out[0] = np.asarray(reference_fascicle_segments_1b[0], dtype=np.float32)
    f_points_count = np.zeros(n, dtype=np.int32)
    f_tracked_count = np.zeros(n, dtype=np.int32)
    f_inlier_count = np.zeros(n, dtype=np.int32)
    f_affine_ok = np.zeros(n, dtype=bool)
    f_affine_matrices = np.full((n, 2, 3), np.nan, dtype=np.float32)

    for frame in range(1, n):
        ok, current_frame = cap.read()
        if not ok:
            out = out[:frame]
            f_points_count = f_points_count[:frame]
            f_tracked_count = f_tracked_count[:frame]
            f_inlier_count = f_inlier_count[:frame]
            f_affine_ok = f_affine_ok[:frame]
            f_affine_matrices = f_affine_matrices[:frame]
            break

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if current_frame.ndim == 3 else current_frame.copy()
        masks = tracking_masks_from_geofeature(
            geofeatures[frame - 1],
            shape=shape,
            super_cut=super_cut,
        )
        points = detect_min_eigen_like(
            prev_gray,
            masks["fascicle_mask"],
            max_corners=cfg.max_fascicle_corners,
            config=cfg,
        )
        points = filter_points_by_mask(points, masks["fcor_mask"])
        old_points, new_points = track_points(prev_gray, gray, points, config=cfg)
        affine, inliers = estimate_affine_matlab_coords(old_points, new_points, config=cfg)

        f_points_count[frame] = len(points)
        f_tracked_count[frame] = len(new_points)
        f_inlier_count[frame] = int(inliers)
        if affine is not None:
            out[frame] = apply_affine_1b(reference_fascicle_segments_1b[frame - 1], affine)
            f_affine_ok[frame] = True
            f_affine_matrices[frame] = affine

        prev_gray = gray
        if progress_every and (frame % int(progress_every) == 0 or frame == n - 1):
            print(f"one-step KLT processed {frame + 1}/{n}")

    cap.release()
    return {
        "fascicle_segments": out,
        "f_points_count": f_points_count,
        "f_tracked_count": f_tracked_count,
        "f_inlier_count": f_inlier_count,
        "f_affine_ok": f_affine_ok,
        "f_affine_matrices": f_affine_matrices,
    }
