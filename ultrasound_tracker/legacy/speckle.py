from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import correlate2d


ROI = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SpeckleCoherenceParams:
    """
    Parameters for block-matching speckle coherence.

    The confidence score combines:
        median NCC,
        fraction of valid blocks,
        coherence of the local displacement field.
    """

    block_size: int = 21
    stride: int = 24
    search_radius: int = 10
    rho_min: float = 0.55
    max_displacement_px: float = 14.0
    sigma_ref_px: float = 2.0
    confidence_min: float = 0.05


def robust_mad_sigma(values: np.ndarray) -> float:
    """Robust sigma estimate from median absolute deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")

    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def speckle_pair_confidence(
    prev_roi: np.ndarray,
    curr_roi: np.ndarray,
    params: Optional[SpeckleCoherenceParams] = None,
) -> Dict[str, float]:
    """
    Compute a speckle-coherence confidence score for one ROI frame pair.

    Returns a dictionary with the global confidence and its components:
    ``c_ncc``, ``c_valid``, ``c_motion``, motion dispersion, median
    displacement, and block counts.
    """
    params = params or SpeckleCoherenceParams()
    prev = np.asarray(prev_roi, dtype=np.float32)
    curr = np.asarray(curr_roi, dtype=np.float32)

    if prev.shape != curr.shape:
        raise ValueError("prev_roi and curr_roi must have the same shape.")
    if prev.ndim != 2:
        raise ValueError("prev_roi and curr_roi must be 2D grayscale images.")
    if params.block_size <= 0 or params.block_size % 2 == 0:
        raise ValueError("block_size must be a positive odd integer.")
    if params.stride <= 0:
        raise ValueError("stride must be positive.")
    if params.search_radius < 0:
        raise ValueError("search_radius must be non-negative.")

    h, w = prev.shape
    half = params.block_size // 2
    search_radius = int(params.search_radius)

    displacements = []
    ncc_scores = []
    total_blocks = 0

    y_values = range(half + search_radius, h - half - search_radius, params.stride)
    x_values = range(half + search_radius, w - half - search_radius, params.stride)

    for y in y_values:
        for x in x_values:
            total_blocks += 1

            template = prev[
                y - half : y + half + 1,
                x - half : x + half + 1,
            ]
            if template.std() < 1e-6:
                continue

            search = curr[
                y - half - search_radius : y + half + search_radius + 1,
                x - half - search_radius : x + half + search_radius + 1,
            ]
            if search.shape[0] < params.block_size or search.shape[1] < params.block_size:
                continue

            corr = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, best_ncc, _, best_loc = cv2.minMaxLoc(corr)
            dx = best_loc[0] - search_radius
            dy = best_loc[1] - search_radius
            displacement_norm = float(np.hypot(dx, dy))

            if best_ncc >= params.rho_min and displacement_norm <= params.max_displacement_px:
                displacements.append((dx, dy))
                ncc_scores.append(float(best_ncc))

    if total_blocks == 0 or len(displacements) == 0:
        return {
            "confidence": float(params.confidence_min),
            "c_ncc": float("nan"),
            "c_valid": 0.0,
            "c_motion": 0.0,
            "sigma_motion_px": float("nan"),
            "median_dx_px": float("nan"),
            "median_dy_px": float("nan"),
            "n_valid_blocks": 0,
            "n_total_blocks": int(total_blocks),
        }

    displacements_arr = np.asarray(displacements, dtype=float)
    ncc_scores_arr = np.asarray(ncc_scores, dtype=float)

    median_displacement = np.median(displacements_arr, axis=0)
    residuals = np.linalg.norm(displacements_arr - median_displacement, axis=1)
    sigma_motion = robust_mad_sigma(residuals)

    c_ncc = float(np.median(ncc_scores_arr))
    c_valid = float(len(displacements_arr) / total_blocks)
    c_motion = (
        float(np.exp(-(sigma_motion**2) / (params.sigma_ref_px**2)))
        if np.isfinite(sigma_motion) and params.sigma_ref_px > 0
        else 0.0
    )
    confidence = float(np.clip(c_ncc * c_valid * c_motion, params.confidence_min, 1.0))

    return {
        "confidence": confidence,
        "c_ncc": c_ncc,
        "c_valid": c_valid,
        "c_motion": c_motion,
        "sigma_motion_px": float(sigma_motion),
        "median_dx_px": float(median_displacement[0]),
        "median_dy_px": float(median_displacement[1]),
        "n_valid_blocks": int(len(displacements_arr)),
        "n_total_blocks": int(total_blocks),
    }


def compute_speckle_confidence_for_frames(
    video_path: str | Path,
    frame_numbers: np.ndarray,
    roi: ROI,
    params: Optional[SpeckleCoherenceParams] = None,
    *,
    max_pairs: Optional[int] = None,
    progress_every: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute speckle coherence for a sequence of frame numbers in one ROI.

    The first processed frame has ``NaN`` confidence because there is no
    previous processed frame to compare against.
    """
    params = params or SpeckleCoherenceParams()
    frame_numbers = np.asarray(frame_numbers, dtype=int)
    if frame_numbers.ndim != 1 or len(frame_numbers) == 0:
        raise ValueError("frame_numbers must be a non-empty 1D array.")

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError("roi must be (x, y, w, h) with positive width and height.")

    target = set(frame_numbers.tolist())
    index_by_frame = {int(f): i for i, f in enumerate(frame_numbers)}

    out = {
        "confidence": np.full(len(frame_numbers), np.nan, dtype=float),
        "c_ncc": np.full(len(frame_numbers), np.nan, dtype=float),
        "c_valid": np.full(len(frame_numbers), np.nan, dtype=float),
        "c_motion": np.full(len(frame_numbers), np.nan, dtype=float),
        "sigma_motion_px": np.full(len(frame_numbers), np.nan, dtype=float),
        "median_dx_px": np.full(len(frame_numbers), np.nan, dtype=float),
        "median_dy_px": np.full(len(frame_numbers), np.nan, dtype=float),
        "n_valid_blocks": np.zeros(len(frame_numbers), dtype=int),
        "n_total_blocks": np.zeros(len(frame_numbers), dtype=int),
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    previous_crop = None
    processed_pairs = 0
    max_frame = int(np.max(frame_numbers))

    for frame_idx in range(max_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx not in target:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        crop = gray[y : y + h, x : x + w]
        result_idx = index_by_frame[frame_idx]

        if previous_crop is not None:
            if max_pairs is not None and processed_pairs >= max_pairs:
                break

            pair = speckle_pair_confidence(previous_crop, crop, params=params)
            for key, value in pair.items():
                out[key][result_idx] = value

            processed_pairs += 1
            if progress_every and processed_pairs % progress_every == 0:
                print(f"Speckle pairs processed: {processed_pairs}")

        previous_crop = crop

    cap.release()
    return out


class SpeckleTracker:
    """
    Speckle tracking par corrélation croisée normalisée (NCC).

    Drift-free et peu bruité — exploite le pattern de speckle
    propre à chaque région tissulaire.
    Particulièrement adapté pour la fascia et les tissus mous
    sans structure linéaire claire (où Hough échoue).
    """

    def __init__(self,
                 kernel_size: int = 21,
                 search_radius: int = 15,
                 step: int = 10,
                 min_ncc: float = 0.5):

        self.kernel_size = kernel_size
        self.search_radius = search_radius
        self.step = step
        self.min_ncc = min_ncc   # seuil de confiance NCC

    def compute_displacement(self,
                             frame1: np.ndarray,
                             frame2: np.ndarray):
        """
        Calcule le champ de déplacement entre frame1 et frame2
        par NCC sur une grille de régions.

        Retourne
        --------
        disp_x     : champ de déplacement horizontal (pixels)
        disp_y     : champ de déplacement vertical (pixels)
        confidence : valeur NCC au pic (0→1) pour chaque région
        """
        h, w = frame1.shape
        hk = self.kernel_size // 2
        sr = self.search_radius

        disp_x = np.full((h, w), np.nan)
        disp_y = np.full((h, w), np.nan)
        confidence = np.zeros((h, w))

        for y in range(hk + sr, h - hk - sr, self.step):
            for x in range(hk + sr, w - hk - sr, self.step):

                # Pattern de référence dans frame1
                kernel = frame1[y-hk:y+hk+1,
                                x-hk:x+hk+1].astype(float)

                # Région de recherche dans frame2
                search = frame2[y-hk-sr:y+hk+sr+1,
                                x-hk-sr:x+hk+sr+1].astype(float)

                # Normalisation
                k_norm = kernel - kernel.mean()
                s_norm = search - search.mean()
                k_std = k_norm.std()
                s_std = s_norm.std()

                if k_std < 1e-6 or s_std < 1e-6:
                    continue   # région homogène, pas de texture

                # Corrélation croisée normalisée
                corr = correlate2d(s_norm, k_norm, mode='valid')
                corr /= (k_std * s_std * self.kernel_size ** 2)

                peak = np.unravel_index(np.argmax(corr), corr.shape)
                ncc_val = corr[peak]

                if ncc_val < self.min_ncc:
                    continue   # match pas assez confiant

                disp_y[y, x] = peak[0] - sr
                disp_x[y, x] = peak[1] - sr
                confidence[y, x] = ncc_val

        return disp_x, disp_y, confidence

    def mean_displacement(self,
                          frame1: np.ndarray,
                          frame2: np.ndarray):
        """
        Retourne le déplacement moyen (dx, dy) pondéré par la
        confiance NCC — format prêt pour intégration dans Kalman.
        """
        dx, dy, conf = self.compute_displacement(frame1, frame2)

        mask = ~np.isnan(dx) & (conf > self.min_ncc)
        if mask.sum() == 0:
            return np.array([0., 0.])

        weights = conf[mask]
        mean_dx = np.average(dx[mask], weights=weights)
        mean_dy = np.average(dy[mask], weights=weights)
        return np.array([mean_dx, mean_dy])
