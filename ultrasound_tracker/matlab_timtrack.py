"""Helpers for MATLAB TimTrack geofeature compatibility.

The raw ``dohough`` path returns an angle from one binary fascicle mask.  Saved
UltraTimTrack results also keep the peak angle stream under
``geofeatures(i).alphas`` and ``geofeatures(i).ws``.  MATLAB's saved
``geofeatures(i).alpha`` is the weighted median of those saved peaks.

Keeping this reconstruction explicit lets parity notebooks distinguish two
different gates:

* raw mask/doHough parity against intermediate MATLAB exports;
* saved full-sequence geofeature parity against ``Fdat`` / ``UTT`` outputs.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple

import numpy as np

from .timtrack_hough import weighted_median


def _as_entry_list(geofeatures: object) -> list[Mapping]:
    if isinstance(geofeatures, Mapping):
        return [geofeatures]
    if isinstance(geofeatures, (list, tuple)):
        return [entry for entry in geofeatures if isinstance(entry, Mapping)]
    return [
        entry
        for entry in np.asarray(geofeatures, dtype=object).reshape(-1)
        if isinstance(entry, Mapping)
    ]


def alpha_from_saved_peaks(alphas: object, weights: object) -> float:
    """Return MATLAB's saved TimTrack alpha from saved peak angles and weights."""

    alpha_arr = np.asarray(alphas, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = min(alpha_arr.size, weight_arr.size)
    if n == 0:
        return float("nan")

    alpha_arr = alpha_arr[:n]
    weight_arr = weight_arr[:n]
    valid = np.isfinite(alpha_arr) & np.isfinite(weight_arr) & (weight_arr > 0)
    if not np.any(valid):
        return float("nan")
    return weighted_median(alpha_arr[valid], weight_arr[valid])


def reconstruct_saved_geofeature_alpha(geofeatures: object) -> np.ndarray:
    """
    Reconstruct ``geofeatures.alpha`` from saved ``alphas`` and ``ws`` arrays.

    MATLAB stores the peak weights as ``ws`` in the full UTT numeric export.
    Some diagnostic exports use ``weights`` instead, so this helper accepts both
    names and prefers ``ws`` when available.
    """

    out = []
    for entry in _as_entry_list(geofeatures):
        weights = entry.get("ws", entry.get("weights", []))
        out.append(alpha_from_saved_peaks(entry.get("alphas", []), weights))
    return np.asarray(out, dtype=np.float64)


def extract_saved_peak_arrays(
    geofeatures: object,
    *,
    max_peaks: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(alpha, peak_alphas, peak_weights)`` from saved geofeatures.

    Arrays are padded with NaNs to ``max_peaks`` columns.  ``alpha`` is
    reconstructed from the saved peaks, not copied from ``entry['alpha']``.
    """

    entries = _as_entry_list(geofeatures)
    peak_alphas = np.full((len(entries), int(max_peaks)), np.nan, dtype=np.float64)
    peak_weights = np.full_like(peak_alphas, np.nan)
    alpha = np.full(len(entries), np.nan, dtype=np.float64)

    for idx, entry in enumerate(entries):
        weights = np.asarray(entry.get("ws", entry.get("weights", [])), dtype=np.float64).reshape(-1)
        alphas = np.asarray(entry.get("alphas", []), dtype=np.float64).reshape(-1)
        n = min(int(max_peaks), len(alphas), len(weights))
        if n:
            peak_alphas[idx, :n] = alphas[:n]
            peak_weights[idx, :n] = weights[:n]
        alpha[idx] = alpha_from_saved_peaks(alphas, weights)

    return alpha, peak_alphas, peak_weights


def saved_alpha_error(saved_geofeatures: object, reference_alpha: Optional[Iterable[float]] = None) -> dict:
    """
    Summarize reconstruction error against an optional saved reference alpha.

    This is mainly a compact notebook helper.  When ``reference_alpha`` is not
    provided, the function compares against ``entry['alpha']`` if present.
    """

    entries = _as_entry_list(saved_geofeatures)
    estimate = reconstruct_saved_geofeature_alpha(entries)
    if reference_alpha is None:
        reference = np.asarray(
            [
                np.asarray(entry.get("alpha", np.nan), dtype=np.float64).reshape(-1)[0]
                for entry in entries
            ],
            dtype=np.float64,
        )
    else:
        reference = np.asarray(reference_alpha, dtype=np.float64).reshape(-1)

    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]
    mask = np.isfinite(reference) & np.isfinite(estimate)
    diff = estimate[mask] - reference[mask]
    return {
        "n": int(np.sum(mask)),
        "max_abs_error_deg": float(np.nanmax(np.abs(diff))) if diff.size else np.nan,
        "rmse_deg": float(np.sqrt(np.nanmean(diff**2))) if diff.size else np.nan,
    }
