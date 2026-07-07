"""
MATLAB-style TimTrack Hough helpers.

This module ports the small pieces used by UltraTimTrack's TimTrack path:
``weightedMedian.m``, ``dohough.m``, and the helper ``hough_bin_pixels.m``.
The helpers are intentionally small so validation notebooks can compare the
MATLAB-style weighted-median angle estimator against the older OpenCV
probabilistic Hough detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import rotate as scipy_rotate

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_rotate = None


@dataclass(frozen=True)
class DoHoughParams:
    """Parameters corresponding to the MATLAB ``parms`` fields used by dohough."""

    houghangles: str = "specified"
    angle_range: Tuple[float, float] = (8.0, 80.0)
    thetares: float = 1.0
    rhores: float = 1.0
    emask_radius: Tuple[float, float] = (1.0, 1.0)
    npeaks: int = 10
    show: bool = False
    replace_diagonal_bias: bool = True


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Port of ``weightedMedian.m``.

    MATLAB flattens ``D'`` and ``W'`` before sorting, which is equivalent to a
    row-major flatten in NumPy. The returned value is the first sorted value for
    which cumulative normalized weight reaches 0.5.
    """
    values_arr = np.asarray(values, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)

    if values_arr.shape != weights_arr.shape:
        raise ValueError("values and weights must have the same shape.")

    weight_sum = np.sum(weights_arr)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return float("nan")

    d = values_arr.ravel(order="C")
    w = (weights_arr / weight_sum).ravel(order="C")

    order = np.argsort(d, kind="mergesort")
    d_sorted = d[order]
    w_sorted = w[order]
    cumulative = np.cumsum(w_sorted)

    index = int(np.argmax(cumulative >= 0.5))
    return float(d_sorted[index])


def matlab_theta_from_range(
    houghangles: str = "specified",
    angle_range: Tuple[float, float] = (8.0, 80.0),
    thetares: float = 1.0,
) -> np.ndarray:
    """Return the theta vector used by MATLAB ``dohough.m``."""
    if houghangles == "default":
        start, stop = -90.0, 89.0
    else:
        start, stop = sorted((90.0 - float(angle_range[0]), 90.0 - float(angle_range[1])))

    # MATLAB ``a:b:c`` includes c when it lands exactly on the grid.
    count = int(np.floor((stop - start) / thetares + 1e-12)) + 1
    return start + thetares * np.arange(count, dtype=np.float64)


def matlab_hough_accumulator(
    binary: np.ndarray,
    theta_degrees: Iterable[float],
    rho_resolution: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB-like Hough accumulator for binary images.

    Coordinates follow MathWorks' helper convention: foreground pixels are
    located with 1-based ``find`` in MATLAB, then transformed using zero-based
    ``x`` and ``y`` coordinates.
    """
    bw = np.asarray(binary).astype(bool)
    theta = np.asarray(list(theta_degrees), dtype=np.float64)

    if bw.ndim != 2:
        raise ValueError("binary must be a 2D image.")
    if rho_resolution <= 0:
        raise ValueError("rho_resolution must be positive.")

    nrows, ncols = bw.shape
    diagonal = np.ceil(np.hypot(nrows - 1, ncols - 1))
    rho = np.arange(-diagonal, diagonal + rho_resolution, rho_resolution, dtype=np.float64)
    accumulator = np.zeros((len(rho), len(theta)), dtype=np.float64)

    y, x = np.nonzero(bw)
    if len(x) == 0 or len(theta) == 0:
        return accumulator, theta, rho

    cos_theta = np.cos(np.deg2rad(theta))
    sin_theta = np.sin(np.deg2rad(theta))
    slope = (len(rho) - 1) / (rho[-1] - rho[0])

    for col, (ct, st) in enumerate(zip(cos_theta, sin_theta)):
        rho_xy = x * ct + y * st
        rows = np.rint(slope * (rho_xy - rho[0])).astype(np.int64)
        valid = (rows >= 0) & (rows < len(rho))
        np.add.at(accumulator[:, col], rows[valid], 1.0)

    return accumulator, theta, rho


def default_hough_peak_neighborhood(shape: Tuple[int, int]) -> Tuple[int, int]:
    """Return MATLAB ``houghpeaks`` default ``NHoodSize``."""

    values = np.maximum(2 * np.ceil((np.asarray(shape, dtype=np.float64) / 50.0) / 2.0) + 1, 1)
    return int(values[0]), int(values[1])


def hough_peaks(
    accumulator: np.ndarray,
    num_peaks: int,
    threshold: float = 0.0,
    neighborhood_size: Optional[Tuple[int, int]] = None,
    theta_degrees: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Port the MATLAB ``houghpeaks`` suppression loop.

    Returns zero-based ``(row, col)`` peak coordinates. After each peak, an odd
    neighborhood is suppressed. The implementation keeps MATLAB's column-major
    max tie handling and theta-boundary antisymmetry behavior.
    """
    h = np.asarray(accumulator, dtype=np.float64).copy()
    if h.ndim != 2:
        raise ValueError("accumulator must be 2D.")
    if num_peaks <= 0:
        return np.empty((0, 2), dtype=np.int64)

    if neighborhood_size is None:
        neighborhood_size = default_hough_peak_neighborhood(h.shape)

    nhood = np.asarray(neighborhood_size, dtype=np.int64).reshape(-1)
    if nhood.size != 2 or np.any(nhood <= 0) or np.any(nhood % 2 == 0):
        raise ValueError("neighborhood_size must contain two positive odd integers.")
    row_half = int(nhood[0]) // 2
    col_half = int(nhood[1]) // 2

    if theta_degrees is None:
        theta = np.arange(-90.0, 90.0, dtype=np.float64)
    else:
        theta = np.asarray(theta_degrees, dtype=np.float64).reshape(-1)
    if theta.size > 1:
        min_theta = float(np.min(theta))
        max_theta = float(np.max(theta))
        theta_resolution = abs(max_theta - min_theta) / (theta.size - 1)
        theta_antisymmetric = abs(min_theta + theta_resolution * int(nhood[1])) <= max_theta
    else:
        theta_antisymmetric = False

    peaks = []
    for _ in range(num_peaks):
        flat_index = int(np.argmax(h.ravel(order="F")))
        row, col = np.unravel_index(flat_index, h.shape, order="F")
        value = float(h[row, col])
        if value < threshold:
            break

        peaks.append((row, col))

        row_values = np.arange(max(row - row_half, 0), min(row + row_half, h.shape[0] - 1) + 1)
        col_values = np.arange(col - col_half, col + col_half + 1)
        qq, pp = np.meshgrid(col_values, row_values)
        pp = pp.reshape(-1)
        qq = qq.reshape(-1)

        if theta_antisymmetric:
            low = qq < 0
            qq[low] = h.shape[1] + qq[low]
            pp[low] = h.shape[0] - pp[low] - 1
            high = qq >= h.shape[1]
            qq[high] = qq[high] - h.shape[1]
            pp[high] = h.shape[0] - pp[high] - 1

        valid = (qq >= 0) & (qq < h.shape[1])
        pp = pp[valid]
        qq = qq[valid]
        h[pp, qq] = 0.0

    return np.asarray(peaks, dtype=np.int64)


def hough_bin_pixels(
    binary: np.ndarray,
    theta_degrees: np.ndarray,
    rho_values: np.ndarray,
    bin_row_col: Tuple[int, int],
) -> np.ndarray:
    """Port of MATLAB ``hough_bin_pixels.m`` using zero-based bin indices."""
    bw = np.asarray(binary).astype(bool)
    theta = np.asarray(theta_degrees, dtype=np.float64)
    rho = np.asarray(rho_values, dtype=np.float64)
    row, col = int(bin_row_col[0]), int(bin_row_col[1])

    y, x = np.nonzero(bw)
    out = np.zeros_like(bw, dtype=bool)

    if len(x) == 0 or len(rho) < 2:
        return out

    theta_c = np.deg2rad(theta[col])
    rho_xy = x * np.cos(theta_c) + y * np.sin(theta_c)
    slope = (len(rho) - 1) / (rho[-1] - rho[0])
    rho_bin_index = np.rint(slope * (rho_xy - rho[0])).astype(np.int64)
    idx = rho_bin_index == row

    out[y[idx], x[idx]] = True
    return out


def ellipse_radius_correction(gamma_degrees: np.ndarray, emask_radius: Tuple[float, float]) -> np.ndarray:
    """Return MATLAB ``r_ellipse_rel`` for the Hough angle correction."""
    gamma = np.asarray(gamma_degrees, dtype=np.float64)
    re_vertical = float(emask_radius[0])
    re_horizontal = float(emask_radius[1])

    if not np.isfinite(re_vertical) or not np.isfinite(re_horizontal):
        return np.ones_like(gamma, dtype=np.float64)
    if abs(re_vertical) <= 1e-12 or abs(re_horizontal) <= 1e-12:
        return np.ones_like(gamma, dtype=np.float64)

    cos_g = np.cos(np.deg2rad(gamma))
    sin_g = np.sin(np.deg2rad(gamma))
    denominator = np.sqrt((re_vertical**2) * (cos_g**2) + (re_horizontal**2) * (sin_g**2))
    return re_vertical / denominator


def rotate_binary_nearest(binary: np.ndarray, angle_degrees: float) -> np.ndarray:
    """MATLAB-like ``imrotate(..., 'nearest', 'crop')`` for binary masks."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for the rotated Hough bias correction.")

    rotated = scipy_rotate(
        np.asarray(binary).astype(np.uint8),
        angle=float(angle_degrees),
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return rotated.astype(bool)


def _line_endpoints_for_peak(
    binary: np.ndarray,
    theta_degrees: np.ndarray,
    rho_values: np.ndarray,
    peak: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return one-based line endpoints for one Hough peak."""

    contributing = hough_bin_pixels(binary, theta_degrees, rho_values, peak)
    yy, xx = np.nonzero(contributing)
    if len(xx) == 0:
        return np.full(2, np.nan), np.full(2, np.nan)

    order = np.lexsort((yy, xx))
    xx = xx[order] + 1
    yy = yy[order] + 1
    return np.asarray([xx[0], xx[-1]], dtype=np.float64), np.asarray([yy[0], yy[-1]], dtype=np.float64)


def local_maxima_indices(values: np.ndarray, num_peaks: int) -> np.ndarray:
    """
    Return strongest one-dimensional local maxima.

    This is the notebook-90 localmax peak source: collapse the Hough accumulator
    by angle first, then pick local maxima along the angle profile.
    """

    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0 or num_peaks <= 0:
        return np.empty(0, dtype=np.int64)

    candidates: list[int] = []
    for idx, value in enumerate(vals):
        if not np.isfinite(value):
            continue
        left = vals[idx - 1] if idx > 0 and np.isfinite(vals[idx - 1]) else -np.inf
        right = vals[idx + 1] if idx + 1 < vals.size and np.isfinite(vals[idx + 1]) else -np.inf
        if value >= left and value >= right:
            candidates.append(idx)

    if not candidates:
        return np.empty(0, dtype=np.int64)

    candidates.sort(key=lambda i: (-vals[i], i))
    return np.asarray(candidates[: int(num_peaks)], dtype=np.int64)


def dohough_angle_profile_localmax(binary: np.ndarray, params: DoHoughParams | dict) -> dict:
    """
    Variant of :func:`dohough` using angle-profile local maxima as peaks.

    The accumulator, diagonal-bias replacement, radius correction, and weighted
    median stay MATLAB-style. Only the peak source changes from 2D
    ``houghpeaks`` suppression to one-dimensional local maxima of
    ``max(H, axis=rho)``.
    """

    if isinstance(params, dict):
        params = DoHoughParams(**params)

    bw = np.asarray(binary).astype(bool)
    theta = matlab_theta_from_range(params.houghangles, params.angle_range, params.thetares)
    hmat, theta, rho = matlab_hough_accumulator(bw, theta, params.rhores)

    if (
        params.replace_diagonal_bias
        and params.angle_range[0] < 45 < params.angle_range[1]
        and np.any(theta == 45)
    ):
        rot_angle = 20.0
        rotated = rotate_binary_nearest(bw, rot_angle)
        replacement_theta = np.asarray([90.0 - (45.0 + rot_angle)])
        hmat_rot, _, _ = matlab_hough_accumulator(rotated, replacement_theta, params.rhores)
        hmat[:, theta == 45] = hmat_rot

    gamma = 90.0 - theta
    radius_correction = ellipse_radius_correction(gamma, params.emask_radius)
    hmat_corrected = np.rint(hmat / radius_correction[np.newaxis, :])
    h_by_angle = np.max(hmat_corrected, axis=0) if hmat_corrected.size else np.asarray([])

    peak_cols = local_maxima_indices(h_by_angle, params.npeaks)
    if len(peak_cols):
        peak_cols = peak_cols[np.argsort(h_by_angle[peak_cols])[::-1]]
    peak_rows = (
        np.asarray([int(np.nanargmax(hmat_corrected[:, col])) for col in peak_cols], dtype=np.int64)
        if len(peak_cols)
        else np.empty(0, dtype=np.int64)
    )
    peaks = np.column_stack([peak_rows, peak_cols]) if len(peak_cols) else np.empty((0, 2), dtype=np.int64)
    weights = np.asarray([h_by_angle[col] for col in peak_cols], dtype=np.float64)
    alphas = np.asarray([gamma[col] for col in peak_cols], dtype=np.float64)

    x_lines = np.full((len(peaks), 2), np.nan, dtype=np.float64)
    y_lines = np.full((len(peaks), 2), np.nan, dtype=np.float64)
    for i, peak in enumerate(peaks):
        x_lines[i], y_lines[i] = _line_endpoints_for_peak(bw, theta, rho, tuple(peak))

    return {
        "alpha": weighted_median(alphas, weights) if len(alphas) else np.nan,
        "alphas": alphas,
        "weights": weights,
        "h_by_angle": h_by_angle,
        "hmat": hmat,
        "hmat_corrected": hmat_corrected,
        "theta": theta,
        "rho": rho,
        "gamma": gamma,
        "peaks": peaks,
        "X": x_lines,
        "Y": y_lines,
        "peak_source": "angle_profile_localmax",
    }


def candidate_mass_below(
    alpha: float,
    alphas: np.ndarray,
    weights: np.ndarray,
    margin_degrees: float,
) -> float:
    """Return normalized candidate weight at least ``margin_degrees`` below ``alpha``."""

    alpha_arr = np.asarray(alphas, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = min(len(alpha_arr), len(weight_arr))
    if n == 0 or not np.isfinite(alpha):
        return float("nan")

    alpha_arr = alpha_arr[:n]
    weight_arr = weight_arr[:n]
    valid = np.isfinite(alpha_arr) & np.isfinite(weight_arr) & (weight_arr > 0)
    if not np.any(valid):
        return float("nan")

    alpha_arr = alpha_arr[valid]
    weight_arr = weight_arr[valid]
    total = float(np.sum(weight_arr))
    if total <= 0:
        return float("nan")

    return float(np.sum(weight_arr[alpha_arr <= float(alpha) - float(margin_degrees)]) / total)


def gap_to_nearest_lower(alpha: float, alphas: np.ndarray) -> float:
    """Return distance from ``alpha`` to the nearest lower candidate alpha."""

    alpha_arr = np.asarray(alphas, dtype=np.float64).reshape(-1)
    if alpha_arr.size == 0 or not np.isfinite(alpha):
        return float("nan")
    lower = alpha_arr[np.isfinite(alpha_arr) & (alpha_arr < float(alpha))]
    if lower.size == 0:
        return float("nan")
    return float(float(alpha) - np.max(lower))


def should_use_localmax_fallback(
    alpha: float,
    alphas: np.ndarray,
    weights: np.ndarray,
    *,
    mass_margin_degrees: float = 10.0,
    min_mass_below: float = 0.25,
    min_gap_to_lower_degrees: float = 4.0,
) -> tuple[bool, float, float]:
    """Return notebook-90 mass/gap detector decision and diagnostics."""

    mass = candidate_mass_below(alpha, alphas, weights, mass_margin_degrees)
    gap = gap_to_nearest_lower(alpha, alphas)
    use_fallback = (
        np.isfinite(mass)
        and np.isfinite(gap)
        and mass >= float(min_mass_below)
        and gap >= float(min_gap_to_lower_degrees)
    )
    return bool(use_fallback), float(mass), float(gap)


def dohough_with_localmax_fallback(
    binary: np.ndarray,
    params: DoHoughParams | dict,
    *,
    min_mass_below_10deg: float = 0.25,
    min_gap_to_lower_deg: float = 4.0,
) -> dict:
    """
    Run baseline ``dohough`` and conditionally replace it with localmax output.

    This promotes notebook 90's best Python-only rule:
    ``mass_below_10deg >= 0.25 and gap_to_lower_deg >= 4.0``.
    """

    if isinstance(params, dict):
        params = DoHoughParams(**params)

    baseline = dohough(binary, params)
    use_fallback, mass, gap = should_use_localmax_fallback(
        float(baseline["alpha"]),
        np.asarray(baseline["alphas"], dtype=np.float64),
        np.asarray(baseline["weights"], dtype=np.float64),
        mass_margin_degrees=10.0,
        min_mass_below=float(min_mass_below_10deg),
        min_gap_to_lower_degrees=float(min_gap_to_lower_deg),
    )

    if use_fallback:
        result = dict(dohough_angle_profile_localmax(binary, params))
    else:
        result = dict(baseline)

    result["localmax_fallback_used"] = bool(use_fallback)
    result["localmax_fallback_mass_below_10deg"] = float(mass)
    result["localmax_fallback_gap_to_lower_deg"] = float(gap)
    result["localmax_fallback_min_mass_below_10deg"] = float(min_mass_below_10deg)
    result["localmax_fallback_min_gap_to_lower_deg"] = float(min_gap_to_lower_deg)
    result["baseline_alpha"] = float(baseline["alpha"])
    result["selected_peak_source"] = "angle_profile_localmax" if use_fallback else "houghpeaks"
    return result


def dohough(binary: np.ndarray, params: DoHoughParams | dict) -> dict:
    """
    Port of MATLAB ``dohough.m``.

    Returns a dictionary with raw accumulator products and the weighted median
    angle. Peak coordinates are zero-based, unlike MATLAB's one-based ``P``.
    """
    if isinstance(params, dict):
        params = DoHoughParams(**params)

    bw = np.asarray(binary).astype(bool)
    theta = matlab_theta_from_range(params.houghangles, params.angle_range, params.thetares)
    hmat, theta, rho = matlab_hough_accumulator(bw, theta, params.rhores)

    if (
        params.replace_diagonal_bias
        and params.angle_range[0] < 45 < params.angle_range[1]
        and np.any(theta == 45)
    ):
        rot_angle = 20.0
        rotated = rotate_binary_nearest(bw, rot_angle)
        replacement_theta = np.asarray([90.0 - (45.0 + rot_angle)])
        hmat_rot, _, _ = matlab_hough_accumulator(rotated, replacement_theta, params.rhores)
        hmat[:, theta == 45] = hmat_rot

    gamma = 90.0 - theta
    radius_correction = ellipse_radius_correction(gamma, params.emask_radius)
    hmat_corrected = np.rint(hmat / radius_correction[np.newaxis, :])
    h_by_angle = np.max(hmat_corrected, axis=0) if hmat_corrected.size else np.asarray([])

    peaks = hough_peaks(hmat_corrected, params.npeaks, threshold=0.0, theta_degrees=theta)
    weights = np.asarray([hmat_corrected[row, col] for row, col in peaks], dtype=np.float64)
    alphas = np.asarray([gamma[col] for _, col in peaks], dtype=np.float64)

    x_lines = np.full((len(peaks), 2), np.nan, dtype=np.float64)
    y_lines = np.full((len(peaks), 2), np.nan, dtype=np.float64)
    for i, peak in enumerate(peaks):
        x_lines[i], y_lines[i] = _line_endpoints_for_peak(bw, theta, rho, tuple(peak))

    return {
        "alpha": weighted_median(alphas, weights) if len(alphas) else np.nan,
        "alphas": alphas,
        "weights": weights,
        "h_by_angle": h_by_angle,
        "hmat": hmat,
        "hmat_corrected": hmat_corrected,
        "theta": theta,
        "rho": rho,
        "gamma": gamma,
        "peaks": peaks,
        "X": x_lines,
        "Y": y_lines,
        "peak_source": "houghpeaks",
    }


def estimate_fascicle_alpha_dohough(
    binary_mask: np.ndarray,
    *,
    emask_radius: Optional[Tuple[float, float]] = None,
    angle_range: Tuple[float, float] = (8.0, 80.0),
    thetares: float = 1.0,
    rhores: float = 1.0,
    npeaks: int = 10,
    houghangles: str = "specified",
    replace_diagonal_bias: bool = True,
) -> dict:
    """
    Estimate fascicle alpha from a binary fascicle mask using MATLAB-style Hough.

    This is the compatibility layer for the current Python sequence workflow:
    keep the existing fascicle mask, but replace the selected-segment angle with
    the weighted median alpha returned by UltraTimTrack's ``dohough.m`` path.
    """
    bw = np.asarray(binary_mask).astype(bool)
    if bw.ndim != 2:
        raise ValueError("binary_mask must be a 2D image.")

    if emask_radius is None:
        emask_radius = (bw.shape[0] / 2.0, bw.shape[1] / 2.0)

    params = DoHoughParams(
        houghangles=houghangles,
        angle_range=angle_range,
        thetares=thetares,
        rhores=rhores,
        emask_radius=emask_radius,
        npeaks=npeaks,
        replace_diagonal_bias=replace_diagonal_bias,
    )
    return dohough(bw, params)


def fascicle_ellipse_mask(
    shape: Tuple[int, int],
    superficial_y: float,
    deep_y: float,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Port the ellipse geometry from ``get_fasMask.m`` for diagnostics.

    ``superficial_y`` and ``deep_y`` are one-based MATLAB-style depths when
    read directly from saved geofeatures. The returned radius is
    ``(vertical, horizontal)``.
    """
    nrows, ncols = int(shape[0]), int(shape[1])
    m_super = np.rint(superficial_y)
    m_deep = np.rint(deep_y)

    if not np.isfinite(m_super) or not np.isfinite(m_deep) or m_deep <= m_super:
        return np.ones((nrows, ncols), dtype=bool), (nrows / 2.0, ncols / 2.0)

    r_vertical = (m_deep - m_super) / 2.0
    r_horizontal = ncols / 2.0
    x_center = r_horizontal
    y_center = m_super + r_vertical

    y, x = np.mgrid[1 : nrows + 1, 1 : ncols + 1]
    normalized = ((x - x_center) / r_horizontal) ** 2 + ((y - y_center) / r_vertical) ** 2
    mask = normalized <= 1.0
    return mask, (float(r_vertical), float(r_horizontal))
