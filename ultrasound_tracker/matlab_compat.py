"""
Compatibility helpers for validating Python outputs against UltraTimTrack.

The functions in this module deliberately focus on the data products saved by
the MATLAB GUI:

    TrackingData
    Fdat.Region.FL / PEN / ANG
    Fdat.geofeatures

They are not part of the tracking algorithm itself. They are a stable ruler for
checking whether the Python implementation is converging toward MATLAB parity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


def load_matlab_result(path: str | Path) -> Dict:
    """
    Load an UltraTimTrack MATLAB result file.

    Parameters
    ----------
    path:
        Path to a .mat file saved by UltraTimTrack.

    Returns
    -------
    dict
        A scipy-loaded MATLAB dictionary with private keys removed.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to load MATLAB .mat files.") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = loadmat(path, simplify_cells=True)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def get_nested(data: Mapping, path: str):
    """
    Read a nested dict value by dot path.

    Example
    -------
    get_nested(mat, "Fdat.Region.FL")
    """
    obj = data
    for part in path.split("."):
        if isinstance(obj, Mapping):
            obj = obj[part]
        else:
            raise KeyError(f"Cannot read {part!r} from object of type {type(obj)!r}")
    return obj


def to_1d_float(values) -> np.ndarray:
    """Convert MATLAB-loaded scalar/list/array values to 1D float arrays."""
    arr = np.asarray(values, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def first_region(mat: Mapping) -> Mapping:
    """
    Return the first Fdat.Region entry.

    UltraTimTrack usually saves a scalar region for this workflow. This helper
    also handles a list/array of regions so the comparison code stays explicit.
    """
    region = get_nested(mat, "Fdat.Region")

    if isinstance(region, Mapping):
        return region

    if isinstance(region, (list, tuple)):
        if not region:
            raise ValueError("Fdat.Region is empty.")
        return region[0]

    arr = np.asarray(region, dtype=object).reshape(-1)
    if arr.size == 0:
        raise ValueError("Fdat.Region is empty.")
    if not isinstance(arr[0], Mapping):
        raise TypeError(f"Unsupported Fdat.Region entry type: {type(arr[0])!r}")
    return arr[0]


def extract_final_region_arrays(mat: Mapping) -> Dict[str, np.ndarray | float]:
    """
    Extract final MATLAB output arrays used for parity validation.

    Returns keys:
        time_s, length_mm, pennation_deg, fascicle_angle_deg, image_depth_mm
    """
    region = first_region(mat)
    tracking = mat.get("TrackingData", {})

    image_depth_mm = np.nan
    if isinstance(tracking, Mapping) and "res" in tracking:
        image_depth_mm = float(np.asarray(tracking["res"]).reshape(-1)[0])

    return {
        "time_s": to_1d_float(region["Time"]),
        "length_mm": to_1d_float(region["FL"]),
        "pennation_deg": to_1d_float(region["PEN"]),
        "fascicle_angle_deg": to_1d_float(region["ANG"]),
        "image_depth_mm": image_depth_mm,
    }


def _iter_geofeature_entries(mat: Mapping) -> Iterable[Mapping]:
    geofeatures = get_nested(mat, "Fdat.geofeatures")

    if isinstance(geofeatures, Mapping):
        yield geofeatures
        return

    if isinstance(geofeatures, (list, tuple)):
        iterator = geofeatures
    else:
        iterator = np.asarray(geofeatures, dtype=object).reshape(-1)

    for entry in iterator:
        if isinstance(entry, Mapping):
            yield entry


def _entry_scalar(entry: Mapping, key: str) -> float:
    if key not in entry:
        return np.nan
    arr = np.asarray(entry[key], dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.nan
    return float(arr[0])


def _entry_vector_value(entry: Mapping, key: str, index: int) -> float:
    if key not in entry:
        return np.nan
    arr = np.asarray(entry[key], dtype=np.float64).reshape(-1)
    if arr.size <= index:
        return np.nan
    return float(arr[index])


def extract_geofeature_arrays(mat: Mapping) -> Dict[str, np.ndarray]:
    """
    Extract MATLAB TimTrack intermediate arrays from Fdat.geofeatures.

    Important keys for parity work:
        alpha: TimTrack fascicle angle
        phi: pennation-like fascicle-superficial angle
        faslen_px: TimTrack fascicle length before mm conversion
        gamma: deep aponeurosis angle
        betha: superficial aponeurosis angle
    """
    entries = list(_iter_geofeature_entries(mat))

    scalar_keys = {
        "alpha": "alpha_deg",
        "phi": "phi_deg",
        "faslen": "faslen_px",
        "gamma": "deep_apo_angle_deg",
        "betha": "super_apo_angle_deg",
        "thickness": "muscle_thickness_px",
        "brightness": "brightness",
        "extrapolated_fraction": "extrapolated_fraction",
    }

    out: Dict[str, np.ndarray] = {}

    for matlab_key, output_key in scalar_keys.items():
        out[output_key] = np.asarray(
            [_entry_scalar(entry, matlab_key) for entry in entries],
            dtype=np.float64,
        )

    for key, output_prefix in [
        ("super_pos", "super_pos_y"),
        ("deep_pos", "deep_pos_y"),
    ]:
        out[f"{output_prefix}1"] = np.asarray(
            [_entry_vector_value(entry, key, 0) for entry in entries],
            dtype=np.float64,
        )
        out[f"{output_prefix}2"] = np.asarray(
            [_entry_vector_value(entry, key, 1) for entry in entries],
            dtype=np.float64,
        )

    return out


def valid_pair(reference, estimate) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite paired values from reference and estimate arrays."""
    ref = to_1d_float(reference)
    est = to_1d_float(estimate)
    n = min(len(ref), len(est))
    ref = ref[:n]
    est = est[:n]
    mask = np.isfinite(ref) & np.isfinite(est)
    return ref[mask], est[mask]


def align_by_index(reference, estimate, estimate_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two series by index with an optional offset on the estimate.

    estimate_offset > 0 means estimate starts later than reference.
    estimate_offset < 0 means reference starts later than estimate.
    """
    ref = to_1d_float(reference)
    est = to_1d_float(estimate)

    if estimate_offset > 0:
        est = est[estimate_offset:]
    elif estimate_offset < 0:
        ref = ref[-estimate_offset:]

    n = min(len(ref), len(est))
    return ref[:n], est[:n]


def compute_metrics(reference, estimate) -> Dict[str, float]:
    """
    Compute standard parity metrics for one signal pair.

    The returned bias is estimate - reference.
    """
    ref, est = valid_pair(reference, estimate)

    if len(ref) == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "corr": np.nan,
        }

    diff = est - ref

    if len(ref) > 1 and np.nanstd(ref) > 0 and np.nanstd(est) > 0:
        corr = float(np.corrcoef(ref, est)[0, 1])
    else:
        corr = np.nan

    return {
        "n": int(len(ref)),
        "bias": float(np.nanmean(diff)),
        "mae": float(np.nanmean(np.abs(diff))),
        "rmse": float(np.sqrt(np.nanmean(diff**2))),
        "corr": corr,
    }


def metric_row(name: str, reference, estimate) -> Dict[str, float | str]:
    """Return one named metric row."""
    row: Dict[str, float | str] = {"comparison": name}
    row.update(compute_metrics(reference, estimate))
    return row


def format_metric_rows(rows: List[Mapping[str, float | str]]) -> str:
    """Format metric rows as a compact plain-text table."""
    if not rows:
        return "(no metrics)"

    header = f"{'comparison':36s} {'n':>6s} {'bias':>10s} {'mae':>10s} {'rmse':>10s} {'corr':>8s}"
    lines = [header, "-" * len(header)]

    for row in rows:
        lines.append(
            f"{str(row['comparison'])[:36]:36s} "
            f"{int(row['n']):6d} "
            f"{float(row['bias']):10.4f} "
            f"{float(row['mae']):10.4f} "
            f"{float(row['rmse']):10.4f} "
            f"{float(row['corr']):8.4f}"
        )

    return "\n".join(lines)
