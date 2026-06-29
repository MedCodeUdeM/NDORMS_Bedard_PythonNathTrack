"""
Aponeurosis detection for ultrasound tracking.

This module mirrors the MATLAB TimTrack / UltraTimTrack logic as closely
as possible for aponeurosis detection, with emphasis on the default
Frangi-based method.

Main usage:
    detector = AponeurosisDetector(method="frangi")
    result = detector.detect(roi_image, kind="superficial")

ROI convention:
    image is a cropped ROI that already contains only the superficial
    or deep aponeurosis region.
"""

from typing import Dict, Tuple, Optional, Union

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import frangi
from skimage.measure import label, regionprops


ROI = Tuple[int, int, int, int]


def normalize_to_float01(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    img -= img.min()
    max_val = img.max()
    if max_val > 0:
        img /= max_val
    return img


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    img = normalize_to_float01(image)
    return (img * 255).astype(np.uint8)


def adaptive_binary(image: np.ndarray,
                    sensitivity: float = 0.5,
                    block_size: int = 51) -> np.ndarray:
    """
    Approximation of MATLAB:
        imbinarize(image, 'adaptive', 'sensitivity', s)

    Returns uint8 binary mask {0,1}.
    """
    if block_size % 2 == 0:
        block_size += 1

    img8 = normalize_to_uint8(image)

    # Simple practical mapping:
    # sensitivity 0.5 -> C = 0
    # higher sensitivity -> slightly more permissive
    C = int(round((0.5 - sensitivity) * 20))

    binary = cv2.adaptiveThreshold(
        img8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    return (binary > 0).astype(np.uint8)


def make_apox(width: int,
              apomargin: int = 20,
              napo: int = 10) -> np.ndarray:
    """
    MATLAB-like aponeurosis sample x locations.
    """
    if width <= 2 * apomargin:
        apomargin = max(1, width // 10)

    apox = np.round(
        np.linspace(apomargin, width - apomargin - 1, napo)
    ).astype(int)

    apox = np.clip(apox, 0, width - 1)
    return apox


def select_best_component(binary_mask: np.ndarray,
                          intensity_image: np.ndarray,
                          maxlengthratio: float = 0.9):
    """
    MATLAB-like selection:
      1. keep 2 largest by major axis length
      2. if close call, choose one with highest mean intensity
      3. else choose the longest
    """
    lbl = label(binary_mask > 0, connectivity=2)
    props = regionprops(lbl, intensity_image=intensity_image)

    if len(props) == 0:
        empty = np.zeros_like(binary_mask, dtype=np.uint8)
        return empty, empty, None, []

    props_sorted = sorted(
        props,
        key=lambda p: p.major_axis_length,
        reverse=True
    )

    top2 = props_sorted[:2]
    top2_labels = [p.label for p in top2]
    top2_mask = np.isin(lbl, top2_labels).astype(np.uint8)

    if len(top2) == 1:
        best = top2[0]
    else:
        lengths = [p.major_axis_length for p in top2]
        ratio = min(lengths) / max(lengths) if max(lengths) > 0 else 0.0

        if ratio > maxlengthratio:
            best = max(top2, key=lambda p: p.mean_intensity)
        else:
            best = max(top2, key=lambda p: p.major_axis_length)

    best_mask = (lbl == best.label).astype(np.uint8)
    return best_mask, top2_mask, best, top2


def extract_boundary_super(mask: np.ndarray,
                           apox: np.ndarray,
                           fillgap: int = 50):
    """
    Python port of apo_func logic for the superficial case.

    Returns
    -------
    raw_boundary : array of y values
    gap_filled_mask : binary mask after vertical gap filling
    """
    mask = binary_fill_holes(mask > 0)
    mask = mask.astype(bool)

    n, m = mask.shape

    deep_edge = np.full(len(apox), np.nan, dtype=np.float32)
    super_edge = np.full(len(apox), np.nan, dtype=np.float32)

    # first pass: find top and bottom edges
    for i, x in enumerate(apox):
        ys = np.where(mask[:, x])[0]
        if ys.size > 0:
            deep_edge[i] = ys.max()
            super_edge[i] = ys.min()

    # fill vertical gaps if the gap is not too large
    gap_filled = mask.copy()

    for i, x in enumerate(apox):
        if np.isfinite(super_edge[i]) and np.isfinite(deep_edge[i]):
            y0 = int(super_edge[i])
            y1 = int(deep_edge[i])

            sl = gap_filled[y0:y1 + 1, x]
            if sl.size > 0 and np.sum(sl) > (len(sl) - fillgap):
                gap_filled[y0:y1 + 1, x] = True

    # choose first black pixel below the object
    raw_boundary = np.full(len(apox), np.nan, dtype=np.float32)

    for i, x in enumerate(apox):
        ys = np.where(gap_filled[:, x])[0]
        if ys.size == 0:
            continue

        top_y = ys.min()
        candidates = np.where((~gap_filled[:, x]) & (np.arange(n) > top_y))[0]

        if candidates.size > 0:
            raw_boundary[i] = candidates.min()

    return raw_boundary, gap_filled.astype(np.uint8)


def extract_aponeurosis_vector(mask: np.ndarray,
                               apox: np.ndarray,
                               kind: str,
                               sigma: float = 10.0,
                               fillgap: int = 50):
    """
    MATLAB-like vector extraction:
      - superficial: apo_func(super_obj) then - 0.5*sigma
      - deep: n - apo_func(flipud(deep_obj)) then + 0.5*sigma
    """
    n, _ = mask.shape

    if kind.lower() == "superficial":
        raw_boundary, gap_filled = extract_boundary_super(mask, apox, fillgap=fillgap)
        vector = raw_boundary - 0.5 * sigma
        return raw_boundary, vector, gap_filled

    elif kind.lower() == "deep":
        flipped = np.flipud(mask)
        raw_flipped, gap_filled_flipped = extract_boundary_super(flipped, apox, fillgap=fillgap)

        raw_boundary = np.full_like(raw_flipped, np.nan)
        valid = np.isfinite(raw_flipped)
        raw_boundary[valid] = (n - 1) - raw_flipped[valid]

        vector = raw_boundary + 0.5 * sigma
        gap_filled = np.flipud(gap_filled_flipped)

        return raw_boundary, vector, gap_filled

    else:
        raise ValueError("kind must be 'superficial' or 'deep'")


def fit_line_constrained(x: np.ndarray,
                         y: np.ndarray,
                         maxangle: float = 0.5,
                         fit_method: str = "enforce_maxangle") -> Optional[np.ndarray]:
    """
    MATLAB-like fit_apo for first-order fit.
    """
    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return None

    coef = np.polyfit(x, y, 1).astype(np.float32)
    slope, intercept = coef

    fit_angle = -np.degrees(np.arctan2(slope, 1.0))

    if fit_method == "enforce_maxangle" and fit_angle > maxangle:
        slope = -np.tan(np.deg2rad(maxangle)).astype(np.float32)
        intercept = np.mean(y - slope * x).astype(np.float32)
        coef = np.array([slope, intercept], dtype=np.float32)

    return coef


def line_from_coef(coef: np.ndarray, width: int) -> Optional[np.ndarray]:
    if coef is None:
        return None

    slope, intercept = coef
    x1 = 0
    x2 = width - 1
    y1 = slope * x1 + intercept
    y2 = slope * x2 + intercept

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def y_from_line_segment(line: np.ndarray, x: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = line.astype(np.float32)
    dx = x2 - x1

    out = np.full_like(x.astype(np.float32), np.nan)

    if abs(dx) < 1e-8:
        return out

    m = (y2 - y1) / dx
    out = y1 + m * (x - x1)
    return out.astype(np.float32)


class AponeurosisDetector:
    def __init__(self,
                 method: str = "frangi",
                 sigma: float = 10.0,
                 th: float = 0.5,
                 filtfac: float = 1.0,
                 maxlengthratio: float = 0.9,
                 frangi_scale_range: Tuple[int, int] = (18, 20),
                 frangi_scale_ratio: int = 1,
                 frangi_black_ridges: bool = False,
                 apomargin: int = 20,
                 napo: int = 10,
                 fillgap: int = 50,
                 fit_method: str = "enforce_maxangle",
                 maxangle: float = 0.5,
                 adaptive_block_size: int = 51,
                 hough_detector=None):
        self.method = method.lower()

        self.sigma = sigma
        self.th = th
        self.filtfac = filtfac
        self.maxlengthratio = maxlengthratio

        self.frangi_scale_range = frangi_scale_range
        self.frangi_scale_ratio = frangi_scale_ratio
        self.frangi_black_ridges = frangi_black_ridges

        self.apomargin = apomargin
        self.napo = napo
        self.fillgap = fillgap

        self.fit_method = fit_method
        self.maxangle = maxangle
        self.adaptive_block_size = adaptive_block_size

        self.hough_detector = hough_detector

    def detect(self, image: np.ndarray, kind: str = "superficial") -> Dict:
        if self.method == "frangi":
            return self._detect_frangi(image, kind=kind)
        elif self.method == "hough":
            return self._detect_hough(image, kind=kind)
        else:
            raise ValueError("method must be 'frangi' or 'hough'")

    def _detect_frangi(self, image: np.ndarray, kind: str = "superficial") -> Dict:
        img = normalize_to_float01(image)
        h, w = img.shape

        apox = make_apox(w, apomargin=self.apomargin, napo=self.napo)

        sigmas = list(range(
            self.frangi_scale_range[0],
            self.frangi_scale_range[1] + 1,
            self.frangi_scale_ratio
        ))

        frangi_img = frangi(
            img,
            sigmas=sigmas,
            black_ridges=self.frangi_black_ridges
        )
        frangi_img = normalize_to_float01(frangi_img)

        raw_binary = adaptive_binary(img,
                                     sensitivity=self.th,
                                     block_size=self.adaptive_block_size)

        combined = raw_binary.astype(np.float32) * (frangi_img ** self.filtfac)

        gaussian = cv2.GaussianBlur(
            combined.astype(np.float32),
            (0, 0),
            sigmaX=self.sigma,
            sigmaY=self.sigma
        )

        gaussian_binary = adaptive_binary(
            gaussian,
            sensitivity=self.th,
            block_size=self.adaptive_block_size
        )

        selected_mask, top2_mask, best_prop, top2_props = select_best_component(
            gaussian_binary,
            intensity_image=frangi_img,
            maxlengthratio=self.maxlengthratio
        )

        raw_boundary, vector_y, gap_filled_mask = extract_aponeurosis_vector(
            selected_mask,
            apox=apox,
            kind=kind,
            sigma=self.sigma,
            fillgap=self.fillgap
        )

        coef = fit_line_constrained(
            apox.astype(np.float32),
            vector_y.astype(np.float32),
            maxangle=self.maxangle,
            fit_method=self.fit_method
        )

        line_local = line_from_coef(coef, w)
        vector_points = np.column_stack([apox, vector_y]).astype(np.float32)

        return {
            "method": "frangi",
            "kind": kind,
            "image": img,
            "frangi_img": frangi_img,
            "raw_binary": raw_binary,
            "combined": combined,
            "gaussian": gaussian,
            "gaussian_binary": gaussian_binary,
            "top2_mask": top2_mask,
            "selected_mask": selected_mask,
            "gap_filled_mask": gap_filled_mask,
            "apox": apox,
            "raw_boundary": raw_boundary,
            "vector_y": vector_y,
            "vector_points": vector_points,
            "fit_coef": coef,
            "line_local": line_local,
            "best_prop": best_prop,
            "top2_props": top2_props,
        }

    def _detect_hough(self, image: np.ndarray, kind: str = "superficial") -> Dict:
        if self.hough_detector is None:
            raise ValueError("A hough_detector instance must be provided for method='hough'.")

        img = normalize_to_float01(image)
        h, w = img.shape

        lines, angles, lengths = self.hough_detector.detect(normalize_to_uint8(img))
        apox = make_apox(w, apomargin=self.apomargin, napo=self.napo)

        if lines is None or len(lines) == 0:
            return {
                "method": "hough",
                "kind": kind,
                "image": img,
                "line_local": None,
                "vector_points": None,
                "apox": apox,
                "vector_y": None,
                "fit_coef": None,
            }

        idx = int(np.argmax(lengths))
        best_line = lines[idx].astype(np.float32)

        vector_y = y_from_line_segment(best_line, apox.astype(np.float32))
        coef = fit_line_constrained(
            apox.astype(np.float32),
            vector_y.astype(np.float32),
            maxangle=self.maxangle,
            fit_method=self.fit_method
        )
        line_local = line_from_coef(coef, w)
        vector_points = np.column_stack([apox, vector_y]).astype(np.float32)

        return {
            "method": "hough",
            "kind": kind,
            "image": img,
            "lines_local": lines,
            "angles": angles,
            "lengths": lengths,
            "line_local": line_local if line_local is not None else best_line,
            "vector_points": vector_points,
            "apox": apox,
            "vector_y": vector_y,
            "fit_coef": coef,
        }