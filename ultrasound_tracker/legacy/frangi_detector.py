"""
Frangi vesselness filter for ultrasound fascicle detection.

The Frangi filter (Frangi et al. 1998, "Multiscale vessel enhancement filtering")
detects elongated/tubular structures by analysing the eigenvalues of the
Hessian matrix across multiple Gaussian scales.

For ultrasound fascicles (bright or dark elongated bands), this gives:
  - More complete detection than Hough on low-contrast or noisy frames
  - Better handling of slightly curved fascicles
  - Sensitivity to fascicle *width* via the sigma range
"""

import cv2
import numpy as np
from typing import Tuple

from ultrasound_tracker.legacy.preprocessing import preprocess
from ultrasound_tracker import geometry

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# LOW-LEVEL: HESSIAN EIGENVALUES
# ============================================================================

def _hessian_eigenvalues(image: np.ndarray,
                         sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the two eigenvalues of the scale-normalised 2D Hessian.

    Frangi convention:
        |lam1| <= |lam2|
    """
    image = image.astype(np.float64)

    # Robust normalization: constant image should give zero Hessian.
    image = image - image.min()
    max_val = image.max()

    if max_val <= 1e-10:
        zero = np.zeros_like(image, dtype=np.float64)
        return zero, zero

    image = image / max_val

    s2 = sigma ** 2

    Ixx = gaussian_filter(image, sigma, order=[0, 2]) * s2
    Iyy = gaussian_filter(image, sigma, order=[2, 0]) * s2
    Ixy = gaussian_filter(image, sigma, order=[1, 1]) * s2

    half_trace = (Ixx + Iyy) / 2.0
    discriminant = np.sqrt(
        np.maximum(((Ixx - Iyy) / 2.0) ** 2 + Ixy ** 2, 0.0)
    )

    lambda_a = half_trace - discriminant
    lambda_b = half_trace + discriminant

    swap = np.abs(lambda_a) > np.abs(lambda_b)

    lam1 = np.where(swap, lambda_b, lambda_a)
    lam2 = np.where(swap, lambda_a, lambda_b)

    return lam1, lam2


def _vesselness_at_scale(lam1: np.ndarray, lam2: np.ndarray,
                         alpha: float, beta: float,
                         black_ridges: bool) -> np.ndarray:
    """
    Frangi vesselness measure from eigenvalues at one scale.

    Formula (Frangi 1998):
        V = 0                                              if wrong sign
            exp(−Rb² / 2α²) × (1 − exp(−S² / 2β²))       otherwise

    where
        Rb = |lam1| / |lam2|   ← blobness (small = tubular)
        S  = sqrt(lam1² + lam2²)  ← structure strength (Frobenius norm)
        α  controls sensitivity to blobness
        β  controls background suppression

    Parameters
    ----------
    lam1, lam2   : np.ndarray  eigenvalues (|lam1| ≤ |lam2|)
    alpha        : float       blobness sensitivity (typical 0.5)
    beta         : float       background suppression (typical 5–20)
    black_ridges : bool        True = dark ridges on bright background
                               False = bright ridges on dark background

    Returns
    -------
    V : np.ndarray  float32, values in [0, 1]
    """
    # Sign condition: for bright ridges lam2 < 0; for dark ridges lam2 > 0
    if black_ridges:
        valid = lam2 > 0.0
    else:
        valid = lam2 < 0.0

    # Blobness Rb = |lam1| / |lam2|  (≈0 for pure tubular, ≈1 for blob)
    Rb = np.zeros_like(lam1)
    denom = np.abs(lam2)
    has_denom = valid & (denom > 1e-10)
    Rb[has_denom] = np.abs(lam1[has_denom]) / denom[has_denom]

    # Structure strength S
    S = np.sqrt(lam1 ** 2 + lam2 ** 2)

    # Vesselness
    V = np.where(
        valid,
        np.exp(-Rb ** 2 / (2.0 * alpha ** 2))
        * (1.0 - np.exp(-S ** 2 / (2.0 * beta ** 2))),
        0.0
    )

    return V.astype(np.float32)


# ============================================================================
# PUBLIC FILTER FUNCTION
# ============================================================================

def frangi_filter(image: np.ndarray,
                  sigmas: Tuple[float, ...] = (1, 2, 4),
                  alpha: float = 0.5,
                  beta: float = 15.0,
                  black_ridges: bool = False) -> np.ndarray:
    """
    Multi-scale Frangi vesselness filter.

    Runs the Hessian analysis at each scale in `sigmas` and takes the
    pixel-wise maximum — so fascicles of different apparent widths are
    all captured.

    Parameters
    ----------
    image        : np.ndarray  2D grayscale (uint8 or float)
    sigmas       : tuple       Gaussian scales to scan.
                               Tip: use small values (1–2) for thin fascicles,
                               larger (4–8) for wide bands.
    alpha        : float       Blobness sensitivity. 0.5 is the standard value.
    beta         : float       Background suppression.
                               •  5 → very sensitive, more noise
                               • 15 → balanced (recommended for ultrasound)
                               • 30 → only strong structures
    black_ridges : bool        False → bright fascicles on dark background (default)
                               True  → dark fascicles on bright background

    Returns
    -------
    vesselness : np.ndarray  float32, same shape as image, normalised to [0, 1]

    Raises
    ------
    ImportError if scipy is not installed.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for the Frangi filter.\n"
            "Install with:  pip install scipy"
        )

    img = image.astype(np.float64)
    vesselness = np.zeros(img.shape, dtype=np.float32)

    for sigma in sigmas:
        if sigma <= 0:
            continue

        lam1, lam2 = _hessian_eigenvalues(img, sigma)
        V = _vesselness_at_scale(lam1, lam2, alpha, beta, black_ridges)
        np.maximum(vesselness, V, out=vesselness)

    vmax = vesselness.max()

    # Critical: do not normalize numerical noise into signal.
    if vmax <= 1e-6:
        return np.zeros_like(vesselness, dtype=np.float32)

    vesselness /= vmax

    return vesselness.astype(np.float32)


# ============================================================================
# DETECTOR CLASS  (drop-in compatible with HoughDetector)
# ============================================================================

class FrangiDetector:
    """
    Fascicle detector using the Frangi vesselness filter.

    Pipeline
    --------
    1. Preprocess frame  (CLAHE contrast + Gaussian blur)
    2. Multi-scale Frangi filter  →  vesselness map ∈ [0, 1]
    3. Threshold vesselness       →  binary mask
    4. HoughLinesP on binary mask →  candidate line segments
    5. Filter by fascicle angle range  (angle_min … angle_max)

    Compared to HoughDetector
    -------------------------
    + More robust on low-contrast or noisy ultrasound frames
    + Captures slightly curved or interrupted fascicles better
    + Multi-scale: handles fascicles of different apparent widths
    − Slower (Hessian at each scale)
    − One extra parameter to tune (threshold)

    API is identical to HoughDetector — use as a drop-in replacement:
        lines, angles, lengths = detector.detect(frame)
        angle, length          = detector.estimate(frame)
        vmap                   = detector.get_vesselness_map(frame)

    Parameters
    ----------
    sigmas          : tuple of float
        Gaussian scales for the Hessian.
        Rule of thumb: fascicle apparent half-width in pixels.
        Default (1, 2, 4) covers thin to medium fascicles.
    alpha           : float
        Blobness sensitivity (0.5 recommended).
    beta            : float
        Background suppression (5–20; 15 recommended for ultrasound).
    black_ridges    : bool
        False → bright fascicles on dark background (most ultrasound)
        True  → dark fascicles on bright background
    threshold       : float  ∈ [0, 1]
        Vesselness threshold for binarisation.
        Lower → more sensitive (more false positives).
        Higher → stricter (may miss faint fascicles).
    angle_min       : float  degrees
        Minimum fascicle angle to keep. (absolute value)
    angle_max       : float  degrees
        Maximum fascicle angle to keep. (absolute value)
    hough_threshold : int
        HoughLinesP accumulator threshold.
    min_line_length : int    pixels
        Minimum segment length kept after Hough.
    max_line_gap    : int    pixels
        Maximum collinear gap bridged by HoughLinesP.
    """

    def __init__(self,
                 sigmas: Tuple[float, ...] = (1, 2, 4),
                 alpha: float = 0.5,
                 beta: float = 15.0,
                 black_ridges: bool = False,
                 threshold: float = 0.1,
                 angle_min: float = 10.0,
                 angle_max: float = 40.0,
                 hough_threshold: int = 20,
                 min_line_length: int = 30,
                 max_line_gap: int = 15):

        self.sigmas          = sigmas
        self.alpha           = alpha
        self.beta            = beta
        self.black_ridges    = black_ridges
        self.threshold       = threshold
        self.angle_min       = angle_min
        self.angle_max       = angle_max
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap    = max_line_gap

    # ------------------------------------------------------------------
    def get_vesselness_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the raw Frangi vesselness map — useful for parameter tuning
        and debugging.

        Parameters
        ----------
        frame : np.ndarray  2D grayscale

        Returns
        -------
        vesselness : np.ndarray  float32, same shape as frame, in [0, 1]
        """
        enhanced = preprocess(frame, contrast=True, blur=True)
        return frangi_filter(enhanced, self.sigmas,
                             self.alpha, self.beta, self.black_ridges)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray):
        """
        Detect fascicle line segments in a frame.

        Parameters
        ----------
        frame : np.ndarray  2D grayscale frame (uint8)

        Returns
        -------
        lines   : np.ndarray (N, 4) float32  [x1, y1, x2, y2]  OR  None
        angles  : np.ndarray (N,)   float32  absolute angle (°) OR  None
        lengths : np.ndarray (N,)   float32  length (px)        OR  None
        """
        # ── Step 1: preprocess ────────────────────────────────────────
        enhanced = preprocess(frame, contrast=True, blur=True)

        # ── Step 2: Frangi vesselness ─────────────────────────────────
        vesselness = frangi_filter(
            enhanced, self.sigmas, self.alpha, self.beta, self.black_ridges
        )

        # ── Step 3: threshold → uint8 binary mask ────────────────────
        binary = (vesselness >= self.threshold).astype(np.uint8) * 255

        # ── Step 4: HoughLinesP on vesselness mask ────────────────────
        raw_lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if raw_lines is None:
            return None, None, None

        lines = raw_lines[:, 0, :].astype(np.float32)   # (N, 4)

        # ── Step 5: angles & lengths via geometry module ──────────────
        signed_angles = geometry.line_angles_batch(lines, degrees=True)
        abs_angles    = np.abs(signed_angles)    # fascicle angle is symmetric
        lengths       = geometry.line_lengths_batch(lines)

        # ── Step 6: angular filter (fascicle range) ───────────────────
        # We filter on absolute angles so both orientations are captured.
        filtered_lines, filtered_abs_angles = geometry.filter_lines_by_angle(
            lines, abs_angles, self.angle_min, self.angle_max
        )
        filtered_lengths = lengths[
            (abs_angles >= self.angle_min) & (abs_angles <= self.angle_max)
        ]

        if len(filtered_lines) == 0:
            return None, None, None

        return filtered_lines, filtered_abs_angles, filtered_lengths

    # ------------------------------------------------------------------
    def estimate(self, frame: np.ndarray):
        """
        Return a single (angle, length) estimate for Kalman filter fusion.

        Mirrors HoughDetector.estimate() — drop-in compatible.

        Returns
        -------
        (angle_median, length_median) : (float, float)
        OR (None, None) if no fascicle detected.
        """
        lines, angles, lengths = self.detect(frame)
        if angles is None or len(angles) == 0:
            return None, None
        return float(np.median(angles)), float(np.median(lengths))
