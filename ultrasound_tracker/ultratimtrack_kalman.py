"""
UltraTimTrack-style geometric Kalman fusion.

This module is intended to be closer to MATLAB UltraTimTrack than the simple
angle/length Kalman filter in kalman_fusion.py.

State vector
------------
x = [
    x_sup_attachment,
    y_sup_attachment,
    fascicle_angle_deg,
    fascicle_length_px,
]

This follows the documented UltraTimTrack idea that the fascicle state is based on:
    1. superficial aponeurosis horizontal intersection point
    2. superficial aponeurosis vertical intersection point
    3. fascicle angle with the horizontal
    4. fascicle length

Usage concept
-------------
KLT / UltraTrack-like output:
    used as the prediction.

TimTrack output:
    used as the independent drift-free measurement.

Kalman output:
    low-noise, drift-corrected geometric state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .geometry import (
    line_angle_from_array,
    line_length_from_array,
    normalize_angle,
    pennation_angle,
    compute_fascicle_geometry,
)


IDX_X_SUP = 0
IDX_Y_SUP = 1
IDX_ANGLE = 2
IDX_LENGTH = 3
STATE_SIZE = 4


def _as_float_array(values, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Convert input to float32 numpy array and optionally validate shape.
    """
    arr = np.asarray(values, dtype=np.float32)

    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")

    return arr


def _is_valid_state(state: Optional[np.ndarray]) -> bool:
    """
    Check whether a state vector exists and has finite values.
    """
    if state is None:
        return False

    state = np.asarray(state)

    return state.shape == (STATE_SIZE,) and np.all(np.isfinite(state))


def _angle_difference_deg(measured: float, predicted: float) -> float:
    """
    Smallest signed angular difference in degrees.

    This avoids large jumps when angles cross a wrapping boundary.
    """
    diff = (measured - predicted + 180.0) % 360.0 - 180.0
    return float(diff)


def segment_from_state(state: np.ndarray) -> np.ndarray:
    """
    Convert UltraTimTrack state to a fascicle segment.

    Parameters
    ----------
    state : np.ndarray shape (4,)
        [x_sup, y_sup, fascicle_angle_deg, fascicle_length_px]

    Returns
    -------
    segment : np.ndarray shape (4,)
        [x_sup, y_sup, x_deep, y_deep]

    Notes
    -----
    The project geometry convention used earlier was:
        angle = +45° for a line going up-right
        angle = -45° for a line going down-right

    Therefore:
        x_deep = x_sup + L*cos(theta)
        y_deep = y_sup - L*sin(theta)
    """
    state = _as_float_array(state, shape=(STATE_SIZE,))

    x_sup = float(state[IDX_X_SUP])
    y_sup = float(state[IDX_Y_SUP])
    angle_deg = float(state[IDX_ANGLE])
    length_px = float(state[IDX_LENGTH])

    theta = np.deg2rad(angle_deg)

    x_deep = x_sup + length_px * np.cos(theta)
    y_deep = y_sup - length_px * np.sin(theta)

    return np.array(
        [x_sup, y_sup, x_deep, y_deep],
        dtype=np.float32,
    )


def state_from_segment(
    fascicle_segment: np.ndarray,
    ensure_superficial_to_deep: bool = True,
) -> np.ndarray:
    """
    Convert a fascicle segment to UltraTimTrack state.

    Parameters
    ----------
    fascicle_segment : np.ndarray shape (4,)
        [x_sup, y_sup, x_deep, y_deep] ideally.
    ensure_superficial_to_deep : bool
        If True, enforce that the first point is the more superficial point
        by y-coordinate, i.e. smaller y first.

    Returns
    -------
    state : np.ndarray shape (4,)
        [x_sup, y_sup, fascicle_angle_deg, fascicle_length_px]
    """
    segment = _as_float_array(fascicle_segment, shape=(4,)).copy()

    if ensure_superficial_to_deep:
        y1 = segment[1]
        y2 = segment[3]

        if y2 < y1:
            segment = np.array(
                [segment[2], segment[3], segment[0], segment[1]],
                dtype=np.float32,
            )

    x_sup = float(segment[0])
    y_sup = float(segment[1])

    angle_deg = normalize_angle(
        line_angle_from_array(segment),
        degrees=True,
    )

    length_px = line_length_from_array(segment)

    return np.array(
        [x_sup, y_sup, angle_deg, length_px],
        dtype=np.float32,
    )


def state_from_geometry(
    superficial_apo_line: Optional[np.ndarray] = None,
    deep_apo_line: Optional[np.ndarray] = None,
    fascicle_line: Optional[np.ndarray] = None,
    fascicle_segment: Optional[np.ndarray] = None,
    ensure_superficial_to_deep: bool = True,
) -> np.ndarray:
    """
    Convert tracked geometry to UltraTimTrack state.

    You can provide either:
        A) fascicle_segment directly
    or:
        B) superficial_apo_line + deep_apo_line + fascicle_line

    Parameters
    ----------
    superficial_apo_line : np.ndarray shape (4,), optional
    deep_apo_line : np.ndarray shape (4,), optional
    fascicle_line : np.ndarray shape (4,), optional
    fascicle_segment : np.ndarray shape (4,), optional
    ensure_superficial_to_deep : bool

    Returns
    -------
    state : np.ndarray shape (4,)
    """
    if fascicle_segment is not None:
        return state_from_segment(
            fascicle_segment,
            ensure_superficial_to_deep=ensure_superficial_to_deep,
        )

    if (
        superficial_apo_line is None
        or deep_apo_line is None
        or fascicle_line is None
    ):
        raise ValueError(
            "Provide either fascicle_segment, or "
            "superficial_apo_line + deep_apo_line + fascicle_line."
        )

    features = compute_fascicle_geometry(
        superficial_apo_line=superficial_apo_line,
        deep_apo_line=deep_apo_line,
        fascicle_line=fascicle_line,
    )

    return state_from_segment(
        features["fascicle_segment_between_apos"],
        ensure_superficial_to_deep=ensure_superficial_to_deep,
    )


def geometry_from_state(
    state: np.ndarray,
    deep_apo_line: Optional[np.ndarray] = None,
) -> Dict:
    """
    Convert UltraTimTrack state back to geometry features.

    Parameters
    ----------
    state : np.ndarray shape (4,)
        [x_sup, y_sup, fascicle_angle_deg, fascicle_length_px]
    deep_apo_line : np.ndarray shape (4,), optional
        If provided, deep aponeurosis angle and pennation angle are computed.

    Returns
    -------
    features : dict
    """
    state = _as_float_array(state, shape=(STATE_SIZE,))

    segment = segment_from_state(state)

    sup_attachment = segment[:2].copy()
    deep_attachment = segment[2:].copy()

    fascicle_angle_deg = float(state[IDX_ANGLE])
    fascicle_length_px = float(state[IDX_LENGTH])

    features = {
        "state": state.copy(),
        "sup_attachment": sup_attachment,
        "deep_attachment": deep_attachment,
        "fascicle_segment_between_apos": segment,
        "fascicle_length_px": fascicle_length_px,
        "fascicle_angle_deg": fascicle_angle_deg,
        "deep_apo_angle_deg": np.nan,
        "pennation_angle_deg": np.nan,
    }

    if deep_apo_line is not None:
        deep_apo_angle_deg = normalize_angle(
            line_angle_from_array(deep_apo_line),
            degrees=True,
        )

        pennation_angle_deg = pennation_angle(
            fascicle_angle_deg,
            deep_apo_angle_deg,
            degrees=True,
        )

        features["deep_apo_angle_deg"] = float(deep_apo_angle_deg)
        features["pennation_angle_deg"] = float(pennation_angle_deg)

    return features


@dataclass
class UltraTimTrackKalmanConfig:
    """
    Configuration for UltraTimTrack-style geometric Kalman fusion.

    process_noise_covariance_parameter:
        Equivalent conceptually to the MATLAB GUI parameter.
        Smaller -> trust KLT/UltraTrack prediction more.
        Larger -> allow stronger correction by TimTrack.

    measurement_noise_*:
        Diagonal measurement noise covariance for TimTrack measurements.
        The MATLAB GUI explicitly exposes x-coordinate superficial attachment
        measurement covariance. Here we expose all 4 state components so we can
        tune them separately in Python.
    """

    process_noise_covariance_parameter: float = 0.01

    measurement_noise_x_sup: float = 100.0
    measurement_noise_y_sup: float = 25.0
    measurement_noise_angle: float = 4.0
    measurement_noise_length: float = 100.0

    initial_covariance: float = 1_000.0

    process_weight_x_sup: float = 1.0
    process_weight_y_sup: float = 1.0
    process_weight_angle: float = 1.0
    process_weight_length: float = 1.0

    min_covariance: float = 1e-9


class UltraTimTrackGeometricKalman:
    """
    UltraTimTrack-style geometric Kalman filter.

    This class fuses:
        prediction = KLT / UltraTrack-like geometry
        measurement = TimTrack independent geometry

    State:
        [x_sup_attachment, y_sup_attachment, fascicle_angle_deg, fascicle_length_px]
    """

    def __init__(self, config: Optional[UltraTimTrackKalmanConfig] = None):
        self.config = config if config is not None else UltraTimTrackKalmanConfig()

        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None

        self.initialized: bool = False

        self.last_prior_state: Optional[np.ndarray] = None
        self.last_posterior_state: Optional[np.ndarray] = None
        self.last_kalman_gain: Optional[np.ndarray] = None
        self.last_innovation: Optional[np.ndarray] = None

        self.Q = self._make_process_covariance()
        self.R = self._make_measurement_covariance()

    def _make_process_covariance(self) -> np.ndarray:
        c = float(self.config.process_noise_covariance_parameter)

        weights = np.array(
            [
                self.config.process_weight_x_sup,
                self.config.process_weight_y_sup,
                self.config.process_weight_angle,
                self.config.process_weight_length,
            ],
            dtype=np.float32,
        )

        return np.diag(c * weights).astype(np.float32)

    def _make_measurement_covariance(self) -> np.ndarray:
        values = np.array(
            [
                self.config.measurement_noise_x_sup,
                self.config.measurement_noise_y_sup,
                self.config.measurement_noise_angle,
                self.config.measurement_noise_length,
            ],
            dtype=np.float32,
        )

        return np.diag(values).astype(np.float32)

    def reset(self) -> None:
        self.x = None
        self.P = None
        self.initialized = False

        self.last_prior_state = None
        self.last_posterior_state = None
        self.last_kalman_gain = None
        self.last_innovation = None

    def initialize(self, state: np.ndarray, covariance: Optional[float] = None) -> None:
        """
        Initialize the filter with a valid geometric state.
        """
        state = _as_float_array(state, shape=(STATE_SIZE,))

        if not np.all(np.isfinite(state)):
            raise ValueError("Cannot initialize Kalman filter with non-finite state.")

        self.x = state.copy()

        init_cov = (
            float(covariance)
            if covariance is not None
            else float(self.config.initial_covariance)
        )

        self.P = np.eye(STATE_SIZE, dtype=np.float32) * init_cov

        self.x[IDX_ANGLE] = normalize_angle(
            self.x[IDX_ANGLE],
            degrees=True,
        )

        self.initialized = True
        self.last_prior_state = self.x.copy()
        self.last_posterior_state = self.x.copy()

    def predict(self, klt_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prediction step.

        If klt_state is provided, it becomes the predicted state.
        If not, the previous posterior state is carried forward.

        This is intentionally UltraTrack-like:
            KLT provides a full sequential geometry prediction.
        """
        if not self.initialized:
            if _is_valid_state(klt_state):
                self.initialize(klt_state)
                return self.x.copy()

            raise RuntimeError("Kalman filter is not initialized.")

        if klt_state is not None:
            klt_state = _as_float_array(klt_state, shape=(STATE_SIZE,))

            # Use KLT prediction where finite, otherwise keep previous state.
            finite = np.isfinite(klt_state)

            x_pred = self.x.copy()
            x_pred[finite] = klt_state[finite]
        else:
            x_pred = self.x.copy()

        x_pred[IDX_ANGLE] = normalize_angle(
            x_pred[IDX_ANGLE],
            degrees=True,
        )

        self.x = x_pred
        self.P = self.P + self.Q

        self.last_prior_state = self.x.copy()

        return self.x.copy()

    def update(self, timtrack_state: Optional[np.ndarray]) -> np.ndarray:
        """
        Measurement update using TimTrack geometry.

        Partial updates are allowed: if some state components are NaN,
        only finite components are used.
        """
        if timtrack_state is None:
            if not self.initialized:
                raise RuntimeError("Kalman filter is not initialized.")
            return self.x.copy()

        timtrack_state = _as_float_array(timtrack_state, shape=(STATE_SIZE,))

        if not self.initialized:
            if _is_valid_state(timtrack_state):
                self.initialize(timtrack_state)
                return self.x.copy()

            raise RuntimeError("Cannot initialize from invalid TimTrack state.")

        measurement_mask = np.isfinite(timtrack_state)

        if not np.any(measurement_mask):
            return self.x.copy()

        indices = np.flatnonzero(measurement_mask)

        H = np.zeros((len(indices), STATE_SIZE), dtype=np.float32)

        for row, idx in enumerate(indices):
            H[row, idx] = 1.0

        z = timtrack_state[indices].astype(np.float32)
        R_sub = self.R[np.ix_(indices, indices)]

        y = z - H @ self.x

        # Angle innovation must be wrapped.
        for row, idx in enumerate(indices):
            if idx == IDX_ANGLE:
                y[row] = _angle_difference_deg(
                    measured=float(z[row]),
                    predicted=float(self.x[IDX_ANGLE]),
                )

        S = H @ self.P @ H.T + R_sub
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        self.x[IDX_ANGLE] = normalize_angle(
            self.x[IDX_ANGLE],
            degrees=True,
        )

        I = np.eye(STATE_SIZE, dtype=np.float32)

        # Joseph stabilized covariance update.
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_sub @ K.T

        # Prevent numerical collapse to exactly zero.
        diag = np.diag(self.P).copy()
        diag = np.maximum(diag, self.config.min_covariance)

        self.P = self.P.copy()

        for i in range(STATE_SIZE):
            self.P[i, i] = diag[i]

        self.last_kalman_gain = K.copy()
        self.last_innovation = y.copy()
        self.last_posterior_state = self.x.copy()

        return self.x.copy()

    def step(
        self,
        klt_state: Optional[np.ndarray],
        timtrack_state: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        One full Kalman step:
            1. predict from KLT / UltraTrack-like state
            2. update from TimTrack independent state
        """
        if not self.initialized:
            if _is_valid_state(timtrack_state):
                self.initialize(timtrack_state)
                return self.x.copy()

            if _is_valid_state(klt_state):
                self.initialize(klt_state)
                return self.x.copy()

            raise RuntimeError("Cannot initialize Kalman filter: no valid state.")

        self.predict(klt_state=klt_state)
        self.update(timtrack_state=timtrack_state)

        return self.x.copy()

    def get_state(self) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized.")

        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized.")

        return self.P.copy()

    def get_uncertainty(self) -> np.ndarray:
        """
        Return diagonal covariance:
            [x_sup_unc, y_sup_unc, angle_unc, length_unc]
        """
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized.")

        return np.diag(self.P).copy()

    def get_geometry(self, deep_apo_line: Optional[np.ndarray] = None) -> Dict:
        """
        Convert current filtered state to geometry dict.
        """
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized.")

        return geometry_from_state(
            self.x,
            deep_apo_line=deep_apo_line,
        )

    def get_last_gain_diagonal(self) -> Optional[np.ndarray]:
        """
        Return diagonal-like gain values for diagnostic plotting.

        If the last update used all 4 states, this returns np.diag(K).
        If the last update was partial, returns a length-4 array with NaNs
        for states that were not measured.
        """
        if self.last_kalman_gain is None:
            return None

        gain_diag = np.full(STATE_SIZE, np.nan, dtype=np.float32)

        K = self.last_kalman_gain

        rows = min(K.shape[0], K.shape[1])

        for i in range(rows):
            gain_diag[i] = K[i, i]

        return gain_diag


def make_state_sequence_from_arrays(
    fascicle_segments: np.ndarray,
) -> np.ndarray:
    """
    Convert an array of fascicle segments to UltraTimTrack states.

    Parameters
    ----------
    fascicle_segments : np.ndarray shape (N, 4)

    Returns
    -------
    states : np.ndarray shape (N, 4)
    """
    fascicle_segments = np.asarray(fascicle_segments, dtype=np.float32)

    states = np.full(
        (len(fascicle_segments), STATE_SIZE),
        np.nan,
        dtype=np.float32,
    )

    for i, segment in enumerate(fascicle_segments):
        if np.all(np.isfinite(segment)):
            states[i] = state_from_segment(segment)

    return states


def run_geometric_ultratimtrack_fusion(
    timtrack_segments: np.ndarray,
    klt_segments: np.ndarray,
    timtrack_deep_apo_lines: Optional[np.ndarray] = None,
    config: Optional[UltraTimTrackKalmanConfig] = None,
) -> Dict:
    """
    Convenience function to run geometric UltraTimTrack-like fusion on arrays.

    Parameters
    ----------
    timtrack_segments : np.ndarray shape (N, 4)
        Independent TimTrack fascicle segments.
    klt_segments : np.ndarray shape (N, 4)
        Sequential KLT/UltraTrack-like fascicle segments.
    timtrack_deep_apo_lines : np.ndarray shape (N, 4), optional
        Used only to compute pennation angle.
    config : UltraTimTrackKalmanConfig, optional

    Returns
    -------
    results : dict
    """
    timtrack_states = make_state_sequence_from_arrays(timtrack_segments)
    klt_states = make_state_sequence_from_arrays(klt_segments)

    n = len(timtrack_states)

    filt = UltraTimTrackGeometricKalman(config=config)

    filtered_states = np.full((n, STATE_SIZE), np.nan, dtype=np.float32)
    filtered_segments = np.full((n, 4), np.nan, dtype=np.float32)
    uncertainties = np.full((n, STATE_SIZE), np.nan, dtype=np.float32)

    length_px = np.full(n, np.nan, dtype=np.float32)
    fascicle_angle_deg = np.full(n, np.nan, dtype=np.float32)
    deep_apo_angle_deg = np.full(n, np.nan, dtype=np.float32)
    pennation_angle_deg = np.full(n, np.nan, dtype=np.float32)

    success = np.zeros(n, dtype=bool)
    errors = np.array([""] * n, dtype=object)

    for i in range(n):
        tim_state = timtrack_states[i]
        klt_state = klt_states[i]

        if not _is_valid_state(tim_state):
            tim_state = None

        if not _is_valid_state(klt_state):
            klt_state = None

        try:
            state = filt.step(
                klt_state=klt_state,
                timtrack_state=tim_state,
            )

            deep_apo_line = None

            if timtrack_deep_apo_lines is not None:
                candidate_deep = timtrack_deep_apo_lines[i]

                if np.all(np.isfinite(candidate_deep)):
                    deep_apo_line = candidate_deep

            geom_features = filt.get_geometry(
                deep_apo_line=deep_apo_line,
            )

            filtered_states[i] = state
            filtered_segments[i] = geom_features["fascicle_segment_between_apos"]
            uncertainties[i] = filt.get_uncertainty()

            length_px[i] = geom_features["fascicle_length_px"]
            fascicle_angle_deg[i] = geom_features["fascicle_angle_deg"]
            deep_apo_angle_deg[i] = geom_features["deep_apo_angle_deg"]
            pennation_angle_deg[i] = geom_features["pennation_angle_deg"]

            success[i] = True

        except Exception as exc:
            errors[i] = str(exc)

    return {
        "success": success,
        "error": errors,
        "filtered_states": filtered_states,
        "filtered_segments": filtered_segments,
        "uncertainties": uncertainties,
        "fascicle_length_px": length_px,
        "fascicle_angle_deg": fascicle_angle_deg,
        "deep_apo_angle_deg": deep_apo_angle_deg,
        "pennation_angle_deg": pennation_angle_deg,
    }