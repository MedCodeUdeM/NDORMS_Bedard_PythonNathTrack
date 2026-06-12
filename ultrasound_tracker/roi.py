"""
ROI selection utilities for ultrasound fascicle tracking.

ROI convention:
    roi = (x, y, w, h)

where:
    x, y = top-left corner
    w, h = width and height
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import cv2
import numpy as np


ROI = Tuple[int, int, int, int]


def ensure_uint8_image(frame: np.ndarray) -> np.ndarray:
    """
    Convert an image to uint8 for OpenCV display.
    Accepts grayscale or BGR/RGB-like images.
    """
    if frame is None:
        raise ValueError("frame is None")

    img = frame.copy()

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    img -= img.min()

    max_val = img.max()
    if max_val > 0:
        img = img / max_val * 255.0

    return img.astype(np.uint8)


def extract_roi(frame: np.ndarray, roi: ROI) -> np.ndarray:
    """
    Extract a rectangular ROI from an image.

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    roi : tuple
        (x, y, w, h)

    Returns
    -------
    cropped : np.ndarray
    """
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def draw_roi(
    frame: np.ndarray,
    roi: ROI,
    color: Tuple[int, int, int] = (0, 255, 0),
    label: Optional[str] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw one ROI on an image.

    Parameters
    ----------
    frame : np.ndarray
        Grayscale or BGR image.
    roi : tuple
        (x, y, w, h)
    color : tuple
        BGR color.
    label : str or None
        Optional text label.
    thickness : int

    Returns
    -------
    vis : np.ndarray
        Image with ROI overlay.
    """
    img = ensure_uint8_image(frame)

    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    x, y, w, h = roi
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

    if label is not None:
        cv2.putText(
            vis,
            label,
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis


def draw_rois(frame: np.ndarray, rois: Dict[str, ROI]) -> np.ndarray:
    """
    Draw multiple named ROIs.

    Expected names:
        - superficial
        - deep
        - fascicle, optional
    """
    img = ensure_uint8_image(frame)

    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    colors = {
        "superficial": (255, 0, 0),   # blue in BGR
        "deep": (0, 255, 0),          # green
        "fascicle": (0, 255, 255),    # yellow
    }

    labels = {
        "superficial": "Superficial aponeurosis",
        "deep": "Deep aponeurosis",
        "fascicle": "Fascicle ROI",
    }

    for name, roi in rois.items():
        color = colors.get(name, (0, 0, 255))
        label = labels.get(name, name)
        vis = draw_roi(vis, roi, color=color, label=label)

    return vis


def select_single_roi_cv2(
    frame: np.ndarray,
    window_name: str = "Select ROI",
    instruction: str = "Select ROI, then press ENTER or SPACE",
) -> ROI:
    """
    Select one ROI using OpenCV's interactive selector.

    Controls:
        - draw box with mouse
        - press ENTER or SPACE to confirm
        - press C to cancel
    """
    img = ensure_uint8_image(frame)

    if img.ndim == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display = img.copy()

    cv2.putText(
        display,
        instruction,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    roi = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi

    if w == 0 or h == 0:
        raise ValueError("ROI selection cancelled or empty ROI selected.")

    return int(x), int(y), int(w), int(h)


def select_aponeurosis_rois_cv2(frame: np.ndarray) -> Dict[str, ROI]:
    """
    Select superficial and deep aponeurosis ROIs on the first frame.

    Returns
    -------
    rois : dict
        {
            "superficial": (x, y, w, h),
            "deep": (x, y, w, h)
        }
    """
    superficial = select_single_roi_cv2(
        frame,
        window_name="Select superficial aponeurosis ROI",
        instruction="Select SUPERFICIAL aponeurosis ROI, then press ENTER",
    )

    vis = draw_roi(
        frame,
        superficial,
        color=(255, 0, 0),
        label="Superficial aponeurosis",
    )

    deep = select_single_roi_cv2(
        vis,
        window_name="Select deep aponeurosis ROI",
        instruction="Select DEEP aponeurosis ROI, then press ENTER",
    )

    return {
        "superficial": superficial,
        "deep": deep,
    }


def select_all_rois_cv2(
    frame: np.ndarray,
    include_fascicle_roi: bool = True,
) -> Dict[str, ROI]:
    """
    Select superficial aponeurosis, deep aponeurosis, and optionally fascicle ROI.

    Returns
    -------
    rois : dict
        {
            "superficial": (x, y, w, h),
            "deep": (x, y, w, h),
            "fascicle": (x, y, w, h), optional
        }
    """
    rois = select_aponeurosis_rois_cv2(frame)

    if include_fascicle_roi:
        vis = draw_rois(frame, rois)

        fascicle = select_single_roi_cv2(
            vis,
            window_name="Select fascicle ROI",
            instruction="Select FASCICLE ROI, then press ENTER",
        )

        rois["fascicle"] = fascicle

    return rois


def save_rois(rois: Dict[str, ROI], path: Union[str, Path]) -> None:
    """
    Save ROI dictionary to JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        name: [int(v) for v in roi]
        for name, roi in rois.items()
    }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=4)


def load_rois(path: Union[str, Path]) -> Dict[str, ROI]:
    """
    Load ROI dictionary from JSON.
    """
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    return {
        name: tuple(int(v) for v in roi)
        for name, roi in data.items()
    }