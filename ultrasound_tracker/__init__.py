from .preprocessing import load_video, preprocess
from .klt_tracker import KLTTracker
from .hough_detector import HoughDetector
from .frangi_detector import FrangiDetector
from .matlab_aponeurosis import (
    MatlabHoughAponeurosisConfig,
    MatlabHoughAponeurosisDetector,
    adaptive_threshold_matlab_style,
    detect_matlab_hough_aponeuroses,
    fit_apo_matlab_like,
    get_aponeurosis_line_hough_matlab_like,
    line_segment_from_polyfit_1b,
    zero_outside_vertical_cut,
)
from .kalman_fusion import FascicleKalman
from .speckle import (
    SpeckleCoherenceParams,
    SpeckleTracker,
    compute_speckle_confidence_for_frames,
    robust_mad_sigma,
    speckle_pair_confidence,
)
from .optical_flow_dense import DenseFlowTracker
from .utils import plot_results, save_results
from .final_output import (
    aponeurosis_thickness_px,
    final_outputs_from_components,
    final_outputs_from_lines,
    image_depth_to_mm_per_pixel,
)
from . import geometry

__version__ = "1.0.0"
__author__ = "Alexandre Bédard — Oxford NDORMS, supervised by Jack Tu "
