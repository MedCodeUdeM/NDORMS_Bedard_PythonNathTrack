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
from .ultratimtrack_matlab_2state import (
    MatlabTwoStateKalmanConfig,
    reconstruct_fascicle_from_state,
    run_matlab_2state_kalman,
)
from .matlab_timtrack import (
    alpha_from_saved_peaks,
    extract_saved_peak_arrays,
    reconstruct_saved_geofeature_alpha,
)
from .ultratrack_klt import (
    UltraTrackKLTConfig,
    apply_affine_1b,
    estimate_affine_matlab_coords,
    run_one_step_affine_video,
    run_one_step_affine_sequence,
    tracking_masks_from_geofeature,
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

__version__ = "1.1.0"
__author__ = "Alexandre Bédard — Oxford NDORMS, supervised by Jack Tu "
