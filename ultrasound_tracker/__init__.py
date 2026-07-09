"""Public API for the current strict UltraTimTrack pipeline.

Legacy prototypes are available from :mod:`ultrasound_tracker.legacy`, but they
are intentionally not re-exported here.  This keeps ``import ultrasound_tracker``
focused on the supported MATLAB-compatible tracking path.
"""

from . import geometry, roi
from .matlab_aponeurosis import make_matlab_apox
from .matlab_timtrack import (
    detect_timtrack_geofeature_from_image,
    fascicle_segment_from_geofeature,
    run_timtrack_geofeatures_from_video,
)
from .speckle_confidence import (
    SpeckleConfidenceConfig,
    anisotropic_confidence_to_r_scales,
    combine_anisotropic_confidence_metrics,
    combine_confidence_metrics,
    adapt_anisotropic_measurement_covariance,
    compute_feature_detection_reliability,
    compute_geometry_stability,
    compute_motion_consistency,
    compute_speckle_coherence,
    confidence_to_r_scale,
)
from .strict_fascicle_seed import (
    FascicleSeedScoringConfig,
    cluster_seed_candidates,
    extract_fascicle_seed_candidates,
    select_autonomous_fascicle_seed,
)
from .ultratrack_klt import (
    UltraTrackKLTConfig,
    propagate_cumulative_affines,
    read_gray_frames,
    run_persistent_affine_video,
    run_one_step_affine_video,
)
from .ultratimtrack_aponeurosis import run_matlab_aponeurosis_state_video
from .ultratimtrack_matlab_2state import (
    MatlabTwoStateKalmanConfig,
    run_matlab_2state_kalman,
)

__all__ = [
    "FascicleSeedScoringConfig",
    "MatlabTwoStateKalmanConfig",
    "SpeckleConfidenceConfig",
    "UltraTrackKLTConfig",
    "adapt_anisotropic_measurement_covariance",
    "anisotropic_confidence_to_r_scales",
    "cluster_seed_candidates",
    "combine_anisotropic_confidence_metrics",
    "combine_confidence_metrics",
    "compute_feature_detection_reliability",
    "compute_geometry_stability",
    "compute_motion_consistency",
    "compute_speckle_coherence",
    "confidence_to_r_scale",
    "detect_timtrack_geofeature_from_image",
    "extract_fascicle_seed_candidates",
    "fascicle_segment_from_geofeature",
    "geometry",
    "make_matlab_apox",
    "propagate_cumulative_affines",
    "read_gray_frames",
    "roi",
    "run_matlab_2state_kalman",
    "run_matlab_aponeurosis_state_video",
    "run_persistent_affine_video",
    "run_one_step_affine_video",
    "run_timtrack_geofeatures_from_video",
    "select_autonomous_fascicle_seed",
]

__version__ = "5.0.0"
