# Phantom Ground-Truth Validation

## Data Audit
- Video: `/Users/grosbedou/PycharmProjects/NDORMS/data/raw/june29_3.mp4`
- Video opened: `True`
- Frames/FPS/size: `642` frames, `25` fps, `1024x768` px
- Strict runner NPZ: `/Users/grosbedou/PycharmProjects/NDORMS/results/strict_ultratimtrack_runs/june29_3/june29_3_strict_results.npz`
- Estimate sources: `final_kalman_midpoint, final_kalman_end_midpoint, klt_prior_midpoint, fixed_kalman_midpoint, klt_tracker_median_cumulative`
- mm_per_pixel: `0.06601562500000001`
- Ground truth: `synthetic_linear_cumulative_x15mm`
- GT columns used: `{'frame': 'video_frame', 'time_s': 'frame/fps', 'x': 'synthetic_total_x_mm', 'y': None, 'scalar': None}`
- GT axis type: `x`
- GT source note: synthetic linear cumulative ramp from the supplied total displacement; replace with an actuator/encoder trace if the plate motion was not constant-speed over the full video.
- Failure threshold: `0.5` mm

## Coordinate And Unit Contract
- Image origin is the top-left pixel. x increases to the right; y increases downward.
- Strict line segments are stored in one-based MATLAB-style pixel coordinates, but displacement is computed relative to frame 0, so the one-pixel origin offset cancels.
- Axial displacement corresponds to image y positive downward. Lateral displacement corresponds to image x positive rightward.
- The current scalar `mm_per_pixel` is depth/height-derived axial spacing. Vector or lateral validation needs independent lateral spacing unless square pixels are confirmed.
- `fascicle_segments` are final Kalman output; `klt_prior_segments` are cumulative/persistent KLT prior segments; `fixed_fascicle_segments` is the normal fixed-R Kalman comparator when present.

## Metrics
| source | comparison | axis | lag_frames | estimate_sign | n | mae_mm | rmse_mm | bias_mm | error_sd_mm | max_abs_error_mm | r2 | pearson_r | endpoint_error_mm | drift_slope_mm_per_s | failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_kalman_midpoint | raw | x/lateral | 0 | 1 | 642 | 15.72 | 17.16 | 15.72 | 6.895 | 27.37 | -14.66 | 0.9889 | 24.56 | 0.9023 | 0.9502 |
| final_kalman_midpoint | best_aligned | x/lateral | -10 | 1 | 632 | 15.35 | 16.8 | 15.34 | 6.858 | 27.13 | -14.49 | 0.989 | 24.52 | 0.912 | 0.9494 |
| final_kalman_end_midpoint | raw | x/lateral | 0 | 1 | 642 | 15.24 | 16.63 | 15.24 | 6.662 | 26.29 | -13.71 | 0.9867 | 23.63 | 0.8656 | 0.9502 |
| final_kalman_end_midpoint | best_aligned | x/lateral | -10 | 1 | 632 | 14.88 | 16.28 | 14.87 | 6.63 | 26.05 | -13.55 | 0.9867 | 23.54 | 0.8754 | 0.9494 |
| klt_prior_midpoint | raw | x/lateral | 0 | 1 | 642 | 2.244 | 3 | -2.165 | 2.079 | 6.527 | 0.5215 | 0.9637 | -6.527 | -0.2659 | 0.6994 |
| klt_prior_midpoint | best_aligned | x/lateral | 10 | 1 | 632 | 2.132 | 2.858 | -1.964 | 2.078 | 6.293 | 0.5519 | 0.9632 | -6.293 | -0.2707 | 0.6899 |
| fixed_kalman_midpoint | raw | x/lateral | 0 | 1 | 642 | 17.6 | 19.18 | 17.6 | 7.636 | 30.13 | -18.56 | 0.9875 | 27.22 | 0.9978 | 0.9502 |
| fixed_kalman_midpoint | best_aligned | x/lateral | -10 | 1 | 632 | 17.22 | 18.81 | 17.21 | 7.599 | 29.89 | -18.42 | 0.9876 | 27.18 | 1.009 | 0.9494 |
| klt_tracker_median_cumulative | raw | x/lateral | 0 | 1 | 642 | 2.077 | 2.756 | -1.979 | 1.921 | 6.02 | 0.596 | 0.9702 | -6.02 | -0.2445 | 0.7009 |
| klt_tracker_median_cumulative | best_aligned | x/lateral | 10 | 1 | 632 | 1.964 | 2.614 | -1.775 | 1.921 | 5.786 | 0.6251 | 0.9698 | -5.786 | -0.249 | 0.6835 |

Interpret the `raw` rows as the mathematically direct comparison. The `best_aligned` rows are diagnostic: a non-zero lag or sign flip may reveal a frame offset or coordinate convention error and should not be hidden in a paper result.

## Orthogonal Motion / Bounce Audit
| status | reason | source | estimate_axis | orthogonal_axis | n | x_end_mm | x_range_mm | y_end_mm | y_range_mm | magnitude_end_mm | magnitude_range_mm | failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| not_validated | missing_ground_truth | final_kalman_midpoint | x | y | 642 | 39.56 | 40.54 | -0.7279 | 0.8638 | 39.57 | 40.55 | 0 |
| not_validated | missing_ground_truth | final_kalman_end_midpoint | x | y | 642 | 38.63 | 39.46 | -0.9348 | 1.083 | 38.64 | 39.47 | 0 |
| not_validated | missing_ground_truth | klt_prior_midpoint | x | y | 642 | 8.473 | 8.68 | -1.045 | 1.342 | 8.538 | 8.736 | 0 |
| not_validated | missing_ground_truth | fixed_kalman_midpoint | x | y | 642 | 42.22 | 43.3 | -0.7278 | 0.8645 | 42.22 | 43.31 | 0 |
| not_validated | missing_ground_truth | klt_tracker_median_cumulative | x | y | 642 | 8.98 | 9.229 | -1.013 | 1.27 | 9.037 | 9.274 | 0 |

This table is not an additional ground-truth validation. For the lateral phantom trial, it checks whether the tracker reports axial y motion even though the imposed motion was x-only.

## Reviewer Interpretation
- Raw RMSE exceeds the configured `0.5` mm threshold; this is not yet strong phantom evidence.
- Check whether any best-aligned result requires sign flip or lag. If yes, fix/report the coordinate or synchronization cause before claiming validation.
- For strain, small displacement bias can be amplified because strain divides a length change by a baseline length; validate displacement and length first.

## Plots
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/estimated_lateral_x_by_stage.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/estimated_axial_bounce_by_stage.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/estimated_xy_trajectory_by_stage.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_midpoint_estimated_vs_ground_truth.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_midpoint_error_over_time.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_midpoint_scatter_identity.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_midpoint_bland_altman.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_midpoint_cumulative_drift.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_end_midpoint_estimated_vs_ground_truth.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_end_midpoint_error_over_time.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_end_midpoint_scatter_identity.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_end_midpoint_bland_altman.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/final_kalman_end_midpoint_cumulative_drift.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_prior_midpoint_estimated_vs_ground_truth.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_prior_midpoint_error_over_time.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_prior_midpoint_scatter_identity.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_prior_midpoint_bland_altman.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_prior_midpoint_cumulative_drift.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/fixed_kalman_midpoint_estimated_vs_ground_truth.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/fixed_kalman_midpoint_error_over_time.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/fixed_kalman_midpoint_scatter_identity.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/fixed_kalman_midpoint_bland_altman.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/fixed_kalman_midpoint_cumulative_drift.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_tracker_median_cumulative_estimated_vs_ground_truth.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_tracker_median_cumulative_error_over_time.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_tracker_median_cumulative_scatter_identity.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_tracker_median_cumulative_bland_altman.png`
- `/Users/grosbedou/PycharmProjects/NDORMS/results/phantom_ground_truth_validation/june29_3/plots/klt_tracker_median_cumulative_cumulative_drift.png`

## Paper-Strength Validation Target
- Pre-register the phantom displacement waveform, units, axis convention, frame synchronization, and tolerance.
- Report raw, not only best-aligned, MAE/RMSE/bias/limits-of-agreement over multiple amplitudes and speeds.
- Include failure rate and show overlays at worst-error frames.
- Demonstrate low drift over long sequences and compare against at least KLT-only and fixed-R Kalman baselines.
- Phantom validation does not replace in vivo validation because tissue deformation, out-of-plane motion, probe pressure, anisotropic speckle decorrelation, and manual ROI variability are different failure modes.

## Machine-Readable Audit
```json
{
  "video": {
    "path": "/Users/grosbedou/PycharmProjects/NDORMS/data/raw/june29_3.mp4",
    "opened": true,
    "frame_count": 642,
    "fps": 25.0,
    "width_px": 1024,
    "height_px": 768
  },
  "strict_outputs": {
    "npz": "/Users/grosbedou/PycharmProjects/NDORMS/results/strict_ultratimtrack_runs/june29_3/june29_3_strict_results.npz",
    "csv": "/Users/grosbedou/PycharmProjects/NDORMS/results/strict_ultratimtrack_runs/june29_3/june29_3_strict_FL_PEN_ANG.csv",
    "metadata": "/Users/grosbedou/PycharmProjects/NDORMS/results/strict_ultratimtrack_runs/june29_3/june29_3_strict_metadata.json"
  },
  "strict_audit": {
    "strict_npz": "/Users/grosbedou/PycharmProjects/NDORMS/results/strict_ultratimtrack_runs/june29_3/june29_3_strict_results.npz",
    "strict_metadata": "/Users/grosbedou/PycharmProjects/NDORMS/results/strict_ultratimtrack_runs/june29_3/june29_3_strict_metadata.json",
    "mm_per_pixel": 0.06601562500000001,
    "mm_per_pixel_interpretation": "scalar axial spacing from image depth / image height unless overridden upstream",
    "frames_in_output": 642,
    "available_arrays": [
      "frame",
      "time_s",
      "sup_apo_lines",
      "deep_apo_lines",
      "klt_prior_segments",
      "klt_affine_ok",
      "klt_affine_matrices",
      "klt_points_count",
      "klt_tracked_count",
      "klt_inlier_count",
      "klt_tracker_redetected",
      "klt_tracker_found_fraction",
      "klt_tracker_state_points",
      "klt_tracked_old_points",
      "klt_tracked_new_points",
      "fascicle_segments",
      "fascicle_end_segments",
      "ANG_deg",
      "PEN_deg",
      "FL_px",
      "timtrack_alpha_deg",
      "raw_timtrack_alpha_deg",
      "hough_baseline_alpha_deg",
      "hough_localmax_fallback_used",
      "hough_localmax_fallback_mass_below_10deg",
      "hough_localmax_fallback_gap_to_lower_deg",
      "hough_peak_source",
      "fascicle_candidate_selected_idx",
      "fascicle_candidate_raw_rejected",
      "fascicle_candidate_selection_reason",
      "selected_seed_segment",
      "selected_seed_alpha_deg",
      "mm_per_pixel",
      "apo_measurement_states",
      "apo_accepted_measurement_states",
      "apo_gating_r_scale",
      "apo_rejected_endpoints",
      "apo_soft_downweighted_endpoints",
      "apo_line_rejected",
      "apo_line_soft_downweighted",
      "apo_gating_reasons",
      "apo_consecutive_rejections",
      "X_plus",
      "X_smooth",
      "X_minus",
      "fas_p",
      "fas_p_smooth",
      "fas_p_minus",
      "forward_X_plus",
      "forward_fas_p",
      "kalman_gain",
      "smoother_gain",
      "measurement_R_diag",
      "measurement_r_scale",
      "measurement_r_scale_theta",
      "measurement_r_scale_length",
      "predicted_segments",
      "previous_corrected_segments",
      "prediction_used_affine",
      "forward_fascicle_segments",
      "forward_fascicle_end_segments",
      "forward_ANG_deg",
      "forward_PEN_deg",
      "forward_FL_px",
      "forward_FL_mm",
      "FL_mm",
      "fixed_fascicle_segments",
      "fixed_fascicle_end_segments",
      "fixed_X_plus",
      "fixed_X_smooth",
      "fixed_X_minus",
      "fixed_fas_p",
      "fixed_fas_p_smooth",
      "fixed_fas_p_minus",
      "fixed_forward_X_plus",
      "fixed_forward_fas_p",
      "fixed_kalman_gain",
      "fixed_smoother_gain",
      "fixed_measurement_R_diag",
      "fixed_measurement_r_scale",
      "fixed_measurement_r_scale_theta",
      "fixed_measurement_r_scale_length",
      "fixed_predicted_segments",
      "fixed_previous_corrected_segments",
      "fixed_prediction_used_affine",
      "fixed_forward_fascicle_segments",
      "fixed_forward_fascicle_end_segments",
      "fixed_forward_ANG_deg",
      "fixed_forward_PEN_deg",
      "fixed_forward_FL_px",
      "fixed_forward_FL_mm",
      "fixed_ANG_deg",
      "delta_vs_fixed_ANG_deg",
      "fixed_PEN_deg",
      "delta_vs_fixed_PEN_deg",
      "fixed_FL_px",
      "delta_vs_fixed_FL_px",
      "fixed_FL_mm",
      "delta_vs_fixed_FL_mm",
      "speckle_zncc",
      "speckle_confidence",
      "forward_backward_error",
      "valid_patch_fraction",
      "n_valid_patches",
      "n_total_patches",
      "motion_consistency",
      "motion_spread_px",
      "n_motion_points",
      "feature_reliability",
      "feature_peak_score",
      "feature_peak_count_score",
      "feature_mask_score",
      "feature_mask_density",
      "geometry_stability",
      "geometry_alpha_score",
      "geometry_pennation_score",
      "geometry_length_score",
      "geometry_angle_jump_deg",
      "geometry_angle_jump_score",
      "geometry_length_jump_px",
      "geometry_length_jump_score",
      "confidence_theta",
      "confidence_length",
      "combined_confidence",
      "r_scale",
      "r_scale_theta",
      "r_scale_length",
      "detection_success",
      "R_t_x_variance",
      "R_t_alpha_variance",
      "R_t_length_variance",
      "R_t_theta_variance",
      "kalman_measurement_r_scale",
      "kalman_measurement_r_scale_theta",
      "kalman_measurement_r_scale_length"
    ],
    "estimate_sources": [
      "final_kalman_midpoint",
      "final_kalman_end_midpoint",
      "klt_prior_midpoint",
      "fixed_kalman_midpoint",
      "klt_tracker_median_cumulative"
    ]
  },
  "ground_truth_path": "synthetic_linear_cumulative_x15mm",
  "ground_truth_kind": "synthetic_linear_cumulative",
  "synthetic_ground_truth": {
    "total_x_mm": 15.0,
    "total_y_mm": null,
    "start_frame": null,
    "end_frame": null
  },
  "data_phantom_files": [],
  "ground_truth_loaded": true,
  "axis_requested": "x",
  "failure_threshold_mm": 0.5
}
```