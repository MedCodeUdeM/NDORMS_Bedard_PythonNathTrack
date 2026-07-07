# Updated MATLAB-vs-Python comparison

The updated fixed-R Python run is closer to MATLAB on both primary outputs.

- Same video: `/Users/grosbedou/PycharmProjects/NDORMS/data/raw/UltraTimTrack_test.mp4`
- Same ROI JSON: `/Users/grosbedou/PycharmProjects/NDORMS/data/rois/UltraTimTrack_test_rois.json` with `{'superficial': [20, 18, 685, 54], 'deep': [16, 302, 689, 76], 'fascicle': [23, 78, 679, 218]}`
- MATLAB reference: `/Users/grosbedou/PycharmProjects/NDORMS/data/matlab/slow_low_01_DOWN_tracked_Q=001.mat`
- Matched rows: 2666; Python frame offset: 1

## Fixed-R (MATLAB-like) before/after

- Fascicle length: RMSE 2.5494 -> 1.9954 mm (-21.7%); CCC 0.9638 -> 0.9782.
- Fascicle angle: RMSE 1.3937 -> 1.1661 deg (-16.3%); CCC 0.9586 -> 0.9712.
- Pennation angle: RMSE 1.3308 -> 1.0946 deg (-17.7%); CCC 0.9572 -> 0.9711.

## Updated tracker state

- Affine success: 100.0% of frames.
- Tracker redetections (including initialization): 0.
- Mean tracker found fraction: 1.000.
- Fixed Kalman predictions using the saved affine: 100.0% of frames.

- Final agreement table: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook75_persistent_tracker_matlab_python/agreement_metrics.csv`
- Before/after table: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook75_persistent_tracker_matlab_python/before_after_improvement.csv`
- Tracker/Kalman table: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook75_persistent_tracker_matlab_python/tracker_kalman_state_metrics.csv`
- Final output plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook75_persistent_tracker_matlab_python/matlab_python_before_after.png`
- Internal-state plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook75_persistent_tracker_matlab_python/tracker_kalman_state_comparison.png`
- ROI verification image: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook75_persistent_tracker_matlab_python/same_input_roi_check.png`
