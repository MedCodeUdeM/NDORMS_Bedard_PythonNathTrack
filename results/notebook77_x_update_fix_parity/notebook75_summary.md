# Updated MATLAB-vs-Python comparison

The updated fixed-R Python run is not uniformly closer to MATLAB on both primary outputs.

- Same video: `/Users/grosbedou/PycharmProjects/NDORMS/data/raw/UltraTimTrack_test.mp4`
- Same ROI JSON: `/Users/grosbedou/PycharmProjects/NDORMS/data/rois/UltraTimTrack_test_rois.json` with `{'superficial': [20, 18, 685, 54], 'deep': [16, 302, 689, 76], 'fascicle': [23, 78, 679, 218]}`
- MATLAB reference: `/Users/grosbedou/PycharmProjects/NDORMS/data/matlab/slow_low_01_DOWN_tracked_Q=001.mat`
- Matched rows: 2666; Python frame offset: 1

## Fixed-R (MATLAB-like) before/after

- Fascicle length: RMSE 1.9954 -> 2.0117 mm (+0.8%); CCC 0.9782 -> 0.9778.
- Fascicle angle: RMSE 1.1661 -> 1.1661 deg (-0.0%); CCC 0.9712 -> 0.9712.
- Pennation angle: RMSE 1.0946 -> 1.0946 deg (-0.0%); CCC 0.9711 -> 0.9711.

## Updated tracker state

- Affine success: 100.0% of frames.
- Tracker redetections (including initialization): 0.
- Mean tracker found fraction: 1.000.
- Fixed Kalman predictions using the saved affine: 100.0% of frames.

- Final agreement table: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook77_x_update_fix_parity/agreement_metrics.csv`
- Before/after table: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook77_x_update_fix_parity/before_after_improvement.csv`
- Tracker/Kalman table: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook77_x_update_fix_parity/tracker_kalman_state_metrics.csv`
- Final output plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook77_x_update_fix_parity/matlab_python_before_after.png`
- Internal-state plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook77_x_update_fix_parity/tracker_kalman_state_comparison.png`
- ROI verification image: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook77_x_update_fix_parity/same_input_roi_check.png`
