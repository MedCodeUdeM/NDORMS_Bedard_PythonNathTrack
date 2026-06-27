# Notebook 70 MATLAB vs Python normal/adaptive Kalman summary

Compared 2666 matched frames from `UltraTimTrack_test.mp4`.
MATLAB rows were aligned to Python/video frames using Python offset 1.
Final variables are matched: MATLAB `FL/fas_length`, `ANG/fas_ang`, `PEN/fas_pen` versus Python `FL_mm`, `ANG_deg`, `PEN_deg`.
Normal and adaptive Python series come from the same Python intermediate detections; only Kalman measurement covariance differs.

Python normal Kalman met the pre-specified mean-equivalence criterion for both primary endpoints.
- FL_mm: bias -2.095 mm (block-bootstrap 95% CI -2.566 to -1.761), RMSE 2.867, CCC 0.955.
- ANG_deg: bias 1.192 deg (block-bootstrap 95% CI 0.996 to 1.444), RMSE 1.599, CCC 0.947.

Python adaptive Kalman met the pre-specified mean-equivalence criterion for both primary endpoints.
- FL_mm: bias -2.171 mm (block-bootstrap 95% CI -2.614 to -1.832), RMSE 2.886, CCC 0.954.
- ANG_deg: bias 1.230 deg (block-bootstrap 95% CI 1.050 to 1.462), RMSE 1.601, CCC 0.947.

Adaptive-minus-normal summary CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook70_matlab_normal_adaptive_kalman_comparison/adaptive_minus_normal_summary.csv`
Adaptive confidence summary CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook70_matlab_normal_adaptive_kalman_comparison/adaptive_confidence_summary.csv`
Length plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook70_matlab_normal_adaptive_kalman_comparison/fascicle_length_matlab_python_normal_adaptive_over_time.png`
Angle plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook70_matlab_normal_adaptive_kalman_comparison/fascicle_angle_matlab_python_normal_adaptive_over_time.png`
Aligned data CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook70_matlab_normal_adaptive_kalman_comparison/aligned_matlab_python_normal_adaptive_outputs.csv`
Statistics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook70_matlab_normal_adaptive_kalman_comparison/matlab_python_normal_adaptive_equivalence_statistics.csv`