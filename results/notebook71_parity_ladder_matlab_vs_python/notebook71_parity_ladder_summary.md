# Notebook 71 parity ladder summary

Compared 2666 matched MATLAB/Python rows from `UltraTimTrack_test.mp4` using Python offset 1.
First saved rung outside tolerance: rung 2, TimTrack geofeature measurement.

The current MATLAB `.mat` file is sufficient to locate the first available saved mismatch, but not sufficient to inspect every raw image-processing operation below that mismatch.

Additional MATLAB exports needed only if we want to fix exact low-level parity:
- Per-frame MATLAB filtered ultrasound image after filter_usimage, before thresholding.
- Per-frame MATLAB fascicle/aponeurosis binary masks used for Hough detection.
- MATLAB Hough accumulator/Theta/Rho arrays and houghpeaks output before final alpha selection.
- MATLAB KLT feature point coordinates/status before and after tracking for fascicle and aponeurosis masks.
- Numeric affine matrices for MATLAB warp/awarp, exported as plain 3x3 matrices or affine parameters rather than MCOS objects.
- If Kalman parity is still suspect after upstream parity, per-frame numeric X_minus, X_plus, P, K, Q, and R values are useful; X_plus is already partly available here.

Ladder metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook71_parity_ladder_matlab_vs_python/parity_ladder_metrics.csv`
Rung summary CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook71_parity_ladder_matlab_vs_python/parity_ladder_rung_summary.csv`
Rung RMSE plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook71_parity_ladder_matlab_vs_python/parity_ladder_worst_rmse_by_rung.png`
Time-series checkpoint plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook71_parity_ladder_matlab_vs_python/parity_ladder_time_series_checkpoints.png`