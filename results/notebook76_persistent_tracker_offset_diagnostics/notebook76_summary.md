# Notebook 76 — persistent-tracker offset diagnosis

Aligned 2666 MATLAB/Python samples with Python offset 1.

## Main findings

- The raw tracker offset is already present at frame 0 in the selected seed. Relative to MATLAB raw frame 0, the Python seed is +18.60 px on superficial x, -18.60 px on deep x, and -0.92 deg on angle.
- Over time, raw KLT shape is close after removing the mean bias: x_sup bias +15.97 px with demeaned RMSE 2.08 px; angle bias -1.49 deg with demeaned RMSE 0.58 deg.
- The forward x-state offset is consistent with the current Kalman x update anchoring to the initial x measurement. In a diagnostic variant that uses the current frame's x measurement instead, forward x bias drops from +18.68 px to +1.38 px.
- The angle offset changes sign at the TimTrack measurement stage: raw KLT angle bias is -1.49 deg, selected TimTrack alpha bias is +1.20 deg, and forward alpha bias is +1.01 deg.
- The final smoothed alpha remains mostly a constant offset rather than a shape mismatch: bias +1.03 deg, demeaned RMSE 0.54 deg.

## Interpretation

- There are two offset sources, and they happen at different stages.
- First, the autonomous seed already starts from a different fascicle segment than MATLAB, so the persistent KLT tracker inherits a near-constant spatial shift from frame 0.
- Second, the two-state Kalman x branch appears to preserve or amplify x bias because it is effectively tied to the initial x measurement, while the alpha branch inherits the positive TimTrack alpha baseline.
- That means the curves are genuinely very close in shape; the remaining difference is mostly stage-wise baseline offset, not runaway tracking error.

- Stage summary CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook76_persistent_tracker_offset_diagnostics/stage_offset_summary.csv`
- Seed/frame-0 CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook76_persistent_tracker_offset_diagnostics/seed_frame0_offset.csv`
- X-measurement variant CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook76_persistent_tracker_offset_diagnostics/x_measurement_variant_summary.csv`
- Difference plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook76_persistent_tracker_offset_diagnostics/offset_stage_differences.png`
- Frame-0 seed overlay: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook76_persistent_tracker_offset_diagnostics/frame0_seed_overlay.png`
