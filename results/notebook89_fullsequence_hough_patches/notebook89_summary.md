# Notebook 89 — full-sequence replay of notebook-only Hough patches

Recomputed or loaded the full baseline minimal geofeature stream for 2666 aligned frames, then replayed selected notebook-only Hough patches on the same per-frame binary fascicle masks.

This notebook answers the next practical question after notebook 88: do the mathematically suspicious Hough changes improve final FL/parity on the full sequence, not just on selected worst frames?

## Full-sequence findings

- The baseline variant reproduces the saved Python raw alpha closely (raw-vs-saved RMSE 0.000000 deg), so the replay is anchored to the strict run.
- The best raw-alpha RMSE against MATLAB geofeature alpha is `baseline_current` at 2.8086 deg, versus 2.8086 deg for baseline.
- The best downstream FL RMSE after the same persistence and Kalman path is `baseline_current` at 2.0116 mm, versus 2.0116 mm for baseline.

## Interpretation

- If the no-radius Hough patches improve full-sequence raw alpha and FL together, that is strong evidence the ellipse/radius correction is the main mathematical anomaly.
- If they help worst-frame raw alpha but hurt or fail to improve full-sequence FL, then the knob is still compensatory and needs a narrower patch or a sequence-aware follow-up.
- Comparing `no_radius_correction` against `angle_profile_localmax_no_radius` also tells us whether the remaining gain comes mostly from removing the correction or from changing peak extraction across rho.

- Raw metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook89_fullsequence_hough_patches/fullsequence_raw_metrics.csv`
- Persistence metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook89_fullsequence_hough_patches/fullsequence_persistence_metrics.csv`
- Downstream metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook89_fullsequence_hough_patches/fullsequence_downstream_metrics.csv`
- Combined summary CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook89_fullsequence_hough_patches/fullsequence_variant_summary.csv`
- Per-frame CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook89_fullsequence_hough_patches/fullsequence_per_frame.csv`
- Summary plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook89_fullsequence_hough_patches/fullsequence_hough_patch_summary.png`
- Trace plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook89_fullsequence_hough_patches/fullsequence_hough_patch_traces.png`
- Baseline cache: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook89_fullsequence_hough_patches/baseline_minimal_geofeatures.pkl`
