# Notebook 86 — Hough candidate-set shaping before aggregation

Recomputed the baseline TimTrack candidate stream for 2666 aligned frames and cached it at `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook86_hough_candidate_shaping/baseline_geofeatures.pkl`.

This notebook stays upstream of any code patch. It keeps the same raw TimTrack image path, then tests whether reshaping the Hough peak set before the weighted-median alpha can improve parity.

## Candidate-shaping findings

- The baseline identity variant reproduces the saved Python raw alpha exactly (raw-vs-saved RMSE 0.000000 deg), so the notebook is testing the same candidate stream as the strict run.
- The best raw-alpha RMSE against MATLAB geofeature alpha is `baseline_current` at 2.8086 deg, versus 2.8086 deg for the unshaped baseline.
- The best downstream FL RMSE after the same persistence and Kalman path is `top_peak_family_5deg` at 1.5137 mm, versus 2.0116 mm for the baseline.
- The lowest rate of 'good candidate present but raw >5 deg wrong' is `top_peak_family_5deg` at 0.0034, versus 0.0544 for baseline.

## Interpretation

- If candidate-set shaping helps on the same source peaks, that points to family structure and duplicate/weight bias inside the raw Hough candidate stream.
- If shaping helps FL but not raw MATLAB parity, it is still a compensating modeling change rather than a clean parity fix.
- If baseline remains best, the unresolved difference is even earlier: candidate generation and mask construction, not just peak-family shaping.

- Raw metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook86_hough_candidate_shaping/shaping_raw_metrics.csv`
- Persistence metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook86_hough_candidate_shaping/shaping_persistence_metrics.csv`
- Downstream metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook86_hough_candidate_shaping/shaping_downstream_metrics.csv`
- Combined summary CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook86_hough_candidate_shaping/shaping_full_summary.csv`
- Per-frame CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook86_hough_candidate_shaping/shaping_per_frame.csv`
- Summary plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook86_hough_candidate_shaping/shaping_variant_summary.png`
- Trace plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook86_hough_candidate_shaping/shaping_variant_traces.png`
