# Notebook 85 — raw Hough aggregation alternatives

Recomputed the full baseline TimTrack candidate stream for 2666 aligned frames, then held those candidate sets fixed while swapping only the raw Hough aggregation rule.

This notebook does not patch production code. It asks whether a different raw aggregation rule improves parity on the same candidate sets, or merely compensates for upstream candidate bias.

## Raw aggregation findings

- The current weighted-median rule reproduces the saved Python raw alpha exactly (raw-vs-saved RMSE 0.000000 deg), so it is the correct baseline.
- On the same candidate sets, the best raw-alpha RMSE against MATLAB geofeature alpha is `weighted_median_current` at 2.8086 deg, versus 2.8086 deg for the current weighted median.
- The best downstream FL RMSE after the same persistence and Kalman path is `weighted_mean` at 1.2649 mm, versus 2.0116 mm for the current weighted median.

## Interpretation

- If a non-MATLAB aggregation rule improves raw alpha and downstream FL on the same candidate sets, that does not prove MATLAB's aggregation is wrong.
- It means the current Python candidate distribution is biased enough that a different aggregator can partially compensate for upstream errors.
- So any future code patch here would be a modeling change, not a pure parity fix, unless we also show the candidate-generation side is already correct.
- This notebook therefore helps rank ideas, but it is not by itself a green light to change the production aggregation rule.

- Raw metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook85_raw_hough_aggregation_alternatives/aggregation_raw_metrics.csv`
- Persistence metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook85_raw_hough_aggregation_alternatives/aggregation_persistence_metrics.csv`
- Downstream metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook85_raw_hough_aggregation_alternatives/aggregation_downstream_metrics.csv`
- Combined summary CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook85_raw_hough_aggregation_alternatives/aggregation_full_summary.csv`
- Per-frame CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook85_raw_hough_aggregation_alternatives/aggregation_per_frame.csv`
- Summary plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook85_raw_hough_aggregation_alternatives/aggregation_alternative_summary.png`
- Trace plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook85_raw_hough_aggregation_alternatives/aggregation_alternative_traces.png`
