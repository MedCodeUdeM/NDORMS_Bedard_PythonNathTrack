# Notebook 87 — per-frame Hough internals on representative worst frames

Audited 16 representative worst aligned frames chosen from the full-sequence raw-alpha error ranking (minimum spacing 20 frames).

This notebook compares exact saved MATLAB Hough peaks/weights/selected median against the current Python per-frame Hough output on the same frames.

## Ground-truth boundary

- Exact MATLAB saved peak angles, weights, peak lines, and selected weighted-median alpha are available in the local UTT export.
- Exact MATLAB full fascicle Hough accumulator/profile is not present in the saved exports, and no local MATLAB/Octave runtime is available here to regenerate it.
- So the per-frame accumulator heatmap/profile panels are exact Python-side internals with MATLAB peaks overlaid, not a full MATLAB-vs-Python accumulator matrix diff.

## Key findings

- On these representative worst frames, MATLAB alpha is reconstructed from the saved MATLAB peaks with max absolute error 0.000000 deg, so the MATLAB peak/weight reference is self-consistent.
- Python still contains a peak within 2 deg of MATLAB on 13/16 representative worst frames.
- Yet the raw Python alpha is still >5 deg wrong on 13/16 frames even though such a close candidate exists.
- Bias-mode counts: lower-angle overweight 0, higher-angle overweight 13, candidate absent 3.

## Interpretation

- When a MATLAB-like Python peak is present but the Python weighted median still misses badly, the implementation of weighted median itself is not the main issue.
- The failure is that the Python candidate-family weight distribution crosses 0.5 on the wrong side before the MATLAB-like peak can dominate.
- Frames where the MATLAB-like peak is absent point further upstream to candidate generation / mask construction rather than aggregation logic.

- Selected-frame summary CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook87_hough_internals_worst_frames/selected_worst_frame_summary.csv`
- Bias mode counts CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook87_hough_internals_worst_frames/bias_mode_counts.csv`
- Python Hough profile CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook87_hough_internals_worst_frames/python_hough_profiles_long.csv`
- Representative frame list: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook87_hough_internals_worst_frames/representative_frames.json`
- Overview plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook87_hough_internals_worst_frames/hough_internals_overview.png`
- Per-frame debug panels: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook87_hough_internals_worst_frames/frame_debug`
