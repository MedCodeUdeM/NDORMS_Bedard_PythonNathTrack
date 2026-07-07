# Notebook 78 — seed offset attack

Aligned 2666 MATLAB/Python samples with Python offset 1.

The notebook evaluates seed changes using notebook 77's saved persistent-KLT affine matrices, TimTrack alpha, and aponeurosis lines. This is valid because the tracker points/affines are independent of the chosen fascicle seed; the seed only determines the propagated segment geometry.

## Main findings

- Sweeping the frame-0 seed alpha over 16.5–19.1 deg does not improve final FL over the current autonomous seed. The best FL RMSE in that sweep is 2.0117 mm at alpha 17.6 deg; the current seed is 17.6 deg with FL RMSE 2.0117 mm.
- The oracle MATLAB raw frame-0 seed does not help final FL. It produces FL RMSE 2.0943 mm, worse than baseline.
- Uniform x translation of the frame-0 seed changes final FL much more than changing seed alpha. In the tested range, FL RMSE improves from 2.0117 mm at dx=0 to 1.9192 mm at dx=120 px, while ANG RMSE stays essentially flat.
- Matching the MATLAB frame-0 x_sup offset directly is not beneficial: the left-shift variant that matches MATLAB x_sup has FL RMSE 2.0273 mm, worse than baseline.

## Interpretation

- The observed raw seed-vs-MATLAB angle offset is not the lever that controls final FL parity here.
- Final FL is much more sensitive to how the seed anchors x through the propagated prior than to small seed-angle changes.
- Because even the oracle MATLAB raw seed does not improve FL, the remaining FL difference is unlikely to be solved by copying MATLAB's frame-0 seed geometry alone.
- The next target should be the x/length anchoring model downstream of the seed, not just autonomous seed-angle selection.

- Alpha sweep CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook78_seed_offset_attack/seed_alpha_sweep_metrics.csv`
- X-shift sweep CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook78_seed_offset_attack/seed_x_translation_sweep_metrics.csv`
- Named variants CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook78_seed_offset_attack/named_seed_variant_metrics.csv`
- Sweep plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook78_seed_offset_attack/seed_offset_sweeps_fl_rmse.png`
