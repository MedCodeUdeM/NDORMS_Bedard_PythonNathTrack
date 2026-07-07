# Notebook 79 — x/length anchoring audit

Aligned 2666 MATLAB/Python samples with Python offset 1.

This notebook targets the x/length anchor model directly while reusing notebook 77's saved KLT prior, affines, TimTrack alpha, and aponeurosis lines.

## Main findings

- Sweeping the x measurement variance changes final FL only trivially. Across the tested range 10 to 1000, FL RMSE only moves from 2.0105 to 2.0119 mm, while forward-state x bias stays near 1.36–1.43 px.
- Sweeping the constant fixed superficial y anchor changes final FL much more. In the tested range, FL RMSE improves from 2.0117 mm at dy=0 to 1.9174 mm at dy=60 px.
- That apparent FL improvement is not a faithful parity improvement: at the same dy=60 px point, final-end endpoint geometry gets much worse. `fas_x_end` x_sup RMSE grows from 5.95 px to 123.69 px, and x_deep RMSE grows from 25.80 px to 144.63 px.
- Using MATLAB's own frame-0 fixed-y anchor does not help. The MATLAB-frame0-y variant gives FL RMSE 2.0141 mm, not better than baseline.

## Audit cross-check

- The crucial unresolved difference is still final-end geometry (`fas_x_end` / `fas_y_end`), especially the x coordinates. Earlier parity audits already showed those endpoint RMSEs were large, and notebook 79 confirms that simple anchor tweaks only trade FL against endpoint correctness.
- I do not see another must-use MATLAB function being ignored that obviously outranks this issue. The main contract fields for the current mismatch are already in play: `X_plus`, `X_minus`, `fas_x`, `fas_y`, `fas_x_end`, `fas_y_end`, and `alpha`.
- One lower-priority contract field that is still not explicitly mirrored as a named output is MATLAB `Fascicle.A`. It is worth keeping on the checklist, but nothing in these anchor sweeps suggests it is the dominant blocker for current FL parity.

## Interpretation

- The remaining FL gap is not primarily a scalar tuning problem in the x measurement update.
- The constant-y anchor is powerful enough to improve FL, but only by compensating in a way that breaks endpoint geometry. That makes it a diagnostic clue, not a production fix.
- The next useful direction is to inspect how the final fascicle end segment is reconstructed from `[x_sup, alpha]` and the aponeurosis lines, rather than continuing to tune scalar variances.

- X-variance sweep CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook79_x_length_anchoring/x_measurement_variance_sweep.csv`
- Fixed-y sweep CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook79_x_length_anchoring/fixed_y_anchor_sweep.csv`
- Named variant CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook79_x_length_anchoring/named_x_length_variants.csv`
- Sweep plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook79_x_length_anchoring/x_length_anchor_sweeps.png`
