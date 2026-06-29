# Notebook 74 revised three-way comparison

Matched frames: 2666; Python offset: 1; runtime: 769.0 s.

## Revised agreement
- normal FL_mm: bias -1.775 mm, MAE 2.112, RMSE 2.549, CCC 0.964.
- normal ANG_deg: bias 1.006 deg, MAE 1.138, RMSE 1.394, CCC 0.959.
- adaptive FL_mm: bias -1.874 mm, MAE 2.136, RMSE 2.594, CCC 0.962.
- adaptive ANG_deg: bias 1.056 deg, MAE 1.153, RMSE 1.407, CCC 0.958.

## Change from Notebook 70
- normal FL_mm: RMSE 2.867 -> 2.549 (closer by 0.318).
- normal ANG_deg: RMSE 1.599 -> 1.394 (closer by 0.206).
- adaptive FL_mm: RMSE 2.886 -> 2.594 (closer by 0.292).
- adaptive ANG_deg: RMSE 1.601 -> 1.407 (closer by 0.194).

## `71` clarification
- MATLAB BlockSize `[21, 71]` is the KLT point-tracker window, not the ROI.
- The old `71x71` threshold approximation affected masks but was not the only source of drift.
- The parity ladder also identified independent KLT prior drift after the image-measurement stage.