# UltraTimTrack Python Parity Plan

Goal: reproduce MATLAB UltraTimTrack final outputs (`Fdat.Region.FL`, `PEN`, and `ANG`) from the same input video, with a compatibility layer that can compare Python and MATLAB numerically.

## Guiding Decision

For this port, prioritize numerical compatibility over algorithmic novelty. The Python internals can be cleaner than MATLAB, but every replacement needs a parity check against MATLAB outputs.

## Validation Gates

1. Final-output gate:
   - Compare Python outputs against MATLAB `Fdat.Region.FL`, `PEN`, and `ANG`.
   - Use `scripts/compare_ultratimtrack_parity.py`.
   - Initial target: clear positive correlation and no sign inversion.
   - Practical target: `FL` RMSE below 2 mm, `PEN`/`ANG` RMSE below 1 deg.

2. TimTrack-intermediate gate:
   - Compare Python TimTrack-like outputs against MATLAB `Fdat.geofeatures`.
   - Required signals: `alpha`, `phi`, `faslen`, `gamma`, `betha`, `super_pos`, `deep_pos`.
   - Do not tune Kalman until this gate is close.

3. UltraTrack/KLT gate:
   - Match affine propagation behavior, point selection, masks, and aponeurosis/fascicle warps.
   - Compare tracked geometry before state estimation.

4. Kalman/smoothing gate:
   - Match MATLAB's scalar state estimation:
     - aponeurosis y endpoints are filtered independently,
     - fascicle state is `[superficial attachment x, fascicle alpha]`,
     - `Q = Q_parameter * dx^2`,
     - final pass uses the MATLAB Rauch-Tung-Striebel-style smoother.

## Current Known Mismatches

- The current Python TimTrack-like detector is not yet equivalent to MATLAB TimTrack. MATLAB uses Frangi filtering, an aponeurosis-bounded fascicle mask, a standard Hough accumulator, and a weighted median of Hough peak angles. The current Python path uses OpenCV probabilistic Hough line segments and selects a best line.
- The current 4-state geometric Kalman filter is a useful experiment, but it is not MATLAB UltraTimTrack's state estimator.
- Current fused segments can reconstruct outside the image because angle normalization removes segment direction.
- The `geometry.py` angle comments disagree with the implemented convention in places.
- The project test requirements currently omit `pytest`.

## Recommended Work Order

1. Freeze validation:
   - Use one MATLAB `.mat` result and one video as the reference pair.
   - Use the MATLAB-saved final and intermediate arrays as the source of truth.
   - Avoid judging algorithm changes from plots alone.

2. Port TimTrack first:
   - Port `filter_usimage.m`, `apo_func.m`, `fit_apo.m`, `get_fasMask.m`, `dohough.m`, and `weightedMedian.m`.
   - Match MATLAB indexing and image-size conventions explicitly.
   - Validate `alpha`, `phi`, `faslen`, `gamma`, `betha`, `super_pos`, and `deep_pos`.

3. Port UltraTrack second:
   - Match `detectMinEigenFeatures` settings as closely as OpenCV allows.
   - Match affine transform estimation and masks.
   - Save predicted fascicle/aponeurosis geometry before Kalman.

4. Port MATLAB state estimator third:
   - Implement a compatibility estimator that mirrors MATLAB's two-state fascicle model.
   - Keep the existing experimental 4-state filter separate.

5. Finalize output compatibility:
   - Produce a Python object or NPZ/CSV with MATLAB-compatible names:
     - `FL`, `PEN`, `ANG`, `Time`
     - optional `geofeatures`
   - Keep notebook plots as visualization, not the primary validation method.

## Command

From the repository root:

```bash
.venv/bin/python scripts/compare_ultratimtrack_parity.py
```

This writes:

```text
results/matlab_comparison/parity_metrics.csv
```
