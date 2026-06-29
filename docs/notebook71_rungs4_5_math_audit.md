# Notebook 71 rungs 4–5: MATLAB versus Python mathematical audit

## Scope and conclusion

This audit compares the production MATLAB `UltraTimTrack.m` forward path with the Python path used by notebook 71 for:

1. rung 4, **KLT / fascicle prior**; and
2. rung 5, **fascicle Kalman state / output geometry**.

The current Python pipeline does **not** yet implement the same complete mathematical logic as UltraTimTrack.

- The one-frame LK/affine transition is close, but the tracker state, initial seed, affine estimator, and cumulative integration are not exact.
- The scalar Kalman correction and geometry reconstruction are close ports of MATLAB.
- The Kalman prediction is not exact: MATLAB applies each saved affine to the **previous corrected fascicle**, while Python adds motion measured from the raw cumulative KLT segment.
- Notebook 71 incorrectly compares MATLAB's forward `X_plus` with Python's smoothed output. Those two rows are not a like-for-like state comparison.
- The large final deep-intersection error is geometric amplification of upstream x/angle differences, not evidence of a 52 px Kalman-state jump.

The first saved mismatch in notebook 71 is already rung 2. Rungs 4 and 5 therefore contain both their own implementation differences and inherited TimTrack/aponeurosis differences.

Notebook 71 aligns Python row 1 with MATLAB row 0 because the Python result contains one additional leading video frame. All notebook-71 numerical comparisons below use that offset.

### Artifact freshness

Notebook 71 loads notebook 70's cached Python run, which predates the exact adaptive-threshold and `houghpeaks` changes validated in notebook 73. The equations and structural findings in this audit still apply, but notebook 71's numerical bars are not the newest Python result.

Using notebook 74's fresh post-fix run changes the main fixed-R numbers as follows:

| Comparison | Notebook 71 / pre-fix cache | Notebook 74 / revised run |
|---|---:|---:|
| KLT-prior angle RMSE | 2.4128 deg | 2.1907 deg |
| KLT-prior length RMSE | 38.4950 px | 32.4684 px |
| final FL RMSE | 2.8670 mm | 2.5494 mm |
| final PEN RMSE | 1.5245 deg | 1.3308 deg |
| final ANG RMSE | 1.5992 deg | 1.3937 deg |

The low-level fixes help, but the KLT and Kalman-prediction mismatches remain material.

## Coordinate and endpoint contract

MATLAB stores each fascicle as two points in this order:

```text
fas_x = [x_deep; x_superficial]
fas_y = [y_deep; y_superficial]
```

Python stores the same geometry as:

```text
[x_superficial, y_superficial, x_deep, y_deep]
```

Both use one-based coordinates at the MATLAB-parity boundary. Angles use the image-coordinate convention

```text
alpha = atan2d(-(y_deep - y_superficial), x_deep - x_superficial)
```

with equivalent 180-degree line-orientation normalization in Python.

## The oval `Emask`: related, but upstream of rungs 4–5

MATLAB production behavior is:

1. compute mean superficial and deep aponeurosis depths;
2. set vertical radius `r_y = (mean_deep - mean_superficial) / 2`;
3. set horizontal radius `r_x = image_width / 2`;
4. sample an ellipse boundary with 100 angles;
5. rasterize it with `inpolygon`; and
6. save and reuse `parms.fas.Emask` unless `redo_ROI` is true.

Sources: `get_fasMask.m:3–29` and `auto_ultrasound.m:35–44` in the UltraTimTrack repository.

Python's analytic port uses

```text
((x - cx) / r_x)^2 + ((y - cy) / r_y)^2 <= 1
```

in `ultrasound_tracker/matlab_timtrack.py:299–327`. With identical depths, the analytic-versus-polygon rasterization difference is small. The larger screenshot difference comes mainly from different aponeurosis depths/radii, plus the fact that notebook 72's debug harness recomputes a dynamic mask for every exported frame.

That debug behavior is not the production MATLAB reuse contract. The export pattern can be seen in `results/notebook36_mask_parity/export_notebook36_masks.m:27–46`, while production MATLAB reuses the stored mask.

There is also a separate production-parity mismatch in the strict Python runner. It overwrites the exported MATLAB mask with an ellipse made from the local ROI JSON (`scripts/run_strict_ultratimtrack_video.py:241–274`). For `UltraTimTrack_test`:

| Mask | Radius `[r_y, r_x]` px | 1-based bounds `(x1,y1,x2,y2)` | Foreground pixels |
|---|---:|---:|---:|
| saved MATLAB `parms.fas.Emask` | `[138, 353]` | `(1, 46, 706, 320)` | 152,870 |
| Python ROI-derived mask | `[109, 339.5]` | `(24, 80, 703, 296)` | 116,212 |

Their Dice score is 0.8638, with 36,658 different pixels. This changes TimTrack Hough candidates and the autonomous fascicle seed before KLT begins.

Notebook 72 is also superseded in one important respect by notebook 73: after the MATLAB `houghpeaks` tie/order patch, the same-mask Hough accumulator, peaks, endpoints, weights, and alpha are exact on all 15 debug frames. The remaining low-level differences are therefore mask/filter/dtype inputs, not the Hough equations themselves.

## Rung 4: KLT / fascicle prior

### MATLAB step-by-step

For transition `f-1 -> f`, MATLAB does the following in `UltraTimTrack.m:1697–1967`.

1. **Build local fascicle image mask.** Each TimTrack Hough candidate line is widened by +/-5 px, and the union is intersected with the central 80% of the space between the current superficial/deep aponeuroses.
2. **Detect points.** `detectMinEigenFeatures(FilterSize=11, MinQuality=0.005)`, then `selectStrongest(300)` for a Hough ROI.
3. **Track persistent points.** `vision.PointTracker(NumPyramidLevels=4, MaxIterations=50, MaxBidirectionalError=inf, BlockSize=[21,71])`.
4. **Estimate a full affine.** `wf = estimateGeometricTransform2D(old_found, new_found, 'affine', MaxDistance=50)`.
5. **Propagate the previous raw fascicle.** `fas_new = transformPointsForward(wf, fas_prev)`.
6. **Save the pre-Kalman reference.** Copy `fas_new` to `fas_x_original/fas_y_original`.
7. **Keep tracker state.** Continue with `fpointsNew`; redetect below 100 points, apply the current ROI filter, and call `setPoints`.

The Kalman estimator runs only after the whole UltraTrack pass has finished (`UltraTimTrack.m:1650–1669`). Therefore `fas_x_original/fas_y_original` is a valid raw-KLT target and has no same-pass Kalman feedback.

### Python step-by-step

The current Python route is `run_strict_ultratimtrack_video.py:1358–1369` plus `ultrasound_tracker/ultratrack_klt.py`.

1. Build a line-union and central aponeurosis mask (`ultratrack_klt.py:53–116`).
2. Detect OpenCV min-eigen corners with `maxCorners=300`, `qualityLevel=0.005`, `minDistance=1`, and `blockSize=11` (`:127–147`).
3. Run forward pyramidal LK with a `(71,21)` window, four pyramid levels in OpenCV indexing, 50 iterations, and epsilon 0.01 (`:165–195`).
4. Estimate a full OpenCV RANSAC affine after converting point coordinates to one-based coordinates (`:198–224`).
5. Repeat detection independently for every transition; there is no persistent MATLAB `PointTracker` state (`:395–430`).
6. Apply the sequence of local affines cumulatively to one autonomous seed (`:237–274`).

### Variable map

| Concept | MATLAB | Python | Parity status |
|---|---|---|---|
| local Hough lines | `geofeatures(f).x/y` | `entry['x'/'y']` | same contract; upstream values differ |
| central fascicle ROI | `ROIy_fcor`, `fmask` | `fcor_mask`, `fascicle_mask` | same intended geometry; rasterizer differs slightly |
| old feature points | `fpoints(isFound,:)` | `old_points` | same role |
| new feature points | `fpointsNew(isFound,:)` | `new_points` | same role |
| tracker state | persistent `fpointTracker` + `setPoints` | no persistent object; redetect every transition | **different** |
| affine | `wf` / `Region.warp(:,:,f-1)` | `f_affine_matrices[f]` | same 2D affine role; estimator semantics differ |
| prior segment | `fas_prev` | previous cumulative Python segment | same recurrence form |
| propagated segment | `fas_new` | `klt_prior_segments[f]` | same output role |
| saved pre-Kalman target | `fas_x_original/fas_y_original` | `klt_prior_segments` | notebook 71 comparison target |

### Exact discrepancies

1. **Initial seed/aligned first prior differs.** Python's selected frame-0 seed is `[753.8305, 55.9259, -47.8305, 308.6887]`; after its first affine transition, the aligned comparison is:

   ```text
   MATLAB row 0: [733.0000, 54.4188, -27.0000, 309.0134]
   Python aligned row 1: [753.8105, 55.9280, -47.8488, 308.6903]
   Python - MATLAB: [+20.8105, +1.5093, -20.8488, -0.3231] px
   ```

   Python selected its seed on video frame 0 at 17.5 degrees; notebook 71 aligns Python row 1 to MATLAB row 0. The first aligned Python prior is 1.0205 degrees lower and 39.0531 px longer than MATLAB.

2. **Persistent versus redetect-every-frame tracker.** MATLAB advances one point set and conditionally refreshes it. Python detects a new point set for each pair of frames. These can give close local transforms but a different affine stream.
3. **Feature detector implementations differ.** OpenCV `goodFeaturesToTrack` is an approximation of MATLAB `detectMinEigenFeatures`, not a pixel-identical port.
4. **Affine robust estimation differs.** OpenCV `estimateAffine2D(RANSAC, threshold=50)` does not guarantee the same sample selection, scoring, refinement, or returned transform as MATLAB `estimateGeometricTransform2D(..., MaxDistance=50)`.
5. **Rasterization differs.** `cv2.fillPoly` plus rounded vertices only approximates `poly2mask`/`inpolygon`.
6. **Failure behavior differs.** Python holds the previous cumulative segment when an affine is missing. MATLAB's normal path expects an affine object or throws into the outer `try/catch`.

### Numerical evidence

Notebook 45 shows that the local one-step affine is close when applied to MATLAB's previous raw geometry:

| One-step comparison | RMSE |
|---|---:|
| angle | 0.3013 deg |
| superficial x | 1.2426 px |
| superficial y | 0.2867 px |
| deep x | 5.7251 px |
| deep y | 0.6049 px |
| length | 5.1680 px / 0.4662 mm |

The current cumulative notebook-71 prior is much farther from MATLAB:

| Rung-4 comparison | RMSE | Bias (Python - MATLAB) |
|---|---:|---:|
| superficial x | 15.3613 px | +2.1883 px |
| superficial y | 2.6764 px | -0.5324 px |
| deep x | 32.1006 px | +18.2190 px |
| deep y | 4.4568 px | -4.2671 px |
| length | 38.4950 px | -15.0337 px |
| angle | 2.4128 deg | +0.6765 deg |

The cumulative length error starts at +39.05 px, passes through approximately zero near frame 1250, and reaches -81.12 px on the last frame. This is compounded, time-varying affine error on top of a different seed—not a fixed offset caused only by the oval mask.

## Rung 5: two-state fascicle Kalman and output geometry

### State definition and initialization

MATLAB state:

```text
X = [x_superficial, alpha]
```

where `x_superficial = fas_x(2)`. On the first frame:

```text
alpha0[k] = angle(fas_x_original[k], fas_y_original[k])
X_plus[0] = [x_superficial_raw[0], mean(alpha0)]
P_plus[0] = [0, var(alpha0)]
X_minus[0] = X_plus[0]
P_minus[0] = P_plus[0]
```

The Python initialization in `ultratimtrack_matlab_2state.py:268–281` matches this contract for the supplied Python KLT segments.

### MATLAB prediction

For every frame, MATLAB first applies the saved affine to the **previous corrected fascicle** (`UltraTimTrack.m:2532–2544`):

```text
fas_new = warp(fas_prev_corrected)
dalpha = abs(angle(fas_new)) - abs(angle(fas_prev_corrected))
x_minus = [fas_new.superficial_x, alpha_prev_corrected + dalpha]
```

Process noise is motion dependent:

```text
q_x     = Q_parameter * ||superficial_new - superficial_prev||^2
dalpha_q = 0 if abs(dalpha) < 0.005 else abs(dalpha)
q_alpha = Q_parameter * dalpha_q^2
```

For this run, `Q_parameter=0.01`.

### Python prediction

Python does not have usable numeric MATLAB affine matrices in the `.mat` file; SciPy sees opaque MCOS `affine2d` objects. It therefore uses raw-KLT differences (`ultratimtrack_matlab_2state.py:283–323`):

```text
dx_sup = klt[f].x_sup - klt[f-1].x_sup
dy_sup = klt[f].y_sup - klt[f-1].y_sup
x_minus = previous_corrected_x + dx_sup

dalpha = abs(klt_alpha[f]) - abs(klt_alpha[f-1])
alpha_minus = previous_corrected_alpha + dalpha

q_x = Q_parameter * hypot(dx_sup, dy_sup)^2
q_alpha = Q_parameter * threshold(abs(dalpha), 0.005)^2
```

This is exact only for translation-like motion. Under rotation, scale, or shear, applying the affine to the corrected endpoint is not equivalent to adding the displacement observed at a different raw endpoint. The angular effect of a general affine also depends on the incoming line direction.

### Scalar correction

For either state component, both implementations use:

```text
P_minus = P_prev + q
K = P_minus / (P_minus + R)
x_plus = x_minus + K * (measurement - x_minus)
P_plus = (1 - K) * P_minus
```

For this run:

| Component | Measurement | Variance `R` |
|---|---|---:|
| superficial x | constant initial superficial x | `X = 100` px^2 |
| alpha | current TimTrack/Hough alpha | `R(1) = 3.05529211` deg^2 |

The Python fixed-R path matches these scalar equations (`ultratimtrack_matlab_2state.py:79–102`). The adaptive-R modes are extensions and are not part of normal UltraTimTrack parity.

### Backward smoother

Both implementations use, independently for x and alpha:

```text
A_f = P_plus_f / P_minus_(f+1)
X_smooth_f = X_plus_f + A_f * (X_smooth_(f+1) - X_minus_(f+1))
P_smooth_f = P_plus_f + A_f * (1 - P_minus_(f+1)) * A_f
```

The unusual constant `1` is intentional: MATLAB initializes `Psmooth = ones(1,2)` in each smoother call (`UltraTimTrack.m:2281–2301`). Python preserves that behavior (`ultratimtrack_matlab_2state.py:337–352`).

### Geometry reconstruction

Let the filtered superficial and deep aponeuroses be

```text
y_super(x) = m_super*x + b_super
y_deep(x)  = m_deep*x  + b_deep
```

MATLAB fixes the superficial y coordinate to its first-frame value `y0`. From smoothed state `[x_s, alpha]`:

```text
m_fas = -tan(alpha)
b_fas = y0 - m_fas*x_s

x_deep = (b_fas - b_deep) / (m_deep - m_fas)
y_deep = b_deep + m_deep*x_deep

x_super_end = (b_fas - b_super) / (m_super - m_fas)
y_super_end = b_super + m_super*x_super_end
```

It stores two related segments:

```text
fas_x/fas_y         = [deep intersection; (state x, fixed y0)]
fas_x_end/fas_y_end = [deep intersection; superficial-apo intersection]
```

Python `reconstruct_fascicle_from_state` implements the same equations (`ultratimtrack_matlab_2state.py:105–145`).

Final outputs also match the MATLAB definitions:

```text
ANG = angle(fas_x/fas_y)
gamma = angle(deep aponeurosis)
PEN = ANG - gamma
FL_mm = (ID / image_height) * length(fas_x_end/fas_y_end)
```

For this run, `ID/image_height = 50.7/562` mm/px.

### State/output variable map

| Concept | MATLAB variable | Correct Python variable | Status |
|---|---|---|---|
| forward predicted state | `X_minus` | `X_minus` | same role; prediction input differs |
| forward corrected state | `X_plus` | `forward_X_plus` | same role |
| forward predicted covariance | `fas_p_minus` | `fas_p_minus` | same role |
| forward corrected covariance | `fas_p` before smoothing | `forward_fas_p` | not separately preserved in saved MATLAB final file |
| smoothed state | encoded in final `fas_x(2)` and `alpha` | Python result `X_plus` | values have same role, **name does not** |
| smoothed covariance | final `fas_p` | Python result `fas_p` | same role |
| smoother gain | `A` | `smoother_gain` | MATLAB saves only the last loop component per frame; Python keeps both |
| alpha correction gain | `K` | `kalman_gain[:,1]` | same role |
| reconstructed state segment | `fas_x/fas_y` | `fascicle_segments` | same geometry |
| endpoint-to-endpoint segment | `fas_x_end/fas_y_end` | `fascicle_end_segments` | same geometry |
| final outputs | `fas_ang/fas_pen/fas_length` | `ANG_deg/PEN_deg/FL_mm` | same formulas |

### Notebook 71 `X_plus` comparison error

Notebook 71 currently compares:

```text
MATLAB forward X_plus x     versus Python smoothed fascicle x
MATLAB forward X_plus alpha versus Python smoothed ANG
```

MATLAB's smoother does not overwrite `X_plus`; it overwrites `fas_x`, `fas_y`, `alpha`, and `fas_p`. Therefore the correct comparisons are:

```text
MATLAB X_plus               versus Python forward_X_plus
MATLAB final fas_x(2),alpha versus Python smoothed X_plus
```

With the current autonomous Python inputs:

| Like-for-like comparison | RMSE |
|---|---:|
| forward `X_plus.x` | 20.0387 px |
| forward `X_plus.alpha` | 1.8735 deg |
| smoothed x via final `fas_x` | 21.3857 px |
| smoothed alpha via final `Fascicle.alpha` | 1.5992 deg |

The correction changes the interpretation, but it does not make the current pipeline pass because the seed, KLT prior, and TimTrack/aponeurosis inputs already differ.

### Why deep x reaches about 52 px RMSE

Current rung-5 geometry errors are:

| Output geometry | RMSE |
|---|---:|
| smoothed/state superficial x | 21.39 px |
| fixed superficial y | 1.507 px exactly |
| deep intersection x | 52.45 px |
| deep intersection y | 1.86 px |
| end-segment length | about 31.8 px |

The fixed 1.507 px y error is inherited exactly from the Python seed's first-frame superficial y. The much larger deep-x error comes from the line-intersection division by `m_deep - m_fas`: a modest state-x or alpha error can create a much larger horizontal intersection displacement when the lines are relatively close in slope. This is expected geometric sensitivity.

### Isolated Kalman/geometry parity with MATLAB inputs

Feeding the Python two-state implementation the saved MATLAB raw KLT, MATLAB TimTrack alpha, and MATLAB filtered aponeuroses gives:

| Like-for-like output | RMSE |
|---|---:|
| forward state x | 3.5921 px |
| forward state alpha | 0.2935 deg |
| smoothed state x | 3.2466 px |
| smoothed state alpha | 0.2657 deg |
| final FL | 0.5605 mm |
| final PEN | 0.2657 deg |
| final ANG | 0.2657 deg |

This demonstrates that the scalar update, smoother, and reconstruction are largely correct. The remaining state-x difference under oracle inputs is consistent with the missing exact affine-on-corrected-state prediction.

With the current autonomous inputs, notebook 71 reports:

| Final output | RMSE |
|---|---:|
| FL | 2.8670 mm |
| PEN | 1.5245 deg |
| ANG | 1.5992 deg |

## Required changes for mathematical parity

Implement in this order:

1. **Use one fixed production `Emask`.** Either import the exact run's saved `parms.fas.Emask`, or create it once from the same first-frame aponeurosis vectors and freeze it. Do not replace it with the current ROI-rectangle ellipse in parity mode.
2. **Match the initial fascicle.** Add a parity mode that uses the same first-frame Hough/ROI selection rule as MATLAB. Keep the 11-frame autonomous cluster selector as a separate non-parity mode.
3. **Export numeric MATLAB affines and KLT checkpoints.** Save each `wf.T`, `wa.T`, old/new points, `isFound`, and post-`setPoints` point set as ordinary numeric arrays. The existing MCOS objects cannot be audited reliably in SciPy.
4. **Implement persistent tracker semantics.** Carry the point set forward, preserve the MATLAB ROI filtering order, and redetect only at the same threshold.
5. **Validate the affine stream before cumulative geometry.** Compare every numeric 3x3 transform and its action on standard probe points.
6. **Apply the affine to the previous corrected state in the Kalman predictor.** Reconstruct `fas_prev` from the previous corrected state, transform both endpoints, then derive `x_minus`, `dalpha`, and both process-noise values from that transformed segment.
7. **Correct state naming and notebook comparisons.** Preserve `X_plus` as the forward corrected state for MATLAB compatibility; expose the backward result separately as `X_smooth`. Compare forward-to-forward and smooth-to-smooth.
8. **Validate per-frame internals.** Gate `X_minus`, forward `X_plus`, both `P_minus` values, both forward `P_plus` values, both gains, smoothed state/covariance, both reconstructed segments, and only then `FL/PEN/ANG`.

Do not tune adaptive `R`, anatomical gating, or confidence weights to close this parity gap. Those are useful extensions, but they change the model and should be assessed only after the fixed UltraTimTrack path is exact.

## Conditional MATLAB branches not covered by the current Python parity path

These do not affect the present forward, non-manual, finite-parameter run, but they are part of full UltraTimTrack logic:

- manual x and angle measurements applied as a second sequential correction;
- manual measurement variance conversion for angle;
- `Q = Inf` and `X = Inf` override branches;
- backward optical-flow tracking and reverse smoother order;
- MATLAB failure behavior when affine estimation/tracking throws.

## Evidence used

- `notebooks/71_parity_ladder_matlab_vs_python.ipynb`
- `notebooks/72_timtrack_low_level_matlab_export_parity.ipynb`
- `notebooks/73_exact_matlab_threshold_houghpeaks_parity.ipynb`
- `ultrasound_tracker/ultratrack_klt.py`
- `ultrasound_tracker/ultratimtrack_matlab_2state.py`
- `scripts/run_strict_ultratimtrack_video.py`
- `results/notebook45_ultratrack_klt_one_step_affine_diagnostics/klt_one_step_affine_metrics.csv`
- `results/notebook46_klt_drift_vs_kalman_boundary/klt_kalman_boundary_metrics.csv`
- `results/notebook71_parity_ladder_matlab_vs_python/parity_ladder_metrics.csv`
- `/Users/grosbedou/Documents/GitHub/UltraTimTrack/UltraTimTrack.m`
- `/Users/grosbedou/Documents/GitHub/UltraTimTrack/ultrasound-automated-algorithm/Filter/get_fasMask.m`
- `/Users/grosbedou/Documents/GitHub/UltraTimTrack/ultrasound-automated-algorithm/auto_ultrasound.m`
