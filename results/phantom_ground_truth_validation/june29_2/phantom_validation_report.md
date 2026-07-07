# Phantom Ground-Truth Validation

## Data Audit
- Video: `/Users/grosbedou/PycharmProjects/NDORMS/data/raw/june29_2.mp4`
- Video opened: `True`
- Frames/FPS/size: `669` frames, `25` fps, `1024x768` px
- Strict runner NPZ: **not found** for this video.
- Ground truth: `synthetic_linear_cumulative_x15mm`
- GT columns used: `{'frame': 'video_frame', 'time_s': 'frame/fps', 'x': 'synthetic_total_x_mm', 'y': None, 'scalar': None}`
- GT axis type: `x`
- GT source note: synthetic linear cumulative ramp from the supplied total displacement; replace with an actuator/encoder trace if the plate motion was not constant-speed over the full video.
- Failure threshold: `0.5` mm

## Coordinate And Unit Contract
- Image origin is the top-left pixel. x increases to the right; y increases downward.
- Strict line segments are stored in one-based MATLAB-style pixel coordinates, but displacement is computed relative to frame 0, so the one-pixel origin offset cancels.
- Axial displacement corresponds to image y positive downward. Lateral displacement corresponds to image x positive rightward.
- The current scalar `mm_per_pixel` is depth/height-derived axial spacing. Vector or lateral validation needs independent lateral spacing unless square pixels are confirmed.
- `fascicle_segments` are final Kalman output; `klt_prior_segments` are cumulative/persistent KLT prior segments; `fixed_fascicle_segments` is the normal fixed-R Kalman comparator when present.

## Validation Status
Ground truth was available, but no strict runner output was found for this video. Run the tracker first or pass `--strict-npz`.

## Reviewer Interpretation
- Accuracy cannot be concluded yet. Missing ground truth and/or missing tracker output prevents a numerical validation.

## Paper-Strength Validation Target
- Pre-register the phantom displacement waveform, units, axis convention, frame synchronization, and tolerance.
- Report raw, not only best-aligned, MAE/RMSE/bias/limits-of-agreement over multiple amplitudes and speeds.
- Include failure rate and show overlays at worst-error frames.
- Demonstrate low drift over long sequences and compare against at least KLT-only and fixed-R Kalman baselines.
- Phantom validation does not replace in vivo validation because tissue deformation, out-of-plane motion, probe pressure, anisotropic speckle decorrelation, and manual ROI variability are different failure modes.

## Machine-Readable Audit
```json
{
  "video": {
    "path": "/Users/grosbedou/PycharmProjects/NDORMS/data/raw/june29_2.mp4",
    "opened": true,
    "frame_count": 669,
    "fps": 25.0,
    "width_px": 1024,
    "height_px": 768
  },
  "strict_outputs": {
    "npz": null,
    "csv": null,
    "metadata": null
  },
  "strict_audit": null,
  "ground_truth_path": "synthetic_linear_cumulative_x15mm",
  "ground_truth_kind": "synthetic_linear_cumulative",
  "synthetic_ground_truth": {
    "total_x_mm": 15.0,
    "total_y_mm": null,
    "start_frame": null,
    "end_frame": null
  },
  "data_phantom_files": [],
  "ground_truth_loaded": true,
  "axis_requested": "x",
  "failure_threshold_mm": 0.5
}
```