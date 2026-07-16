# PythonNathTrack — Python UltraTimTrack for Ultrasound Fascicle Tracking

A Python port and extension of the MATLAB **UltraTimTrack** pipeline for tracking
muscle fascicles, aponeuroses, and pennation angles in B-mode ultrasound video.

Developed in collaboration with the **Nuffield Department of Orthopaedics,
Rheumatology and Musculoskeletal Sciences (NDORMS)**, University of Oxford
(Jack Tu Lab).

> Original MATLAB reference: [timvanderzee/UltraTimTrack](https://github.com/timvanderzee/UltraTimTrack)

---

## Table of Contents

- [What Does It Do?](#what-does-it-do)
- [Pipeline Overview](#pipeline-overview)
- [What Is New in This Algorithm](#what-is-new-in-this-algorithm)
  - [1. Speckle Confidence Factor (Adaptive Measurement Noise)](#1-speckle-confidence-factor-adaptive-measurement-noise)
  - [2. Autonomous Fascicle Seed Selection](#2-autonomous-fascicle-seed-selection)
  - [3. Hough Localmax Fallback](#3-hough-localmax-fallback)
  - [4. One-Step KLT (Drift-Free)](#4-one-step-klt-drift-free)
  - [5. Adaptive Anisotropic Kalman Filter](#5-adaptive-anisotropic-kalman-filter)
- [Key Variables Reference](#key-variables-reference)
  - [Speckle Confidence Variables](#speckle-confidence-variables)
  - [Kalman Filter State & Noise Variables](#kalman-filter-state--noise-variables)
  - [Hough / TimTrack Geofeature Variables](#hough--timtrack-geofeature-variables)
  - [Aponeurosis Variables](#aponeurosis-variables)
  - [KLT / Optical Flow Variables](#klt--optical-flow-variables)
  - [Final Output Variables](#final-output-variables)
  - [Fascicle Seed Selection Variables](#fascicle-seed-selection-variables)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [GUI Mode](#gui-mode)
  - [Command-Line Mode](#command-line-mode)
  - [Adaptive Confidence Mode](#adaptive-confidence-mode)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

---

## What Does It Do?

Given a B-mode ultrasound video of a muscle and user-defined regions of interest
(ROI) for the superficial aponeurosis, deep aponeurosis, and fascicle region,
this pipeline produces frame-by-frame measurements of:

| Output | Symbol | Description |
|--------|--------|-------------|
| **Fascicle angle** | `ANG` (α) | The orientation of the tracked fascicle relative to horizontal |
| **Pennation angle** | `PEN` | `α − aponeurosis_angle` — the angle between the fascicle and the aponeurosis |
| **Fascicle length** | `FL` | `thickness / sin(PEN)` — the length of the fascicle between the two aponeuroses |

These three quantities are the standard morphometric outputs used in
musculoskeletal biomechanics to quantify muscle architecture from ultrasound.

The pipeline follows the MATLAB UltraTimTrack methodology — adaptive
thresholding, Frangi vesselness filtering, Hough-transform fascicle angle
estimation, KLT optical-flow tracking, and a 2-state Kalman filter — but adds
several novel improvements (see below).

---

## Pipeline Overview

```
Ultrasound video + ROI JSON
        │
        ▼
┌──────────────────────────────────┐
│  1. ROI Loading (roi.py)         │  Load superficial/deep/fascicle ROIs
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  2. Aponeurosis Detection        │  Adaptive threshold → Hough line fit
│  (matlab_aponeurosis.py)         │  → superficial & deep aponeurosis lines
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  3. Fascicle Mask + Frangi       │  Elliptical mask → aponeurosis subtraction
│  (matlab_timtrack.py)            │  → Frangi vesselness filtering
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  4. Hough Angle Estimation       │  Weighted-median α from Hough peaks
│  (timtrack_hough.py)             │  + localmax fallback
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  5. Autonomous Seed Selection    │  Score/cluster candidate segments
│  (strict_fascicle_seed.py)       │  → best first-frame fascicle seed
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  6. KLT Optical-Flow Tracking    │  OpenCV Lucas-Kanade affine tracking
│  (ultratrack_klt.py)             │  (one-step, drift-free)
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  7. Speckle Confidence Metrics   │  4-component confidence → R_t scaling
│  (speckle_confidence.py)         │  (NOVEL — see below)
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  8. 2-State Kalman Filter        │  State: [x_sup, α] with adaptive R
│  (ultratimtrack_matlab_2state.py)│  + optional RTS smoother
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  9. Final Output (final_output)  │  ANG, PEN, FL (px + mm)
│                                  │  + CSV, NPZ, plots, annotated MP4
└──────────────────────────────────┘
```

---

## What Is New in This Algorithm

The following features go **beyond** the original MATLAB UltraTimTrack
implementation and are novel contributions of this Python port.

### 1. Speckle Confidence Factor (Adaptive Measurement Noise)

**Module:** `ultrasound_tracker/speckle_confidence.py`

This is the primary novel contribution. Instead of treating every frame's
measurement equally, the algorithm computes a **per-frame confidence score**
from four independent ultrasound-specific metrics and uses it to dynamically
scale the Kalman filter's measurement noise covariance **R_t**.

> **Core principle:** Low confidence does *not* mean a frame is discarded. It
> means the Kalman filter should **trust the measurement less** by increasing
> `R_t`. High-confidence frames pull the estimate toward the measurement;
> low-confidence frames let the prediction dominate.

#### The Four Confidence Components

| Component | What It Measures | Module Function |
|-----------|-----------------|-----------------|
| **Speckle coherence** | Forward-backward patch matching ZNCC on a grid of speckle patches. High ZNCC = speckle pattern is trackable. | `compute_speckle_coherence()` |
| **Motion consistency** | Robust MAD of displacement vectors from the median motion. Low spread = coherent motion. | `compute_motion_consistency()` |
| **Feature detection reliability** | Combines Hough peak count, peak strength/separation, and fascicle mask density. | `compute_feature_detection_reliability()` |
| **Geometry stability** | Anatomical plausibility of α, pennation, and length + temporal stability (angle/length jumps between frames). | `compute_geometry_stability()` |

#### How They Combine

The four components are merged via a **weighted geometric mean**:

```
confidence = speckle^w1 × motion^w2 × feature^w3 × geometry^w4
```

Default weights: `speckle=0.35, motion=0.25, feature=0.25, geometry=0.15`

#### R-Scale Mapping

Confidence is mapped to a measurement-noise scaling factor:

```
R_scale = R_min + (R_max − R_min) × (1 − confidence)^γ
```

With defaults `R_min=0.5`, `R_max=20.0`, `γ=1.5`, so:
- `confidence = 1.0` → `R_scale ≈ 0.5` (trust measurement strongly)
- `confidence = 0.5` → `R_scale ≈ 5.6` (moderate trust)
- `confidence = 0.0` → `R_scale = 20.0` (trust prediction, not measurement)

The final adaptive covariance is: **`R_t = R_base × R_scale`**

#### Anisotropic Extension

An **anisotropic** variant computes separate confidence scores for the angle
(θ) and length (FL) dimensions, producing independent `R_scale_θ` and
`R_scale_length`. This recognizes that angle measurements and length
measurements can have different reliability on the same frame — e.g., the
fascicle angle may be clearly visible but the aponeurosis intersection (which
determines length) may be ambiguous.

```
R_t = diag([R_θ_base × s_θ, R_length_base × s_length])
```

---

### 2. Autonomous Fascicle Seed Selection

**Module:** `ultrasound_tracker/strict_fascicle_seed.py`

The original UltraTimTrack requires manual selection of the initial fascicle.
This module performs **fully autonomous** first-frame seed selection:

1. **Candidate extraction**: Sweeps an angle grid (14°–24° at 0.1° steps)
   combined with the top Hough peaks to generate candidate fascicle segments.
2. **Multi-criteria scoring**: Each candidate is scored on mask support (35%),
   raw mask support (18%), lateral span (12%), Hough peak strength (5%),
   pennation plausibility (8%), inside-muscle score (8%), and boundary penalty
   (1%). The score is deliberately **depth-independent** to avoid biasing
   selection toward shallow or deep fascicles.
3. **Multi-frame clustering**: Candidates are clustered by `(alpha, x_mid,
   length)` across frames. Clusters must appear in ≥ 8 frames to be
   considered stable.
4. **Stable cluster selection**: Prefers Hough-branch clusters within a
   tolerance margin; falls back to highest cluster score.

This enables the pipeline to run end-to-end without manual fascicle
identification.

---

### 3. Hough Localmax Fallback

**Module:** `ultrasound_tracker/timtrack_hough.py` — `dohough_with_localmax_fallback()`

When the standard `houghpeaks`-based angle estimation produces a spurious result
(typically when too much Hough accumulator mass sits below 10°), the algorithm
automatically switches to a **1D local-maxima** approach on the angle profile
`max(H, axis=0)`. The decision rule (from validation Notebook 90):

```python
if mass_below_10deg >= 0.25 and gap_to_lower_deg >= 4.0:
    use_localmax_variant()
```

This improves robustness on frames where the fascicle mask is noisy or the
fascicle is nearly horizontal.

---

### 4. One-Step KLT (Drift-Free)

**Module:** `ultrasound_tracker/ultratrack_klt.py`

The original UltraTimTrack uses sequential (cumulative) KLT tracking, which
**accumulates drift** over long sequences. This implementation offers a
**one-step** mode:

- Each affine is estimated from frame `i−1 → i` and applied **only to the
  reference segment** at `i−1`, not to a running state.
- This produces drift-free per-frame geometry that feeds directly into the
  Kalman filter, which handles temporal smoothing.
- A persistent (state-carrying) tracker is also available for comparison.

> The key insight: applying small frame-to-frame affines to a long-running
> segment compounds drift, while handing the local transition to the Kalman
> gate preserves the validated one-step behavior.

---

### 5. Adaptive Anisotropic Kalman Filter

**Module:** `ultrasound_tracker/ultratimtrack_matlab_2state.py`

The 2-state Kalman filter (state: `[x_sup, α]`) supports two modes:

| Mode | Behavior |
|------|----------|
| **Fixed-R** (MATLAB parity) | Constant measurement noise `R` — matches original UltraTimTrack |
| **Adaptive anisotropic** | `R_t` scaled per-frame by the speckle confidence factor, with separate scaling for angle and length dimensions |

The adaptive mode uses the speckle confidence metrics (Section 1) to
dynamically adjust how much the filter trusts each frame's measurement, which
is the central novelty of this implementation.

Additional features:
- **Affine-based prediction**: When available, uses one-step affine matrices
  for the prediction step (preferred over raw KLT deltas).
- **Adaptive process noise**: `Q_x = q × Δx²`, `Q_α = q × Δα²` — scales with
  observed motion magnitude.
- **Optional RTS smoother**: Backward pass for smoothed state estimates.

---

## Key Variables Reference

### Speckle Confidence Variables

| Variable | Type | Description |
|----------|------|-------------|
| `speckle_zncc` | float | Median zero-mean normalized cross-correlation across matched patches. Range [−1, 1]. |
| `speckle_confidence` | float | Clipped, fraction-weighted speckle confidence. Range [0, 1]. |
| `forward_backward_error` | float | Median forward-backward patch matching displacement error (px). |
| `valid_patch_fraction` | float | Fraction of grid patches with sufficient texture variance. |
| `motion_consistency` | float | Exponential decay of displacement-vector spread (MAD). Range [0, 1]. |
| `feature_reliability` | float | Combined Hough peak count, strength, and mask density score. Range [0, 1]. |
| `geometry_stability` | float | Anatomical plausibility + temporal stability score. Range [0, 1]. |
| `confidence_theta` | float | Anisotropic confidence for the angle (θ) dimension. |
| `confidence_length` | float | Anisotropic confidence for the length dimension. |
| `combined_confidence` | float | Weighted geometric mean of all components. |
| `r_scale` | float | Isotropic R-scaling factor: `R_min + (R_max−R_min)×(1−c)^γ` |
| `r_scale_theta` | float | Anisotropic R-scaling for angle. |
| `r_scale_length` | float | Anisotropic R-scaling for length. |
| `detection_success` | bool | Whether feature detection succeeded for this frame. |

**Configuration** (`SpeckleConfidenceConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 21 | Patch size for ZNCC matching (px) |
| `stride` | 24 | Grid spacing between patches (px) |
| `search_radius` | 8 | Forward-backward search radius (px) |
| `min_texture_variance` | 5.0 | Minimum patch variance to be considered valid |
| `zncc_low` | 0.45 | ZNCC below this → low speckle confidence |
| `zncc_high` | 0.90 | ZNCC above this → high speckle confidence |
| `confidence_floor` | 0.05 | Minimum confidence value (never zero) |
| `confidence_ceiling` | 1.0 | Maximum confidence value |
| `r_min_scale` | 0.5 | Minimum R-scaling (high confidence) |
| `r_max_scale` | 20.0 | Maximum R-scaling (low confidence) |
| `r_gamma` | 1.5 | Non-linearity exponent in R-scale mapping |
| `weights` | `{speckle:0.35, motion:0.25, feature:0.25, geometry:0.15}` | Component weights |

---

### Kalman Filter State & Noise Variables

| Variable | Type | Description |
|----------|------|-------------|
| `x_sup` | float | State[0] — superficial fascicle attachment x-position (px) |
| `alpha` | float | State[1] — fascicle angle (degrees) |
| `P` (p_prev, p_plus, p_minus) | float | Error covariance (scalar per state) |
| `Q` (q_x, q_alpha) | float | Process noise: `q × Δx²`, `q × Δα²` |
| `R` (R_base) | float | Base measurement noise covariance |
| `R_t` | float | Adaptive measurement noise: `R_base × r_scale` |
| `K` (kalman_gain) | float | Kalman gain (clipped to [0, 1]) |
| `X_plus` | array | Corrected (a posteriori) state sequence |
| `X_minus` | array | Predicted (a priori) state sequence |
| `X_smooth` | array | RTS-smoothed state sequence |
| `prediction_used_affine` | bool | Whether affine prediction was used (vs. KLT delta fallback) |

**Configuration** (`MatlabTwoStateKalmanConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q_parameter` | 0.01 | Process noise scaling coefficient |
| `x_measurement_variance` | 100.0 | Base R for x_sup measurement |
| `alpha_measurement_variance` | 3.055 | Base R for alpha measurement |
| `n_start_frames` | 1 | Frames used for initial variance estimation |
| `run_smoother` | True | Enable RTS backward smoother |
| `use_adaptive_R` | False | Enable speckle-confidence adaptive R |

---

### Hough / TimTrack Geofeature Variables

| Variable | Type | Description |
|----------|------|-------------|
| `alpha` | float | Weighted-median fascicle angle from Hough peaks (degrees) |
| `alphas` | array | Per-peak fascicle angles (degrees) |
| `ws` (weights) | array | Per-peak Hough accumulator weights |
| `phi` | float | Fascicle line orientation parameter |
| `faslen` | float | Raw fascicle segment length (px) |
| `betha` | float | Fascicle line intercept parameter |
| `gamma` | float | Ellipse radius correction parameter |
| `thickness` | float | Muscle thickness between aponeuroses (px) |
| `hmat` | array | Raw Hough accumulator matrix |
| `hmat_corrected` | array | Ellipse-radius-corrected accumulator |
| `theta` | array | Hough angle parameter vector (degrees) |
| `rho` | array | Hough distance parameter vector (px) |
| `peaks` | array | Hough peak indices [(row, col), ...] |
| `peak_source` | str | `"houghpeaks"` or `"angle_profile_localmax"` |
| `localmax_fallback_used` | bool | Whether localmax fallback was triggered |
| `Emask` | array | Elliptical fascicle mask (2D boolean) |
| `filtered` | array | Frangi-filtered image |

**Configuration** (`DoHoughParams`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `houghangles` | `"specified"` | Angle range mode (`"default"` or `"specified"`) |
| `angle_range` | (8.0, 80.0) | Hough angle search range (degrees) |
| `thetares` | 1.0 | Hough theta resolution (degrees) |
| `rhores` | 1.0 | Hough rho resolution (px) |
| `npeaks` | 10 | Maximum Hough peaks |
| `replace_diagonal_bias` | True | Correct 45° diagonal bias in accumulator |

---

### Aponeurosis Variables

| Variable | Type | Description |
|----------|------|-------------|
| `apox_1b` | array | One-based x-sample positions for aponeurosis profiling |
| `super_vec_1b` / `deep_vec_1b` | array | Raw Hough y-vectors at apox positions (1-based) |
| `super_coef_linear_1b` / `deep_coef_linear_1b` | array | Linear polyfit coefficients (1-based) |
| `super_line_0b` / `deep_line_0b` | array | Aponeurosis line segments `[x1, y1, x2, y2]` (0-based) |
| `super_apo_angle_deg` / `deep_apo_angle_deg` | float | Aponeurosis angles (degrees) |
| `super_cut` / `deep_cut` | tuple | Vertical cut fractions for superficial/deep regions |
| `apo_thres` | array | Adaptive-thresholded aponeurosis mask |

**Configuration** (`MatlabHoughAponeurosisConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `super_cut` | (0.0, 0.5) | Superficial aponeurosis vertical cut range (fraction of height) |
| `deep_cut` | (0.5, 1.0) | Deep aponeurosis vertical cut range |
| `threshold_sensitivity` | 0.5 | Adaptive threshold sensitivity |
| `threshold_method` | `"mean"` | Local statistic for adaptive threshold |
| `hough_theta_step_deg` | 1.0 | Hough theta resolution for aponeurosis detection |
| `fit_method` | `"enforce_maxangle"` | Aponeurosis fit method |
| `super_maxangle` / `deep_maxangle` | 0.5 | Maximum allowed aponeurosis slope angle (degrees) |
| `super_order` / `deep_order` | 1 | Polynomial fit order |
| `apomargin` | 20 | Margin from frame edges (px) |
| `napo` | 10 | Number of aponeurosis sample points |

---

### KLT / Optical Flow Variables

| Variable | Type | Description |
|----------|------|-------------|
| `fascicle_segments` | array | Tracked fascicle segments per frame `(N, 4)` `[x1,y1,x2,y2]` |
| `f_affine_matrices` | array | Per-frame affine transform matrices `(N, 2, 3)` |
| `f_affine_ok` | array | Boolean: whether affine estimation succeeded per frame |
| `tracker_redetected` | array | Boolean: whether points were redetected (persistent mode) |
| `tracker_found_fraction` | float | Fraction of frames with successful tracking |
| `n_inliers` | int | RANSAC inliers for affine estimation |

**Configuration** (`UltraTrackKLTConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_fascicle_corners` | 300 | Maximum goodFeaturesToTrack corners in fascicle mask |
| `quality_level` | 0.005 | Corner quality threshold |
| `min_distance` | 1 | Minimum corner spacing (px) |
| `block_size` | 11 | Corner detection block size |
| `lk_win_size` | (81, 81) | Lucas-Kanade search window size |
| `lk_max_level` | 3 | Pyramid levels |
| `lk_max_iter` | 50 | Maximum LK iterations |
| `lk_epsilon` | 0.01 | LK convergence threshold |
| `ransac_reproj_threshold` | 50.0 | RANSAC reprojection threshold (px) |

---

### Final Output Variables

| Variable | Type | Description |
|----------|------|-------------|
| `ANG_deg` | float/array | Fascicle angle α (degrees) |
| `PEN_deg` | float/array | Pennation angle = α − aponeurosis_angle (degrees) |
| `FL_px` | float/array | Fascicle length in pixels |
| `FL_mm` | float/array | Fascicle length in millimeters (if `mm_per_pixel` provided) |
| `muscle_thickness_px` | float/array | Muscle thickness between aponeuroses (px) |
| `super_apo_angle_deg` | float/array | Superficial aponeurosis angle (degrees) |
| `deep_apo_angle_deg` | float/array | Deep aponeurosis angle (degrees) |
| `mm_per_pixel` | float | Pixel-to-mm scale: `image_depth_mm / image_height_px` |

**Key formulas:**

```
PEN = α − aponeurosis_angle
FL  = thickness / sin(PEN)
thickness = (deep_y(x) − superficial_y(x)) × cos(superficial_angle)
```

---

### Fascicle Seed Selection Variables

| Variable | Type | Description |
|----------|------|-------------|
| `score` | float | Total candidate score (weighted sum of sub-scores) |
| `mask_support` | float | Dilated mask intersection fraction along segment |
| `raw_mask_support` | float | Undilated mask intersection fraction |
| `lateral_span` | float | Normalized segment width across the frame |
| `inside_muscle` | float | Distance-based score for segment being between aponeuroses |
| `phi_score` / `pennation_score` | float | Soft range penalties for anatomical plausibility |
| `boundary_penalty` | float | Exponential decay for segments leaving the frame |
| `cluster_id` | str | Cluster label: `a{alpha:.2f}_x{xmid:.0f}_lp{length:.0f}` |
| `frame_coverage` | int | Number of unique frames in a cluster |
| `cluster_score` | float | Aggregate stability score for a cluster |
| `selected_alpha_deg` | float | Final chosen fascicle angle from best cluster |
| `selected_seed_segment` | array | Final chosen fascicle segment `[x1,y1,x2,y2]` |

**Configuration** (`FascicleSeedScoringConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `angle_min_deg` | 14.0 | Minimum candidate angle (degrees) |
| `angle_max_deg` | 24.0 | Maximum candidate angle (degrees) |
| `angle_step_deg` | 0.1 | Angle grid step (degrees) |
| `top_peak_limit` | 10 | Maximum Hough peaks to consider |
| `min_cluster_frame_coverage` | 8 | Minimum frames for a stable cluster |
| `weight_mask_support` | 0.35 | Weight for dilated mask support |
| `weight_raw_mask_support` | 0.18 | Weight for raw mask support |
| `weight_hough` | 0.05 | Weight for Hough peak strength |
| `weight_lateral_span` | 0.12 | Weight for lateral span |
| `weight_phi` | 0.12 | Weight for phi score |
| `weight_pennation` | 0.08 | Weight for pennation plausibility |
| `weight_inside_muscle` | 0.08 | Weight for inside-muscle score |
| `weight_boundary` | 0.01 | Weight for boundary penalty |

---

## Repository Structure

```
NDORMS_Bedard_PythonNathTrack/
│
├── READ_ME.md                          # This file
├── daily_log.md                        # Development progress log
├── requirements                        # Python dependencies
│
├── ultrasound_tracker/                 # Core Python package (v6.0.0)
│   ├── __init__.py                     # Public API exports
│   ├── geometry.py                     # Line/angle/intersection geometry
│   ├── roi.py                          # ROI selection & persistence
│   ├── matlab_compat.py                # MATLAB compatibility helpers
│   ├── utils.py                        # Plotting & visualization
│   ├── plot_timtrack.py                # TimTrack-specific plotting
│   │
│   ├── matlab_timtrack.py              # Main TimTrack geofeature pipeline
│   ├── timtrack_hough.py               # Hough transform + weighted median
│   ├── matlab_aponeurosis.py           # Aponeurosis Hough detection
│   ├── ultratimtrack_aponeurosis.py    # Aponeurosis state tracking
│   │
│   ├── strict_fascicle_seed.py         # Autonomous seed selection (NOVEL)
│   ├── speckle_confidence.py           # Speckle confidence factor (NOVEL)
│   │
│   ├── ultratrack_klt.py               # KLT optical-flow tracking
│   ├── ultratimtrack_matlab_2state.py  # 2-state Kalman filter
│   ├── final_output.py                 # ANG/PEN/FL computation
│   │
│   └── legacy/                         # Earlier prototype modules
│
├── scripts/                            # Runnable entry points
│   ├── strict_ultratimtrack_gui.py     # GUI launcher (primary)
│   ├── run_strict_ultratimtrack_video.py  # CLI runner
│   ├── run_ultratimtrack_adaptive_confidence.py  # Adaptive-R wrapper
│   └── run_new_video.py                # New-video processing script
│
├── data/
│   ├── raw/                            # Input ultrasound videos
│   └── rois/                           # ROI JSON files
│
└── results/                            # Output directory (CSV, NPZ, plots, MP4)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/OOEC-Engineers/NDORMS_Bedard_PythonNathTrack.git
cd NDORMS_Bedard_PythonNathTrack

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements
```

---

## Usage

### GUI Mode

The GUI is the recommended entry point. It exposes all pipeline parameters
and writes each analysis run to `results/strict_ultratimtrack_runs/`.

```bash
python scripts/strict_ultratimtrack_gui.py
```

The GUI defaults to the current best configuration:
- Conditional Hough localmax fallback
- Automatic fascicle angle orientation
- Candidate persistence
- Aponeurosis gating
- Adaptive anisotropic Kalman (with fixed-R Kalman saved as comparison)

Each run produces:
- **CSV** — frame-by-frame ANG, PEN, FL
- **NPZ** — compressed arrays for all intermediate states
- **JSON** — run metadata and parameters
- **PNG** — time-series plots
- **MP4** — annotated video with overlaid tracking

### Command-Line Mode

```bash
python scripts/run_strict_ultratimtrack_video.py \
    data/raw/your_video.mp4 \
    --roi-path data/rois/your_roi.json
```

### Adaptive Confidence Mode

To enable the speckle confidence factor with adaptive measurement noise:

```bash
python scripts/run_ultratimtrack_adaptive_confidence.py \
    data/raw/your_video.mp4 \
    --roi-path data/rois/your_roi.json \
    --kalman-mode adaptive-anisotropic \
    --save-confidence-plots \
    --compare-to-fixed-kalman
```

---

## Dependencies

```
opencv-contrib-python
numpy
scipy
matplotlib
filterpy
imageio
scikit-image
pandas
pillow
jupyter
pytest
```

---

## Acknowledgements

- **Original MATLAB UltraTimTrack**: [timvanderzee/UltraTimTrack](https://github.com/timvanderzee/UltraTimTrack)
- **NDORMS, University of Oxford** — Jack Tu Lab
- Developed as part of a collaboration between OOEC-Engineers and the
  Nuffield Department of Orthopaedics, Rheumatology and Musculoskeletal Sciences
- Thank you to my mom, Nathalie Beauregard, who passed away on june 2rd 2026, who pushed me to come to Oxford and pursue my goal of bridging the engineering world to the medecine world
---

## Version

Current version: **6.0.0**
