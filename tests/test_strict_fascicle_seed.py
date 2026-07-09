import numpy as np
import pandas as pd

from ultrasound_tracker.matlab_timtrack import fascicle_segment_from_aponeuroses_and_alpha
from ultrasound_tracker.strict_fascicle_seed import (
    FascicleSeedScoringConfig,
    choose_stable_seed_cluster,
    cluster_seed_candidates,
    extract_fascicle_seed_candidates,
    sample_segment_points,
    score_fascicle_seed_candidate,
    select_autonomous_fascicle_seed,
)


def _entry(frame: int = 0):
    gamma = np.arange(14.0, 24.5, 0.5)
    return {
        "frame": frame,
        "fascicle_masked": np.ones((80, 120), dtype=bool),
        "super_coef": np.array([0.0, 15.0]),
        "deep_coef": np.array([0.0, 65.0]),
        "super_coef_linear": np.array([0.0, 15.0]),
        "deep_coef_linear": np.array([0.0, 65.0]),
        "betha": 0.0,
        "gamma": 0.0,
        "alphas": np.array([17.5, 19.0]),
        "weights": np.array([10.0, 8.0]),
        "hough_result": {
            "gamma": gamma,
            "h_by_angle": np.exp(-0.5 * ((gamma - 17.5) / 1.5) ** 2),
        },
    }


def test_score_fascicle_seed_candidate_returns_finite_score():
    entry = _entry()
    segment = fascicle_segment_from_aponeuroses_and_alpha(
        entry["super_coef"],
        entry["deep_coef"],
        17.5,
        120,
        super_coef_linear_1b=entry["super_coef_linear"],
        deep_coef_linear_1b=entry["deep_coef_linear"],
    )

    scores = score_fascicle_seed_candidate(entry, segment, 17.5, mm_per_px=0.1, frame_shape=(80, 120))

    assert scores["score"] > 0.0
    assert scores["mask_support_score"] == 1.0
    assert scores["inside_muscle_score"] > 0.9
    assert scores["length_px"] > 0.0


def test_seed_scoring_is_independent_of_pixel_to_mm_scale():
    entry = _entry()
    segment = fascicle_segment_from_aponeuroses_and_alpha(
        entry["super_coef"],
        entry["deep_coef"],
        17.5,
        120,
        super_coef_linear_1b=entry["super_coef_linear"],
        deep_coef_linear_1b=entry["deep_coef_linear"],
    )

    matlab_depth_score = score_fascicle_seed_candidate(
        entry,
        segment,
        17.5,
        mm_per_px=50.7 / 800.0,
        frame_shape=(80, 120),
    )
    entered_depth_score = score_fascicle_seed_candidate(
        entry,
        segment,
        17.5,
        mm_per_px=69.0 / 800.0,
        frame_shape=(80, 120),
    )

    assert matlab_depth_score["length_mm"] != entered_depth_score["length_mm"]
    assert matlab_depth_score["length_score"] == entered_depth_score["length_score"]
    assert matlab_depth_score["score"] == entered_depth_score["score"]


def test_seed_scoring_prefers_mask_supported_segment_over_stale_length_prior():
    height, width = 80, 120
    super_coef = np.array([0.0, 15.0])
    deep_coef = np.array([0.0, 65.0])
    steep_alpha = 55.0
    shallow_alpha = 20.0
    steep_segment = fascicle_segment_from_aponeuroses_and_alpha(
        super_coef,
        deep_coef,
        steep_alpha,
        width,
        super_coef_linear_1b=super_coef,
        deep_coef_linear_1b=deep_coef,
    )
    shallow_segment = fascicle_segment_from_aponeuroses_and_alpha(
        super_coef,
        deep_coef,
        shallow_alpha,
        width,
        super_coef_linear_1b=super_coef,
        deep_coef_linear_1b=deep_coef,
    )

    mask = np.zeros((height, width), dtype=bool)
    xs, ys = sample_segment_points(steep_segment, n_samples=300)
    for x, y in zip(xs, ys):
        xi = int(round(x)) - 1
        yi = int(round(y)) - 1
        mask[max(0, yi - 2) : min(height, yi + 3), max(0, xi - 2) : min(width, xi + 3)] = True

    entry = {
        "fascicle_masked": mask,
        "super_coef": super_coef,
        "deep_coef": deep_coef,
        "super_coef_linear": super_coef,
        "deep_coef_linear": deep_coef,
        "betha": 0.0,
        "gamma": 0.0,
        "hough_result": {
            "gamma": np.array([shallow_alpha, steep_alpha]),
            "h_by_angle": np.array([1.0, 1.0]),
        },
    }

    steep_scores = score_fascicle_seed_candidate(
        entry,
        steep_segment,
        steep_alpha,
        mm_per_px=0.5,
        frame_shape=(height, width),
    )
    shallow_scores = score_fascicle_seed_candidate(
        entry,
        shallow_segment,
        shallow_alpha,
        mm_per_px=0.5,
        frame_shape=(height, width),
    )

    assert steep_scores["mask_support_score"] > shallow_scores["mask_support_score"]
    assert steep_scores["score"] > shallow_scores["score"]


def test_stable_seed_cluster_prefers_near_tied_hough_branch():
    candidates = pd.DataFrame(
        [
            {"frame": frame, "candidate_source": "hough_peak", "alpha_deg": 13.0, "score": 0.5}
            for frame in range(8)
        ]
    )
    clusters = pd.DataFrame(
        [
            {
                "cluster_id": "a45.00_x400_l25",
                "cluster_score": 0.975,
                "median_alpha_deg": 45.0,
                "mean_hough_score": 0.55,
                "mean_mask_support": 0.55,
            },
            {
                "cluster_id": "a13.00_x400_l75",
                "cluster_score": 0.960,
                "median_alpha_deg": 13.0,
                "mean_hough_score": 0.78,
                "mean_mask_support": 0.39,
            },
        ]
    )

    selected = choose_stable_seed_cluster(candidates, clusters)

    assert selected["cluster_id"] == "a13.00_x400_l75"


def test_stable_seed_cluster_keeps_clear_top_cluster():
    candidates = pd.DataFrame(
        [
            {"frame": frame, "candidate_source": "hough_peak", "alpha_deg": 13.0, "score": 0.5}
            for frame in range(8)
        ]
    )
    clusters = pd.DataFrame(
        [
            {
                "cluster_id": "a45.00_x400_l25",
                "cluster_score": 0.990,
                "median_alpha_deg": 45.0,
                "mean_hough_score": 0.55,
                "mean_mask_support": 0.55,
            },
            {
                "cluster_id": "a13.00_x400_l75",
                "cluster_score": 0.940,
                "median_alpha_deg": 13.0,
                "mean_hough_score": 0.78,
                "mean_mask_support": 0.39,
            },
        ]
    )

    selected = choose_stable_seed_cluster(candidates, clusters)

    assert selected["cluster_id"] == "a45.00_x400_l25"


def test_extract_cluster_and_select_autonomous_seed():
    entries = [_entry(frame=i) for i in range(9)]
    frames = [np.zeros((80, 120), dtype=np.uint8) for _ in entries]
    config = FascicleSeedScoringConfig(
        angle_min_deg=17.5,
        angle_max_deg=19.0,
        angle_step_deg=1.5,
        min_cluster_frame_coverage=8,
    )

    candidates = extract_fascicle_seed_candidates(entries, frames, mm_per_px=0.1, config=config)
    candidates, clusters = cluster_seed_candidates(candidates, min_frame_coverage=8)
    selection = select_autonomous_fascicle_seed(candidates, clusters, entries[0], width_px=120)

    assert not candidates.empty
    assert not clusters.empty
    assert np.isfinite(selection["selected_alpha_deg"])
    assert np.all(np.isfinite(selection["selected_seed_segment"]))
    assert len(selection["per_frame_best"]) >= 8
