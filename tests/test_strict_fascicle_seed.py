import numpy as np

from ultrasound_tracker.matlab_timtrack import fascicle_segment_from_aponeuroses_and_alpha
from ultrasound_tracker.strict_fascicle_seed import (
    FascicleSeedScoringConfig,
    cluster_seed_candidates,
    extract_fascicle_seed_candidates,
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
