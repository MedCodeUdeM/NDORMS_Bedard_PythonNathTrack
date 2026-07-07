import numpy as np

from scripts.run_strict_ultratimtrack_video import (
    FascicleCandidatePersistenceConfig,
    apply_aponeurosis_maxangle_overrides,
    apply_fascicle_angle_overrides,
    candidate_signed_fascicle_angle_ranges,
    current_fascicle_angle_range,
    draw_overlay_frame,
    fascicle_angle_abs_bounds,
    kalman_comparison_rows,
    kalman_mode_uses_confidence,
    prompt_kalman_mode,
    run_fascicle_kalman_mode,
    select_best_fascicle_angle_range_result,
    select_fascicle_candidate_persistence,
)
from ultrasound_tracker.ultratimtrack_matlab_2state import MatlabTwoStateKalmanConfig


def _toy_inputs():
    klt = np.array(
        [
            [80.0, 10.0, 30.0, 60.0],
            [81.0, 10.0, 31.0, 60.0],
        ],
        dtype=float,
    )
    superficial = np.tile(np.array([[1.0, 10.0, 101.0, 10.0]]), (2, 1))
    deep = np.tile(np.array([[1.0, 60.0, 101.0, 60.0]]), (2, 1))
    alpha = np.array([45.0, 44.0])
    config = MatlabTwoStateKalmanConfig(
        x_measurement_variance=100.0,
        alpha_measurement_variance=3.0,
        run_smoother=False,
    )
    return klt, alpha, superficial, deep, config


def test_kalman_mode_confidence_flags():
    assert not kalman_mode_uses_confidence("fixed")
    assert kalman_mode_uses_confidence("adaptive-scalar")
    assert kalman_mode_uses_confidence("adaptive-anisotropic")


def test_prompt_kalman_mode_accepts_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert prompt_kalman_mode(default="fixed") == "fixed"


def test_prompt_kalman_mode_accepts_number_and_alias(monkeypatch):
    answers = iter(["2", "anisotropic"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert prompt_kalman_mode(default="fixed") == "adaptive-scalar"
    assert prompt_kalman_mode(default="fixed") == "adaptive-anisotropic"


def test_aponeurosis_maxangle_override_updates_exported_parms_copy():
    parms = {"apo": {"super": {"maxangle": 0.5}, "deep": {"maxangle": 0.0}}}

    apply_aponeurosis_maxangle_overrides(parms, apo_maxangle=10.0, deep_apo_maxangle=6.0)

    assert parms["apo"]["super"]["maxangle"] == 10.0
    assert parms["apo"]["deep"]["maxangle"] == 6.0
    assert parms["apo"]["super"]["fit_method"] == "enforce_maxangle"
    assert parms["apo"]["deep"]["fit_method"] == "enforce_maxangle"


def test_fascicle_angle_override_updates_timtrack_range():
    parms = {"fas": {"range": np.array([10.0, 40.0])}}

    apply_fascicle_angle_overrides(parms, fas_angle_min=5.0, fas_angle_max=55.0)

    np.testing.assert_allclose(parms["fas"]["range"], [5.0, 55.0])


def test_fascicle_angle_auto_candidates_include_reverse_signed_range():
    parms = {"fas": {"range": np.array([5.0, 60.0])}}

    assert current_fascicle_angle_range(parms) == (5.0, 60.0)
    assert fascicle_angle_abs_bounds(parms) == (5.0, 60.0)
    assert candidate_signed_fascicle_angle_ranges(parms) == [(5.0, 60.0), (-60.0, -5.0)]


def test_fascicle_angle_auto_candidates_preserve_negative_current_range_first():
    parms = {"fas": {"range": np.array([-50.0, -10.0])}}

    assert current_fascicle_angle_range(parms) == (-50.0, -10.0)
    assert fascicle_angle_abs_bounds(parms) == (10.0, 50.0)
    assert candidate_signed_fascicle_angle_ranges(parms) == [(-50.0, -10.0), (10.0, 50.0)]


def test_fascicle_angle_auto_near_tie_prefers_positive_orientation():
    scored = [
        {"angle_min_deg": -60.0, "angle_max_deg": -5.0, "score": 0.9716},
        {"angle_min_deg": 5.0, "angle_max_deg": 60.0, "score": 0.9651},
    ]

    best = select_best_fascicle_angle_range_result(scored)

    assert best["angle_min_deg"] == 5.0
    assert best["angle_max_deg"] == 60.0


def test_fascicle_angle_auto_keeps_clear_negative_winner():
    scored = [
        {"angle_min_deg": -60.0, "angle_max_deg": -5.0, "score": 0.98},
        {"angle_min_deg": 5.0, "angle_max_deg": 60.0, "score": 0.93},
    ]

    best = select_best_fascicle_angle_range_result(scored)

    assert best["angle_min_deg"] == -60.0
    assert best["angle_max_deg"] == -5.0


def test_candidate_persistence_prefers_near_previous_candidate_on_raw_jump():
    geofeatures = [
        {
            "alphas": np.array([20.0]),
            "weights": np.array([1.0]),
            "x": np.array([[1.0, 20.0]]),
            "y": np.array([[10.0, 40.0]]),
        },
        {
            "alphas": np.array([21.0, 40.0]),
            "weights": np.array([0.8, 1.0]),
            "x": np.array([[1.0, 20.0], [1.0, 20.0]]),
            "y": np.array([[11.0, 42.0], [10.0, 80.0]]),
        },
    ]

    out = select_fascicle_candidate_persistence(
        geofeatures,
        np.array([20.0, 40.0]),
        config=FascicleCandidatePersistenceConfig(
            enabled=True,
            angle_min_deg=5.0,
            angle_max_deg=60.0,
            max_angle_step_deg=5.0,
            hough_weight_bonus_deg=2.0,
        ),
    )

    np.testing.assert_allclose(out["selected_alpha_deg"], [20.0, 21.0])
    assert out["selected_candidate_idx"][1] == 0
    assert bool(out["raw_alpha_rejected"][1])
    assert "selected candidate" in out["selection_reason"][1]
    assert any(row["selected"] for row in out["candidate_rows"] if row["Frame"] == 1)


def test_candidate_persistence_disabled_keeps_raw_alpha_but_writes_rows():
    geofeatures = [
        {
            "alphas": np.array([20.0, 35.0]),
            "weights": np.array([0.5, 1.0]),
            "x": np.array([[1.0, 20.0], [1.0, 20.0]]),
            "y": np.array([[10.0, 40.0], [10.0, 70.0]]),
        }
    ]

    out = select_fascicle_candidate_persistence(
        geofeatures,
        np.array([35.0]),
        config=FascicleCandidatePersistenceConfig(enabled=False),
    )

    np.testing.assert_allclose(out["selected_alpha_deg"], [35.0])
    assert not bool(out["raw_alpha_rejected"][0])
    assert len(out["candidate_rows"]) == 2
    assert len(out["selection_rows"]) == 1


def test_candidate_persistence_handles_reverse_negative_angle_range():
    geofeatures = [
        {
            "alphas": np.array([-20.0]),
            "weights": np.array([1.0]),
            "x": np.array([[1.0, 20.0]]),
            "y": np.array([[40.0, 10.0]]),
        },
        {
            "alphas": np.array([-21.0, -40.0]),
            "weights": np.array([0.8, 1.0]),
            "x": np.array([[1.0, 20.0], [1.0, 20.0]]),
            "y": np.array([[42.0, 11.0], [80.0, 10.0]]),
        },
    ]

    out = select_fascicle_candidate_persistence(
        geofeatures,
        np.array([-20.0, -40.0]),
        config=FascicleCandidatePersistenceConfig(
            enabled=True,
            angle_min_deg=-60.0,
            angle_max_deg=-5.0,
            max_angle_step_deg=5.0,
            hough_weight_bonus_deg=2.0,
        ),
    )

    np.testing.assert_allclose(out["selected_alpha_deg"], [-20.0, -21.0])
    assert out["selected_candidate_idx"][1] == 0
    assert bool(out["raw_alpha_rejected"][1])


def test_overlay_can_draw_fixed_red_and_adaptive_blue():
    frame = np.zeros((30, 40), dtype=np.uint8)
    arrays = {
        "frame": np.array([0]),
        "sup_apo_lines": np.array([[1.0, 2.0, 40.0, 2.0]]),
        "deep_apo_lines": np.array([[1.0, 28.0, 40.0, 28.0]]),
        "klt_prior_segments": np.array([[1.0, 15.0, 40.0, 15.0]]),
        "fascicle_segments": np.array([[1.0, 20.0, 40.0, 20.0]]),
        "fixed_fascicle_segments": np.array([[1.0, 10.0, 40.0, 10.0]]),
        "ANG_deg": np.array([20.0]),
        "PEN_deg": np.array([10.0]),
        "FL_px": np.array([30.0]),
    }

    vis = draw_overlay_frame(frame, {}, arrays, 0, show_kalman_comparison=True)

    assert np.any(np.all(vis == np.array([0, 0, 255], dtype=np.uint8), axis=2))
    assert np.any(np.all(vis == np.array([255, 0, 0], dtype=np.uint8), axis=2))


def test_script_scalar_adaptive_mode_scales_both_measurements():
    klt, alpha, superficial, deep, config = _toy_inputs()

    out = run_fascicle_kalman_mode(
        "adaptive-scalar",
        klt,
        alpha,
        superficial,
        deep,
        config,
        confidence_arrays={"r_scale": np.array([1.0, 5.0])},
    )

    np.testing.assert_allclose(out["measurement_R_diag"][1], [500.0, 15.0])


def test_script_anisotropic_mode_keeps_length_fixed_when_only_theta_is_low():
    klt, alpha, superficial, deep, config = _toy_inputs()

    out = run_fascicle_kalman_mode(
        "adaptive-anisotropic",
        klt,
        alpha,
        superficial,
        deep,
        config,
        confidence_arrays={
            "r_scale": np.array([1.0, 5.0]),
            "r_scale_theta": np.array([1.0, 5.0]),
            "r_scale_length": np.array([1.0, 1.0]),
        },
    )

    np.testing.assert_allclose(out["measurement_R_diag"][1], [100.0, 15.0])


def test_kalman_comparison_rows_report_selected_minus_fixed_rmse():
    selected = {
        "ANG_deg": np.array([2.0, 4.0, 6.0]),
        "PEN_deg": np.array([10.0, 9.0, 8.0]),
        "FL_mm": np.array([50.0, 52.0, 54.0]),
    }
    fixed = {
        "ANG_deg": np.array([1.0, 2.0, 3.0]),
        "PEN_deg": np.array([10.0, 10.0, 10.0]),
        "FL_mm": np.array([49.0, 49.0, 49.0]),
    }

    rows = {row["metric"]: row for row in kalman_comparison_rows(selected, fixed)}

    assert rows["ANG"]["unit"] == "deg"
    assert rows["FL"]["unit"] == "mm"
    np.testing.assert_allclose(rows["ANG"]["mean_delta"], 2.0)
    np.testing.assert_allclose(rows["ANG"]["rmse_delta"], np.sqrt((1.0 + 4.0 + 9.0) / 3.0))
    np.testing.assert_allclose(rows["PEN"]["mean_delta"], -1.0)
    np.testing.assert_allclose(rows["FL"]["max_abs_delta"], 5.0)
