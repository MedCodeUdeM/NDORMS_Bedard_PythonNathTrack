from scripts.strict_ultratimtrack_gui import _has_metric_values, _read_metrics_csv


def test_read_metrics_csv_loads_fixed_kalman_comparison_columns(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Frame,Time,FL,PEN,ANG,fixed_FL_mm,fixed_PEN_deg,fixed_ANG_deg",
                "0,0.0,22.0,45.0,45.5,23.0,44.0,44.5",
                "1,0.1,24.0,40.0,40.5,25.0,39.0,39.5",
            ]
        )
        + "\n"
    )

    metrics = _read_metrics_csv(csv_path)

    assert metrics["FL"] == [22.0, 24.0]
    assert metrics["PEN"] == [45.0, 40.0]
    assert metrics["ANG"] == [45.5, 40.5]
    assert metrics["FixedFL"] == [23.0, 25.0]
    assert metrics["FixedPEN"] == [44.0, 39.0]
    assert metrics["FixedANG"] == [44.5, 39.5]
    assert _has_metric_values(metrics["FixedFL"])


def test_read_metrics_csv_keeps_fixed_columns_empty_when_not_comparing(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("Frame,Time,FL,PEN,ANG\n0,0.0,22.0,45.0,45.5\n")

    metrics = _read_metrics_csv(csv_path)

    assert metrics["FL"] == [22.0]
    assert metrics["FixedFL"] == []
    assert not _has_metric_values(metrics["FixedFL"])
