"""
Plot TimTrack / UltraTimTrack CSV results.

Run directly from the terminal without importing the full ultrasound_tracker package.

Examples
--------
From the NDORMS project root:

    python3 ultrasound_tracker/plot_timtrack.py results/Test2_timtrack_final_features.csv

Selected variables:

    python3 ultrasound_tracker/plot_timtrack.py results/Test2_timtrack_final_features.csv --values ANG_deg PEN_deg FL_mm --smooth 25

Save figures:

    python3 ultrasound_tracker/plot_timtrack.py results/Test2_timtrack_final_features.csv --values ANG_deg PEN_deg FL_mm --smooth 25 --save results/plots/timtrack.png

List CSV columns:

    python3 ultrasound_tracker/plot_timtrack.py results/Test2_timtrack_final_features.csv --list-columns
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_VALUES = [
    "ANG_deg",
    "PEN_deg",
    "FL_mm",
    "FL_px",
    "fascicle_angle_deg",
    "pennation_angle_deg",
    "fascicle_length_px",
    "final_fascicle_length_px",
    "super_apo_angle_deg",
    "deep_apo_angle_deg",
    "muscle_thickness_px",
    "dohough_mask_density",
    "n_fascicle_candidates",
]


def _safe_filename(name):
    """
    Convert a column name into a safe filename.
    """
    safe = str(name)

    for char in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        safe = safe.replace(char, "_")

    return safe


def _get_output_path(save_path, column_name, n_values):
    """
    Decide where to save each figure.

    Cases
    -----
    1. save_path is a folder:
        results/plots
        -> results/plots/ANG_deg.png

    2. save_path is a file and multiple columns are plotted:
        results/plots/timtrack.png
        -> results/plots/timtrack_ANG_deg.png

    3. save_path is a file and only one column is plotted:
        results/plots/timtrack.png
        -> results/plots/timtrack.png
    """

    save_path = Path(save_path)
    safe_col = _safe_filename(column_name)

    if save_path.suffix == "":
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path / f"{safe_col}.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if n_values == 1:
        return save_path

    return save_path.with_name(f"{save_path.stem}_{safe_col}{save_path.suffix}")


def _apply_axis_style(ax):
    """
    Add tick marks and grid lines to a matplotlib axis.
    """

    ax.minorticks_on()

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=7,
        width=1.0,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=4,
        width=0.8,
    )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.8,
        alpha=0.45,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.6,
        alpha=0.30,
    )

    ax.set_axisbelow(True)


def plot_timtrack_results(
    csv_path,
    values=None,
    x_col=None,
    smooth_window=15,
    success_only=True,
    save_path=None,
    show=True,
    figsize=(11, 7),
    dpi=300,
    baseline_duration_s=2.0,
    baseline_points=30,
):
    """
    Plot TimTrack / UltraTimTrack values over time with drift visualization.

    Each selected variable is plotted in its own independent figure.

    Each figure contains:
    - top panel: raw data + smoothed data + baseline
    - bottom panel: drift = smoothed data - baseline
    - major tick marks
    - minor tick marks
    - major grid
    - minor grid

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    values : list[str] or None
        Columns to plot.
        If None, DEFAULT_VALUES are used.

    x_col : str or None
        Column used for the x-axis.
        If None, uses 'time_s' if available, otherwise 'frame'.

    smooth_window : int or None
        Rolling mean window for smoothing.
        Example: 10, 15, 25.
        Use 1 to disable smoothing.

    success_only : bool
        If True, only rows with success == True are plotted.

    save_path : str or Path or None
        Optional path to save figures.
        Can be a file path or a folder path.

    show : bool
        If True, displays the figures.

    figsize : tuple
        Size of each independent figure.

    dpi : int
        Figure resolution when saving.

    baseline_duration_s : float
        Duration in seconds used to compute the baseline when x_col is time_s.

    baseline_points : int
        Number of first points used to compute the baseline if time_s is not available.

    Returns
    -------
    figures : list
        List of (fig, ax_main, ax_drift) tuples.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if success_only and "success" in df.columns:
        success_mask = df["success"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        df = df[success_mask].copy()

    if x_col is None:
        if "time_s" in df.columns:
            x_col = "time_s"
        elif "frame" in df.columns:
            x_col = "frame"
        else:
            x_col = None

    if x_col is not None:
        if x_col not in df.columns:
            raise ValueError(
                f"x_col '{x_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        x = pd.to_numeric(df[x_col], errors="coerce")
        x_label = "Time (s)" if x_col == "time_s" else x_col

    else:
        x = pd.Series(df.index, index=df.index)
        x_label = "Index"

    if values is None:
        values = DEFAULT_VALUES

    available_values = [col for col in values if col in df.columns]
    missing_values = [col for col in values if col not in df.columns]

    for col in missing_values:
        print(f"Skipping missing column: {col}")

    if len(available_values) == 0:
        raise ValueError(
            "No valid columns selected to plot.\n"
            f"Available columns: {list(df.columns)}"
        )

    figures = []

    for col in available_values:
        y_raw = pd.to_numeric(df[col], errors="coerce")

        valid_mask = ~(pd.isna(x) | pd.isna(y_raw))
        x_plot = x[valid_mask]
        y_raw_plot = y_raw[valid_mask]

        if len(y_raw_plot) == 0:
            print(f"Skipping {col}: no valid numeric data.")
            continue

        if smooth_window is not None and smooth_window > 1:
            y_smooth = y_raw_plot.rolling(
                window=smooth_window,
                center=True,
                min_periods=1,
            ).mean()
            smooth_label = f"Smoothed data, window={smooth_window}"
        else:
            y_smooth = y_raw_plot.copy()
            smooth_label = "Smoothed data disabled"

        # ------------------------------------------------------------
        # Baseline calculation
        # ------------------------------------------------------------
        if x_col == "time_s":
            x_start = x_plot.iloc[0]
            baseline_mask = x_plot <= x_start + baseline_duration_s

            if baseline_mask.sum() < 3:
                baseline_mask = pd.Series(False, index=x_plot.index)
                baseline_mask.iloc[: min(baseline_points, len(x_plot))] = True

        else:
            baseline_mask = pd.Series(False, index=x_plot.index)
            baseline_mask.iloc[: min(baseline_points, len(x_plot))] = True

        baseline_value = y_smooth[baseline_mask].mean()

        if pd.isna(baseline_value):
            baseline_value = y_smooth.iloc[0]

        drift = y_smooth - baseline_value
        final_drift = drift.iloc[-1]

        # ------------------------------------------------------------
        # Drift slope approximation
        # ------------------------------------------------------------
        if x_col == "time_s":
            duration_min = (x_plot.iloc[-1] - x_plot.iloc[0]) / 60.0

            if duration_min > 0:
                drift_slope = final_drift / duration_min
            else:
                drift_slope = 0.0

            drift_slope_text = f"{drift_slope:.3f} {col}/min"

        else:
            duration_points = len(x_plot)

            if duration_points > 0:
                drift_slope = final_drift / duration_points
            else:
                drift_slope = 0.0

            drift_slope_text = f"{drift_slope:.3f} {col}/point"

        # ------------------------------------------------------------
        # Figure with 2 panels
        # ------------------------------------------------------------
        fig, (ax_main, ax_drift) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # ------------------------------------------------------------
        # Main plot: raw + smoothed + baseline
        # ------------------------------------------------------------
        ax_main.plot(
            x_plot,
            y_raw_plot,
            label="Raw data",
            linewidth=1.0,
            alpha=0.45,
        )

        ax_main.plot(
            x_plot,
            y_smooth,
            label=smooth_label,
            linewidth=2.2,
        )

        ax_main.axhline(
            baseline_value,
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline = {baseline_value:.3f}",
        )

        ax_main.set_title(f"{col} over time with drift — {csv_path.name}")
        ax_main.set_ylabel(col)

        drift_text = (
            f"Baseline = {baseline_value:.3f}\n"
            f"Final drift = {final_drift:.3f}\n"
            f"Drift slope = {drift_slope_text}"
        )

        ax_main.text(
            0.02,
            0.95,
            drift_text,
            transform=ax_main.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.18),
        )

        ax_main.legend(loc="best")

        # ------------------------------------------------------------
        # Drift plot
        # ------------------------------------------------------------
        ax_drift.plot(
            x_plot,
            drift,
            label="Drift from baseline",
            linewidth=1.8,
        )

        ax_drift.axhline(
            0,
            linestyle="--",
            linewidth=1.2,
        )

        ax_drift.set_ylabel("Drift")
        ax_drift.set_xlabel(x_label)
        ax_drift.legend(loc="best")

        # ------------------------------------------------------------
        # Ticks and grids
        # ------------------------------------------------------------
        _apply_axis_style(ax_main)
        _apply_axis_style(ax_drift)

        fig.tight_layout()

        # ------------------------------------------------------------
        # Save
        # ------------------------------------------------------------
        if save_path is not None:
            out_path = _get_output_path(
                save_path=save_path,
                column_name=col,
                n_values=len(available_values),
            )

            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure to: {out_path}")

        figures.append((fig, ax_main, ax_drift))

    if show:
        plt.show()

    return figures


def list_columns(csv_path):
    """
    Print all available columns in the CSV file.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("Available columns:")

    for col in df.columns:
        print(f"  - {col}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot TimTrack / UltraTimTrack CSV results over time."
    )

    parser.add_argument(
        "csv_path",
        help="Path to the TimTrack CSV file.",
    )

    parser.add_argument(
        "--values",
        nargs="+",
        default=None,
        help=(
            "Columns to plot. "
            "Example: --values ANG_deg PEN_deg FL_mm"
        ),
    )

    parser.add_argument(
        "--x-col",
        default=None,
        help=(
            "Column for x-axis. "
            "Default: time_s if available, otherwise frame."
        ),
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=15,
        help=(
            "Rolling mean smoothing window. "
            "Use 1 to disable smoothing. Default: 15."
        ),
    )

    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=2.0,
        help=(
            "Number of initial seconds used to compute baseline "
            "when x-axis is time_s. Default: 2.0."
        ),
    )

    parser.add_argument(
        "--baseline-points",
        type=int,
        default=30,
        help=(
            "Number of initial points used to compute baseline "
            "when time_s is not available. Default: 30."
        ),
    )

    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Plot all rows, including rows where success is False.",
    )

    parser.add_argument(
        "--save",
        default=None,
        help=(
            "Optional path to save figures. "
            "Example file: --save results/plots/timtrack.png "
            "Example folder: --save results/plots"
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures. Useful when only saving.",
    )

    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print available CSV columns and exit.",
    )

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(11, 7),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size. Default: --figsize 11 7",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures. Default: 300.",
    )

    args = parser.parse_args()

    if args.list_columns:
        list_columns(args.csv_path)
        return

    plot_timtrack_results(
        csv_path=args.csv_path,
        values=args.values,
        x_col=args.x_col,
        smooth_window=args.smooth,
        success_only=not args.all_rows,
        save_path=args.save,
        show=not args.no_show,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        baseline_duration_s=args.baseline_seconds,
        baseline_points=args.baseline_points,
    )


if __name__ == "__main__":
    main()

# python3 ultrasound_tracker/plot_timtrack.py results/Test2_timtrack_final_features.csv --smooth 25
#