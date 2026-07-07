#!/usr/bin/env python3
"""Run UltraTimTrack with the optional ultrasound confidence/adaptive-R gate.

Examples
--------
Fixed-R baseline, same behavior as the strict runner:
    python scripts/run_ultratimtrack_adaptive_confidence.py data/raw/UltraTimTrack_test.mp4 \
        --roi-path data/rois/UltraTimTrack_test_rois.json

Adaptive-R confidence mode:
    python scripts/run_ultratimtrack_adaptive_confidence.py data/raw/UltraTimTrack_test.mp4 \
        --roi-path data/rois/UltraTimTrack_test_rois.json \
        --kalman-mode adaptive-anisotropic --save-confidence-plots --compare-to-fixed-kalman

This script intentionally delegates to ``run_strict_ultratimtrack_video.py`` so
the validated UltraTimTrack implementation remains the single executable path.
"""

from __future__ import annotations

from run_strict_ultratimtrack_video import main


if __name__ == "__main__":
    main()
