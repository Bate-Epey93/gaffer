"""Backtesting: replay a completed season and score the model against it.

``backtest_season`` is the entry point; ``metrics`` holds the scoring functions,
which are pure and reusable on any (prediction, actual) pair.
"""
from __future__ import annotations

from gaffer.backtest.metrics import (
    calibration,
    captain_metrics,
    evaluate_projections,
    full_report,
    hit_rate_top_n,
    mae,
    merge_pred_actual,
    rmse,
    spearman,
    verdict,
)
from gaffer.backtest.runner import (
    SeasonReplay,
    backtest_season,
    format_report,
    report_path,
    write_report,
)

__all__ = [
    "SeasonReplay",
    "backtest_season",
    "calibration",
    "captain_metrics",
    "evaluate_projections",
    "format_report",
    "full_report",
    "hit_rate_top_n",
    "mae",
    "merge_pred_actual",
    "report_path",
    "rmse",
    "spearman",
    "verdict",
    "write_report",
]
