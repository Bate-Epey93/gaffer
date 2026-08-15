"""Scoring a set of projections against what actually happened.

Everything here operates on a single tidy frame with one row per
(player, gameweek) and one column per predictor, so the model and every
baseline are measured on exactly the same rows. That matters more than it
sounds: a metric computed over a different universe is not a comparison, and
the whole point of this module is the comparison.

Three families of number, in increasing order of how much they should change a
decision:

1. **Fit** — RMSE, MAE, Spearman. Familiar, and mostly a measure of how well
   you predict zeros. Reported overall, by position and by price tier.
2. **Decision quality** — top-20 hit rate and captain accuracy. These are what
   an FPL manager actually consumes: who to own and who to double. A model can
   win on RMSE and lose here, and if it does, the RMSE is the number to ignore.
3. **Calibration** — is a stated 70% chance of 60 minutes really a 70% chance?
   A miscalibrated minutes model corrupts every other component, because every
   component is multiplied by it.

RMSE and MAE are meaningless for a predictor on a different scale (price, in
tenths of a million), so rank-only predictors are marked as such and their
error columns come back as NaN rather than as an impressive-looking number.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gaffer.core import scoring

# Price bands in £m. Chosen to separate the decisions a manager actually makes:
# bench fodder, budget starters, mid-price, premium and elite.
PRICE_TIERS: Tuple[Tuple[str, float, float], ...] = (
    ("<=4.5", 0.0, 4.5),
    ("4.6-5.5", 4.5, 5.5),
    ("5.6-7.0", 5.5, 7.0),
    ("7.1-9.0", 7.0, 9.0),
    (">9.0", 9.0, 1e9),
)

CALIBRATION_BINS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)

TOP_N_DEFAULT = 20


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def price_tier(price: float) -> str:
    """Label for a price in £m."""
    for label, lo, hi in PRICE_TIERS:
        if lo < price <= hi or (lo == 0.0 and price <= hi):
            return label
    return PRICE_TIERS[-1][0]


def _pairs(pred: Sequence[float], actual: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    n = min(len(p), len(a))
    p, a = p[:n], a[:n]
    ok = np.isfinite(p) & np.isfinite(a)
    return p[ok], a[ok]


def rmse(pred: Sequence[float], actual: Sequence[float]) -> float:
    p, a = _pairs(pred, actual)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - a) ** 2)))


def mae(pred: Sequence[float], actual: Sequence[float]) -> float:
    p, a = _pairs(pred, actual)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - a)))


def spearman(pred: Sequence[float], actual: Sequence[float]) -> float:
    """Rank correlation, ties averaged.

    Identical to ``core.stats.spearman`` (a unit test asserts that) but written
    on pandas ranks because this runs over ~25k rows per season and the pure
    Python version is a minute of the wall clock.
    """
    p, a = _pairs(pred, actual)
    if p.size < 2:
        return float("nan")
    rp = pd.Series(p).rank()
    ra = pd.Series(a).rank()
    if rp.std(ddof=0) == 0 or ra.std(ddof=0) == 0:
        return float("nan")
    return float(rp.corr(ra))


def hit_rate_top_n(
    pred: Sequence[float], actual: Sequence[float], n: int = TOP_N_DEFAULT
) -> float:
    """Share of the n highest-predicted that landed in the actual top n.

    Ties in the actual points are resolved by threshold, not by an arbitrary
    cut: everyone who scored at least as much as the nth best counts as top n.
    Ties in the prediction are cut arbitrarily but deterministically, since a
    manager picking a squad has to cut them somewhere too.
    """
    p, a = _pairs(pred, actual)
    if p.size == 0:
        return float("nan")
    n = min(n, p.size)
    top_pred = np.argsort(-p, kind="stable")[:n]
    threshold = np.sort(a)[::-1][n - 1]
    return float(np.mean(a[top_pred] >= threshold))


# ---------------------------------------------------------------------------
# Per-predictor scoring
# ---------------------------------------------------------------------------


def score_predictor(
    frame: pd.DataFrame,
    pred_col: str,
    actual_col: str = "actual_points",
    gw_col: str = "gw",
    rank_only: bool = False,
    top_n: int = TOP_N_DEFAULT,
) -> Dict[str, Any]:
    """Fit, decision and bias metrics for one prediction column."""
    sub = frame[[pred_col, actual_col, gw_col]].dropna(subset=[pred_col, actual_col])
    pred = sub[pred_col].to_numpy(dtype=float)
    actual = sub[actual_col].to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "n": int(len(sub)),
        "rank_only": bool(rank_only),
        "rmse": float("nan") if rank_only else rmse(pred, actual),
        "mae": float("nan") if rank_only else mae(pred, actual),
        "spearman": spearman(pred, actual),
        "mean_pred": float(np.mean(pred)) if len(pred) else float("nan"),
        "mean_actual": float(np.mean(actual)) if len(actual) else float("nan"),
        "bias": float("nan") if rank_only else (
            float(np.mean(pred - actual)) if len(pred) else float("nan")),
    }
    # Rank metrics are per gameweek: a top-20 that pools the whole season is a
    # different (and much easier) question than "who do I own this week".
    per_gw_spearman: List[float] = []
    per_gw_hit: List[float] = []
    for _, grp in sub.groupby(gw_col):
        if len(grp) < 2:
            continue
        per_gw_spearman.append(spearman(grp[pred_col], grp[actual_col]))
        per_gw_hit.append(hit_rate_top_n(grp[pred_col], grp[actual_col], top_n))
    good = [v for v in per_gw_spearman if v == v]
    out["spearman_per_gw_mean"] = float(np.mean(good)) if good else float("nan")
    # A predictor that is undefined in a gameweek (last gameweek's points at
    # GW1 is a column of zeros) contributes no rank correlation there, so the
    # per-gameweek mean covers fewer gameweeks. Report it or the comparison is
    # quietly unlike-for-unlike.
    out["n_gws_ranked"] = len(good)
    out["n_gws"] = int(sub[gw_col].nunique())
    good_hit = [v for v in per_gw_hit if v == v]
    out["top%d_hit_rate" % top_n] = float(np.mean(good_hit)) if good_hit else float("nan")
    out["top%d_hit_rate_per_gw" % top_n] = [round(v, 4) for v in per_gw_hit]
    return out


def captain_metrics(
    frame: pd.DataFrame,
    pred_col: str,
    actual_col: str = "actual_points",
    gw_col: str = "gw",
    id_col: str = "player_id",
) -> Dict[str, Any]:
    """How good is "captain the highest projected player"?

    ``accuracy`` is the share of gameweeks where that pick was (one of) the
    highest actual scorer(s). ``mean_points_lost`` is the gap to the perfect
    captain, doubled, because the captain armband doubles the miss as well as
    the hit — this is the number that decides seasons.
    """
    picks: List[Dict[str, Any]] = []
    for gw, grp in frame.dropna(subset=[pred_col, actual_col]).groupby(gw_col):
        if grp.empty:
            continue
        # Deterministic tie-break on player id so a rerun picks the same player.
        ordered = grp.sort_values([pred_col, id_col], ascending=[False, True])
        pick = ordered.iloc[0]
        scores = np.sort(grp[actual_col].to_numpy(dtype=float))[::-1]
        best = float(scores[0])
        actual = float(pick[actual_col])
        picks.append({
            "gw": int(gw),
            "player_id": int(pick[id_col]),
            "pred": float(pick[pred_col]),
            "actual": actual,
            "best_actual": best,
            "points_lost": best - actual,
            # Being the single top scorer out of ~800 is a lottery; landing in
            # the top handful is the thing a captain pick can actually control.
            "in_top5": bool(actual >= scores[min(4, len(scores) - 1)]),
            "in_top10": bool(actual >= scores[min(9, len(scores) - 1)]),
            "field_mean": float(grp[actual_col].mean()),
        })
    if not picks:
        return {"n_gws": 0, "accuracy": float("nan"), "mean_points_lost": float("nan")}
    actual = np.array([p["actual"] for p in picks], dtype=float)
    best = np.array([p["best_actual"] for p in picks], dtype=float)
    return {
        "n_gws": len(picks),
        "accuracy": float(np.mean(actual >= best)),
        "top5_rate": float(np.mean([p["in_top5"] for p in picks])),
        "top10_rate": float(np.mean([p["in_top10"] for p in picks])),
        "mean_captain_points": float(np.mean(actual)),
        "mean_perfect_points": float(np.mean(best)),
        "mean_field_points": float(np.mean([p["field_mean"] for p in picks])),
        "mean_points_lost": float(np.mean(best - actual)),
        "mean_points_lost_doubled": float(2.0 * np.mean(best - actual)),
        "total_points_lost_doubled": float(2.0 * np.sum(best - actual)),
        "blanks": int(np.sum(actual <= 2)),
        "picks": picks,
    }


def split_metrics(
    frame: pd.DataFrame,
    pred_col: str,
    by: str,
    actual_col: str = "actual_points",
    rank_only: bool = False,
    order: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """``score_predictor`` per group, keyed by the group label."""
    out: Dict[str, Dict[str, Any]] = {}
    keys = list(order) if order is not None else sorted(frame[by].dropna().unique())
    for key in keys:
        sub = frame[frame[by] == key]
        if sub.empty:
            continue
        scored = score_predictor(sub, pred_col, actual_col, rank_only=rank_only)
        scored.pop("top%d_hit_rate_per_gw" % TOP_N_DEFAULT, None)
        out[str(key)] = scored
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration(
    frame: pd.DataFrame,
    prob_col: str,
    outcome_col: str,
    bins: Sequence[float] = CALIBRATION_BINS,
) -> Dict[str, Any]:
    """Bucketed reliability of a probability against a 0/1 outcome.

    Returns the per-bucket table plus a Brier score and the expected
    calibration error (the count-weighted mean absolute gap), which is the
    single number to watch: over 0.05 and the probability is decorative.
    """
    sub = frame[[prob_col, outcome_col]].dropna()
    if sub.empty:
        return {"n": 0, "brier": float("nan"), "ece": float("nan"), "buckets": []}
    p = sub[prob_col].to_numpy(dtype=float)
    y = sub[outcome_col].to_numpy(dtype=float)
    buckets: List[Dict[str, Any]] = []
    total_gap = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            buckets.append({"lo": lo, "hi": min(hi, 1.0), "n": 0,
                            "mean_pred": float("nan"), "realised": float("nan"),
                            "gap": float("nan")})
            continue
        mean_pred = float(np.mean(p[mask]))
        realised = float(np.mean(y[mask]))
        buckets.append({"lo": lo, "hi": min(hi, 1.0), "n": count,
                        "mean_pred": mean_pred, "realised": realised,
                        "gap": mean_pred - realised})
        total_gap += count * abs(mean_pred - realised)
    return {
        "n": int(len(sub)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": total_gap / float(len(sub)),
        "mean_pred": float(np.mean(p)),
        "realised": float(np.mean(y)),
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# The SPEC 3.13 entry point
# ---------------------------------------------------------------------------


def evaluate_projections(pred: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
    """Headline metrics for one set of projections against one set of actuals.

    ``pred`` needs ``player_id``, ``gw`` and ``xp``; ``actual`` needs
    ``player_id``, ``gw`` and ``total_points``. Optional columns are used when
    present: ``position``/``element_type`` and ``price`` add the splits,
    ``minutes`` plus ``p_60`` add the minutes calibration.

    The return type is a flat ``Dict[str, float]`` as the spec declares; the
    richer nested breakdowns come from ``full_report``.
    """
    merged = merge_pred_actual(pred, actual)
    if merged.empty:
        return {"n": 0.0}
    scored = score_predictor(merged, "xp")
    out: Dict[str, float] = {
        "n": float(scored["n"]),
        "rmse": scored["rmse"],
        "mae": scored["mae"],
        "spearman": scored["spearman"],
        "spearman_per_gw_mean": scored["spearman_per_gw_mean"],
        "top20_hit_rate": scored["top%d_hit_rate" % TOP_N_DEFAULT],
        "mean_pred": scored["mean_pred"],
        "mean_actual": scored["mean_actual"],
        "bias": scored["bias"],
    }
    cap = captain_metrics(merged, "xp")
    out["captain_accuracy"] = float(cap["accuracy"])
    out["captain_points_lost"] = float(cap["mean_points_lost"])
    if "p_60" in merged.columns and "played_60" in merged.columns:
        cal = calibration(merged, "p_60", "played_60")
        out["p60_brier"] = float(cal["brier"])
        out["p60_ece"] = float(cal["ece"])
    return out


def merge_pred_actual(pred: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (player_id, gw), adding the derived columns metrics need."""
    keys = ["player_id", "gw"]
    right = actual.copy()
    if "actual_points" not in right.columns and "total_points" in right.columns:
        right = right.rename(columns={"total_points": "actual_points"})
    merged = pred.merge(right, on=keys, how="inner", suffixes=("", "_actual"))
    if "minutes" in merged.columns:
        merged["played_60"] = (merged["minutes"] >= scoring.CLEAN_SHEET_MIN_MINUTES).astype(float)
        merged["played"] = (merged["minutes"] > 0).astype(float)
    if "starts" in merged.columns:
        merged["started"] = (merged["starts"] > 0).astype(float)
    if "position" not in merged.columns and "element_type" in merged.columns:
        merged["position"] = merged["element_type"]
    if "position" in merged.columns and "pos" not in merged.columns:
        merged["pos"] = merged["position"].map(scoring.POS_NAME)
    if "price" in merged.columns and "price_tier" not in merged.columns:
        merged["price_tier"] = merged["price"].map(price_tier)
    return merged


# ---------------------------------------------------------------------------
# Full comparison report
# ---------------------------------------------------------------------------


def full_report(
    frame: pd.DataFrame,
    predictors: Sequence[Tuple[str, str, bool]],
    actual_col: str = "actual_points",
    top_n: int = TOP_N_DEFAULT,
) -> Dict[str, Any]:
    """Score every predictor on identical rows.

    ``predictors`` is a sequence of ``(key, column, rank_only)``. The first
    entry is treated as the model under test; everything after it is a baseline
    it has to beat.
    """
    out: Dict[str, Any] = {}
    for key, column, rank_only in predictors:
        if column not in frame.columns:
            continue
        entry = score_predictor(frame, column, actual_col, rank_only=rank_only, top_n=top_n)
        entry["column"] = column
        entry["captain"] = captain_metrics(frame, column, actual_col)
        entry["by_position"] = split_metrics(
            frame, column, "pos", actual_col, rank_only=rank_only,
            order=[scoring.POS_NAME[p] for p in scoring.POSITIONS],
        )
        entry["by_price_tier"] = split_metrics(
            frame, column, "price_tier", actual_col, rank_only=rank_only,
            order=[t[0] for t in PRICE_TIERS],
        )
        out[key] = entry
    return out


def verdict(report: Dict[str, Any], model_key: str = "model") -> Dict[str, Any]:
    """Does the model actually beat the baselines? Plain answer, no hedging.

    A baseline counts as beaten on a metric only if the model is strictly
    better. Ties go to the baseline: the baselines are free and the model is
    not, so matching a baseline is a loss.
    """
    model = report.get(model_key)
    lines: List[str] = []
    if model is None:
        return {"beats_all": False, "lines": ["no model results to judge"], "detail": {}}

    detail: Dict[str, Any] = {}
    beaten_everywhere = True
    hit_key = "top%d_hit_rate" % TOP_N_DEFAULT
    for key, entry in report.items():
        if key == model_key:
            continue
        comparisons: Dict[str, Any] = {}
        for metric, higher_is_better in (
            ("rmse", False), ("mae", False), ("spearman", True),
            ("spearman_per_gw_mean", True), (hit_key, True),
        ):
            mine, theirs = model.get(metric), entry.get(metric)
            if mine is None or theirs is None or mine != mine or theirs != theirs:
                comparisons[metric] = None  # not applicable (rank-only baseline)
                continue
            better = mine > theirs if higher_is_better else mine < theirs
            comparisons[metric] = {
                "model": mine, "baseline": theirs,
                "delta": mine - theirs, "model_better": bool(better),
            }
        cap_mine = model["captain"]["mean_points_lost"]
        cap_theirs = entry["captain"]["mean_points_lost"]
        comparisons["captain_points_lost"] = {
            "model": cap_mine, "baseline": cap_theirs,
            "delta": cap_mine - cap_theirs,
            "model_better": bool(cap_mine < cap_theirs),
        }
        applicable = [c for c in comparisons.values() if c is not None]
        won = sum(1 for c in applicable if c["model_better"])
        comparisons["_won"] = won
        comparisons["_of"] = len(applicable)
        # The decision metrics are the ones that matter; the fit metrics are
        # context. A model that loses on rank correlation has lost, full stop.
        decisive = comparisons.get("spearman_per_gw_mean") or comparisons.get("spearman")
        lost_decisive = decisive is not None and not decisive["model_better"]
        if lost_decisive:
            beaten_everywhere = False
        detail[key] = comparisons
        lines.append(
            "%-22s model %s baseline on rank correlation (%.4f vs %.4f), "
            "wins %d/%d metrics"
            % (key,
               "BEATS" if (decisive and decisive["model_better"]) else "DOES NOT BEAT",
               (decisive or {}).get("model", float("nan")),
               (decisive or {}).get("baseline", float("nan")),
               won, len(applicable))
        )
    return {"beats_all": bool(beaten_everywhere), "lines": lines, "detail": detail}


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str = "%.3f") -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN
            return "n/a"
    except TypeError:
        return str(value)
    return spec % value


def comparison_table(report: Dict[str, Any], labels: Dict[str, str]) -> str:
    """The headline table: every predictor, one row each."""
    hit_key = "top%d_hit_rate" % TOP_N_DEFAULT
    header = ("%-26s %7s %7s %7s %9s %6s %8s %8s %8s"
              % ("predictor", "RMSE", "MAE", "rho", "rho/GW", "nGW",
                 "top20", "capt%", "capt-lost"))
    lines = [header, "-" * len(header)]
    for key, entry in report.items():
        cap = entry.get("captain", {})
        lines.append(
            "%-26s %7s %7s %7s %9s %6d %8s %8s %8s"
            % (labels.get(key, key),
               _fmt(entry.get("rmse"), "%.3f"),
               _fmt(entry.get("mae"), "%.3f"),
               _fmt(entry.get("spearman"), "%.4f"),
               _fmt(entry.get("spearman_per_gw_mean"), "%.4f"),
               entry.get("n_gws_ranked", 0),
               _fmt(entry.get(hit_key), "%.3f"),
               _fmt(cap.get("accuracy"), "%.3f"),
               _fmt(cap.get("mean_points_lost"), "%.2f"))
        )
    return "\n".join(lines)


def split_table(report: Dict[str, Any], key: str, split: str, labels: Dict[str, str]) -> str:
    """One predictor's metrics broken down by position or price tier."""
    entry = report.get(key, {})
    table = entry.get(split, {})
    header = "%-12s %7s %7s %7s %8s %9s %9s" % (
        split.replace("by_", ""), "n", "RMSE", "MAE", "rho", "mean_pred", "mean_act")
    lines = [header, "-" * len(header)]
    for label, metrics in table.items():
        lines.append("%-12s %7d %7s %7s %8s %9s %9s" % (
            label, metrics["n"], _fmt(metrics["rmse"]), _fmt(metrics["mae"]),
            _fmt(metrics["spearman"], "%.4f"), _fmt(metrics["mean_pred"], "%.2f"),
            _fmt(metrics["mean_actual"], "%.2f")))
    return "\n".join(lines)


def calibration_table(cal: Dict[str, Any], title: str) -> str:
    header = "%-14s %8s %11s %11s %8s" % (title, "n", "predicted", "realised", "gap")
    lines = [header, "-" * len(header)]
    for bucket in cal.get("buckets", []):
        if not bucket["n"]:
            continue
        lines.append("%-14s %8d %11s %11s %8s" % (
            "%.1f-%.1f" % (bucket["lo"], bucket["hi"]), bucket["n"],
            _fmt(bucket["mean_pred"]), _fmt(bucket["realised"]), _fmt(bucket["gap"], "%+.3f")))
    lines.append("%-14s %8d %11s %11s %8s" % (
        "ALL", cal.get("n", 0), _fmt(cal.get("mean_pred")), _fmt(cal.get("realised")),
        _fmt((cal.get("mean_pred") or 0) - (cal.get("realised") or 0), "%+.3f")))
    lines.append("Brier %.4f, expected calibration error %.4f"
                 % (cal.get("brier", float("nan")), cal.get("ece", float("nan"))))
    return "\n".join(lines)


if __name__ == "__main__":
    # Metrics are pure functions of two columns, so the smoke test builds a
    # frame whose right answer is known by construction: a predictor that is
    # the truth plus noise must beat one that is pure noise on every metric.
    rng = np.random.default_rng(11)
    rows = []
    for gw in range(1, 11):
        for pid in range(1, 121):
            truth = float(rng.poisson(2.0)) + (6.0 if pid % 40 == 0 else 0.0)
            rows.append({
                "player_id": pid,
                "gw": gw,
                "actual_points": truth,
                "minutes": 90 if rng.random() < 0.6 else 0,
                "xp": truth * 0.7 + rng.normal(0, 1.0) + 0.6,
                "noise": rng.normal(0, 3.0),
                "price": 4.0 + (pid % 10),
                "position": 1 + (pid % 4),
                "p_60": min(0.99, max(0.01, rng.random())),
            })
    frame = merge_pred_actual(
        pd.DataFrame(rows).drop(columns=["actual_points", "minutes"]),
        pd.DataFrame(rows)[["player_id", "gw", "actual_points", "minutes"]],
    )
    rep = full_report(frame, [("model", "xp", False), ("noise", "noise", False)])
    print(comparison_table(rep, {"model": "signal + noise", "noise": "pure noise"}))
    print()
    print(split_table(rep, "model", "by_position", {}))
    print()
    print(calibration_table(calibration(frame, "p_60", "played_60"), "p60 bucket"))
    print()
    v = verdict(rep)
    for line in v["lines"]:
        print("  " + line)
    print("\nheadline evaluate_projections():")
    flat = evaluate_projections(
        pd.DataFrame(rows).drop(columns=["actual_points", "minutes"]),
        pd.DataFrame(rows)[["player_id", "gw", "actual_points", "minutes"]].rename(
            columns={"actual_points": "total_points"}),
    )
    for k in sorted(flat):
        print("  %-24s %s" % (k, _fmt(flat[k], "%.4f")))
    assert rep["model"]["spearman"] > rep["noise"]["spearman"]
    assert rep["model"]["rmse"] < rep["noise"]["rmse"]
    assert v["beats_all"] is True
    print("\nOK: the informative predictor beats the noise predictor on every metric.")
