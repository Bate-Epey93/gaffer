"""Statistical helpers shared by the model modules.

Deliberately dependency-light: numpy is used where it helps, but every
function here has a pure-Python fallback path so unit tests stay fast.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

try:  # pragma: no cover - numpy is in requirements but keep the import soft
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def poisson_sf(k: int, lam: float) -> float:
    """P(X > k)."""
    return max(0.0, 1.0 - poisson_cdf(k, lam))


def poisson_at_least(k: int, lam: float) -> float:
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    return poisson_sf(k - 1, lam)


def p_clean_sheet(lam_conceded: float) -> float:
    """P(0 goals conceded) under Poisson."""
    return math.exp(-max(0.0, lam_conceded))


def poisson_vector(lam: float, max_k: int = 12) -> List[float]:
    return [poisson_pmf(k, lam) for k in range(0, max_k + 1)]


# ---------------------------------------------------------------------------
# Negative binomial - used where counts are overdispersed relative to Poisson
# (defensive contributions, saves, shots).
# ---------------------------------------------------------------------------


def negbin_pmf(k: int, mean: float, dispersion: float) -> float:
    """Negative binomial with mean `mean` and variance mean + mean^2/dispersion.

    dispersion -> infinity recovers the Poisson.
    """
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    if dispersion <= 0 or dispersion > 1e6:
        return poisson_pmf(k, mean)
    r = dispersion
    p = r / (r + mean)
    return math.exp(
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p) + k * math.log(1 - p)
    )


def negbin_at_least(k: int, mean: float, dispersion: float, max_k: int = 60) -> float:
    """P(X >= k) for the negative binomial."""
    if k <= 0:
        return 1.0
    below = sum(negbin_pmf(i, mean, dispersion) for i in range(0, k))
    return max(0.0, min(1.0, 1.0 - below))


def fit_dispersion(mean: float, variance: float) -> float:
    """Method-of-moments dispersion from an observed mean and variance."""
    if variance <= mean or mean <= 0:
        return 1e6  # effectively Poisson
    return (mean * mean) / (variance - mean)


# ---------------------------------------------------------------------------
# Dixon-Coles correction for low-score dependence
# ---------------------------------------------------------------------------


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(
    lam_home: float, lam_away: float, rho: float = -0.06, max_goals: int = 10
) -> List[List[float]]:
    """Joint probability of every scoreline, Dixon-Coles adjusted."""
    home = poisson_vector(lam_home, max_goals)
    away = poisson_vector(lam_away, max_goals)
    matrix = []
    total = 0.0
    for x in range(max_goals + 1):
        row = []
        for y in range(max_goals + 1):
            p = home[x] * away[y] * dixon_coles_tau(x, y, lam_home, lam_away, rho)
            p = max(p, 0.0)
            row.append(p)
            total += p
        matrix.append(row)
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]
    return matrix


def match_outcome_probs(
    lam_home: float, lam_away: float, rho: float = -0.06, max_goals: int = 10
) -> Tuple[float, float, float]:
    """(P(home win), P(draw), P(away win))."""
    m = score_matrix(lam_home, lam_away, rho, max_goals)
    home = draw = away = 0.0
    for x, row in enumerate(m):
        for y, p in enumerate(row):
            if x > y:
                home += p
            elif x == y:
                draw += p
            else:
                away += p
    return home, draw, away


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------


def remove_vig(odds: Sequence[float]) -> List[float]:
    """Convert decimal odds to normalised implied probabilities."""
    raw = [1.0 / o for o in odds if o and o > 0]
    total = sum(raw)
    if total <= 0:
        return []
    return [r / total for r in raw]


def lambdas_from_odds(
    p_home: float, p_draw: float, p_away: float, total_goals_line: float = 2.7
) -> Tuple[float, float]:
    """Recover (lambda_home, lambda_away) that reproduce given 1X2 probabilities.

    Solves by bisection on the goal supremacy while holding the total fixed.
    """
    lo, hi = -3.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        lam_h = max(0.05, (total_goals_line + mid) / 2.0)
        lam_a = max(0.05, (total_goals_line - mid) / 2.0)
        ph, _, pa = match_outcome_probs(lam_h, lam_a)
        if (ph - pa) < (p_home - p_away):
            lo = mid
        else:
            hi = mid
    supremacy = (lo + hi) / 2.0
    return (
        max(0.05, (total_goals_line + supremacy) / 2.0),
        max(0.05, (total_goals_line - supremacy) / 2.0),
    )


# ---------------------------------------------------------------------------
# Shrinkage and blending
# ---------------------------------------------------------------------------


def shrink(
    observed_rate: float, sample_90s: float, prior_rate: float, prior_strength: float
) -> float:
    """Empirical-Bayes shrinkage of a per-90 rate towards a prior.

    sample_90s is the number of 90-minute equivalents behind observed_rate.
    """
    if sample_90s <= 0:
        return prior_rate
    w = sample_90s / (sample_90s + max(prior_strength, 1e-9))
    return w * observed_rate + (1.0 - w) * prior_rate


def beta_binomial_rate(
    successes: float, trials: float, prior_rate: float, prior_strength: float
) -> float:
    """Posterior mean of a rate with a Beta prior expressed as pseudo-trials."""
    a = prior_rate * prior_strength + successes
    b = (1.0 - prior_rate) * prior_strength + (trials - successes)
    if a + b <= 0:
        return prior_rate
    return a / (a + b)


def weighted_blend(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None and w > 0]
    if not pairs:
        return 0.0
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in pairs) / total_w


def exponential_decay_weight(days_ago: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (days_ago / half_life_days)


# ---------------------------------------------------------------------------
# Top-k probabilities (bonus points)
# ---------------------------------------------------------------------------


def top_k_probabilities(
    scores: Dict[int, float], sd: Dict[int, float], k: int = 3, iterations: int = 2000,
    seed: int = 7,
) -> Dict[int, List[float]]:
    """Monte-Carlo P(rank 1), P(rank 2), P(rank 3) for a set of noisy scores.

    Returns {player_id: [p_first, p_second, p_third]}. Ties are split evenly,
    matching FPL's rule that tied BPS scores share the higher bonus.
    """
    ids = list(scores.keys())
    if not ids:
        return {}
    result = {i: [0.0] * k for i in ids}
    if np is not None:
        rng = np.random.default_rng(seed)
        mu = np.array([scores[i] for i in ids], dtype=float)
        sigma = np.array([max(sd.get(i, 1.0), 1e-6) for i in ids], dtype=float)
        draws = rng.normal(mu, sigma, size=(iterations, len(ids)))
        order = np.argsort(-draws, axis=1)
        for rank in range(min(k, len(ids))):
            counts = np.bincount(order[:, rank], minlength=len(ids))
            for idx, cid in enumerate(ids):
                result[cid][rank] = counts[idx] / float(iterations)
        return result
    # Pure-python fallback
    import random

    rnd = random.Random(seed)
    for _ in range(iterations):
        draw = sorted(
            ((scores[i] + rnd.gauss(0, max(sd.get(i, 1.0), 1e-6)), i) for i in ids),
            reverse=True,
        )
        for rank in range(min(k, len(draw))):
            result[draw[rank][1]][rank] += 1.0 / iterations
    return result


# ---------------------------------------------------------------------------
# Scoring metrics for backtesting
# ---------------------------------------------------------------------------


def rmse(pred: Sequence[float], actual: Sequence[float]) -> float:
    pairs = list(zip(pred, actual))
    if not pairs:
        return float("nan")
    return math.sqrt(sum((p - a) ** 2 for p, a in pairs) / len(pairs))


def mae(pred: Sequence[float], actual: Sequence[float]) -> float:
    pairs = list(zip(pred, actual))
    if not pairs:
        return float("nan")
    return sum(abs(p - a) for p, a in pairs) / len(pairs)


def spearman(pred: Sequence[float], actual: Sequence[float]) -> float:
    pairs = list(zip(pred, actual))
    n = len(pairs)
    if n < 2:
        return float("nan")

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for m in range(i, j + 1):
                out[order[m]] = avg
            i = j + 1
        return out

    rp, ra = ranks([p for p, _ in pairs]), ranks([a for _, a in pairs])
    mp, ma_ = sum(rp) / n, sum(ra) / n
    num = sum((rp[i] - mp) * (ra[i] - ma_) for i in range(n))
    den = math.sqrt(sum((rp[i] - mp) ** 2 for i in range(n)) * sum((ra[i] - ma_) ** 2 for i in range(n)))
    return num / den if den else float("nan")
