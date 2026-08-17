"""Bookmaker odds for upcoming fixtures, from The Odds API.

Why this exists. Before a ball is kicked the model's team ratings are fitted on
last season's results, and three of this season's clubs have no Premier League
history at all. The betting market has none of those problems: it has priced in
the summer's transfers, the manager changes and this morning's team news, and it
is the single sharpest publicly available forecast of a football match. Feeding
its implied goal rates into the fixture forecast is the cheapest real accuracy
gain available to a model that cannot afford paid stats.

The model already knew how to use odds — ``TeamRatingModel.set_odds`` converts a
1X2 price plus an over/under into (lambda_home, lambda_away), and the backtest
has been fitting against historical prices all along. What was missing was
anything fetching *live* prices for fixtures that have not been played. This is
that piece.

The budget is the whole design constraint. The free tier is 500 requests a
month; the site rebuilds hourly, which is 720. So a naive fetch-per-build
exhausts the quota in three weeks and then silently stops working. Instead:

  - one request covers every upcoming Premier League fixture, not one per match
  - the response is cached to disk and reused until it is older than
    ``min_refresh_hours`` (default 6), so ~120 requests a month, comfortably
    inside the free tier with room for manual builds
  - the remaining monthly quota is read back from the response headers and
    logged, so running dry is visible before it happens
  - every failure is non-fatal: no key, no network, quota exhausted, malformed
    response all degrade to "no odds", and the model falls back to its own
    fitted ratings exactly as it does today

Never put the key in the exported site. It is read from ODDS_API_KEY in the
environment, used server-side or in CI, and the derived lambdas are what ship.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_epl"
CACHE_NAME = "odds_epl.json"

# One request buys every upcoming fixture in one region/market combination.
DEFAULT_REGION = "uk"
DEFAULT_MARKETS = "h2h,totals"
DEFAULT_MIN_REFRESH_HOURS = 6.0

# Team names as The Odds API writes them -> FPL short codes. Anything missing is
# reported rather than guessed: a silently unmatched club means that fixture
# quietly keeps the fitted rating while its neighbours use the market, which is
# the kind of inconsistency that is very hard to see in an output table.
ODDS_NAME_TO_SHORT: Dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "AFC Bournemouth": "BOU",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton and Hove Albion": "BHA",
    "Brighton & Hove Albion": "BHA",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Coventry City": "COV",
    "Coventry": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull City": "HUL",
    "Hull": "HUL",
    "Ipswich Town": "IPS",
    "Ipswich": "IPS",
    "Leeds United": "LEE",
    "Leeds": "LEE",
    "Leicester City": "LEI",
    "Liverpool": "LIV",
    "Luton Town": "LUT",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle United": "NEW",
    "Nottingham Forest": "NFO",
    "Sheffield United": "SHU",
    "Southampton": "SOU",
    "Sunderland": "SUN",
    "Tottenham Hotspur": "TOT",
    "Tottenham": "TOT",
    "West Ham United": "WHU",
    "Wolverhampton Wanderers": "WOL",
    "Wolves": "WOL",
}


class OddsUnavailable(RuntimeError):
    """Odds could not be fetched. Always non-fatal to the caller."""


def api_key(explicit: Optional[str] = None) -> Optional[str]:
    key = explicit or os.environ.get("ODDS_API_KEY") or ""
    key = key.strip()
    return key or None


def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, CACHE_NAME)


def _read_cache(cache_dir: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (ValueError, OSError):
        return None


def _write_cache(cache_dir: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(_cache_path(cache_dir), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        pass  # a cache we cannot write is a slower next run, not a failure


def cache_age_hours(cache_dir: str) -> Optional[float]:
    entry = _read_cache(cache_dir)
    if not entry or not entry.get("fetched_at"):
        return None
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"])
    except ValueError:
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0


def _fetch(key: str, region: str, markets: str, timeout: float) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    query = urllib.parse.urlencode({
        "apiKey": key,
        "regions": region,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    })
    url = "%s/sports/%s/odds/?%s" % (API_BASE, SPORT_KEY, query)
    request = urllib.request.Request(url, headers={"User-Agent": "gaffer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            headers = {
                "remaining": response.headers.get("x-requests-remaining") or "",
                "used": response.headers.get("x-requests-used") or "",
            }
            return body, headers
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:200]
        except Exception:
            pass
        if exc.code == 401:
            raise OddsUnavailable("ODDS_API_KEY rejected (401). %s" % detail)
        if exc.code == 429:
            raise OddsUnavailable("odds quota exhausted (429). %s" % detail)
        raise OddsUnavailable("odds API returned HTTP %d. %s" % (exc.code, detail))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        raise OddsUnavailable("could not reach the odds API: %s" % exc)


def _best_prices(event: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Best available price per outcome across bookmakers.

    Taking the best price per outcome slightly over-rounds the book, but the
    conversion downstream removes the vig anyway, and the alternative — trusting
    a single bookmaker — is noisier.
    """
    home_name = event.get("home_team")
    away_name = event.get("away_team")
    out: Dict[str, float] = {}
    for book in event.get("bookmakers") or []:
        for market in book.get("markets") or []:
            key = market.get("key")
            for outcome in market.get("outcomes") or []:
                price = outcome.get("price")
                if not isinstance(price, (int, float)) or price <= 1.0:
                    continue
                name = outcome.get("name")
                if key == "h2h":
                    if name == home_name:
                        out["home"] = max(out.get("home", 0.0), float(price))
                    elif name == away_name:
                        out["away"] = max(out.get("away", 0.0), float(price))
                    elif name == "Draw":
                        out["draw"] = max(out.get("draw", 0.0), float(price))
                elif key == "totals":
                    # Only the 2.5 line; it is the one the conversion expects.
                    if abs(float(outcome.get("point", 0)) - 2.5) > 1e-9:
                        continue
                    if name == "Over":
                        out["over25"] = max(out.get("over25", 0.0), float(price))
                    elif name == "Under":
                        out["under25"] = max(out.get("under25", 0.0), float(price))
    if not all(k in out for k in ("home", "draw", "away")):
        return None
    return out


def load_odds(
    cache_dir: str,
    key: Optional[str] = None,
    min_refresh_hours: float = DEFAULT_MIN_REFRESH_HOURS,
    region: str = DEFAULT_REGION,
    markets: str = DEFAULT_MARKETS,
    timeout: float = 20.0,
    force: bool = False,
) -> Dict[str, Any]:
    """Odds for upcoming fixtures, from cache when it is fresh enough.

    Never raises: the return always carries ``events`` (possibly empty) and a
    human-readable ``status`` explaining what happened.
    """
    cached = _read_cache(cache_dir)
    age = cache_age_hours(cache_dir)
    if cached and not force and age is not None and age < min_refresh_hours:
        return {
            "events": cached.get("events") or [],
            "status": "cache (%.1fh old, refresh after %.0fh)" % (age, min_refresh_hours),
            "fetched_at": cached.get("fetched_at"),
            "remaining": cached.get("remaining"),
            "from_cache": True,
        }

    token = api_key(key)
    if not token:
        return {
            "events": (cached or {}).get("events") or [],
            "status": "no ODDS_API_KEY set; the model uses its own fitted ratings"
                      + (" (serving a stale cache)" if cached else ""),
            "from_cache": bool(cached),
            "remaining": None,
        }

    try:
        events, headers = _fetch(token, region, markets, timeout)
    except OddsUnavailable as exc:
        # A stale cache beats nothing: yesterday's market is far closer to the
        # truth than a rating fitted before the transfer window shut.
        return {
            "events": (cached or {}).get("events") or [],
            "status": "%s%s" % (exc, " (serving a stale cache)" if cached else ""),
            "from_cache": bool(cached),
            "remaining": None,
        }

    payload = {
        "events": events,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "remaining": headers.get("remaining"),
        "used": headers.get("used"),
    }
    _write_cache(cache_dir, payload)
    return {
        "events": events,
        "status": "fetched %d event(s); %s request(s) left this month"
                  % (len(events), headers.get("remaining") or "?"),
        "fetched_at": payload["fetched_at"],
        "remaining": headers.get("remaining"),
        "from_cache": False,
    }


def apply_to_model(model: Any, state: Any, odds: Dict[str, Any]) -> Dict[str, Any]:
    """Push market prices into a fitted TeamRatingModel.

    Matches on (home short code, away short code) and only for fixtures that
    have not kicked off. Returns a report: how many fixtures were priced, and
    every club name the map did not recognise.
    """
    events = odds.get("events") or []
    if not events:
        return {"applied": 0, "unmatched_names": [], "unmatched_fixtures": 0,
                "status": odds.get("status", "no odds")}

    short_by_id = {}
    for team in state.teams.values() if hasattr(state.teams, "values") else state.teams:
        short_by_id[int(team.id)] = team.short_name

    # (home_short, away_short) -> prices
    priced: Dict[Tuple[str, str], Dict[str, float]] = {}
    unmatched: set = set()
    for event in events:
        home = ODDS_NAME_TO_SHORT.get(event.get("home_team") or "")
        away = ODDS_NAME_TO_SHORT.get(event.get("away_team") or "")
        if not home:
            unmatched.add(event.get("home_team") or "?")
        if not away:
            unmatched.add(event.get("away_team") or "?")
        if not home or not away:
            continue
        prices = _best_prices(event)
        if prices:
            priced[(home, away)] = prices

    applied = 0
    unmatched_fixtures = 0
    for fixture in state.fixtures:
        if getattr(fixture, "finished", False):
            continue
        home = short_by_id.get(int(fixture.team_h))
        away = short_by_id.get(int(fixture.team_a))
        prices = priced.get((home, away))
        if not prices:
            unmatched_fixtures += 1
            continue
        ok = model.set_odds(
            int(fixture.id),
            prices["home"], prices["draw"], prices["away"],
            prices.get("over25"), prices.get("under25"),
        )
        if ok:
            applied += 1

    if applied:
        # Only worth trusting the market path once something is actually priced.
        model.use_odds = True

    return {
        "applied": applied,
        "unmatched_names": sorted(unmatched),
        "unmatched_fixtures": unmatched_fixtures,
        "status": odds.get("status", ""),
    }
