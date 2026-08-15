"""Historical data sources: football-data.co.uk results/odds and the vaastav
FPL archive.

Both are static per season, so they cache for 24h by default rather than the
5-minute live TTL.

The single biggest trap here is team naming. Three vocabularies are in play:

    football-data.co.uk  "Man United"  "Tottenham"     "Nott'm Forest"
    FPL long name        "Man Utd"     "Spurs"         "Nott'm Forest"
    FPL short name       "MUN"         "TOT"           "NFO"

vaastav's ``merged_gw.csv`` uses the FPL long names, football-data uses its own.
Everything here normalises to the FPL **short name**, and an unmapped name is a
hard error listing exactly what failed — a silently dropped team would quietly
corrupt every team rating downstream.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
import requests

from gaffer.core.config import USER_AGENT
from gaffer.core.types import Team
from gaffer.data.cache import TTL_HISTORICAL, Cache

log = logging.getLogger(__name__)

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


class UnmappedTeamError(KeyError):
    """A team name in a source file has no entry in the mapping table."""


# football-data.co.uk name -> FPL short name. Covers every club that has
# appeared in E0 recently plus the 2026/27 promoted sides. Extend rather than
# guess: an unmapped name raises.
FOOTBALL_DATA_TO_FPL_SHORT: Dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Birmingham": "BIR",
    "Blackburn": "BLB",
    "Blackpool": "BLA",
    "Bolton": "BOL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Cardiff": "CAR",
    "Chelsea": "CHE",
    "Coventry": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Huddersfield": "HUD",
    "Hull": "HUL",
    "Ipswich": "IPS",
    "Leeds": "LEE",
    "Leicester": "LEI",
    "Liverpool": "LIV",
    "Luton": "LUT",
    "Man City": "MCI",
    "Man United": "MUN",
    "Middlesbrough": "MID",
    "Newcastle": "NEW",
    "Norwich": "NOR",
    "Nott'm Forest": "NFO",
    "Portsmouth": "POR",
    "QPR": "QPR",
    "Reading": "REA",
    "Sheffield United": "SHU",
    "Southampton": "SOU",
    "Stoke": "STK",
    "Sunderland": "SUN",
    "Swansea": "SWA",
    "Tottenham": "TOT",
    "Watford": "WAT",
    "West Brom": "WBA",
    "West Ham": "WHU",
    "Wigan": "WIG",
    "Wolves": "WOL",
}

# FPL long name (as used by bootstrap `teams[].name` and vaastav merged_gw)
# -> FPL short name. Only needed where the long name is not derivable.
FPL_LONG_TO_SHORT: Dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Coventry City": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull": "HUL",
    "Hull City": "HUL",
    "Ipswich": "IPS",
    "Ipswich Town": "IPS",
    "Coventry": "COV",
    "Leeds": "LEE",
    "Leeds United": "LEE",
    "Nott'm Forest": "NFO",
    "Nottingham Forest": "NFO",
    "Tottenham": "TOT",
    "Leicester": "LEI",
    "Liverpool": "LIV",
    "Luton": "LUT",
    "Man City": "MCI",
    "Man Utd": "MUN",
    "Newcastle": "NEW",
    "Nott'm Forest": "NFO",
    "Sheffield Utd": "SHU",
    "Southampton": "SOU",
    "Spurs": "TOT",
    "Sunderland": "SUN",
    "West Ham": "WHU",
    "Wolves": "WOL",
}

VAASTAV_POSITION_TO_ELEMENT_TYPE: Dict[str, int] = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}


def team_name_map() -> Dict[str, str]:
    """football-data.co.uk names -> FPL short names."""
    return dict(FOOTBALL_DATA_TO_FPL_SHORT)


# ---------------------------------------------------------------------------
# Season name handling. Three formats in play:
#   "2026-27"  config / vaastav directory
#   "2026/27"  FPL element-summary history_past season_name
#   "2627"     football-data.co.uk directory code
# ---------------------------------------------------------------------------

_SEASON_RE = re.compile(r"^(\d{4})[-/](\d{2})$")


def season_start_year(season: str) -> int:
    season = season.strip()
    m = _SEASON_RE.match(season)
    if m:
        return int(m.group(1))
    if re.match(r"^\d{4}$", season):  # football-data code, e.g. "2526"
        return 2000 + int(season[:2])
    raise ValueError("unrecognised season format: %r" % season)


def season_dash(season: str) -> str:
    """Canonical '2026-27' form."""
    y = season_start_year(season)
    return "%d-%02d" % (y, (y + 1) % 100)


def season_slash(season: str) -> str:
    """FPL history_past form, '2026/27'."""
    y = season_start_year(season)
    return "%d/%02d" % (y, (y + 1) % 100)


def fd_season_code(season: str) -> str:
    """football-data.co.uk directory code, '2627'."""
    y = season_start_year(season)
    return "%02d%02d" % (y % 100, (y + 1) % 100)


def prior_season(season: str, back: int = 1) -> str:
    y = season_start_year(season) - back
    return "%d-%02d" % (y, (y + 1) % 100)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _fetch_text(url: str, timeout: float = 90.0) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    # football-data ships a UTF-8 BOM; utf-8-sig strips it so the first column
    # header parses as "Div" and not "﻿Div".
    return resp.content.decode("utf-8-sig")


def _cached_csv(
    url: str,
    key: str,
    cache: Optional[Cache],
    ttl: int,
    force: bool = False,
) -> pd.DataFrame:
    cache = cache or Cache()
    text = None
    if not force:
        text = cache.get(key, ttl=ttl)
    if text is None:
        log.info("fetching %s", url)
        text = _fetch_text(url)
        cache.set(key, text)
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------
# football-data.co.uk
# ---------------------------------------------------------------------------

# Preference order per market: the market average is the most stable signal,
# then Bet365 (present in every season), then the best available price.
_ODDS_COLUMNS = {
    "odds_home": ("AvgH", "B365H", "MaxH", "BbAvH"),
    "odds_draw": ("AvgD", "B365D", "MaxD", "BbAvD"),
    "odds_away": ("AvgA", "B365A", "MaxA", "BbAvA"),
    "odds_over25": ("Avg>2.5", "B365>2.5", "Max>2.5", "BbAv>2.5"),
    "odds_under25": ("Avg<2.5", "B365<2.5", "Max<2.5", "BbAv<2.5"),
}

_STAT_COLUMNS = {
    "home_goals": "FTHG",
    "away_goals": "FTAG",
    "home_ht_goals": "HTHG",
    "away_ht_goals": "HTAG",
    "home_shots": "HS",
    "away_shots": "AS",
    "home_sot": "HST",
    "away_sot": "AST",
    "home_corners": "HC",
    "away_corners": "AC",
    "home_fouls": "HF",
    "away_fouls": "AF",
    "home_yellows": "HY",
    "away_yellows": "AY",
    "home_reds": "HR",
    "away_reds": "AR",
}

FOOTBALL_DATA_COLUMNS: List[str] = (
    ["season", "date", "time", "home_team", "away_team", "home_short", "away_short", "result"]
    + list(_STAT_COLUMNS.keys())
    + list(_ODDS_COLUMNS.keys())
)


def validate_team_names(names: Iterable[str], source: str = "football-data") -> None:
    """Raise listing every name with no mapping. Never drop rows silently."""
    unmapped = sorted({n for n in names if isinstance(n, str) and n not in FOOTBALL_DATA_TO_FPL_SHORT})
    if unmapped:
        raise UnmappedTeamError(
            "%s: %d unmapped team name(s): %s. Add them to "
            "gaffer.data.history.FOOTBALL_DATA_TO_FPL_SHORT."
            % (source, len(unmapped), ", ".join(repr(n) for n in unmapped))
        )


def load_football_data(
    seasons: Sequence[str],
    cache: Optional[Cache] = None,
    ttl: int = TTL_HISTORICAL,
    force: bool = False,
) -> pd.DataFrame:
    """Match results, shots and 1X2 / over-2.5 odds for one or more seasons.

    ``seasons`` accepts "2425", "2024-25" or "2024/25". Team names are kept in
    their raw football-data form under ``home_team`` / ``away_team`` and the
    canonical FPL short names are added as ``home_short`` / ``away_short``.
    """
    frames = []
    for season in seasons:
        code = fd_season_code(season)
        raw = _cached_csv(
            FOOTBALL_DATA_URL.format(code=code),
            "football_data_%s" % code,
            cache,
            ttl,
            force=force,
        )
        raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()].copy()
        validate_team_names(
            list(raw["HomeTeam"]) + list(raw["AwayTeam"]),
            source="football-data %s" % code,
        )

        out = pd.DataFrame(index=raw.index)
        out["season"] = season_dash(season)
        # football-data uses UK day-first dates, in both 2- and 4-digit years.
        out["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce")
        out["time"] = raw["Time"] if "Time" in raw.columns else pd.NA
        out["home_team"] = raw["HomeTeam"]
        out["away_team"] = raw["AwayTeam"]
        out["home_short"] = raw["HomeTeam"].map(FOOTBALL_DATA_TO_FPL_SHORT)
        out["away_short"] = raw["AwayTeam"].map(FOOTBALL_DATA_TO_FPL_SHORT)
        out["result"] = raw["FTR"] if "FTR" in raw.columns else pd.NA

        for name, src in _STAT_COLUMNS.items():
            out[name] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else pd.NA

        for name, candidates in _ODDS_COLUMNS.items():
            series = pd.Series([pd.NA] * len(raw), index=raw.index, dtype="object")
            for col in candidates:
                if col in raw.columns:
                    series = series.where(series.notna(), raw[col])
            out[name] = pd.to_numeric(series, errors="coerce")

        frames.append(out[FOOTBALL_DATA_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=FOOTBALL_DATA_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "home_team"], kind="mergesort").reset_index(drop=True)
    return df


def attach_fpl_team_ids(df: pd.DataFrame, teams: Dict[int, Team]) -> pd.DataFrame:
    """Add nullable ``home_team_id`` / ``away_team_id`` from an FPL team table.

    Teams not in the current Premier League (relegated sides in older seasons)
    get <NA> rather than a fabricated id.
    """
    short_to_id = {t.short_name: t.id for t in teams.values()}
    out = df.copy()
    out["home_team_id"] = out["home_short"].map(short_to_id).astype("Int64")
    out["away_team_id"] = out["away_short"].map(short_to_id).astype("Int64")
    return out


def teams_in_results(df: pd.DataFrame) -> Set[str]:
    """FPL short names of every club appearing in a football-data frame."""
    return set(df["home_short"].dropna()) | set(df["away_short"].dropna())


def detect_promoted(teams: Dict[int, Team], prior_results: pd.DataFrame) -> List[int]:
    """Team ids with no appearance in the previous season's top-flight results."""
    seen = teams_in_results(prior_results)
    return sorted(t.id for t in teams.values() if t.short_name not in seen)


# ---------------------------------------------------------------------------
# vaastav Fantasy-Premier-League archive
# ---------------------------------------------------------------------------


def load_vaastav_gws(
    season: str,
    cache: Optional[Cache] = None,
    ttl: int = TTL_HISTORICAL,
    force: bool = False,
) -> pd.DataFrame:
    """Per-GW player rows for a completed season (``merged_gw.csv``).

    Adds ``season``, ``element_type`` (int, from the string ``position``) and
    ``team_short``. ``GW`` is the gameweek; ``element`` is the FPL player id
    *for that season only* — ids are reassigned every summer, so join across
    seasons on the player ``code`` or on name, never on ``element``.
    """
    season = season_dash(season)
    df = _cached_csv(
        "%s/%s/gws/merged_gw.csv" % (VAASTAV_BASE, season),
        "vaastav_%s_merged_gw" % season,
        cache,
        ttl,
        force=force,
    )
    df["season"] = season
    if "position" in df.columns:
        df["element_type"] = df["position"].map(VAASTAV_POSITION_TO_ELEMENT_TYPE).astype("Int64")
    if "team" in df.columns:
        df["team_short"] = df["team"].map(FPL_LONG_TO_SHORT)
        missing = sorted({t for t in df["team"].dropna().unique() if t not in FPL_LONG_TO_SHORT})
        if missing:
            raise UnmappedTeamError(
                "vaastav %s: unmapped team name(s): %s. Add them to "
                "gaffer.data.history.FPL_LONG_TO_SHORT." % (season, ", ".join(missing))
            )
    if "kickoff_time" in df.columns:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    return df


def load_vaastav_players_raw(
    season: str,
    cache: Optional[Cache] = None,
    ttl: int = TTL_HISTORICAL,
    force: bool = False,
) -> pd.DataFrame:
    """End-of-season ``players_raw.csv`` (the final bootstrap element rows)."""
    season = season_dash(season)
    df = _cached_csv(
        "%s/%s/players_raw.csv" % (VAASTAV_BASE, season),
        "vaastav_%s_players_raw" % season,
        cache,
        ttl,
        force=force,
    )
    df["season"] = season
    return df


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cache = Cache()

    print("=== season name helpers ===")
    for s in ("2026-27", "2026/27", "2627"):
        print("  %-8s -> dash=%s slash=%s fd=%s prior=%s"
              % (s, season_dash(s), season_slash(s), fd_season_code(s), prior_season(s)))

    print("\n=== football-data.co.uk ===")
    t0 = time.time()
    fd = load_football_data(["2324", "2425", "2526"], cache=cache)
    print("  rows=%d cols=%d (%.2fs)" % (len(fd), len(fd.columns), time.time() - t0))
    print("  columns: %s" % ", ".join(fd.columns))
    for season, grp in fd.groupby("season"):
        print("    %s: %d rows, %d clubs, %s..%s, odds_home null=%d"
              % (season, len(grp), len(teams_in_results(grp)),
                 grp["date"].min().date(), grp["date"].max().date(),
                 int(grp["odds_home"].isna().sum())))
    assert fd["home_short"].isna().sum() == 0 and fd["away_short"].isna().sum() == 0
    print("  every team name mapped: OK")
    print(fd[["date", "home_team", "away_team", "home_goals", "away_goals", "home_sot",
              "away_sot", "odds_home", "odds_draw", "odds_away", "odds_over25"]].tail(3).to_string(index=False))

    print("\n=== vaastav 2025-26 ===")
    t0 = time.time()
    gws = load_vaastav_gws("2025-26", cache=cache)
    print("  merged_gw shape=%s (%.2fs)" % (str(gws.shape), time.time() - t0))
    print("  columns: %s" % ", ".join(gws.columns))
    print("  GW range %s..%s, %d unique players, %d teams"
          % (gws["GW"].min(), gws["GW"].max(), gws["element"].nunique(), gws["team"].nunique()))
    praw = load_vaastav_players_raw("2025-26", cache=cache)
    print("  players_raw shape=%s" % str(praw.shape))

    print("\n=== warm reload ===")
    t0 = time.time()
    load_football_data(["2324", "2425", "2526"], cache=cache)
    load_vaastav_gws("2025-26", cache=cache)
    print("  %.2fs" % (time.time() - t0))
