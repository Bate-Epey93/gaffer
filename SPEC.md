# gaffer — build specification

An expected-points engine and squad optimizer for Fantasy Premier League 2026/27.
Goal: produce the highest-expected-points legal decision every gameweek, with a
full audit trail of *why*.

This document is the integration contract. Every module is built against the
types in `gaffer/core/types.py` and the constants in `gaffer/core/scoring.py`.
Do not invent parallel data structures. Do not change a shared signature without
updating this file.

---

## 0. Environment (verified 2026-08-14)

- Python **3.9.6** at `/usr/bin/python3`. No homebrew, no pyenv, no uv.
  **All code must be Python 3.9 compatible**: use `from __future__ import annotations`,
  `Optional[X]` not `X | None`, `Dict`/`List` from `typing` in signatures.
- Virtualenv at `.venv`, created by `setup.sh`. Verified installable:
  `pandas 2.3.3`, `numpy 2.0.2`, `pulp 3.3.1` (CBC bundled), `highspy`, `requests`,
  `fastapi`, `uvicorn`.
- No internet access is assumed at solve time — everything caches to `data/cache`.

## 1. Verified facts about the 2026/27 season

Read from the live API on 2026-08-14. `gaffer/core/scoring.py --verify` re-checks all of it.

| Fact | Value |
| --- | --- |
| Season | 2026/27, 38 gameweeks, 380 fixtures, 10 per GW (no blanks/doubles scheduled yet) |
| GW1 deadline | `2026-08-21T17:30:00Z` |
| Players | 587 |
| Budget | £100.0m (`now_cost` is in tenths, so 1000) |
| Squad | 15 = 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 per club |
| Lineup | 11, 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD |
| Transfers | 1 free per GW, bank up to 5, extra transfers cost -4, cap 20/GW |
| Sell-on | keep 50% of profit, rounded down to 0.1m |
| Chips | 8 total: 2x each of wildcard, freehit, bboost, 3xc. First set GW1/2-19 (expires at the GW19 deadline, 2027-01-02), second set GW20-38 |
| Goals | GKP 10, DEF 6, MID 5, FWD 4 |
| Assists | 3 |
| Clean sheet | GKP 4, DEF 4, MID 1, FWD 0 — requires 60+ minutes |
| Conceded | -1 per 2 goals, GKP and DEF only |
| Saves | 1 per 3 |
| DEFCON | +2 flat. DEF at 10+ CBIT (clearances+blocks+interceptions+tackles). MID and FWD at 12+ CBIRT (CBIT + recoveries). GKP ineligible. Capped at 2. |
| Cards | yellow -1, red -3. Own goal -2. Penalty save +5, penalty miss -2 |
| Bonus | 3/2/1 to the top three BPS in each match |

**2026/27 BPS changes** (official, affects the bonus model):
1. The -1 BPS for being tackled is removed.
2. Clearances/blocks/interceptions now earn 1 BPS per **3** actions (was per 2) —
   deliberately reducing the DEFCON/bonus double-count.
3. Goalkeeper saves restructured: **any save = 2 BPS**, +1 if the save is inside
   the box, +1 if it is from a big chance. Penalty save BPS cut from 8 to 7.

Net effect: goalkeepers and attacking full-backs gain bonus share; pure
clearance-heavy centre-backs lose some.

**Season-start caveats — these are the hard modelling problems, handle them explicitly:**

1. `teams[].strength_attack_*` and `strength_defence_*` are **all 0** pre-season, and
   `strength` is `null`. Only `strength_overall_home/away` (1-5) is populated. The
   official per-fixture FDR (`team_h_difficulty` / `team_a_difficulty`, 1-5) *is*
   populated and is usable as a weak prior, but we build our own ratings.
2. Three promoted clubs have little or no Premier League history: **Coventry (COV, id 7),
   Hull (HUL, id 11), Ipswich (IPS, id 12)**. They need a promoted-team prior, not a
   silently-missing rating.
3. Pre-season, `elements[]` in bootstrap carries **last season's totals**, not zeros
   (verified: Raya shows 3330 minutes, 162 points). Once GW1 completes these reset to
   season-to-date. Code must not assume either state — always check `events[].finished`
   to determine how many gameweeks of the current season exist, and treat bootstrap
   element stats as current-season-to-date only when `current_gw >= 1`.
4. `element-summary/{id}/history` is empty pre-season; `history_past` holds up to 5
   prior seasons per player with xG, xA, xGC, DEFCON, tackles, recoveries and CBI.

## 2. Verified data sources

| Source | URL | Status | Use |
| --- | --- | --- | --- |
| FPL bootstrap | `https://fantasy.premierleague.com/api/bootstrap-static/` | 200, no auth, 1.4MB | players, teams, events, chips, scoring |
| FPL fixtures | `https://fantasy.premierleague.com/api/fixtures/` | 200 | 380 fixtures with FDR + kickoff |
| Player detail | `https://fantasy.premierleague.com/api/element-summary/{id}/` | 200 | per-GW history, `history_past`, upcoming fixtures |
| Live GW | `https://fantasy.premierleague.com/api/event/{gw}/live/` | 200 (empty pre-season) | in-play stats |
| Manager entry | `https://fantasy.premierleague.com/api/entry/{id}/` and `/history/` | 200 | squad state |
| Manager picks | `https://fantasy.premierleague.com/api/entry/{id}/event/{gw}/picks/` | 404 pre-season, live from GW1 | current squad |
| Event status | `https://fantasy.premierleague.com/api/event-status/` | 200 | bonus finalisation |
| Set pieces | `https://fantasy.premierleague.com/api/team/set-piece-notes/` | 200 | penalty/corner takers |
| Match results + odds | `https://www.football-data.co.uk/mmz4281/{yy}{yy}/E0.csv` | 200, 380 rows for 2526 | team ratings, shots, shots on target, bookmaker 1X2 and over/under 2.5 odds |
| Historical FPL | `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv` | 200, maintained through 2026-27 | per-GW player history for backtesting |

Notes:
- The FPL API sends **no CORS headers** — the browser dashboard must talk to our own
  FastAPI backend, never directly to `fantasy.premierleague.com`.
- Responses are Fastly-cached with `max-age=300`. Never poll faster than 5 minutes.
- Always send a browser-like `User-Agent` (see `config.USER_AGENT`); bare clients
  can get blocked.
- **FBref no longer carries advanced stats** (lost the Opta licence in January 2026).
  Do not build on it. The FPL API's own xG/xA/xGI/xGC covers the need.

---

## 3. Module contracts

### 3.1 `gaffer/data/` — ingest

**`fpl_api.py`**
```python
class FPLClient:
    def __init__(self, config: Config, cache: Cache) -> None
    def bootstrap(self, force: bool = False) -> Dict[str, Any]
    def fixtures(self, force: bool = False) -> List[Dict[str, Any]]
    def element_summary(self, player_id: int, force: bool = False) -> Dict[str, Any]
    def live(self, gw: int, force: bool = False) -> Dict[str, Any]
    def entry(self, entry_id: int) -> Dict[str, Any]
    def entry_history(self, entry_id: int) -> Dict[str, Any]
    def entry_picks(self, entry_id: int, gw: int) -> Dict[str, Any]
    def event_status(self) -> Dict[str, Any]
    def set_piece_notes(self) -> Dict[str, Any]
```
Requirements: retry with backoff on 429/5xx, honour the cache TTL, thread-pool
`element_summary` fetches (587 players — use at most 8 workers and a small delay),
and never raise on a single player failing; log and continue.

**`cache.py`** — a disk cache under `data/cache/{key}.json` with `{"fetched_at": iso, "data": ...}`.
```python
class Cache:
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]
    def set(self, key: str, value: Any) -> None
    def clear(self, pattern: str = "*") -> None
```

**`history.py`**
```python
def load_football_data(seasons: List[str]) -> pd.DataFrame
    # seasons like ["2425", "2526"]; columns normalised to
    # date, home_team, away_team, home_goals, away_goals, home_shots, away_shots,
    # home_sot, away_sot, home_corners, away_corners, odds_home, odds_draw, odds_away,
    # odds_over25, odds_under25
def load_vaastav_gws(season: str) -> pd.DataFrame   # per-GW player rows
def load_vaastav_players_raw(season: str) -> pd.DataFrame
def team_name_map() -> Dict[str, str]  # football-data names -> FPL short names
```
Team-name normalisation is a known trap: football-data uses "Man City", "Man United",
"Nott'm Forest", "Newcastle", "Sheffield United"; FPL uses short codes. Build an
explicit mapping table and **fail loudly** on an unmapped name.

**`loaders.py`** — the single entry point the rest of the app uses.
```python
@dataclass
class GameState:
    teams: Dict[int, Team]
    players: Dict[int, Player]
    fixtures: List[Fixture]
    events: List[Dict[str, Any]]
    current_gw: int          # next gameweek to plan for
    finished_gws: List[int]
    bootstrap: Dict[str, Any]
    player_history: Dict[int, pd.DataFrame]      # per-GW current season
    player_history_past: Dict[int, List[Dict]]   # prior seasons

def load_game_state(config: Config, with_histories: bool = True) -> GameState
def fixtures_by_gw(state: GameState) -> Dict[int, List[Fixture]]
def team_fixtures(state: GameState, team_id: int, gws: List[int]) -> List[Fixture]
    # returns one entry per fixture; a team with two fixtures in a GW yields two,
    # a blank gameweek yields none. Double and blank gameweeks must work.
```

### 3.2 `gaffer/model/team_ratings.py`

```python
class TeamRatingModel:
    def __init__(self, config: Config) -> None
    def fit(self, results: pd.DataFrame, state: GameState) -> None
    def ratings(self) -> Dict[int, TeamStrength]
    def match_lambdas(self, home_id: int, away_id: int) -> Tuple[float, float]
    def forecast_fixture(self, fixture: Fixture) -> MatchForecast
    def forecast_all(self, state: GameState, gws: List[int]) -> Dict[int, MatchForecast]
```

Approach: Dixon-Coles style. Each team gets an attack and a defence multiplier on
the league mean. Fit by weighted maximum likelihood over historical results with
exponential time decay (`ratings_half_life_days`). Then:

```
lambda_home = league_avg * attack[home] * defence[away] * home_advantage
lambda_away = league_avg * attack[away] * defence[home]
```

Requirements:
- Time-decay weights so recent matches dominate.
- **Promoted teams**: no or thin PL history. Use `promoted_attack_prior` /
  `promoted_defence_prior`, and shrink hard until they have real 2026/27 matches.
- **Prior blending**: at GW1 the rating is 100% previous-season-derived. As the new
  season accumulates matches, decay the prior with `ratings_prior_decay_matches`.
- **Odds blending**: when football-data odds exist for a fixture (they do for past
  seasons; for future fixtures they will not), blend model lambdas with
  odds-implied lambdas via `stats.lambdas_from_odds` at weight `odds_blend_weight`.
- Expose `confidence` per team so downstream code can widen uncertainty for
  promoted sides.
- Sanity check the output: league-average lambda should land near 1.45, home
  advantage near 1.1-1.2. Log a warning if a fitted lambda leaves [0.2, 4.0].

### 3.3 `gaffer/model/minutes.py`

```python
class MinutesModel:
    def __init__(self, config: Config) -> None
    def fit(self, state: GameState) -> None
    def forecast(self, player_id: int, gw: int, state: GameState) -> MinutesForecast
    def forecast_all(self, state: GameState, gws: List[int]) -> Dict[Tuple[int, int], MinutesForecast]
```

This is the single highest-leverage component: every other point component is
multiplied by it. Get it right.

Inputs, in priority order:
1. **Hard availability**: `status` — `i`/`s`/`u` means 0. `d` (doubtful) scales by
   `chance_of_playing_next_round / 100` when present, else 0.5. `news` text should be
   parsed for return dates and phrases like "Expected back", "50% chance".
2. **Recent starts**: from `history` this season (once it exists) — the last N
   appearances weighted most heavily.
3. **Prior season**: `starts`, `minutes` from `history_past` for the same season, and
   `starts_per_90`.
4. **Priors when no data exists** (new signings, promoted-club players, pre-season):
   back off to a position-and-price baseline. A £13.0m forward is a nailed starter;
   a £4.0m defender is probably a bench filler. Fit the price-to-start-probability
   curve from last season's actual data rather than hardcoding it.
5. **New club**: if `team_join_date` is within the current window, damp confidence.

Output `p_start`, `p_appear = p_start + (1 - p_start) * p_sub_appearance`,
`p_60`, and `xmins`. `p_60` must be conditioned on starting: a substitute rarely
reaches 60 minutes. Fill `reason` with a short human-readable explanation
("nailed: 34/38 starts, no injury news").

### 3.4 `gaffer/model/attacking.py`

```python
class AttackingModel:
    def __init__(self, config: Config) -> None
    def fit(self, state: GameState) -> None
    def rates(self, player_id: int) -> Tuple[float, float]   # (xG90, xA90) shrunk
    def project(self, player_id: int, fixture_ctx: FixtureContext) -> Tuple[float, float]
        # returns (lambda_goals, lambda_assists) for this player in this fixture
```

Method:
1. Blend per-90 xG and xA across prior season / season-to-date / last 6 GWs using
   the configured weights, renormalising when a source is missing.
2. Shrink towards a position+price baseline with `shrinkage_90s_attacking`.
3. Scale to the fixture: a player's share of team goals should track team lambda.
   `lambda_goals = xG90 * (xmins/90) * (team_lambda / team_avg_lambda) * finishing_adj`.
   Use the team lambda from `TeamRatingModel`, not a raw FDR bucket.
4. **Penalties**: a player with `penalties_order == 1` gets an added penalty
   component — estimate the team's penalties-won rate per match and the ~0.79
   conversion rate. This is a real and frequently mis-modelled edge.
5. Assists should scale with team lambda too, but with a flatter response than goals.
6. Convert to points with `scoring.goal_points` / `assist_points`. Because FPL
   points are linear in event counts, `E[points] = lambda * points_per_event`; the
   Poisson distribution is still needed for the variance and haul probabilities.

### 3.5 `gaffer/model/defending.py`

```python
class DefendingModel:
    def project_clean_sheet(self, player_id, fixture_ctx) -> float
    def project_conceded_points(self, player_id, fixture_ctx) -> float
    def project_saves(self, player_id, fixture_ctx) -> float   # lambda_saves
    def project(self, player_id, fixture_ctx) -> Dict[str, float]
```

- `p_clean_sheet = exp(-lambda_conceded)` from the team model, times `p_60`.
- Conceded points: `scoring.expected_goals_conceded_points`.
- Saves: expected saves scale with opponent shots on target. Derive a team-level
  shots-on-target-faced rate from football-data (`AST`/`HST` columns) mapped to the
  opponent's attacking strength, then multiply by the keeper's historical save share.
  Cross-check against the keeper's own `saves_per_90`.
- Keepers get 2026/27's improved save BPS — that belongs in `bonus.py` but the
  saves lambda computed here feeds it.

### 3.6 `gaffer/model/defcon.py`

```python
class DefconModel:
    def fit(self, state: GameState) -> None
    def project(self, player_id, fixture_ctx) -> float   # P(hits threshold)
```

DEFCON is a **threshold** event, not a rate, so modelling the mean alone is wrong.

1. Build per-90 CBIT (DEF) or CBIRT (MID/FWD) rates per player from
   `history_past` (`clearances_blocks_interceptions + tackles` and `+ recoveries`)
   and from current-season `history` once it exists. Note the API exposes
   `defensive_contribution` directly as the count of qualifying *matches*-worth
   of points — check per-season semantics before trusting it, and prefer summing
   the component stats.
2. Fit a negative binomial per player (mean from the shrunk per-90 rate scaled by
   expected minutes; dispersion fitted from observed variance, pooled by position).
3. `p_defcon = negbin_at_least(threshold, mean, dispersion)`.
4. Adjust the mean by opponent "field tilt": a team that will spend the match
   defending racks up more CBIT. Proxy field tilt with the ratio of the two teams'
   attacking strengths, calibrated on last season's data.
5. Fixture polarity matters and is counter-intuitive: a defender facing a *strong*
   attack has a lower clean-sheet chance but a *higher* DEFCON chance. Both effects
   must appear, with opposite signs.

Reference points from 2025/26 for calibration: the top DEFCON scorers were around
50 points across the season (roughly 25 qualifying matches). The top ten was five
centre-backs and five defensive midfielders, no forwards. A player hitting the
threshold in over 50% of starts is elite.

### 3.7 `gaffer/model/bonus.py`

```python
class BonusModel:
    def fit(self, state: GameState) -> None
    def project_bps(self, player_id, fixture_ctx, projections) -> float
    def project_bonus(self, fixture_id: int, candidates: Dict[int, float]) -> Dict[int, float]
        # {player_id: expected bonus points}, must sum to <= 6 per fixture
```

Implement BPS as an event-weighted score using the **2026/27** weights, then use
`stats.top_k_probabilities` to convert each player's projected BPS (plus fitted
noise) into P(1st)/P(2nd)/P(3rd), then to expected bonus. The constraint that a
match awards exactly 3+2+1 points is a strong one — respect it, and unit-test that
per-fixture expected bonus sums to at most 6.

Do not fabricate exact BPS weights that were not verified. Where an exact weight is
unknown, fit it: regress observed BPS on observed event counts using last season's
`merged_gw.csv` from the vaastav dataset, and record the fitted coefficients in
`reports/bps_fit.json`. Note in comments that CBI weighting and GK save weighting
changed for 2026/27, so 2025/26-fitted coefficients need the documented manual
adjustment on top.

### 3.8 `gaffer/model/xp.py` — the assembler

```python
@dataclass
class FixtureContext:
    fixture: Fixture
    player: Player
    is_home: bool
    team_lambda: float          # goals the player's team is expected to score
    opponent_lambda: float      # goals the player's team is expected to concede
    forecast: MatchForecast
    minutes: MinutesForecast

class XPEngine:
    def __init__(self, config: Config) -> None
    def fit(self, state: GameState) -> None
    def project_player_fixture(self, player_id: int, fixture: Fixture, gw: int) -> PlayerFixtureProjection
    def project(self, state: GameState, gws: List[int]) -> ProjectionSet
    def explain(self, player_id: int, gw: int) -> str   # human-readable breakdown
```

`project` must:
- Handle **double gameweeks** (two fixtures, points sum) and **blank gameweeks**
  (no fixture, xp = 0) correctly.
- Fill every component field on `PlayerFixtureProjection`, not just the total —
  the dashboard shows the breakdown and the user must be able to audit it.
- Compute `sd_total` for captaincy risk, by combining component variances
  (Poisson variance for goals/assists, Bernoulli for clean sheet and DEFCON).
- Be deterministic given the same inputs (seed any Monte Carlo).
- Cache to `data/cache/projections_{first}_{last}.json` and be loadable back.

### 3.9 `gaffer/optimize/squad.py`

```python
def pick_initial_squad(projections: ProjectionSet, state: GameState, config: Config,
                       gws: Optional[List[int]] = None) -> GWDecision
def pick_lineup(squad_ids: List[int], projections: ProjectionSet, gw: int,
                state: GameState, chip: Optional[str] = None) -> Tuple[List[int], List[int], int, int]
    # (lineup, bench_in_order, captain_id, vice_id)
def solve_squad_milp(...) -> ...
```

MILP with PuLP. Binary `x[p]` squad, `y[p]` lineup (`y[p] <= x[p]`), `c[p]` captain
(`c[p] <= y[p]`, sum = 1). Objective, summed over the horizon with `decay**w`:
```
sum_p ( xp[p][w] * y[p][w] + xp[p][w] * c[p][w] + bench_weight * (x - y) )
```
Constraints: budget, 15-man position quotas, max 3 per club, valid formation,
`min_xmins_to_consider` filter, locked in/out.

Solver: try HiGHS via `pulp.HiGHS_CMD` or `highspy`, fall back to `PULP_CBC_CMD`.
Always report the status and reject a non-optimal solution loudly.

Bench order matters: rank the bench by `xp * p_appear` descending, and keep a
playing-goalkeeper rule (the backup GK is always bench slot 0, never ordered
against outfielders).

### 3.10 `gaffer/optimize/planner.py`

```python
def plan(state: GameState, squad: SquadState, projections: ProjectionSet,
         config: Config, chips_available: Optional[List[str]] = None) -> Plan
def evaluate_transfer(...) -> float
def should_take_hit(gain: float, config: Config) -> bool
```

Multi-gameweek MILP: transfer-in/out binaries per gameweek, free-transfer
accumulation (cap 5), hit costs at -4, bank tracking with the sell-on rule, and a
terminal value term crediting leftover free transfers (`ft_value`) and squad value.

Decision rules that must be encoded, from what actually wins:
- A free transfer is worth about 2 points, so a hit must clear roughly 4 points of
  gain **over the horizon**, not over one gameweek. Config: `hit_threshold`.
- The last two world champions took near-zero hits and captained their premium
  asset in the large majority of gameweeks. Bias the planner towards patience:
  it must justify churn, not assume it.
- Squad value is worth something but not much — `value_per_tenth`.

### 3.11 `gaffer/optimize/chips.py`

```python
def detect_double_blank_gws(state: GameState) -> Dict[int, Dict[str, List[int]]]
def recommend_chips(state, squad, projections, config) -> List[Dict[str, Any]]
```
- Detect doubles/blanks by counting each team's fixtures per gameweek (right now
  every team has exactly one per gameweek; doubles appear later as postponed games
  are rescheduled, so this must be recomputed from live data every run, never
  hardcoded).
- Bench Boost wants a double gameweek with 15 playing assets; Triple Captain wants
  a double gameweek for a premium; Free Hit answers a blank; Wildcard precedes a
  fixture swing.
- **Hard deadline: the first set of four chips expires at the GW19 deadline
  (2027-01-02).** Any unused first-half chip must raise an escalating warning from
  about GW14 onward.

### 3.12 `gaffer/optimize/strategy.py`

```python
def effective_ownership(state, projections, gw) -> Dict[int, float]
def captain_options(squad_ids, projections, gw, state, config) -> List[CaptainOption]
def differential_score(player_id, state, projections, gw) -> float
def rank_risk_report(squad, state, projections, gw) -> Dict[str, Any]
```
Effective ownership = ownership% + captaincy%. Captaincy share is not in the API;
approximate it from `selected_by_percent` and projected points (the highest-xp
premium in a template squad absorbs most captaincy), and allow a manual override.
A player at over 100% effective ownership who hauls **costs you rank if you do not
own him** — that is the core rank-chasing insight and must be surfaced explicitly.

### 3.13 `gaffer/backtest/`

```python
def backtest_season(season: str, gws: List[int], config: Config) -> Dict[str, Any]
def evaluate_projections(pred: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]
```
Walk 2025/26 gameweek by gameweek using only data available before each deadline,
project, and compare with actuals from the vaastav dataset. Report RMSE, MAE,
Spearman correlation overall and by position, plus top-20 hit rate and captain
accuracy. Write `reports/backtest_{season}.json` and a readable summary.

**A model nobody has backtested is a guess.** This module is not optional.

### 3.14 `gaffer/api/server.py` and `gaffer/cli.py`

FastAPI endpoints:
```
GET  /api/state                    -> current gw, deadline, counts
GET  /api/players?gw=&pos=&max_cost=&sort=  -> players with xp and components
GET  /api/player/{id}              -> full breakdown + fixtures + explanation
GET  /api/fixtures?gw=             -> fixtures with model lambdas and FDR
GET  /api/projections?first=&last= -> the projection set
GET  /api/squad?entry_id=          -> current squad state
POST /api/optimize                 -> body: horizon, chips, locks -> Plan
GET  /api/captain?gw=              -> ranked captain options
GET  /api/chips                    -> chip recommendations and expiry warnings
POST /api/refresh                  -> force a data refresh
```
Serve `gaffer/web/` as static files at `/`. CORS open to localhost only.

CLI (`python -m gaffer.cli ...`):
`refresh`, `project --gws 1-6`, `squad --budget 100`, `plan --horizon 6`,
`captain --gw 1`, `chips`, `backtest --season 2025-26`, `serve`, `verify`.

Every command prints a readable table, not raw JSON.

### 3.15 `gaffer/web/` — dashboard

Single-page, no build step, no CDN dependencies (vanilla JS + CSS, fetch from our own
API). Dark, dense, information-first — this is a decision cockpit, not a marketing page.

Views:
1. **Squad** — the recommended XI in a pitch layout, bench, captain, total xP,
   with each player showing price, xP and the component breakdown on hover/click.
2. **Players** — sortable table: xP for the next N gameweeks, price, value
   (xP per £m), ownership, minutes probability, form, xG90/xA90, DEFCON probability,
   fixture ticker colour-coded by our own model lambda (not just FDR).
3. **Planner** — the multi-gameweek plan: transfers in/out per gameweek, hits,
   projected points, with an explanation for each move.
4. **Captain** — ranked options with xP, standard deviation, haul probability and
   effective ownership.
5. **Chips** — recommended gameweeks and the GW19 expiry countdown.

Every number that is a model output must be traceable to its components in one click.

---

## 4. Cross-cutting requirements

- **Determinism.** Same inputs, same outputs. Seed everything.
- **No silent failures.** If a data source is missing, say so in the output; never
  substitute a zero and carry on quietly.
- **Explainability.** Every recommendation carries a reason string. If the app
  says "buy Haaland", it must say why: fixture, minutes, xG, ownership risk.
- **Speed.** A full projection run over 587 players and 6 gameweeks should take
  seconds, not minutes, after the first fetch. Cache aggressively.
- **Python 3.9.** Test with `.venv/bin/python`, not the system interpreter.
- **Tests.** `pytest` under `tests/`, covering: scoring maths (hand-checked cases),
  clean-sheet and DEFCON probability edges, sell-price rounding, squad legality,
  MILP output legality (15 players, quotas, budget, 3-per-club), double and blank
  gameweek handling, and the bonus points sum constraint.
