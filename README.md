# gaffer

An expected-points engine and squad optimizer for **Fantasy Premier League 2026/27**.

It projects every player's expected points for the next N gameweeks, picks the best
legal 15 for your budget, plans your transfers, ranks your captain options and tells
you when to play your chips — and it shows its working for every number.

`SPEC.md` is the build specification (the contract the modules are written against).
This file is the user manual.

---

## What it can and cannot do

**Measured, not claimed.** The model was walked forward through the whole of 2025/26
one gameweek at a time, using only data that existed before each deadline
(`reports/backtest_2025-26.json`, 38 gameweeks, no-lookahead audit clean). On the
subset that actually matters — the 8,136 player-gameweeks where the model said before
kickoff that the player would probably start, which is the pool you would ever transfer
from — it scores:

| predictor | RMSE | MAE | Spearman | top-20 hit rate |
| --- | --- | --- | --- | --- |
| **gaffer** | **3.07** | **2.27** | **0.247** | **21.2%** |
| baseline: season points per game | 3.26 | 2.39 | 0.193 | 18.7% |
| baseline: last gameweek's points | 4.31 | 3.02 | 0.135 | 12.8% |
| baseline: price (rank only) | – | – | 0.110 | 19.3% |

So it beats every naive baseline on every metric, and the margin is real but modest.
Read that Spearman of 0.247 honestly: **week-to-week FPL scores are mostly noise, and
this model captures a minority of the signal.** An RMSE of 3.07 points on a player who
averages about 4 means single-gameweek predictions are wide. The edge compounds over a
season through consistently better-than-baseline ranking; it does not make any
individual gameweek predictable.

Captaincy, same backtest: over 37 gameweeks the model's top captain pick was **never
once** the actual highest scorer in the game. It landed in the actual top 5 in 18.9% of
gameweeks and the top 10 in 29.7%, averaging 6.27 points per captain against a
theoretical perfect-hindsight 16.62. That is what a good captain model looks like —
nobody picks the top scorer reliably — but do not expect an oracle.

**What it does not do.** It cannot read press conferences, does not know your league
rivals' squads, has no bookmaker odds for future fixtures (only for historical fit),
and cannot see a manager's rotation intentions. It will not tell you Guardiola is
resting people for a Champions League tie. It has no concept of your rank, your league
position, or how much risk you should be taking — `differential_weight` is a dial you
set, not something it infers.

**Right now, pre-season, it is at its weakest.** Every model input is last season's
data: the API's team attack/defence strengths are all zero before GW1, `element-summary`
per-gameweek history is empty, and the three promoted clubs (Coventry, Hull, Ipswich)
have no Premier League history at all and run on a promoted-team prior. Treat GW1
output as a well-reasoned prior, not a forecast. See **Known issues** below for a
specific pre-season distortion you should know about before you trust the GW1 squad.

---

## Install

Requirements: macOS or Linux, `python3` 3.9 or newer, and an internet connection for
the first run.

```bash
git clone <repo> gaffer
cd gaffer
./setup.sh
```

`setup.sh` creates `.venv/`, installs `requirements.txt`, and prints `deps ok`. It
takes about 25 seconds. It is idempotent — safe to re-run.

**There is no `gaffer` executable on your PATH.** The package is not pip-installed;
every command is run through the virtualenv's interpreter as a module:

```bash
./.venv/bin/python -m gaffer.cli <command>
```

If you would rather type less, alias it:

```bash
alias gaffer="$PWD/.venv/bin/python -m gaffer.cli"
```

Every example below is written as `gaffer <command>`; substitute whichever form you use.

A harmless warning appears on every run on macOS system Python
(`urllib3 v2 only supports OpenSSL 1.1.1+ … LibreSSL 2.8.3`). Ignore it.

No API key, no login, no FPL account is required. Everything the tool reads is public.

---

## The weekly workflow

Run these in order. Each one tells you what to run next.

```bash
gaffer refresh              # 1. pull fresh FPL data
gaffer verify               # 2. check the data and the model are sane
gaffer project --gws 1-6    # 3. expected points for all 587 players
gaffer squad                # 4. best legal 15 from scratch (GW1 or a wildcard)
gaffer plan --horizon 6     # 5. transfer plan for a squad you already own
gaffer captain --gw 1       # 6. ranked armband options
gaffer chips                # 7. when to play chips, and the GW19 expiry clock
```

**The first run is the slow one** — it fetches 587 individual player summaries. Measured
cold, from an empty cache: about 12 seconds to fetch and load, 20 seconds to fit the
models, 4 seconds to project. After that everything is served from `data/cache/` and
each command takes 1–5 seconds.

Progress narration goes to **stderr** and the tables go to **stdout**, so
`gaffer project | less` stays clean.

If you already have a squad, tell it who you are — either export `FPL_ENTRY_ID`, or
put `"entry_id"` in `config.json`. Without an entry id, `plan`, `captain` and `chips`
solve for the optimal squad from scratch and say so.

---

## The commands

### `refresh` — pull fresh FPL data

Re-fetches bootstrap, fixtures and all player summaries into `data/cache/`. Prints the
season state (teams, players, fixtures, current gameweek, deadline), the cache ages,
and **a table of flagged players** — every injury, suspension and departure at £5.5m or
more, with the news text.

Run this before every decision. The FPL API is cached for 5 minutes upstream; there is
no point polling faster.

### `verify` — check the data and the model are sane

Runs 18 checks and prints pass/fail for each. Two groups:

1. **Scoring constants** — re-reads the live API and confirms every 2026/27 rule the
   engine hardcodes (goal values by position, DEFCON thresholds, the BPS changes) still
   matches what the API says.
2. **Data and projection integrity** — 20 teams, 380 fixtures, every club has 38
   fixtures, no NaNs, components sum to the total, no negative gameweek totals, expected
   bonus per fixture within the sanity limit.

If anything here fails, stop and fix it before trusting any other output. This is the
command that catches "FPL changed a rule and nobody noticed".

### `project` — expected points for every player

```bash
gaffer project --gws 1-6
```

Ranks all 587 players by total expected points over the window.

| column | meaning |
| --- | --- |
| `GW1`…`GW6` | expected points in that gameweek. `*` marks a double gameweek, `-` a blank |
| `total` | sum over the window — the ranking column |
| `per £m` | value: total expected points divided by price |
| `xmins` | expected minutes per gameweek. **This is the number to sanity-check first** — every other component is multiplied by it |

Expected points are a *mean*, not a prediction. A player on 5.1 xP most often scores 2,
occasionally 13. The distribution matters as much as the mean, which is why `captain`
reports standard deviation and haul probability separately.

Useful filters: `--pos MID`, `--max-cost 8.0`, `--min-xmins 60`, `--limit 50`,
`--fresh` to recompute instead of reusing the cache.

**`--explain` is the most useful flag in the tool.** It prints the complete audit trail
for any player, by name or id:

```bash
gaffer project --gws 1-6 --explain Haaland,Saka
```

```
Haaland (FWD, MCI, £15.5m) — GW1
  fixture 8: MCI v BOU (H), official FDR 3
    match      : team lambda 1.88, opponent lambda 1.02 | P(win) 56%, P(team CS) 36%
    minutes    : p_start 0.85 p_appear 0.89 p_60 0.81 xmins 73.2
                 nailed: 34/38 starts in 2025/26, no news
    attacking  : xG90 0.505 xA90 0.058, fixture x1.270/1.334, finishing x1.006
                 lambda_goals 0.600 (open play 0.525 + pens 0.074)
    defcon     : 3.08 actions/90 ... mean 2.97 vs threshold 12 -> P=0.0%
    bonus      : xBPS 20.0 -> P(3)=23% P(2)=9% P(1)=5%
    components : appearance +1.707  goals +2.399  assists +0.189  bonus +0.909
                 cards -0.069  penalty -0.040   TOTAL 5.096  sd 4.62
    simulation : mean 5.076 sd 4.620 over 4000 draws (analytic 5.096 / 4.620)
  GAMEWEEK TOTAL: 5.096 xP, sd 4.62, P(>=10) 15.3%
```

Use it whenever a recommendation surprises you. It shows every intermediate quantity,
names the evidence behind the minutes call, and cross-checks the analytic mean against a
Monte Carlo simulation. If the tool is wrong about a player, this is where you find out
why in about ten seconds.

### `squad` — the best legal 15 from scratch

```bash
gaffer squad --budget 100
```

Use this for your GW1 team or when you are wildcarding. It runs a MILP over the whole
horizon (not just the next gameweek) so you get a squad you will not immediately have to
transfer out of.

Prints the XI in position order, the bench **in autosub order**, then a summary:
formation, captain, vice, spend, bank, GW1 expected points, and the solver's status and
objective. The notes underneath explain the choice.

Two totals appear and they answer different questions:
- **GW1 expected points** — what the XI plus captain is worth this week. A real points figure.
- **GW1-6 squad xP** — the sum over all 15 players across the window.

The bench is deliberately cheap. Spending £4.5m on a fourth-choice forward who will
never play is correct: that money buys an upgrade in the XI, and the bench weights in
the objective (`bench_weights`, default 0.12/0.08/0.05/0.02) price the small autosub
insurance value honestly rather than pretending bench points matter.

**If you disagree with a pick, overrule it.** `--lock-in` and `--lock-out` take names or
ids and constrain the solve, so you can see what your own conviction actually costs:

```bash
gaffer squad --lock-in Haaland              # build the best squad that contains him
gaffer squad --lock-out Haaland --budget 100
```

Compare the two objective values and you have priced your opinion in points. This is the
recommended way to handle the premium-forward issue described under Known issues — and
the price is small: forcing Haaland into the GW1 squad costs 1.23 expected points in
GW1 (51.14 against 52.37) and moves the horizon objective from 208.25 to 204.97. Given
that the model under-projects premium forwards pre-season, that is well inside the
margin of error, and the locked squad is a perfectly defensible pick.

Other flags: `--budget`, `--decay`, `--chip {wildcard,freehit,bboost,3xc}` to solve as
if that chip were active in the first gameweek.

### `plan` — transfer planning

```bash
gaffer plan --horizon 6
```

A multi-gameweek MILP over your actual squad. For each gameweek it decides transfers,
hits, captain and formation, tracking free-transfer accumulation (bank up to 5), the
50% sell-on fee, and your bank balance.

Every move comes with an explanation naming the players, the price, the expected-minutes
change, the expected-points gain **over the remaining horizon**, and the fixtures on both
sides of the swap:

```
OUT Kusi-Asare (FUL FWD, 4.5m, 18 mins/GW) -> IN Mateta (CRY FWD, 6.5m, 56 mins/GW):
  +8.20 xP over GW4-6. Fixtures out LIV(A) MUN(H) IPS(A) | in IPS(H) LEE(A) NFO(H).
```

The planner is deliberately patient. A free transfer is valued at 1.6 points
(`ft_value`), so a -4 hit has to clear `hit_threshold` (default 4.0 points over the
horizon, not over one gameweek) before it is recommended. Rolling a transfer is a normal
and frequent answer — "roll" in the table is a decision, not a failure to find one.

`total net expected points` in the summary is the honest undiscounted sum of the
per-gameweek column. `objective` is the solver's internal decayed value and is not a
points forecast — see Known issues.

### `captain` — ranked armband options

```bash
gaffer captain --gw 1
```

| column | meaning |
| --- | --- |
| `xP` | expected points (before doubling) |
| `sd` | standard deviation — how wide the outcome is |
| `P(10+)` | probability of a double-digit haul |
| `EO` | effective ownership: ownership% plus estimated captaincy% |
| `ev vs field` | expected points gained against the average team, from the armband alone |

The prose underneath each option is the point of the command. It states plainly that
**among players you already own, effective ownership does not change the expected value
of a captaincy pick** — the field's armband returns the same whoever you pick. What EO
changes is how *correlated* your gameweek is with everyone else's, which is what decides
whether a good week is a green arrow. A player above 100% EO is close to compulsory and
captaining him is defence, not offence.

Captaincy share is not published by the FPL API. It is approximated from
`selected_by_percent` and projected points, so treat EO as an estimate.

### `chips` — chip timing and the expiry clock

Ranks all four first-half chips by expected gain, with a confidence level, and explains
the reasoning for each. It also prints the double/blank gameweek table (recomputed from
live fixtures every run — never hardcoded) and the countdown to the **GW19 deadline,
when the first set of four chips expires**.

Read the confidence column. Pre-season, with no doubles or blanks in the published
schedule and projections that only run six gameweeks deep, everything is `low` and the
output says so explicitly: *"no double gameweek exists in the published schedule, so this
is the best ordinary gameweek rather than the gameweek Bench Boost is designed for."*
That is the tool telling you to hold, not to play.

### `backtest` — score the model against a finished season

```bash
gaffer backtest --season 2025-26            # whole season, writes reports/
gaffer backtest --gws 2-6 --no-save         # a quick slice, writes nothing
```

Walks the season gameweek by gameweek using only pre-deadline data and compares
projections with what actually happened. Downloads the archive it needs on first run.
The full season takes several minutes.

It reports against four baselines and — importantly — **judges itself only on the
decision-relevant subsets** (players who appeared; players the model expected to start).
The full-universe numbers look better but are inflated by the ~60% of rows that are
players who did not play and scored exactly zero, which any baseline gets right for free.
The report says this itself and excludes it from the verdict.

It also excludes FPL's own `ep_next` from the verdict, because a contamination check
shows the archived value was recomputed after the gameweek was played.

### `serve` — API and dashboard

```bash
gaffer serve --port 8770
```

Serves the dashboard at `http://127.0.0.1:8770/` and a JSON API under `/api/`
(`/api/state`, `/api/players`, `/api/player/{id}`, `/api/fixtures`, `/api/projections`,
`/api/squad`, `/api/captain`, `/api/chips`, `POST /api/optimize`, `POST /api/refresh`).
CORS is restricted to localhost. The dashboard is vanilla JS with no build step and no
CDN dependencies.

---

## Reading the numbers

| term | what it means |
| --- | --- |
| **xP** | expected points: the mean of the outcome distribution, not a prediction |
| **xmins** | expected minutes. The master multiplier — if this is wrong, everything is wrong |
| **p_start / p_60** | probability of starting; probability of reaching 60 minutes (which is what clean sheet points require) |
| **DEFCON** | the defensive-contribution +2. Defenders need 10+ clearances/blocks/interceptions/tackles; midfielders and forwards need 12+ including recoveries. Goalkeepers are ineligible |
| **BPS** | the bonus points system score, which decides who gets the 3/2/1. Not the same thing as bonus points |
| **EO** | effective ownership: ownership% + captaincy%. Above 100% means the average team scores more than one copy of him |
| **FDR** | the official 1-5 fixture difficulty. gaffer builds its own team ratings and uses FDR only as a weak prior |

Every projection decomposes into components that sum exactly to the total:
appearance, goals, assists, clean sheet, goals conceded, saves, DEFCON, bonus, cards,
penalties. `verify` asserts the sum. In the dashboard, click any number to see its parts.

One naming quirk: the `xp_penalty` component is **negative** for penalty takers and zero
for everyone else. It is not "points from penalties" — the expected penalty *goals* are
already inside `lambda_goals`. This component carries only the penalty-miss (-2) and
penalty-save risk, so negative is correct.

---

## Known issues

**Premium forwards are under-projected pre-season, and this distorts the GW1 squad.**
The minutes model blends a player's observed start rate with a price-and-position prior
fitted from last season. The fitted forward curve saturates far lower than the defender
curve: an £8.0m defender is already at the prior cap of 0.921, while a forward does not
reach it at any price the game contains (£15.5m maps to 0.840, and even a hypothetical
£20m forward only reaches 0.896). The result is that a striker who started 34 of 38
games is given a *lower* start probability than a defender who started 30 — the model's
own reason strings print both facts side by side and contradict themselves.

Because expected minutes multiply every other component, this propagates: premium
forwards lose xP, defenders (who also collect clean sheet and DEFCON floor points) gain
it, and the optimizer — behaving correctly given its inputs — leaves Haaland out of the
GW1 squad and captains a £6.0m centre-back in three of the six planned gameweeks.
**If your GW1 squad has no recognised premium forward and the plan wants to captain a
defender, this is why.** Sanity-check those picks against your own judgement.

This is a model problem, not a data problem — `starts` per prior season is present in
the API and is being read correctly; it is simply outweighed by the extrapolated price
prior. It should shrink once real 2026/27 minutes arrive from GW2 onward.

**Two "horizon points" numbers are not comparable.** `squad` reports
`Horizon expected points N over 6 gameweeks`, which is **decay-weighted** by
`optimizer.decay` (default 0.84) and is therefore roughly two thirds of a real points
total. `plan` reports `total net expected points`, which is the honest undiscounted sum.
For the same squad and horizon these read as ~206 and ~303. The `plan` number is the one
that means points.

**Chip advice is a lower bound pre-season.** Projections run six gameweeks deep but the
first-half chips run to GW19. The command warns about this itself.

---

## Configuration

Defaults live in `gaffer/core/config.py`. To override, create `config.json` in the
project root (it is gitignored) with only the keys you want to change:

```json
{
  "entry_id": 1234567,
  "optimizer": {
    "horizon": 8,
    "hit_threshold": 5.0,
    "differential_weight": 0.2,
    "locked_in": [],
    "locked_out": []
  }
}
```

Useful knobs:

| key | default | effect |
| --- | --- | --- |
| `optimizer.horizon` | 6 | how many gameweeks to plan over |
| `optimizer.decay` | 0.84 | how much less a future gameweek counts than this one |
| `optimizer.hit_threshold` | 4.0 | expected gain over the horizon before a -4 is recommended |
| `optimizer.ft_value` | 1.6 | points value of a banked free transfer |
| `optimizer.differential_weight` | 0.0 | 0 = pure expected points, 1 = maximum differential chasing |
| `optimizer.locked_in` / `locked_out` | `[]` | force player ids in or out of every solve |
| `model.rotation_risk_penalty` | 0.0 | extra rotation damping for clubs in Europe |

Environment variables `FPL_ENTRY_ID` and `ODDS_API_KEY` override the file.

---

## Layout

```
gaffer/core/       scoring constants, shared types, config, statistics helpers
gaffer/data/       FPL API client, disk cache, historical data loaders
gaffer/model/      team ratings, minutes, attacking, defending, DEFCON, bonus, xP assembler
gaffer/optimize/   squad MILP, transfer planner, chips, rank strategy
gaffer/backtest/   walk-forward season replay and metrics
gaffer/api/        FastAPI server
gaffer/web/        dashboard (vanilla JS, no build step)
data/cache/        everything fetched, plus cached projections (gitignored)
reports/           fitted model parameters and backtest results (gitignored)
```

`data/cache/` and `reports/` are gitignored and regenerate themselves. Deleting them
costs you one slow run, nothing more. Fitted parameters (`reports/*_fit.json`) are
refit automatically when missing or stale.

## Development

```bash
./.venv/bin/python -m pytest              # the suite
./.venv/bin/python -m pytest -m "not slow"   # skip MILP solves and season replays
```

Two markers: `slow` (real MILP solves, whole-season simulation) and `livedata` (needs a
warm `data/cache`, skipped when absent).

**All code must be Python 3.9 compatible** — `from __future__ import annotations`,
`Optional[X]` not `X | None`, `Dict`/`List` from `typing` in signatures. Test with
`.venv/bin/python`, never the system interpreter.
