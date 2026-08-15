"""Shared data types. Every module in gaffer exchanges these objects.

This file is the integration contract: model modules produce them, the
optimizer and API consume them. Change a field here only with a matching
update to SPEC.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Static entities
# ---------------------------------------------------------------------------


@dataclass
class Team:
    id: int
    name: str
    short_name: str
    code: int
    strength_overall_home: int = 0
    strength_overall_away: int = 0
    strength_attack_home: int = 0
    strength_attack_away: int = 0
    strength_defence_home: int = 0
    strength_defence_away: int = 0
    promoted: bool = False


@dataclass
class Player:
    id: int
    code: int
    web_name: str
    first_name: str
    second_name: str
    team_id: int
    position: int  # 1 GKP, 2 DEF, 3 MID, 4 FWD
    now_cost: int  # tenths of a million
    status: str = "a"  # a available, d doubtful, i injured, s suspended, u unavailable
    news: str = ""
    chance_of_playing_next_round: Optional[int] = None
    chance_of_playing_this_round: Optional[int] = None
    selected_by_percent: float = 0.0
    penalties_order: Optional[int] = None
    corners_order: Optional[int] = None
    direct_freekicks_order: Optional[int] = None
    minutes: int = 0
    starts: int = 0
    total_points: int = 0
    points_per_game: float = 0.0
    form: float = 0.0
    ep_next: float = 0.0
    price_change_percent: float = 0.0
    transfers_in_event: int = 0
    transfers_out_event: int = 0
    # Raw carry-over / season-to-date underlying stats
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    expected_goal_involvements: float = 0.0
    expected_goals_conceded: float = 0.0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    defensive_contribution: int = 0
    clearances_blocks_interceptions: int = 0
    recoveries: int = 0
    tackles: int = 0
    team_join_date: Optional[str] = None

    @property
    def price(self) -> float:
        return self.now_cost / 10.0

    @property
    def name(self) -> str:
        return self.web_name


@dataclass
class Fixture:
    id: int
    code: int
    gw: Optional[int]
    team_h: int
    team_a: int
    kickoff_time: Optional[str]
    team_h_difficulty: int = 3
    team_a_difficulty: int = 3
    finished: bool = False
    team_h_score: Optional[int] = None
    team_a_score: Optional[int] = None

    def opponent_of(self, team_id: int) -> int:
        return self.team_a if team_id == self.team_h else self.team_h

    def is_home_for(self, team_id: int) -> bool:
        return team_id == self.team_h


# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------


@dataclass
class TeamStrength:
    """Attack / defence ratings on a Poisson goal-rate scale.

    attack is a multiplicative factor on league-average goals scored,
    defence is a multiplicative factor on league-average goals conceded.
    1.0 means exactly league average; attack 1.3 means 30% more goals.
    """

    team_id: int
    attack: float = 1.0
    defence: float = 1.0
    home_advantage: float = 1.0
    n_matches: int = 0
    confidence: float = 0.0  # 0-1, how much real data backs this rating


@dataclass
class MatchForecast:
    """Expected goals for both sides of one fixture."""

    fixture_id: int
    gw: int
    team_h: int
    team_a: int
    lambda_h: float
    lambda_a: float
    p_cs_h: float  # P(home team keeps a clean sheet)
    p_cs_a: float
    p_home_win: float = 0.0
    p_draw: float = 0.0
    p_away_win: float = 0.0
    source: str = "model"  # "model", "odds", "blend"


@dataclass
class MinutesForecast:
    player_id: int
    gw: int
    p_appear: float  # P(minutes >= 1)
    p_start: float  # P(named in starting XI)
    p_60: float  # P(minutes >= 60)
    xmins: float  # expected minutes
    reason: str = ""  # human-readable explanation of the call


@dataclass
class PlayerFixtureProjection:
    """Projected points for one player in one fixture.

    A player with two fixtures in a double gameweek gets two of these; the
    gameweek total is the sum.
    """

    player_id: int
    gw: int
    fixture_id: int
    opponent_id: int
    is_home: bool
    difficulty: int = 3

    # Availability
    p_appear: float = 0.0
    p_start: float = 0.0
    p_60: float = 0.0
    xmins: float = 0.0

    # Underlying rates for this fixture
    lambda_goals: float = 0.0
    lambda_assists: float = 0.0
    lambda_conceded: float = 0.0
    lambda_saves: float = 0.0
    p_clean_sheet: float = 0.0
    p_defcon: float = 0.0
    exp_bps: float = 0.0

    # Point components
    xp_appearance: float = 0.0
    xp_goals: float = 0.0
    xp_assists: float = 0.0
    xp_clean_sheet: float = 0.0
    xp_goals_conceded: float = 0.0
    xp_saves: float = 0.0
    xp_defcon: float = 0.0
    xp_bonus: float = 0.0
    xp_cards: float = 0.0
    xp_penalty: float = 0.0  # penalty saves minus penalty misses
    xp_total: float = 0.0
    sd_total: float = 0.0  # standard deviation of points, for captain risk

    def components(self) -> Dict[str, float]:
        return {
            "appearance": self.xp_appearance,
            "goals": self.xp_goals,
            "assists": self.xp_assists,
            "clean_sheet": self.xp_clean_sheet,
            "goals_conceded": self.xp_goals_conceded,
            "saves": self.xp_saves,
            "defcon": self.xp_defcon,
            "bonus": self.xp_bonus,
            "cards": self.xp_cards,
            "penalty": self.xp_penalty,
        }


@dataclass
class PlayerGWProjection:
    """A player's total projection for a gameweek (sums over 0, 1 or 2 fixtures)."""

    player_id: int
    gw: int
    xp: float = 0.0
    sd: float = 0.0
    fixtures: List[PlayerFixtureProjection] = field(default_factory=list)

    @property
    def n_fixtures(self) -> int:
        return len(self.fixtures)

    @property
    def is_blank(self) -> bool:
        return len(self.fixtures) == 0


@dataclass
class ProjectionSet:
    """Everything the optimizer needs: xp[player_id][gw]."""

    season: str
    generated_at: str
    first_gw: int
    last_gw: int
    projections: Dict[int, Dict[int, PlayerGWProjection]] = field(default_factory=dict)
    model_version: str = "0.1.0"

    def xp(self, player_id: int, gw: int) -> float:
        try:
            return self.projections[player_id][gw].xp
        except KeyError:
            return 0.0

    def gws(self) -> List[int]:
        return list(range(self.first_gw, self.last_gw + 1))

    def player_ids(self) -> List[int]:
        return sorted(self.projections.keys())


# ---------------------------------------------------------------------------
# Squad / decisions
# ---------------------------------------------------------------------------


@dataclass
class SquadPick:
    player_id: int
    purchase_price: int  # tenths of a million paid
    selling_price: int  # tenths of a million you would receive now
    multiplier: int = 1  # 0 bench, 1 starter, 2 captain, 3 triple captain
    position_in_squad: int = 0  # 1-15, bench order for 12-15


@dataclass
class SquadState:
    """Where the manager currently stands."""

    gw: int
    picks: List[SquadPick] = field(default_factory=list)
    bank: int = 0  # tenths of a million
    free_transfers: int = 1
    chips_used: List[str] = field(default_factory=list)
    entry_id: Optional[int] = None
    total_points: int = 0
    overall_rank: Optional[int] = None

    @property
    def squad_value(self) -> int:
        return sum(p.selling_price for p in self.picks)

    @property
    def budget(self) -> int:
        return self.squad_value + self.bank

    def player_ids(self) -> List[int]:
        return [p.player_id for p in self.picks]


@dataclass
class Transfer:
    gw: int
    out_id: int
    in_id: int
    out_price: int
    in_price: int


@dataclass
class GWDecision:
    """The recommendation for a single gameweek."""

    gw: int
    transfers: List[Transfer] = field(default_factory=list)
    hits: int = 0  # number of -4 penalties taken
    chip: Optional[str] = None
    lineup: List[int] = field(default_factory=list)  # 11 player ids
    bench: List[int] = field(default_factory=list)  # 4 player ids in order
    captain: Optional[int] = None
    vice_captain: Optional[int] = None
    squad: List[int] = field(default_factory=list)  # 15 player ids after transfers
    bank_after: int = 0
    free_transfers_after: int = 1
    expected_points: float = 0.0
    expected_points_net: float = 0.0  # after hit costs
    notes: List[str] = field(default_factory=list)


@dataclass
class Plan:
    """A multi-gameweek plan produced by the optimizer."""

    first_gw: int
    horizon: int
    decisions: List[GWDecision] = field(default_factory=list)
    objective: float = 0.0
    decay: float = 0.84
    solver_status: str = ""
    solve_seconds: float = 0.0

    @property
    def total_expected_points(self) -> float:
        return sum(d.expected_points_net for d in self.decisions)


@dataclass
class CaptainOption:
    player_id: int
    xp: float
    sd: float
    p_haul: float  # P(points >= 10)
    effective_ownership: float = 0.0
    ev_vs_field: float = 0.0  # expected rank-adjusted gain over the template captain
    rationale: str = ""
