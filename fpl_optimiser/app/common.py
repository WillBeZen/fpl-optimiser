"""What the dashboard's tabs share: the week's results, the season context, and a
few lookups every tab needs.

`WeekResults` is what the weekly run (scripts/run_week.py) produced; `Context` is
the filed season the report is built from. Every tab module has one function,
`add(report, week, ctx)`, and skips whatever its inputs don't support (no team id:
no personal plan; no event ratings: no fixture ticker), so the report always builds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:                     # type hints only; nothing imported at runtime
    from fpl_optimiser.claude.advisor import Advice
    from fpl_optimiser.ingest.my_team import MyTeam
    from fpl_optimiser.ingest.season_store import Season
    from fpl_optimiser.model.predict import Forecast
    from fpl_optimiser.optimise.planner import Plan
    from fpl_optimiser.optimise.squad import Squad


@dataclass
class WeekResults:
    """What the weekly run (scripts/run_week.py) produced; every part but the
    forecast and the from-scratch squad is optional."""
    forecast: Forecast
    wildcard: Squad                    # the best squad from scratch (a wildcard)
    plan: Plan | None = None           # your transfer plan
    baseline: Plan | None = None       # your squad with no transfers
    team: MyTeam | None = None
    advice: Advice | None = None       # None = skipped
    backtest: pd.DataFrame | None = None   # model.backtest.summarise(...)
    table: pd.DataFrame | None = None      # the multi-season training table
    ladder: pd.DataFrame | None = None     # optimise.whatif.transfer_ladder(...)
    budget: pd.DataFrame | None = None     # optimise.whatif.budget_value(...)
    chips: pd.DataFrame | None = None      # optimise.whatif.chip_weeks(...)


@dataclass
class Context:
    """The filed season behind the report (all optional, for tests)."""
    season: Season | None = None
    prior: pd.DataFrame | None = None       # last season per player (prior_season.prior_table)
    profiles: pd.DataFrame | None = None    # opponent profiles with archetype
    types: pd.DataFrame | None = None       # profiled players with archetype_name (player types)


SCOUT_COLS = ["selected_by_percent", "penalties_order", "corners_and_indirect_freekicks_order",
              "direct_freekicks_order", "transfers_in_event", "transfers_out_event", "cost_change_event",
              "cost_change_start", "ep_next", "price_change_percent", "goals_scored", "assists",
              "expected_goals", "expected_assists", "minutes", "total_points", "team_short"]


def player_info(w: WeekResults, ctx: Context | None) -> pd.DataFrame:
    """One row per current player (index = id): who he is, price, status and news,
    plus FPL's scouting fields (ownership, set pieces, transfers, ...) when filed."""
    info = w.forecast.players.set_index("id")
    if ctx is not None and ctx.season is not None:
        extra = ctx.season.players.set_index("id")
        info = info.join(extra[[c for c in SCOUT_COLS if c in extra.columns]], how="left")
    for c in SCOUT_COLS:
        if c not in info:
            info[c] = np.nan
    for c in SCOUT_COLS:
        if c != "team_short":
            info[c] = pd.to_numeric(info[c], errors="coerce")
    return info


def squad_roles(w: WeekResults) -> pd.DataFrame:
    """The players this week is about: your squad today plus this week's buys,
    each marked Keep / Sell / Buy (empty if there's no team)."""
    if w.team is None or w.plan is None:
        return pd.DataFrame(columns=["id", "plan"])
    first = w.plan.weeks[0]
    sold, bought = set(first.sold["id"]), set(first.bought["id"])
    rows = [{"id": i, "plan": "Sell" if i in sold else "Keep"} for i in w.team.squad_ids]
    rows += [{"id": i, "plan": "Buy"} for i in first.bought["id"] if i not in set(w.team.squad_ids)]
    return pd.DataFrame(rows)


def next_fixtures(w: WeekResults) -> pd.DataFrame:
    """The forecast's per-fixture rows for the next gameweek."""
    fx = w.forecast.per_fixture
    if "round" not in fx:
        return fx.iloc[0:0]
    return fx[fx["round"] == w.forecast.next_gw]


def has_events(w: WeekResults) -> bool:
    return w.forecast.ratings is not None and "xp_events" in w.forecast.per_fixture


def money(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"£{v:.1f}m"


def pct(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}"


def own(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}%"
