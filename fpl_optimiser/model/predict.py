"""Expected points for the coming gameweeks — the prediction engine's output.

`forecast()` fits the points model on every filed season (the multi-season table,
sample-weighted), builds feature rows for each current player's fixtures in the
next `horizon` gameweeks (computed as of the next deadline, exactly as training
computes them), predicts expected points per fixture, applies FPL's injury and
suspension flags, and adds up double gameweeks.

Each fixture also gets the event model's breakdown (model/events.py: expected
goals, assists, clean sheet, ... and points by source), and the expected points the
optimiser uses blend the two models:

    xp = (1 - w) x xp_model + w x xp_events        w = config.EVENT_MODEL_WEIGHT

The result has one row per (player, gameweek): `xp` (expected points, after
availability), `xp_raw` (before), `xp_model` and `xp_events` (the two parts),
`p_play`, `p_60`, `fixtures` ("ARS (H), CHE (A)"), plus who he is (id, code, name,
club, position, price, status, news). A blank gameweek gives xp 0. This table is
what the optimiser (Sections 4-5) consumes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from fpl_optimiser.features import build, multi_season
from fpl_optimiser.ingest import season_store
from fpl_optimiser.model import events, points

# FPL status codes: a available, d doubtful, i injured, s suspended,
# u unavailable (left the club / on loan), n not in the squad.
OUT_FOR_GOOD = {"u", "n"}
OUT_NEXT_GW = {"i", "s"}


@dataclass
class Forecast:
    season: str
    next_gw: int
    gameweeks: list[int]
    xp: pd.DataFrame            # one row per (player, gameweek)
    per_fixture: pd.DataFrame   # one row per (player, fixture), with the model's parts
    model: points.PointsModel
    deadline: str = ""          # the next gameweek's deadline, "YYYY-MM-DD HH:MM" UTC
    ratings: events.Ratings | None = None   # team attack/defence as of the next gameweek
    rates: pd.DataFrame | None = None       # each player's per-90 rates (events.player_rates)
    event_weight: float = config.EVENT_MODEL_WEIGHT

    @property
    def players(self) -> pd.DataFrame:
        """One row per current player: id, code, name, club, position, price, status, news."""
        cols = ["id", "code", "web_name", "team_name", "position", "price", "status", "news"]
        return self.xp.drop_duplicates("id")[cols].reset_index(drop=True)

    def wide(self) -> pd.DataFrame:
        """One row per player: xp in each gameweek (GW9, GW10, ...) and their total."""
        info = ["id", "code", "web_name", "team_name", "position", "price", "status", "news"]
        grid = self.xp.pivot_table(index="id", columns="gw", values="xp", aggfunc="sum")
        grid.columns = [f"GW{g}" for g in grid.columns]
        grid["total"] = grid.sum(axis=1)
        players = self.xp.drop_duplicates("id").set_index("id")[info[1:]]
        return players.join(grid).reset_index().sort_values("total", ascending=False)


def next_gameweek(gameweeks: pd.DataFrame) -> int | None:
    """The next gameweek whose deadline hasn't passed (None once the season is over)."""
    if "is_next" in gameweeks:
        nxt = gameweeks.loc[gameweeks["is_next"] == True, "id"]  # noqa: E712
        if len(nxt):
            return int(nxt.iloc[0])
    open_ = gameweeks.loc[gameweeks["finished"] != True, "id"]  # noqa: E712
    return int(open_.min()) if len(open_) else None


_BACK = re.compile(r"(?:expected back|suspended until)\s+(\d{1,2})\s+([a-z]{3})", re.IGNORECASE)


def return_date(news, after: pd.Timestamp) -> pd.Timestamp | None:
    """The date in "Expected back 8 Nov" / "Suspended until 31 Oct" (None if there's
    none), in the year that puts it on or after `after` (minus a few weeks' grace)."""
    m = _BACK.search(news) if isinstance(news, str) else None
    if not m:
        return None
    after = pd.Timestamp(after).tz_localize(None) if pd.Timestamp(after).tzinfo else pd.Timestamp(after)
    try:
        d = pd.Timestamp(f"{m.group(1)} {m.group(2)} {after.year}")
    except ValueError:
        return None
    return d if d >= after - pd.Timedelta(days=60) else d + pd.DateOffset(years=1)


def availability(players: pd.DataFrame, gw: int, next_gw: int,
                 deadlines: dict[int, pd.Timestamp] | None = None) -> pd.Series:
    """Multiplier on each player's chance of playing in `gw` (index = player id).

    Next gameweek: FPL's chance_of_playing_next_round (75 -> 0.75; blank = fit),
    0 for injured / suspended players FPL gives no chance for. Later gameweeks (with
    `deadlines`, gameweek -> deadline): out until the date in his news ("Expected
    back 8 Nov"); injured with "Unknown return date", back gradually at
    config.UNKNOWN_RETURN_RAMP; everyone else assumed fit. Left the club / not in
    the squad: 0 throughout.
    """
    status = players["status"].astype("string").fillna("a")
    chance = pd.to_numeric(players["chance_of_playing_next_round"], errors="coerce")
    if gw == next_gw:
        m = (chance / 100).where(chance.notna(), np.where(status.isin(OUT_NEXT_GW), 0.0, 1.0))
    else:
        m = pd.Series(1.0, index=players.index)
        if deadlines and gw in deadlines and next_gw in deadlines:
            news = players["news"] if "news" in players else pd.Series("", index=players.index)
            when = pd.Timestamp(deadlines[gw])
            when = when.tz_localize(None) if when.tzinfo else when
            ramp = config.UNKNOWN_RETURN_RAMP
            k = gw - next_gw                                    # gameweeks after the next one
            for i, text in news.items():
                back = return_date(text, pd.Timestamp(deadlines[next_gw]))
                if back is not None:
                    m[i] = 1.0 if when.normalize() >= back else 0.0
                elif (status[i] in OUT_NEXT_GW and isinstance(text, str)
                      and "unknown return" in text.lower()):
                    m[i] = ramp[k - 1] if k - 1 < len(ramp) else 1.0
    m = m.where(~status.isin(OUT_FOR_GOOD), 0.0)
    return pd.Series(m.to_numpy(dtype=float), index=players["id"].to_numpy())


def forecast_rows(model: points.PointsModel, cur, prior: pd.DataFrame | None, profiles: pd.DataFrame,
                  as_of: int, gameweeks: list[int], players: pd.DataFrame | None = None,
                  fixtures: pd.DataFrame | None = None, prev=None,
                  half_life: float = config.RECENCY_HALF_LIFE_GW):
    """The shared core of every forecast, live or replayed: one row per (player,
    fixture in `gameweeks`) with every feature as of `as_of`'s deadline, the points
    model's p_play / p_60 / xp, and the fixture text; plus the event model's team
    ratings and player rates as of the same deadline. No injury flags, no blend.

    `players` / `fixtures` default to the filed season's; a replay passes the players
    as they stood at that deadline and the fixtures with later results hidden
    (model/evaluation.py)."""
    players = cur.players if players is None else players
    fixtures = cur.fixtures if fixtures is None else fixtures
    rows = build.build_prediction_table(cur.history, fixtures, players, profiles, as_of_round=as_of,
                                        gameweeks=gameweeks, half_life=half_life, prior=prior)
    fx = rows.join(model.predict(rows))
    short = cur.teams.set_index("id")["short_name"]
    fx["fixture"] = (fx["opponent_team"].map(short) + np.where(fx["was_home"], " (H)", " (A)"))
    ratings = events.team_ratings(cur.fixtures, cur.teams, as_of, cur.history,
                                  prev.fixtures if prev else None, prev.teams if prev else None,
                                  half_life=half_life)
    rates = events.player_rates(cur.history, cur.players, as_of, prior, half_life=half_life)
    return fx, ratings, rates


def forecast(season: str | None = None, horizon: int = config.PLANNING_HORIZON_GW,
             table: pd.DataFrame | None = None,
             event_weight: float = config.EVENT_MODEL_WEIGHT) -> Forecast:
    """Fit on every filed season and predict the next `horizon` gameweeks of `season`."""
    season = season or season_store.available()[-1]
    table = multi_season.build_all() if table is None else table
    model = points.fit(table)

    cur, prior, profiles = multi_season.season_inputs(season)
    nxt = next_gameweek(cur.gameweeks)
    if nxt is None:
        raise ValueError(f"{season} is over: there are no gameweeks left to predict")
    gws = [g for g in range(nxt, nxt + horizon) if g <= int(cur.gameweeks["id"].max())]

    deadlines = dict(zip(cur.gameweeks["id"].astype(int), cur.gameweeks["deadline_time"]))
    prev_season = season_store.previous_season(season)
    prev = season_store.load(prev_season) if prev_season in season_store.available() else None
    fx, ratings, rates = forecast_rows(model, cur, prior, profiles, nxt, gws, prev=prev)
    fx["xp_raw"] = fx["xp"]
    unflagged = fx[["p_play", "p_60"]].copy()          # minutes before FPL's injury flags
    for gw in gws:
        on = fx["round"] == gw
        mult = fx.loc[on, "id"].map(availability(cur.players, gw, nxt, deadlines)).fillna(1.0)
        fx.loc[on, "xp"] = fx.loc[on, "xp_raw"] * mult
        fx.loc[on, "p_play"] = fx.loc[on, "p_play"] * mult
        fx.loc[on, "p_60"] = fx.loc[on, "p_60"] * mult
    # the event model's breakdown, and the blend the optimiser uses
    fx = fx.join(events.expected_events(fx, ratings, rates))
    events_raw = events.expected_events(fx.assign(**unflagged), ratings, rates)["xp_events"]
    fx["xp_model"], fx["xp_model_raw"] = fx["xp"], fx["xp_raw"]
    fx["xp"] = (1 - event_weight) * fx["xp_model"] + event_weight * fx["xp_events"]
    fx["xp_raw"] = (1 - event_weight) * fx["xp_model_raw"] + event_weight * events_raw

    # one row per (player, gameweek): double gameweeks add up, blanks are 0
    per_gw = fx.groupby(["id", "round"]).agg(
        xp=("xp", "sum"), xp_raw=("xp_raw", "sum"), xp_model=("xp_model", "sum"),
        xp_events=("xp_events", "sum"), p_play=("p_play", "max"),
        p_60=("p_60", "max"), fixtures=("fixture", ", ".join))
    grid = pd.MultiIndex.from_product([cur.players["id"], gws], names=["id", "round"])
    per_gw = per_gw.reindex(grid).fillna({"xp": 0.0, "xp_raw": 0.0, "xp_model": 0.0, "xp_events": 0.0,
                                          "p_play": 0.0, "p_60": 0.0, "fixtures": "-"}).reset_index()
    info = cur.players[["id", "code", "web_name", "team_name", "position", "price", "status", "news"]]
    xp = info.merge(per_gw.rename(columns={"round": "gw"}), on="id")
    when = cur.gameweeks.loc[cur.gameweeks["id"] == nxt, "deadline_time"]
    deadline = f"{when.iloc[0]:%Y-%m-%d %H:%M}" if len(when) and pd.notna(when.iloc[0]) else ""
    return Forecast(season=season, next_gw=nxt, gameweeks=gws, xp=xp, per_fixture=fx,
                    model=model, deadline=deadline, ratings=ratings, rates=rates,
                    event_weight=event_weight)
