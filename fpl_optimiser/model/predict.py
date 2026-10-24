"""Expected points for the coming gameweeks — the prediction engine's output.

`forecast()` fits the points model on every filed season (the multi-season table,
sample-weighted), builds feature rows for each current player's fixtures in the
next `horizon` gameweeks (computed as of the next deadline, exactly as training
computes them), predicts expected points per fixture, applies FPL's injury and
suspension flags, and adds up double gameweeks.

The result has one row per (player, gameweek): `xp` (expected points, after
availability), `xp_raw` (before), `p_play`, `p_60`, `fixtures` ("ARS (H), CHE (A)"),
plus who he is (id, code, name, club, position, price, status, news). A blank
gameweek gives xp 0. This table is what the optimiser (Sections 4-5) consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from fpl_optimiser.features import build, multi_season
from fpl_optimiser.ingest import season_store
from fpl_optimiser.model import points

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


def availability(players: pd.DataFrame, gw: int, next_gw: int) -> pd.Series:
    """Multiplier on each player's chance of playing in `gw` (index = player id).

    Next gameweek: FPL's chance_of_playing_next_round (75 -> 0.75; blank = fit),
    0 for injured / suspended players FPL gives no chance for. Later gameweeks: FPL
    publishes nothing, so players are assumed back (the Claude advisor reads the
    news for longer absences). Left the club / not in the squad: 0 throughout.
    """
    status = players["status"].astype("string").fillna("a")
    chance = pd.to_numeric(players["chance_of_playing_next_round"], errors="coerce")
    if gw == next_gw:
        m = (chance / 100).where(chance.notna(), np.where(status.isin(OUT_NEXT_GW), 0.0, 1.0))
    else:
        m = pd.Series(1.0, index=players.index)
    m = m.where(~status.isin(OUT_FOR_GOOD), 0.0)
    return pd.Series(m.to_numpy(dtype=float), index=players["id"].to_numpy())


def forecast(season: str | None = None, horizon: int = config.PLANNING_HORIZON_GW,
             table: pd.DataFrame | None = None) -> Forecast:
    """Fit on every filed season and predict the next `horizon` gameweeks of `season`."""
    season = season or season_store.available()[-1]
    table = multi_season.build_all() if table is None else table
    model = points.fit(table)

    cur, prior, profiles = multi_season.season_inputs(season)
    nxt = next_gameweek(cur.gameweeks)
    if nxt is None:
        raise ValueError(f"{season} is over: there are no gameweeks left to predict")
    gws = [g for g in range(nxt, nxt + horizon) if g <= int(cur.gameweeks["id"].max())]

    rows = build.build_prediction_table(cur.history, cur.fixtures, cur.players, profiles,
                                        as_of_round=nxt, gameweeks=gws, prior=prior)
    fx = rows.join(model.predict(rows))
    fx["xp_raw"] = fx["xp"]
    for gw in gws:
        on = fx["round"] == gw
        mult = fx.loc[on, "id"].map(availability(cur.players, gw, nxt)).fillna(1.0)
        fx.loc[on, "xp"] = fx.loc[on, "xp_raw"] * mult
        fx.loc[on, "p_play"] = fx.loc[on, "p_play"] * mult
        fx.loc[on, "p_60"] = fx.loc[on, "p_60"] * mult
    short = cur.teams.set_index("id")["short_name"]
    fx["fixture"] = (fx["opponent_team"].map(short) + np.where(fx["was_home"], " (H)", " (A)"))

    # one row per (player, gameweek): double gameweeks add up, blanks are 0
    per_gw = fx.groupby(["id", "round"]).agg(
        xp=("xp", "sum"), xp_raw=("xp_raw", "sum"), p_play=("p_play", "max"),
        p_60=("p_60", "max"), fixtures=("fixture", ", ".join))
    grid = pd.MultiIndex.from_product([cur.players["id"], gws], names=["id", "round"])
    per_gw = per_gw.reindex(grid).fillna({"xp": 0.0, "xp_raw": 0.0, "p_play": 0.0,
                                          "p_60": 0.0, "fixtures": "-"}).reset_index()
    info = cur.players[["id", "code", "web_name", "team_name", "position", "price", "status", "news"]]
    xp = info.merge(per_gw.rename(columns={"round": "gw"}), on="id")
    return Forecast(season=season, next_gw=nxt, gameweeks=gws, xp=xp, per_fixture=fx, model=model)
