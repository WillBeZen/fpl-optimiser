"""Backtest — does the model actually beat simple rules? Walk-forward, no peeking.

For each gameweek r of the season under test, fit on every row from BEFORE r
(earlier seasons, plus this season's rounds < r), predict r, and score the
predictions against what happened. Every prediction is therefore made exactly as
it could have been on the day, with only the past.

Compared against two baselines anyone could compute:
  form         the recency-weighted points per gameweek (the form_points_pg feature)
  season avg   the player's plain average points per gameweek so far this season
and, when the season's raw data is passed in (`events_data`), the event model
(model/events.py), which adds up goals, assists, clean sheets... from team and
player rates as of each round, with minutes from the same fitted minutes model.

Scores per method:
  MAE / RMSE   average miss, in points per fixture
  rank corr    how well it ORDERS players in each gameweek (Spearman; what picking needs)
  top-11 pts   actual points of the 11 players it rated highest, per gameweek
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from fpl_optimiser.model import events, points

METHODS = {"xp_blend": "Blend (what the planner uses)", "xp": "Points model",
           "xp_events": "Event model", "form": "Form (recency-weighted)",
           "season_avg": "Season average"}


def _player_team(rows: pd.DataFrame, fixtures: pd.DataFrame) -> pd.Series:
    """The player's own club for each training row (its opponent, round and venue
    identify the fixture)."""
    fx = fixtures.dropna(subset=["gameweek"])
    home = {(int(g), int(a)): int(h) for g, h, a in zip(fx["gameweek"], fx["home_id"], fx["away_id"])}
    away = {(int(g), int(h)): int(a) for g, h, a in zip(fx["gameweek"], fx["home_id"], fx["away_id"])}
    return pd.Series([(home if wh else away).get((int(r), int(o))) for r, o, wh in
                      zip(rows["round"], rows["opponent_team"], rows["was_home"])], index=rows.index)


def walk_forward(table: pd.DataFrame, season: str, rounds: list[int] | None = None,
                 events_data: dict | None = None,
                 half_life: float = config.RECENCY_HALF_LIFE_GW) -> pd.DataFrame:
    """Out-of-sample predictions for `season`'s rounds (default: all but its first).

    table       : multi_season.build_all(half_life=...) — rows with season, round,
                  targets, features (its form features carry the table's half-life).
    events_data : optional events.season_data(season), to backtest the event model
                  too (its team and player rates use `half_life`, so pass the same
                  half-life the table was built with).
    Returns that season's rows for the tested rounds with xp (and xp_events, its
    points by source, and xp_blend), form, season_avg, and `trained_to`: the last
    round of this season the models were fitted on (always before the row's round).
    """
    known = table.dropna(subset=["target_points", "minutes"])
    this = known[known["season"] == season]
    rounds = rounds or sorted(this["round"].unique())[1:]
    out = []
    for r in rounds:
        train = known[(known["season"] < season) | ((known["season"] == season) & (known["round"] < r))]
        test = this[this["round"] == r].copy()
        if test.empty or train.empty:
            continue
        pred = points.fit(train).predict(test)
        this_train = train[train["season"] == season]["round"]
        test["trained_to"] = int(this_train.max()) if len(this_train) else 0
        test["xp"] = pred["xp"].to_numpy()
        test["p_play"], test["p_60"] = pred["p_play"].to_numpy(), pred["p_60"].to_numpy()
        if events_data is not None:
            d = events_data
            rows = test.assign(team=_player_team(test, d["fixtures"])).dropna(subset=["team"])
            ratings = events.team_ratings(d["fixtures"], d["teams"], r, d["history"],
                                          d.get("prev_fixtures"), d.get("prev_teams"), half_life=half_life)
            rates = events.player_rates(d["history"], d["players"], r, d.get("prior"), half_life=half_life)
            ev = events.expected_events(rows, ratings, rates)
            test["events_half_life"] = float(half_life)
            for col in events.SOURCES + ["xp_events"]:
                test[col] = ev[col].reindex(test.index)
            w = config.EVENT_MODEL_WEIGHT
            test["xp_blend"] = (1 - w) * test["xp"] + w * test["xp_events"]
        test["form"] = test["form_points_pg"].fillna(train["target_points"].mean())
        so_far = this[this["round"] < r].groupby("id")["target_points"].mean()
        test["season_avg"] = test["id"].map(so_far).fillna(train["target_points"].mean())
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarise(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per method: MAE, RMSE, mean per-gameweek rank correlation, top-11 points."""
    rows = []
    for col, name in METHODS.items():
        if col not in preds or preds[col].isna().all():
            continue
        err = preds[col] - preds["target_points"]
        corr = preds.groupby("round")[[col, "target_points"]].apply(
            lambda d: d[col].corr(d["target_points"], method="spearman"))
        # a double gameweek counts once: sum each player's fixtures in the round first
        per_player = preds.groupby(["round", "id"])[[col, "target_points"]].sum().reset_index()
        top11 = per_player.groupby("round").apply(
            lambda d: d.nlargest(11, col)["target_points"].sum(), include_groups=False)
        rows.append({"method": name, "MAE": err.abs().mean(), "RMSE": float(np.sqrt((err ** 2).mean())),
                     "rank_corr": corr.mean(), "top11_points": top11.mean(),
                     "gameweeks": preds["round"].nunique(), "predictions": len(preds)})
    return pd.DataFrame(rows)
