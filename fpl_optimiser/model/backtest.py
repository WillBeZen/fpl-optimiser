"""Backtest — does the model actually beat simple rules? Walk-forward, no peeking.

For each gameweek r of the season under test, fit on every row from BEFORE r
(earlier seasons, plus this season's rounds < r), predict r, and score the
predictions against what happened. Every prediction is therefore made exactly as
it could have been on the day, with only the past.

Compared against two baselines anyone could compute:
  form         the recency-weighted points per gameweek (the form_points_pg feature)
  season avg   the player's plain average points per gameweek so far this season

Scores per method:
  MAE / RMSE   average miss, in points per fixture
  rank corr    how well it ORDERS players in each gameweek (Spearman; what picking needs)
  top-11 pts   actual points of the 11 players it rated highest, per gameweek
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_optimiser.model import points

METHODS = {"xp": "Model", "form": "Form (recency-weighted)", "season_avg": "Season average"}


def walk_forward(table: pd.DataFrame, season: str,
                 rounds: list[int] | None = None) -> pd.DataFrame:
    """Out-of-sample predictions for `season`'s rounds (default: all but its first).

    table : multi_season.build_all() — rows with season, round, targets, features.
    Returns that season's rows for the tested rounds with xp, form and season_avg.
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
        test["xp"] = points.fit(train).predict(test)["xp"].to_numpy()
        test["form"] = test["form_points_pg"].fillna(train["target_points"].mean())
        so_far = this[this["round"] < r].groupby("id")["target_points"].mean()
        test["season_avg"] = test["id"].map(so_far).fillna(train["target_points"].mean())
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarise(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per method: MAE, RMSE, mean per-gameweek rank correlation, top-11 points."""
    rows = []
    for col, name in METHODS.items():
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
