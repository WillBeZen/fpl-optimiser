"""Points model — expected FPL points per player per fixture.

    xp = p_play x pts_if_play

p_play comes from the minutes model (minutes.py); pts_if_play is a regressor
trained only on matches the player actually played, so it learns what he scores
WHEN he plays. Splitting it this way means injury news (FPL's chance of playing)
scales exactly the part it should, and the regressor never has to learn "points"
and "will he be picked" as one blurry thing.

Same inputs and the same kind of trees as the minutes model (gradient-boosted,
HistGradientBoosting), trained on the multi-season table with its sample_weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from fpl_optimiser.model import inputs, minutes

PARAMS = dict(minutes.PARAMS)          # same regularised trees, squared-error loss


@dataclass
class PointsModel:
    minutes: minutes.MinutesModel
    if_played: HistGradientBoostingRegressor | DummyRegressor   # E[points | he plays]

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """p_play, p_60, pts_if_play and xp for each row (training or prediction table)."""
        out = self.minutes.predict(rows)
        out["pts_if_play"] = self.if_played.predict(inputs.design_matrix(rows))
        out["xp"] = out["p_play"] * out["pts_if_play"]
        return out


def fit(train: pd.DataFrame) -> PointsModel:
    """Fit on training-table rows whose targets are known."""
    train = train.dropna(subset=["minutes", "target_points"])
    played = train[train["minutes"] > 0]
    X, y, w = inputs.fit_matrix(played), played["target_points"].astype(float), inputs.weights(played)
    if len(played) < minutes.MIN_ROWS:      # too little to learn from: the plain average
        regressor = DummyRegressor(strategy="mean").fit(X, y, sample_weight=w)
    else:
        regressor = HistGradientBoostingRegressor(**PARAMS).fit(X, y, sample_weight=w)
    return PointsModel(minutes=minutes.fit(train), if_played=regressor)
