"""Minutes model — will he play, and will he play 60+ minutes?

Minutes are the single biggest driver of FPL points: no minutes, no points, and 60+
minutes unlocks the extra appearance point and clean sheets. Two classifiers on the
same inputs as the points model (model/inputs.py):

  p_play  P(minutes > 0)
  p_60    P(minutes >= 60)   (never above p_play)

Gradient-boosted trees (scikit-learn's HistGradientBoosting: the LightGBM algorithm,
with native handling of missing values), fitted with each row's sample_weight so an
older season counts less.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from fpl_optimiser.model import inputs

# Deliberately small trees and strong regularisation: a few thousand rows. Chosen
# by walk-forward backtest over 2026/27 GW2-8 (docs/03): against 15-leaf trees
# this setting cut RMSE 2.232 -> 2.189 and lifted the top-11 from 47.3 -> 49.6.
PARAMS = dict(learning_rate=0.05, max_iter=150, max_leaf_nodes=8, min_samples_leaf=80,
              l2_regularization=2.0, early_stopping=False, random_state=0,
              categorical_features="from_dtype")

# Fewer rows than this and there is nothing to learn yet (e.g. GW1 of a first
# season): fall back to plain averages instead of fitting trees on noise.
MIN_ROWS = 2 * PARAMS["min_samples_leaf"]


@dataclass
class MinutesModel:
    play: HistGradientBoostingClassifier | DummyClassifier     # P(minutes > 0)
    sixty: HistGradientBoostingClassifier | DummyClassifier    # P(minutes >= 60)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = inputs.design_matrix(rows)
        p_play = _positive_proba(self.play, X)
        p_60 = np.minimum(_positive_proba(self.sixty, X), p_play)
        return pd.DataFrame({"p_play": p_play, "p_60": p_60}, index=rows.index)


def _positive_proba(clf, X: pd.DataFrame) -> np.ndarray:
    """P(class 1); a model that only ever saw one class answers with that class."""
    if len(clf.classes_) == 1:
        return np.full(len(X), float(clf.classes_[0]))
    return clf.predict_proba(X)[:, list(clf.classes_).index(1)]


def _classifier(X: pd.DataFrame, y: pd.Series, w: pd.Series):
    """Trees, or the plain (weighted) rate if `y` has one class or too few rows."""
    if y.nunique() < 2 or len(y) < MIN_ROWS:
        return DummyClassifier(strategy="prior").fit(X, y, sample_weight=w)
    return HistGradientBoostingClassifier(**PARAMS).fit(X, y, sample_weight=w)


def fit(train: pd.DataFrame) -> MinutesModel:
    """Fit both classifiers on training-table rows (their `minutes` must be known)."""
    X, w = inputs.fit_matrix(train), inputs.weights(train)
    minutes = train["minutes"].astype(float)
    return MinutesModel(play=_classifier(X, (minutes > 0).astype(int), w),
                        sixty=_classifier(X, (minutes >= 60).astype(int), w))
