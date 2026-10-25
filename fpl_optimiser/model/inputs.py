"""The model's inputs: which training-table columns it sees, and in what form.

One list, used by both the minutes model and the points model, so they can never
drift apart. Identity columns (id, names, round) and the targets are never inputs.

Every input must be knowable at the deadline of the match it describes. Two
columns the table still carries for the report are deliberately NOT inputs:
  opp_archetype  the opponent's defensive "style", clustered on its goals conceded
                 over the season so far, which for a past match includes that match
                 (future information: it inflated the backtest's top-11 by ~14 points)
  opp_str_def    FPL's defence rating, a current snapshot (0 for every club in
                 2026/27); its only signal was which season a row came from, now the
                 explicit `earlier_season` flag
"""
from __future__ import annotations

import pandas as pd

NUMERIC = [
    "price",
    "form_points_pg", "form_minutes_pg", "form_xg_pg", "form_xa_pg", "form_xgi_pg",
    "form_defcon_pg", "form_bonus_pg", "form_bps_pg", "form_starts_rate", "games_played",
    "opp_gc_pg_asof", "x_xgi_oppgc",
    "prev_points_per_gw", "prev_minutes_per_gw", "prev_points_per_90",
    "prev_xgi_per_90", "prev_defcon_per_90",
]
FLAGS = ["was_home", "is_cold", "has_prev_season", "prev_changed_team",
         "earlier_season"]   # a row from an earlier season (multi_season); always False when predicting
POSITIONS = ["GK", "DEF", "MID", "FWD"]
FEATURES = NUMERIC + FLAGS + ["position"]
ROUND_TO = 6          # decimal places kept in every numeric input


def design_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Training-table (or prediction-table) rows -> the model's X.

    Numbers stay as floats with NaN meaning "unknown" (the trees handle it), flags
    become 0/1, and position becomes a categorical the trees can split on directly.
    """
    # rounded so the same feature computed by two code paths (the training table's
    # matrix products, the forecast's probe rows) lands in the same tree bin: they
    # differ only by floating-point rounding (~1e-16), which can tip a value sitting
    # exactly on a split point
    X = rows[NUMERIC].astype(float).round(ROUND_TO).copy()
    for col in FLAGS:
        X[col] = rows[col].astype(float) if col in rows else 0.0
    X["position"] = pd.Categorical(rows["position"], categories=POSITIONS)
    return X[FEATURES]


def fit_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """design_matrix for FITTING: a column with no values at all (e.g. every prev_*
    column when no previous season is filed) becomes a constant. The trees then
    simply never split on it (scikit-learn's trees refuse an all-missing column)."""
    X = design_matrix(rows)
    empty = [c for c in NUMERIC + FLAGS if X[c].isna().all()]
    X[empty] = 0.0
    return X


def weights(rows: pd.DataFrame) -> pd.Series:
    """Each row's fitting weight: its season's sample_weight (1 if absent)."""
    if "sample_weight" in rows:
        return rows["sample_weight"].fillna(1.0).astype(float)
    return pd.Series(1.0, index=rows.index)
