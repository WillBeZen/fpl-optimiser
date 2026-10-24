"""The model's inputs: which training-table columns it sees, and in what form.

One list, used by both the minutes model and the points model, so they can never
drift apart. Identity columns (id, names, round) and the targets are never inputs.
"""
from __future__ import annotations

import pandas as pd

NUMERIC = [
    "price",
    "form_points_pg", "form_minutes_pg", "form_xg_pg", "form_xa_pg", "form_xgi_pg",
    "form_defcon_pg", "form_bonus_pg", "form_bps_pg", "form_starts_rate", "games_played",
    "opp_gc_pg_asof", "opp_archetype", "opp_str_def", "x_xgi_oppgc",
    "prev_points_per_gw", "prev_minutes_per_gw", "prev_points_per_90",
    "prev_xgi_per_90", "prev_defcon_per_90",
]
FLAGS = ["was_home", "is_cold", "has_prev_season", "prev_changed_team"]
POSITIONS = ["GK", "DEF", "MID", "FWD"]
FEATURES = NUMERIC + FLAGS + ["position"]


def design_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Training-table (or prediction-table) rows -> the model's X.

    Numbers stay as floats with NaN meaning "unknown" (the trees handle it), flags
    become 0/1, and position becomes a categorical the trees can split on directly.
    """
    X = rows[NUMERIC].astype(float).copy()
    for col in FLAGS:
        X[col] = rows[col].astype(float)
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
