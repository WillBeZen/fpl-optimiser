"""Player profiles and archetypes: "what kind of player is this?", for visual analysis.

The opponent-archetype idea applied to players. Each player is described by per-90
rates (what he does when he plays) plus how much he plays, then each position is
clustered into a few types. The HTML report draws these; they are not (yet) a
model input.

Profiles blend two seasons: this season's totals plus last season's (linked by the
permanent player code) at config.PREV_SEASON_WEIGHT per minute. So a profile early
in the season isn't 300 minutes of noise, and a new arrival still gets a profile
from this season alone once he has played enough.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# season-total columns (fpl_api.parse_players) -> per-90 profile features
PER90 = {
    "pts_90": "total_points",
    "xg_90": "expected_goals",
    "xa_90": "expected_assists",
    "xgi_90": "expected_goal_involvements",
    "defcon_90": "defensive_contribution",
    "saves_90": "saves",
    "xgc_90": "expected_goals_conceded",
    "bonus_90": "bonus",
}

# What separates players WITHIN a position (outcomes like points are described, not clustered on).
ARCHETYPE_FEATURES = {
    "GK": ["saves_90", "xgc_90", "bonus_90", "minutes_share"],
    "DEF": ["defcon_90", "xgi_90", "xgc_90", "bonus_90", "minutes_share"],
    "MID": ["xg_90", "xa_90", "defcon_90", "bonus_90", "minutes_share"],
    "FWD": ["xg_90", "xa_90", "bonus_90", "minutes_share"],
}

LABELS = {
    "pts_90": "Points/90", "xg_90": "xG/90", "xa_90": "xA/90", "xgi_90": "xGI/90",
    "defcon_90": "Def. contributions/90", "saves_90": "Saves/90",
    "xgc_90": "xG conceded/90", "bonus_90": "Bonus/90", "minutes_share": "Minutes share",
}

# Plain words for naming types ("High bonus · High xA"): rates are per 90 throughout.
_WORDS = {
    "pts_90": "points", "xg_90": "xG", "xa_90": "xA", "xgi_90": "xGI",
    "defcon_90": "def. contributions", "saves_90": "saves", "xgc_90": "xG conceded",
    "bonus_90": "bonus", "minutes_share": "minutes",
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def build_profiles(curr: pd.DataFrame, curr_gameweeks: int,
                   prev: pd.DataFrame | None = None, links: pd.DataFrame | None = None,
                   prev_gameweeks: int = config.GAMEWEEKS_IN_SEASON,
                   prev_weight: float = config.PREV_SEASON_WEIGHT) -> pd.DataFrame:
    """One row per current player: per-90 rates and minutes share, blending in last
    season's totals for linked players.

    curr / prev : fpl_api.parse_players tables (season totals) for this / last season
    links       : player_link.link_players(prev, curr).links
    """
    p = curr[["id", "code", "web_name", "team_name", "position", "price"]].copy()
    p["points"] = _num(curr, "total_points").to_numpy()
    p["minutes"] = _num(curr, "minutes").to_numpy()
    totals = {col: _num(curr, col).to_numpy() for col in PER90.values()}
    minutes = p["minutes"].to_numpy(dtype=float)
    available = np.full(len(p), curr_gameweeks * 90.0)       # minutes he could have played

    p["prev_minutes"] = 0.0
    if prev is not None and links is not None and len(links):
        last = prev.set_index("id").reindex(
            p["id"].map(links.set_index("id_curr")["id_prev"]))  # NaN row = not linked
        linked = last.index.notna()
        w = np.where(linked, prev_weight, 0.0)
        for col in PER90.values():
            totals[col] = totals[col] + w * _num(last, col).to_numpy()
        prev_min = _num(last, "minutes").to_numpy()
        minutes = minutes + w * prev_min
        available = available + w * prev_gameweeks * 90.0
        p["prev_minutes"] = np.where(linked, prev_min, 0.0)

    p["has_prev_season"] = p["prev_minutes"] > 0
    p["profile_minutes"] = minutes                            # blended evidence behind the rates
    with np.errstate(invalid="ignore", divide="ignore"):
        for feat, col in PER90.items():
            p[feat] = np.where(minutes > 0, totals[col] / minutes * 90, np.nan)
        p["minutes_share"] = np.clip(minutes / available, 0, 1)
    p["points_per_m"] = p["points"] / p["price"]
    return p


def _name(z: pd.Series) -> str:
    """Name a cluster by what stands out in its centre (z-scores vs its position)."""
    high = [f"High {_WORDS[f]}" for f, v in z.sort_values(ascending=False).items() if v >= 0.5]
    low = [f"Low {_WORDS[f]}" for f, v in z.sort_values().items() if v <= -0.5]
    return " · ".join((high + low)[:2]) or "All-round"


def assign_archetypes(profiles: pd.DataFrame, k: dict[str, int] = config.PLAYER_ARCHETYPE_K,
                      min_minutes: float = config.PROFILE_MIN_MINUTES,
                      random_state: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster each position's profiled players into k types.

    Returns (players, types):
      players  profiled players (profile_minutes >= min_minutes) with `archetype`
               (0 = the type that scores most points per 90) and `archetype_name`
      types    one row per (position, archetype): name, size, mean points/90 and
               the centre's z-score on every feature
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    players, types = [], []
    for pos, feats in ARCHETYPE_FEATURES.items():
        grp = profiles[(profiles["position"] == pos)
                       & (profiles["profile_minutes"] >= min_minutes)].copy()
        n_types = min(k[pos], len(grp))
        if n_types < 2:
            continue
        X = grp[feats].astype(float)
        X = X.fillna(X.mean())
        Z = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=n_types, n_init=10, random_state=random_state).fit_predict(Z)

        # order types by points per 90 so archetype 0 is the most productive
        order = grp.groupby(labels)["pts_90"].mean().sort_values(ascending=False).index
        grp["archetype"] = pd.Series(labels, index=grp.index).map(
            {old: new for new, old in enumerate(order)})
        centres = pd.DataFrame(Z, index=grp.index, columns=feats).groupby(grp["archetype"]).mean()
        names = {a: _name(z) for a, z in centres.iterrows()}
        grp["archetype_name"] = grp["archetype"].map(names)
        players.append(grp)

        summary = grp.groupby("archetype").agg(n_players=("id", "size"), pts_90=("pts_90", "mean"))
        types.append(centres.add_prefix("z_").join(summary)
                     .assign(position=pos, archetype_name=lambda d: d.index.map(names))
                     .reset_index())

    if not players:     # e.g. GW1 of a season with no previous one: nobody has the minutes yet
        return (profiles.iloc[0:0].assign(archetype=pd.Series(dtype=int), archetype_name=""),
                pd.DataFrame(columns=["archetype", "n_players", "pts_90", "position", "archetype_name"]))
    return pd.concat(players, ignore_index=True), pd.concat(types, ignore_index=True)
