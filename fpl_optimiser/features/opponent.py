"""Opponent (team) characteristics and defensive archetypes.

v1 per-team profile:
  gc_pg            goals conceded per game (season; from the opponent_stats adapter)
  str_def_home     FPL's defensive-strength rating at home
  str_def_away     FPL's defensive-strength rating away
  str_overall      FPL overall strength

We standardise these and cluster the 20 teams into `config.ARCHETYPE_K` defensive
archetypes with k-means, giving each team a stable, human-readable 'type' label
(archetype 0 = meanest defence, by mean goals conceded).

Leakage note: these values are season-level / current. For the strict 2025/26
backtest they are mildly optimistic (they peek at full-season quality). The
leakage-free *quantitative* opponent signal — as-of rolling goals conceded — is
built in features.build. Archetypes are coarse, season-stable 'style/quality'
labels, used as a descriptor rather than a precise per-gameweek estimate.

When the opponent source later provides xGA / PPDA (e.g. Understat reopens), add
those columns to PROFILE_FEATURES and re-cluster — nothing else changes.
"""
from __future__ import annotations

import pandas as pd

import config

PROFILE_FEATURES = ["gc_pg", "str_def_home", "str_def_away", "str_overall"]


def build_opponent_profiles(opp_stats: pd.DataFrame, teams: pd.DataFrame,
                            team_name_map: dict | None = None) -> pd.DataFrame:
    """Join adapter goals-conceded onto FPL team strengths → per-team profile.

    opp_stats : opponent_stats.opponent_stats()  (team, goals_conceded_per_game, ...)
    teams     : fpl_api.teams()  (id, name, short_name, strength, strength_defence_*)
    team_name_map : optional {fpl_name: source_name}; if None, matched on normalised name.
    """
    from fpl_optimiser.ingest import team_map

    t = teams.copy()
    if team_name_map is None:
        team_name_map = team_map.reconcile(t["name"], opp_stats["team"])["matched"]
    t["source_team"] = t["name"].map(team_name_map)

    o = opp_stats.rename(columns={"team": "source_team"})
    merged = t.merge(o[["source_team", "goals_conceded_per_game"]],
                     on="source_team", how="left")

    return pd.DataFrame({
        "team_id": merged["id"],
        "team": merged["name"],
        "gc_pg": merged["goals_conceded_per_game"],
        "str_def_home": merged.get("strength_defence_home"),
        "str_def_away": merged.get("strength_defence_away"),
        "str_overall": merged.get("strength"),
    })


def assign_archetypes(profiles: pd.DataFrame, k: int = config.ARCHETYPE_K,
                      features: list[str] = PROFILE_FEATURES,
                      random_state: int = 0) -> pd.DataFrame:
    """Add an `archetype` label (0 = meanest defence) via standardised k-means."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    df = profiles.copy()
    X = df[features].astype(float)
    X = X.loc[:, X.notna().any()]   # skip a feature the source supplied for no team at all
    X = X.fillna(X.mean())
    Xs = StandardScaler().fit_transform(X)

    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(Xs)
    df["archetype"] = labels

    # relabel so archetype index is ordered by mean goals conceded (interpretable)
    order = df.groupby("archetype")["gc_pg"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    df["archetype"] = df["archetype"].map(remap)
    return df


def archetype_summary(profiles_with_arch: pd.DataFrame) -> pd.DataFrame:
    """Human-readable description of each archetype (for docs and Claude's narrative)."""
    return (profiles_with_arch.groupby("archetype")
            .agg(n_teams=("team", "size"),
                 gc_pg=("gc_pg", "mean"),
                 str_def_home=("str_def_home", "mean"),
                 str_def_away=("str_def_away", "mean"),
                 teams=("team", lambda s: ", ".join(sorted(s))))
            .reset_index())