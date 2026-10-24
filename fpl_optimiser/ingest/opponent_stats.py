"""Source-agnostic opponent (team) defensive stats.

Everything downstream (Section 2 onward) calls ONLY `opponent_stats(season)` and
gets a canonical per-team table. The data source is a swappable backend, so adding
or changing a source never touches the feature/model code again.

Canonical columns:
    team                      team name (as the source spells it; reconcile via team_map)
    matches                   matches counted
    goals_conceded_per_game   goals conceded per match — reliable from FBref's schedule
    xga_per_game              expected goals conceded per match — richer, but NaN when the
                              source omits xG (e.g. this soccerdata schedule); populated by
                              Understat or by FBref versions whose schedule includes xG
    (plus any source extras, e.g. ppda_mean from Understat)

Backends:
    "fpl"        goals conceded from the season's own filed FPL results
                 (season_store). No browser, always the right season, and the same
                 signal FBref's schedule gives today. The default.
    "fbref"      via soccerdata.read_schedule() — needs Google Chrome (soccerdata
                 reads FBref through a headless browser).
    "understat"  via understat.py (plain HTTP) — lighter and adds PPDA, but Understat
                 now sits behind Cloudflare and may block plain requests.

`opponent_stats()` tries the configured source first and automatically falls back
to the others if it errors, so a Cloudflare block on one source is not fatal.
"""
from __future__ import annotations

import pandas as pd

import config

_DASHES = ["\u2013", "\u2014", "-"]  # en dash, em dash, hyphen


def _parse_score(score) -> tuple[int | None, int | None]:
    """'2-1' (or '2–1') -> (2, 1); anything unparseable -> (None, None)."""
    if not isinstance(score, str):
        return (None, None)
    for dash in _DASHES:
        if dash in score:
            left, _, right = score.partition(dash)
            try:
                return int(left.strip()), int(right.strip())
            except ValueError:
                return (None, None)
    return (None, None)


def _find_col(cols_lower: dict[str, str], *needles: str) -> str | None:
    """Find the original column whose lowercased name contains all needles."""
    for lower, original in cols_lower.items():
        if all(n in lower for n in needles):
            return original
    return None


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _from_fbref(season: str) -> pd.DataFrame:
    from fpl_optimiser.ingest import fbref

    sched = fbref.schedule(season).reset_index()
    cols = {c.lower(): c for c in sched.columns}

    home_t = _find_col(cols, "home", "team") or cols.get("home")
    away_t = _find_col(cols, "away", "team") or cols.get("away")
    score_c = cols.get("score") or _find_col(cols, "score")
    home_x = _find_col(cols, "home", "xg")
    away_x = _find_col(cols, "away", "xg")
    if not all([home_t, away_t, score_c]):
        raise KeyError(f"FBref schedule columns not as expected: {list(sched.columns)}")

    have_xg = bool(home_x and away_x)  # some soccerdata versions omit xG from the schedule
    df = sched.copy()
    if have_xg:
        df[home_x] = pd.to_numeric(df[home_x], errors="coerce")
        df[away_x] = pd.to_numeric(df[away_x], errors="coerce")

    rows = []
    for _, r in df.iterrows():
        home_goals, away_goals = _parse_score(r.get(score_c))
        if home_goals is None:                  # unplayed / unparseable -> skip
            continue
        # a team concedes the OTHER side's goals (and xG, when available)
        rows.append({"team": r[home_t], "gc": away_goals,
                     "xga": (r[away_x] if have_xg else float("nan"))})
        rows.append({"team": r[away_t], "gc": home_goals,
                     "xga": (r[home_x] if have_xg else float("nan"))})

    long = pd.DataFrame(rows)
    long["xga"] = pd.to_numeric(long["xga"], errors="coerce")
    long["gc"] = pd.to_numeric(long["gc"], errors="coerce")
    return (long.groupby("team")
                .agg(matches=("gc", "size"),
                     goals_conceded_per_game=("gc", "mean"),
                     xga_per_game=("xga", "mean"))
                .reset_index())


def _from_understat(season: str) -> pd.DataFrame:
    from fpl_optimiser.ingest import understat

    ts = understat.team_season(season)
    out = pd.DataFrame({
        "team": ts["team"],
        "matches": ts["matches"],
        "xga_per_game": ts["xga_mean"],
        "goals_conceded_per_game": ts["goals_conceded"] / ts["matches"],
    })
    for extra in ["ppda_mean", "ppda_allowed_mean", "deep_allowed_mean", "npxga_mean"]:
        if extra in ts.columns:
            out[extra] = ts[extra]
    return out


def goals_conceded_from_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Pure: parsed FPL fixtures -> per-team matches and goals conceded per game."""
    fin = fixtures.dropna(subset=["home_score", "away_score"])
    if "finished" in fin.columns:
        fin = fin[fin["finished"] == True]  # noqa: E712
    long = pd.concat([pd.DataFrame({"team": fin["home"], "gc": fin["away_score"]}),
                      pd.DataFrame({"team": fin["away"], "gc": fin["home_score"]})])
    long["gc"] = pd.to_numeric(long["gc"], errors="coerce")
    out = (long.groupby("team")["gc"].agg(matches="size", goals_conceded_per_game="mean")
               .reset_index())
    out["xga_per_game"] = float("nan")       # FPL results carry no team xG
    return out


def _from_fpl(season: str) -> pd.DataFrame:
    from fpl_optimiser.ingest import season_store

    return goals_conceded_from_fixtures(season_store.load_fixtures(season))


_BACKENDS = {"fpl": _from_fpl, "fbref": _from_fbref, "understat": _from_understat}


def opponent_stats(season: str = config.SEASON_TRAIN,
                   source: str | None = None) -> pd.DataFrame:
    """Canonical per-team defensive table from the configured source, with fallback."""
    primary = source or config.ADVANCED_STATS_SOURCE
    order = [primary] + [s for s in _BACKENDS if s != primary]
    errors: dict[str, str] = {}
    for src in order:
        try:
            df = _BACKENDS[src](season)
            df.attrs["source"] = src
            return df
        except Exception as exc:  # noqa: BLE001 — we genuinely want to try the next source
            first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            errors[src] = f"{type(exc).__name__}: {first}"
    raise RuntimeError(f"all opponent-stats sources failed: {errors}")