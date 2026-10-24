"""Event model — what each player should do in each fixture, in football terms.

The points model (points.py) predicts one number per fixture, and that number is
what the optimiser uses. A scout also wants to know *why*: how many goals is his
team expected to score, will they keep a clean sheet, how likely is he to score,
assist or hit his defensive-contribution threshold? This module answers that with
a transparent, classic football model, built in three steps:

1. Team ratings. Each club's attack = recency-weighted xG it creates per game, its
   defence = xG it concedes per game, with last season's goals for/against blended
   in as a fading prior (the same half-life and summer gap as player form;
   promoted clubs start where last season's relegated clubs finished).
2. Fixture expected goals. home goals = league average x home advantage x home
   attack index x away defence index (and the mirror for the away side). A clean
   sheet is P(the opponent scores 0) = exp(-their expected goals).
3. Player events. His recency-weighted per-90 rates (xG, xA, saves, bonus, cards,
   how often he hits the defensive-contribution threshold), shrunk towards his
   position's average when he has few minutes, are scaled by this fixture's
   difficulty and by his expected minutes. Minutes come from the minutes model
   (P(plays), P(60+), after FPL's injury flags), so both models agree on who plays.

Points by source then follow FPL's scoring table in config.SCORING. The total
(`xp_events`) is shown beside the points model's `xp` and backtested against it
(model/backtest.py): it explains, while the points model decides.

Everything is computed "as of" a round from data strictly before it, so the same
code serves the live forecast and an honest backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from fpl_optimiser.features import recency

SOURCES = ["Appearance", "Goals", "Assists", "Clean sheet", "Conceded", "Saves",
           "Def. contributions", "Bonus", "Cards"]

SHRINK_MINUTES = 270        # minutes of "position average" mixed into every per-90 rate
SHRINK_MATCHES = 3          # matches of position average mixed into per-match rates
DEFAULT_MINS_60 = 87.0      # minutes in a 60+ appearance, when he has none yet
DEFAULT_MINS_SUB = 25.0     # minutes in a shorter appearance, when he has none yet
DEFAULT_GOALS = 1.35        # goals per team per game, if there's no data at all
FULL_TEAM_MINUTES = 900     # a team's xG in a match counts only if its players' records
                            # cover at least this many minutes (else goals are used):
                            # a season filed for its top players only can't sum to team xG


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def prior_weight(as_of_round, half_life: float = config.RECENCY_HALF_LIFE_GW,
                 gap_gw: float = config.PREV_SEASON_GAP_GW,
                 n_gw: int = config.GAMEWEEKS_IN_SEASON):
    """Total recency weight of last season (n_gw matches ending gap_gw gameweeks before
    GW1), seen from `as_of_round`: what it counts for, in "this week's matches"."""
    r = np.asarray(as_of_round, dtype=float)
    return recency.decay_weight(r + gap_gw, half_life) * recency.decay_weight(np.arange(n_gw), half_life).sum()


# --------------------------------------------------------------------------- #
# 1. Team ratings
# --------------------------------------------------------------------------- #
@dataclass
class Ratings:
    teams: pd.DataFrame      # team_id, name, short_name, attack, defence (per game),
                             # att_index, def_index (1 = league average), matches, prior_share
    mean_goals: float        # goals per team per game, league average
    home: float              # home side's goals relative to the average (e.g. 1.12)
    away: float
    as_of_round: int


def team_matches(fixtures: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Finished fixtures as one row per team: round, team_id, opponent, home,
    goals for/against and (from players' match records) xG for/against."""
    fx = fixtures[(fixtures["finished"] == True) & fixtures["home_score"].notna()]  # noqa: E712
    rows = pd.concat([
        pd.DataFrame({"fixture_id": fx["fixture_id"], "round": fx["gameweek"], "team_id": fx["home_id"],
                      "opponent": fx["away_id"], "home": True, "gf": fx["home_score"], "ga": fx["away_score"]}),
        pd.DataFrame({"fixture_id": fx["fixture_id"], "round": fx["gameweek"], "team_id": fx["away_id"],
                      "opponent": fx["home_id"], "home": False, "gf": fx["away_score"], "ga": fx["home_score"]}),
    ], ignore_index=True)
    rows["xgf"] = np.nan
    if history is not None and len(history) and "fixture" in history:
        h = history[history["fixture"].isin(fx["fixture_id"])]
        side = fx.set_index("fixture_id")
        team = np.where(h["was_home"].astype(bool), h["fixture"].map(side["home_id"]),
                        h["fixture"].map(side["away_id"]))
        per_team = (pd.DataFrame({"fixture_id": h["fixture"].to_numpy(), "team_id": team,
                                  "xg": _num(h, "expected_goals").to_numpy(),
                                  "mins": _num(h, "minutes").fillna(0).to_numpy()})
                    .groupby(["fixture_id", "team_id"]).agg(xg=("xg", lambda s: s.sum(min_count=1)),
                                                            mins=("mins", "sum")))
        xg = per_team["xg"].where(per_team["mins"] >= FULL_TEAM_MINUTES)
        rows["xgf"] = pd.MultiIndex.from_frame(rows[["fixture_id", "team_id"]]).map(xg.to_dict().get)
        rows["xgf"] = pd.to_numeric(rows["xgf"], errors="coerce")
    xga = rows.set_index(["fixture_id", "team_id"])["xgf"]
    rows["xga"] = [xga.get((f, o), np.nan) for f, o in zip(rows["fixture_id"], rows["opponent"])]
    for col in ("round", "team_id", "opponent"):
        rows[col] = rows[col].astype(int)
    return rows


def _last_season(prev_fixtures: pd.DataFrame | None, prev_teams: pd.DataFrame | None,
                 teams: pd.DataFrame) -> pd.DataFrame:
    """Per current team_id: last season's goals for/against per game (promoted clubs:
    the average of the clubs that went down). Empty if last season isn't filed."""
    if prev_fixtures is None or prev_teams is None or prev_fixtures.empty:
        return pd.DataFrame(columns=["gf", "ga"])
    m = team_matches(prev_fixtures)
    per_game = m.groupby("team_id")[["gf", "ga"]].mean()
    by_code = per_game.rename(index=prev_teams.set_index("id")["code"])
    relegated = by_code[~by_code.index.isin(teams["code"])]
    fallback = relegated.mean() if len(relegated) else by_code.mean()
    out = teams.set_index("id")["code"].map(lambda c: c in by_code.index)
    return pd.DataFrame({
        "gf": [by_code.at[c, "gf"] if ok else fallback["gf"] for c, ok in zip(teams["code"], out)],
        "ga": [by_code.at[c, "ga"] if ok else fallback["ga"] for c, ok in zip(teams["code"], out)],
        "promoted": [not ok for ok in out]}, index=teams["id"].to_numpy())


def team_ratings(fixtures: pd.DataFrame, teams: pd.DataFrame, as_of_round: int,
                 history: pd.DataFrame | None = None, prev_fixtures: pd.DataFrame | None = None,
                 prev_teams: pd.DataFrame | None = None,
                 half_life: float = config.RECENCY_HALF_LIFE_GW,
                 gap_gw: float = config.PREV_SEASON_GAP_GW) -> Ratings:
    """Attack and defence per team, from matches strictly before `as_of_round`.

    This season counts xG (more stable than goals; goals where xG is missing),
    each match weighted by recency; last season's goals per game count as a block
    of matches before GW1 with prior_weight(as_of_round)."""
    m = team_matches(fixtures, history)
    m = m[m["round"] < as_of_round]
    m = m.assign(w=recency.decay_weight(as_of_round - m["round"].to_numpy(dtype=float), half_life),
                 f=m["xgf"].fillna(m["gf"]), a=m["xga"].fillna(m["ga"]))
    last = _last_season(prev_fixtures, prev_teams, teams)
    wp = float(prior_weight(as_of_round, half_life, gap_gw)) if len(last) else 0.0

    rows = []
    for _, t in teams.iterrows():
        mine = m[m["team_id"] == t["id"]]
        w = mine["w"].sum()
        pf = last["gf"].get(t["id"], np.nan) if len(last) else np.nan
        pa = last["ga"].get(t["id"], np.nan) if len(last) else np.nan
        wpt = wp if not np.isnan(pf) else 0.0
        den = w + wpt
        rows.append({"team_id": int(t["id"]), "name": t["name"], "short_name": t["short_name"],
                     "attack": ((mine["w"] * mine["f"]).sum() + wpt * np.nan_to_num(pf)) / den if den else np.nan,
                     "defence": ((mine["w"] * mine["a"]).sum() + wpt * np.nan_to_num(pa)) / den if den else np.nan,
                     "matches": len(mine), "prior_share": wpt / den if den else 0.0,
                     "promoted": bool(last["promoted"].get(t["id"], False)) if len(last) else False})
    tbl = pd.DataFrame(rows)
    mean = float(np.nanmean(pd.concat([tbl["attack"], tbl["defence"]]))) if tbl["attack"].notna().any() else DEFAULT_GOALS
    tbl["attack"] = tbl["attack"].fillna(mean)
    tbl["defence"] = tbl["defence"].fillna(mean)
    tbl["att_index"] = tbl["attack"] / mean
    tbl["def_index"] = tbl["defence"] / mean

    # home advantage from every finished match on file (this season before the
    # round, plus last season): home goals vs away goals
    games = [m[m["home"]]]
    if prev_fixtures is not None and len(prev_fixtures):
        pm = team_matches(prev_fixtures)
        games.append(pm[pm["home"]])
    g = pd.concat(games)
    if len(g) and g["gf"].sum() + g["ga"].sum() > 0:
        home = 2 * g["gf"].sum() / (g["gf"].sum() + g["ga"].sum())
    else:
        home = 1.0
    return Ratings(teams=tbl, mean_goals=mean, home=home, away=2 - home, as_of_round=as_of_round)


def fixture_goals(r: Ratings, team: pd.Series, opponent: pd.Series, home: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Expected goals for and against each (team, opponent, home?) fixture."""
    t = r.teams.set_index("team_id")
    venue_for = np.where(home, r.home, r.away)
    venue_against = np.where(home, r.away, r.home)
    xg_for = r.mean_goals * venue_for * team.map(t["att_index"]).to_numpy() * opponent.map(t["def_index"]).to_numpy()
    xg_against = r.mean_goals * venue_against * opponent.map(t["att_index"]).to_numpy() * team.map(t["def_index"]).to_numpy()
    return xg_for, xg_against


def fixture_outlook(r: Ratings, fixtures: pd.DataFrame, gameweeks: list[int]) -> pd.DataFrame:
    """Every upcoming fixture from both sides: expected goals for/against and the
    clean-sheet chance (P(the opponent scores 0))."""
    fx = fixtures[fixtures["gameweek"].isin(gameweeks)]
    sides = pd.concat([
        pd.DataFrame({"gw": fx["gameweek"], "fixture_id": fx["fixture_id"], "team_id": fx["home_id"],
                      "opponent": fx["away_id"], "home": True}),
        pd.DataFrame({"gw": fx["gameweek"], "fixture_id": fx["fixture_id"], "team_id": fx["away_id"],
                      "opponent": fx["home_id"], "home": False}),
    ], ignore_index=True)
    for col in ("gw", "team_id", "opponent"):
        sides[col] = sides[col].astype(int)
    sides["xg_for"], sides["xg_against"] = fixture_goals(r, sides["team_id"], sides["opponent"], sides["home"])
    sides["p_cs"] = np.exp(-sides["xg_against"])
    short = r.teams.set_index("team_id")["short_name"]
    sides["team"] = sides["team_id"].map(short)
    sides["vs"] = sides["opponent"].map(short) + np.where(sides["home"], " (H)", " (A)")
    return sides.sort_values(["team", "gw", "fixture_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Player rates
# --------------------------------------------------------------------------- #
def _defcon_threshold(position: pd.Series) -> pd.Series:
    return position.map(config.SCORING["defensive_contribution_threshold"]).fillna(np.inf)


def _poisson_at_least(mean: np.ndarray, k: np.ndarray) -> np.ndarray:
    """P(X >= k) for X ~ Poisson(mean), elementwise (k a whole number, inf -> 0)."""
    mean = np.asarray(mean, dtype=float)
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(mean)
    ok = np.isfinite(k) & np.isfinite(mean)
    for i in np.flatnonzero(ok):
        j = np.arange(int(k[i]))
        logp = -mean[i] + j * np.log(max(mean[i], 1e-12)) - np.cumsum(np.log(np.maximum(j, 1)))
        out[i] = max(0.0, 1.0 - np.exp(logp).sum())
    return out


def expected_floor(mean: np.ndarray, per: int, kmax: int = 30) -> np.ndarray:
    """E[floor(X / per)] for X ~ Poisson(mean): e.g. per=2 for goals conceded,
    per=3 for saves."""
    mean = np.nan_to_num(np.asarray(mean, dtype=float))
    k = np.arange(kmax + 1)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(k[1:]))])
    logp = -mean[:, None] + k[None, :] * np.log(np.maximum(mean[:, None], 1e-12)) - logfact[None, :]
    return (np.exp(logp) * (k // per)[None, :]).sum(axis=1)


def player_rates(history: pd.DataFrame, players: pd.DataFrame, as_of_round: int,
                 prior: pd.DataFrame | None = None,
                 half_life: float = config.RECENCY_HALF_LIFE_GW,
                 gap_gw: float = config.PREV_SEASON_GAP_GW) -> pd.DataFrame:
    """Per player (index = id): recency-weighted rates from matches strictly before
    `as_of_round`, with last season (`prior`, prior_season.prior_table) blended in
    where it exists, and every rate shrunk towards the position average.

    xg90, xa90, saves90   per 90 minutes
    defcon_hit            P(reaching the defensive-contribution threshold | 60+ minutes)
    bonus_pa, yellow_pa   per appearance
    mins_60, mins_sub     typical minutes in a 60+ appearance / a shorter one"""
    h = history[history["round"] < as_of_round].copy()
    pos = players.set_index("id")["position"]
    h["position"] = h["id"].map(pos)
    h["w"] = recency.decay_weight(as_of_round - h["round"].to_numpy(dtype=float), half_life)
    mins = _num(h, "minutes").fillna(0.0)
    h["mins"], h["played"], h["full"] = mins, (mins > 0).astype(float), (mins >= 60).astype(float)
    for col in ("expected_goals", "expected_assists", "saves", "bonus", "yellow_cards", "defensive_contribution"):
        h[col] = _num(h, col).fillna(0.0)
    h["hit"] = (h["defensive_contribution"] >= _defcon_threshold(h["position"])).astype(float) * h["full"]
    h["mins_60"] = h["mins"] * h["full"]
    h["sub"] = ((mins > 0) & (mins < 60)).astype(float)
    h["mins_sub"] = h["mins"] * h["sub"]

    cols = ["mins", "played", "full", "expected_goals", "expected_assists", "saves", "bonus",
            "yellow_cards", "hit", "mins_60", "sub", "mins_sub"]
    sums = (h[cols].mul(h["w"], axis=0).assign(id=h["id"]).groupby("id").sum()
            .reindex(players["id"]).fillna(0.0))
    sums.index.name = "id"

    # last season as a block of matches before GW1 (per-gameweek averages x its weight)
    wp = float(prior_weight(as_of_round, half_life, gap_gw))
    pr = (prior.set_index("id") if prior is not None and len(prior) else pd.DataFrame()).reindex(sums.index)
    def pcol(c):
        return pd.to_numeric(pr[c], errors="coerce").fillna(0.0) if c in pr else pd.Series(0.0, index=sums.index)
    p_mins, p_full = pcol("pg_minutes"), pcol("pg_played_60")
    mins_all = sums["mins"] + wp * p_mins
    xg = sums["expected_goals"] + wp * pcol("pg_expected_goals")
    xa = sums["expected_assists"] + wp * pcol("pg_expected_assists")
    apps = sums["played"] + wp * p_full
    bonus = sums["bonus"] + wp * pcol("pg_bonus")
    full = sums["full"] + wp * p_full
    # last season's defensive contributions per 90 -> P(hit) under a Poisson count
    p_defcon90 = np.where(p_mins > 0, pcol("pg_defensive_contribution") / p_mins.where(p_mins > 0) * 90, np.nan)
    p_hit = _poisson_at_least(np.nan_to_num(p_defcon90), _defcon_threshold(pos.reindex(sums.index)).to_numpy())
    hit = sums["hit"] + wp * p_full * np.where(np.isnan(p_defcon90), 0.0, p_hit)

    out = pd.DataFrame(index=sums.index)
    out["position"] = pos.reindex(sums.index)
    out["minutes"] = h.groupby("id")["mins"].sum().reindex(sums.index).fillna(0.0)   # unweighted
    out["raw_xg90"] = (xg / mins_all.where(mins_all > 0)) * 90
    out["raw_xa90"] = (xa / mins_all.where(mins_all > 0)) * 90
    out["raw_saves90"] = (sums["saves"] / sums["mins"].where(sums["mins"] > 0)) * 90
    out["raw_hit"] = hit / full.where(full > 0)
    out["raw_bonus_pa"] = bonus / apps.where(apps > 0)
    out["raw_yellow_pa"] = sums["yellow_cards"] / sums["played"].where(sums["played"] > 0)

    def shrink(raw: pd.Series, evidence: pd.Series, k: float) -> pd.Series:
        """(evidence x raw + k x position average) / (evidence + k)."""
        avg = (raw * evidence).groupby(out["position"]).sum() / evidence.where(raw.notna()).groupby(out["position"]).sum()
        prior_avg = out["position"].map(avg).fillna(0.0)
        return (raw.fillna(0.0) * evidence + k * prior_avg) / (evidence + k)

    out["xg90"] = shrink(out["raw_xg90"], mins_all, SHRINK_MINUTES)
    out["xa90"] = shrink(out["raw_xa90"], mins_all, SHRINK_MINUTES)
    out["saves90"] = shrink(out["raw_saves90"], sums["mins"], SHRINK_MINUTES).where(out["position"] == "GK", 0.0)
    out["defcon_hit"] = shrink(out["raw_hit"], full, SHRINK_MATCHES).where(out["position"] != "GK", 0.0)
    out["bonus_pa"] = shrink(out["raw_bonus_pa"], apps, SHRINK_MATCHES)
    out["yellow_pa"] = shrink(out["raw_yellow_pa"], sums["played"], 5)
    out["mins_60"] = (sums["mins_60"] / sums["full"].where(sums["full"] > 0)).fillna(DEFAULT_MINS_60)
    out["mins_sub"] = (sums["mins_sub"] / sums["sub"].where(sums["sub"] > 0)).fillna(DEFAULT_MINS_SUB)
    out["last_season_share"] = (wp * p_mins / mins_all.where(mins_all > 0)).fillna(0.0)
    return out


# --------------------------------------------------------------------------- #
# 3. Player x fixture
# --------------------------------------------------------------------------- #
def expected_events(rows: pd.DataFrame, ratings: Ratings, rates: pd.DataFrame) -> pd.DataFrame:
    """Events and points by source for each row.

    rows : one per (player, fixture): id, position, team, opponent_team, was_home,
           p_play, p_60 (from the minutes model, after injury flags)."""
    r = rows.reset_index(drop=True)
    rt = rates.reindex(r["id"]).reset_index(drop=True)
    pos = r["position"]
    xg_for, xg_against = fixture_goals(ratings, r["team"], r["opponent_team"], r["was_home"].astype(bool))
    t = ratings.teams.set_index("team_id")
    attack_avg = r["team"].map(t["attack"]).to_numpy()
    defence_avg = r["team"].map(t["defence"]).to_numpy()
    fac_att = xg_for / attack_avg                     # this fixture vs his team's usual
    fac_def = xg_against / defence_avg                # opponent threat vs usual

    p_play = r["p_play"].clip(0, 1).to_numpy(dtype=float)
    p_60 = np.minimum(r["p_60"].clip(0, 1).to_numpy(dtype=float), p_play)
    p_sub = p_play - p_60
    xmins = p_60 * rt["mins_60"].to_numpy() + p_sub * rt["mins_sub"].to_numpy()
    mins_if_play = np.where(p_play > 0, xmins / np.maximum(p_play, 1e-9), 0.0)

    xg = rt["xg90"].to_numpy() * xmins / 90 * fac_att
    xa = rt["xa90"].to_numpy() * xmins / 90 * fac_att
    cs_on_pitch = np.exp(-xg_against * rt["mins_60"].to_numpy() / 90)
    p_cs = p_60 * cs_on_pitch
    conceded = p_play * expected_floor(xg_against * mins_if_play / 90, 2)
    saves = rt["saves90"].to_numpy() * xmins / 90 * fac_def

    s = config.SCORING
    goal_pts = pos.map(s["goal"]).to_numpy(dtype=float)
    cs_pts = pos.map(s["clean_sheet"]).to_numpy(dtype=float)
    gc_pts = pos.map(s["goals_conceded_per_2"]).to_numpy(dtype=float)     # -1 for GK/DEF, else 0
    out = pd.DataFrame({
        "xmins": xmins, "team_xg": xg_for, "team_xga": xg_against,
        "xg": xg, "xa": xa, "p_goal": 1 - np.exp(-xg), "p_assist": 1 - np.exp(-xa),
        "p_cs": p_cs, "p_defcon": p_60 * rt["defcon_hit"].to_numpy(), "saves": saves,
        "Appearance": p_play * s["minutes_under_60"] + p_60 * (s["minutes_60_plus"] - s["minutes_under_60"]),
        "Goals": xg * goal_pts,
        "Assists": xa * s["assist"],
        "Clean sheet": p_cs * cs_pts,
        "Conceded": conceded * gc_pts,
        "Saves": p_play * expected_floor(saves / np.maximum(p_play, 1e-9), 3) * s["saves_per_3"],
        "Def. contributions": p_60 * rt["defcon_hit"].to_numpy() * s["defensive_contribution_points"],
        "Bonus": p_play * rt["bonus_pa"].to_numpy(),
        "Cards": p_play * rt["yellow_pa"].to_numpy() * s["yellow_card"],      # yellow_card is -1
    }, index=rows.index)
    out = out.fillna(0.0)
    out["xp_events"] = out[SOURCES].sum(axis=1)
    return out


def forecast_events(per_fixture: pd.DataFrame, history: pd.DataFrame, players: pd.DataFrame,
                    fixtures: pd.DataFrame, teams: pd.DataFrame, as_of_round: int,
                    prior: pd.DataFrame | None = None, prev_fixtures: pd.DataFrame | None = None,
                    prev_teams: pd.DataFrame | None = None) -> tuple[Ratings, pd.DataFrame, pd.DataFrame]:
    """The live breakdown: ratings, player rates, and `per_fixture` (predict.Forecast's,
    one row per player x upcoming fixture) with every event column added."""
    ratings = team_ratings(fixtures, teams, as_of_round, history, prev_fixtures, prev_teams)
    rates = player_rates(history, players, as_of_round, prior)
    return ratings, rates, per_fixture.join(expected_events(per_fixture, ratings, rates))


def season_data(season: str) -> dict:
    """Everything the event model needs for one filed season (and last season, if
    filed), as keyword arguments for team_ratings / player_rates."""
    from fpl_optimiser.features import multi_season
    from fpl_optimiser.ingest import season_store

    cur, prior, _ = multi_season.season_inputs(season)
    prev_season = season_store.previous_season(season)
    prev = season_store.load(prev_season) if prev_season in season_store.available() else None
    return {"history": cur.history, "players": cur.players, "fixtures": cur.fixtures,
            "teams": cur.teams, "prior": prior,
            "prev_fixtures": prev.fixtures if prev else None, "prev_teams": prev.teams if prev else None}
