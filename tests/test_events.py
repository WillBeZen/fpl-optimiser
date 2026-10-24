"""Event model — team ratings, fixture goals and player events (synthetic, no network)."""
import numpy as np
import pandas as pd
import pytest

import config
from fpl_optimiser.model import events

TEAMS = pd.DataFrame({"id": [1, 2, 3, 4], "code": [11, 12, 13, 14], "name": list("ABCD"),
                      "short_name": ["AAA", "BBB", "CCC", "DDD"]})


def _fixtures(rounds=6, finished_upto=4):
    """Every round: A v B and C v D at home in turn. A scores 3, D concedes 3."""
    rows, fid = [], 0
    for r in range(1, rounds + 1):
        for h, a in ((1, 2), (3, 4)) if r % 2 else ((2, 1), (4, 3)):
            fid += 1
            gh = 3 if h == 1 else (3 if a == 4 else 1)
            ga = 3 if a == 1 else (3 if h == 4 else 1)
            done = r <= finished_upto
            rows.append({"fixture_id": fid, "gameweek": r, "finished": done, "home_id": h, "away_id": a,
                         "home_score": gh if done else np.nan, "away_score": ga if done else np.nan})
    return pd.DataFrame(rows)


def test_team_ratings_rank_attack_and_defence_and_ignore_the_future():
    fx = _fixtures()
    r = events.team_ratings(fx, TEAMS, as_of_round=5)
    t = r.teams.set_index("short_name")
    assert t.loc["AAA", "att_index"] > 1 > t.loc["BBB", "att_index"]
    assert t.loc["DDD", "def_index"] > 1                      # D concedes the most
    assert t["matches"].tolist() == [4, 4, 4, 4]              # rounds 1-4 only
    assert t["prior_share"].eq(0).all()                       # no last season filed

    late = fx.copy()
    late.loc[late["gameweek"] == 5, ["finished", "home_score", "away_score"]] = [True, 9, 9]
    same = events.team_ratings(late, TEAMS, as_of_round=5).teams       # round 5 not yet "known"
    assert np.allclose(same["attack"], r.teams["attack"])


def test_last_season_is_a_fading_prior_and_promoted_clubs_start_low():
    prev_teams = pd.DataFrame({"id": [1, 2, 3, 4], "code": [11, 12, 13, 99], "name": list("ABCZ"),
                               "short_name": ["AAA", "BBB", "CCC", "ZZZ"]})
    prev = _fixtures(rounds=4, finished_upto=4)               # Z (code 99) went down; D is new
    early = events.team_ratings(_fixtures(finished_upto=0), TEAMS, 1, prev_fixtures=prev, prev_teams=prev_teams)
    late = events.team_ratings(_fixtures(finished_upto=4), TEAMS, 5, prev_fixtures=prev, prev_teams=prev_teams)
    e, l = early.teams.set_index("short_name"), late.teams.set_index("short_name")
    assert e.loc["DDD", "promoted"] and not e.loc["AAA", "promoted"]
    assert e["prior_share"].eq(1).all() and (l["prior_share"] < 1).all()
    assert e.loc["DDD", "defence"] == pytest.approx(e.loc["DDD", "defence"])   # set from relegated Z
    w = events.prior_weight(np.array([1, 10, 30]))
    assert w[0] > w[1] > w[2] > 0                             # last season fades


def test_fixture_goals_and_clean_sheets():
    r = events.team_ratings(_fixtures(), TEAMS, 5)
    out = events.fixture_outlook(r, _fixtures(), [5])
    assert len(out) == 4 and set(out["gw"]) == {5}
    a = out.set_index("team").loc["AAA"]
    b = out.set_index("team").loc["BBB"]
    assert a["xg_for"] == pytest.approx(b["xg_against"])      # the same match, both sides
    assert a["p_cs"] == pytest.approx(np.exp(-a["xg_against"]))
    assert a["xg_for"] > b["xg_for"]


def test_poisson_helpers():
    assert events.expected_floor(np.array([0.0]), 2)[0] == pytest.approx(0)
    lam = np.array([1.7])
    k = np.arange(40)
    p = np.exp(-lam[0]) * lam[0] ** k / np.array([float(np.prod(np.arange(1, i + 1))) for i in k])
    assert events.expected_floor(lam, 2)[0] == pytest.approx((p * (k // 2)).sum())
    assert events._poisson_at_least(np.array([2.0]), np.array([1.0]))[0] == pytest.approx(1 - np.exp(-2))
    assert events._poisson_at_least(np.array([2.0]), np.array([np.inf]))[0] == 0


def _history():
    """Striker 1 (team A): 0.6 xG per match; defender 2 (team D): hits his defcon
    threshold every match; keeper 3 (team C): 3 saves; sub 4: 20-minute cameos."""
    fx = _fixtures()
    rows = []
    for r in range(1, 5):
        f = fx[fx["gameweek"] == r]
        def fid(team):
            row = f[(f["home_id"] == team) | (f["away_id"] == team)].iloc[0]
            return int(row["fixture_id"]), bool(row["home_id"] == team)
        for pid, team, mins, extra in ((1, 1, 90, {"expected_goals": 0.6, "expected_assists": 0.1}),
                                       (2, 4, 90, {"defensive_contribution": 12}),
                                       (3, 3, 90, {"saves": 3}),
                                       (4, 2, 20, {"expected_goals": 0.1})):
            f_id, home = fid(team)
            rows.append({"id": pid, "round": r, "fixture": f_id, "was_home": home, "minutes": mins,
                         "expected_goals": 0.0, "expected_assists": 0.0, "saves": 0, "bonus": 0,
                         "yellow_cards": 0, "defensive_contribution": 0, **extra})
    players = pd.DataFrame({"id": [1, 2, 3, 4, 5], "position": ["FWD", "DEF", "GK", "MID", "FWD"]})
    return pd.DataFrame(rows), players


def test_player_rates_are_per_90_shrunk_and_leakage_free():
    history, players = _history()
    rates = events.player_rates(history, players, as_of_round=5)
    assert rates.loc[1, "raw_xg90"] == pytest.approx(0.6)
    assert 0 < rates.loc[1, "xg90"] < 0.6 + 1e-9                 # shrunk a little to the average
    assert rates.loc[2, "raw_hit"] == pytest.approx(1.0) and rates.loc[3, "raw_saves90"] == pytest.approx(3)
    assert rates.loc[4, "mins_sub"] == pytest.approx(20) and rates.loc[1, "mins_60"] == pytest.approx(90)
    assert rates.loc[5, "minutes"] == 0 and rates.loc[5, "xg90"] > 0      # no data: position average
    early = events.player_rates(history, players, as_of_round=3)
    assert early.loc[1, "minutes"] == 180                          # rounds 1-2 only


def test_team_xg_needs_a_full_teams_records():
    history, _ = _history()                        # a few players: far short of a team
    m = events.team_matches(_fixtures(), history)
    assert m["xgf"].isna().all()                   # so goals stand in
    full = history.assign(minutes=990)             # pretend each record covers a whole team
    assert events.team_matches(_fixtures(), full).set_index(["round", "team_id"]).loc[(1, 1), "xgf"] == 0.6


def test_expected_events_add_up_and_follow_the_fixture():
    history, players = _history()
    ratings = events.team_ratings(_fixtures(), TEAMS, 5, history=history)
    rates = events.player_rates(history, players, 5)
    rows = pd.DataFrame({"id": [1, 1, 2, 3, 1], "position": ["FWD", "FWD", "DEF", "GK", "FWD"],
                         "team": [1, 1, 4, 3, 1], "opponent_team": [4, 3, 3, 4, 4],
                         "was_home": [True, True, False, True, True],
                         "p_play": [1.0, 1.0, 1.0, 1.0, 0.0], "p_60": [1.0, 1.0, 1.0, 1.0, 0.0]})
    ev = events.expected_events(rows, ratings, rates)
    assert ev["xp_events"].tolist() == pytest.approx(ev[events.SOURCES].sum(axis=1).tolist())
    assert ev.loc[0, "xg"] > ev.loc[1, "xg"]                       # D concedes 3 a game, C only 1
    assert ev.loc[4, "xp_events"] == 0                             # won't play: no points
    assert ev.loc[0, "Appearance"] == config.SCORING["minutes_60_plus"]
    assert ev.loc[2, "Def. contributions"] > 1                     # hits it every week
    assert ev.loc[3, "Saves"] > 0 and ev.loc[3, "Conceded"] < 0 and ev.loc[0, "Conceded"] == 0
    assert 0 < ev.loc[2, "p_cs"] < 1 and ev.loc[0, "p_goal"] == pytest.approx(1 - np.exp(-ev.loc[0, "xg"]))
