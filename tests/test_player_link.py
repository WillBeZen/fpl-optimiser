"""Cross-season linking of players and clubs — synthetic seasons, no network."""
import pandas as pd

from fpl_optimiser.ingest import player_link

_COLS = ["id", "code", "web_name", "first_name", "second_name", "birth_date", "region",
         "position", "team_name", "price", "total_points", "minutes"]

PREV = pd.DataFrame([
    (1, 154561, "Raya", "David", "Raya Martín", "1995-09-15", 200, "GK", "Arsenal", 5.5, 160, 3420),
    (2, 223094, "Haaland", "Erling", "Haaland", "2000-07-21", 152, "FWD", "Man City", 14.0, 200, 2700),
    (3, 900001, "Smith", "Jon", "Smith", "2001-01-01", 241, "MID", "Burnley", 5.0, 40, 900),
    (4, 900002, "Silva", "João", "da Silva Santos", "1999-03-03", 30, "DEF", "Wolves", 4.5, 60, 2000),
    (5, 900003, "Timer", "Old", "Timer", "1988-05-05", 241, "DEF", "West Ham", 4.0, 10, 300),
    (6, 900004, "Twin", "A", "Twin", "2002-02-02", 241, "MID", "Fulham", 4.5, 20, 500),
    (7, 900005, "Twin", "A", "Twin", "2002-02-02", 241, "MID", "Fulham", 4.5, 15, 400),
    (8, 900006, "Lash", "C", "Lash", "1990-01-01", 241, "DEF", "Brentford", 4.5, 8, 200),
], columns=_COLS)

CURR = pd.DataFrame([
    (10, 154561, "Raya", "David", "Raya Martín", "1995-09-15", 200, "GK", "Arsenal", 5.5, 30, 630),
    (11, 223094, "Haaland", "Erling", "Haaland", "2000-07-21", 152, "FWD", "Man City", 14.5, 60, 600),
    # code changed (e.g. a temporary code made permanent): same name + birth date
    (12, 912345, "Smith", "Jon", "Smith", "2001-01-01", 241, "MID", "Leeds", 5.0, 12, 300),
    # listed name changed: same birth date + nationality + surname
    (13, 912346, "Santos", "Joao", "Santos", "1999-03-03", 30, "DEF", "Ipswich", 4.5, 9, 450),
    (14, 777777, "Signing", "New", "Signing", "2004-04-04", 100, "FWD", "Hull", 6.0, 20, 450),
    # two equally good candidates last season -> ambiguous, never guessed
    (17, 999999, "Twin", "A", "Twin", "2002-02-02", 241, "MID", "Fulham", 4.5, 5, 100),
    # same code but a different birth date -> rejected, not trusted
    (18, 900006, "Lash", "C", "Lash", "1991-01-01", 241, "DEF", "Brentford", 4.5, 5, 90),
], columns=_COLS)


def _linked() -> dict:
    res = player_link.link_players(PREV, CURR)
    return {(r.id_prev, r.id_curr): r.method for r in res.links.itertuples()}


def test_normalise_name_ignores_accents_case_and_punctuation():
    assert player_link.normalise_name("David", "Raya Martín") == "david raya martin"
    assert player_link.normalise_name("N'Golo", "KANTÉ") == "n golo kante"
    assert player_link.normalise_name(None, "Ødegaard") == "odegaard"


def test_links_strictest_pass_first():
    assert _linked() == {(1, 10): "code", (2, 11): "code",
                         (3, 12): "name+dob", (4, 13): "dob+region+surname"}


def test_ambiguous_and_conflicting_candidates_are_not_linked():
    res = player_link.link_players(PREV, CURR)
    assert sorted(res.arrivals["id"]) == [14, 17, 18]
    assert sorted(res.departures["id"]) == [5, 6, 7, 8]
    reasons = set(res.rejected["reason"])
    assert "code match, birth dates differ" in reasons
    assert "ambiguous name+dob" in reasons


def test_links_flag_moves_and_summary_counts():
    res = player_link.link_players(PREV, CURR)
    moved = res.links.set_index("id_curr")["changed_team"]
    assert not moved[10] and moved[12] and moved[13]          # Burnley->Leeds, Wolves->Ipswich
    s = res.summary()
    assert s["linked"] == 4 and s["arrivals"] == 3 and s["departures"] == 4
    assert s["by_method"] == {"code": 2, "name+dob": 1, "dob+region+surname": 1}


def test_link_teams_by_permanent_code():
    prev = pd.DataFrame({"id": [1, 3, 20], "code": [3, 90, 39], "name": ["Arsenal", "Burnley", "Wolves"]})
    curr = pd.DataFrame({"id": [1, 11, 12], "code": [3, 56, 40], "name": ["Arsenal", "Hull City", "Ipswich Town"]})
    t = player_link.link_teams(prev, curr).set_index("code")["status"]
    assert t[3] == "stayed"
    assert t[90] == t[39] == "relegated"
    assert t[56] == t[40] == "promoted"


def test_check_against_history_past():
    links = player_link.link_players(PREV, CURR).links
    past = pd.DataFrame({"id": [10, 11], "season_name": ["2025/26", "2025/26"],
                         "element_code": [154561, 223094], "total_points": [160, 199]})
    check = player_link.check_against_history_past(links, past, PREV, "2025-26")
    assert check == {"checked": 2, "code_agrees": 2, "points_agree": 1}
