"""Season store tests — filing, freezing and loading on mock payloads (no network)."""
import pytest

from fpl_optimiser.ingest import fpl_api
from fpl_optimiser.ingest import season_store as ss

QUIET = {"pause_s": 0, "log": lambda *_: None}


def _bootstrap(first_deadline="2026-08-21T17:30:00Z"):
    return {
        "events": [{"id": 1, "deadline_time": first_deadline, "finished": True},
                   {"id": 2, "deadline_time": "2026-08-28T17:30:00Z", "finished": False}],
        "teams": [{"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
                  {"id": 2, "code": 43, "name": "Man City", "short_name": "MCI"}],
        "elements": [{"id": i, "code": 1000 + i, "web_name": f"P{i}", "first_name": "A",
                      "second_name": f"B{i}", "team": 1 + i % 2, "element_type": 1 + i % 4,
                      "now_cost": 50, "total_points": 10 * i} for i in (1, 2, 3)],
    }


def _summary(pid):
    return {"history": [{"round": 1, "fixture": 1, "minutes": 90, "total_points": pid,
                         "opponent_team": 2, "was_home": True},
                        {"round": 2, "fixture": 2, "minutes": 0, "total_points": 0,   # not played yet
                         "opponent_team": 2, "was_home": False}],
            "history_past": [{"season_name": "2025/26", "element_code": 1000 + pid,
                              "total_points": 100}]}


@pytest.fixture
def fake_api(monkeypatch):
    """Stand-in for the live API; records every path requested."""
    calls = []

    def fetch(path, retries=3):
        calls.append(path)
        if path == "bootstrap-static":
            return _bootstrap()
        if path == "fixtures":
            return [{"id": 1, "event": 1, "team_h": 1, "team_a": 2, "finished": True,
                     "team_h_score": 1, "team_a_score": 0},
                    {"id": 2, "event": 2, "team_h": 2, "team_a": 1, "finished": False}]
        if path.startswith("event/"):
            return {"elements": [{"id": i, "stats": {"minutes": 90, "total_points": i},
                                  "explain": [{"fixture": 1, "stats": [
                                      {"identifier": "minutes", "points": 2, "value": 90},
                                      {"identifier": "bonus", "points": i - 2, "value": i - 2}]}]}
                                 for i in (1, 2, 3)]}
        return _summary(int(path.split("/")[1]))

    monkeypatch.setattr(fpl_api, "fetch", fetch)
    return calls


def test_season_of_reads_the_first_deadline():
    def events(*deadlines):
        return {"events": [{"id": i, "deadline_time": d} for i, d in enumerate(deadlines, 1)]}
    assert ss.season_of(events("2026-08-21T17:30:00Z", "2027-05-23T13:30:00Z")) == "2026-27"
    assert ss.season_of(events("2026-05-24T13:30:00Z", "2025-08-15T17:30:00Z")) == "2025-26"
    assert ss.season_of(events("2099-08-01T00:00:00Z")) == "2099-00"


def test_pull_live_files_by_season_and_resumes(tmp_path, fake_api):
    assert ss.pull_live(root=tmp_path, **QUIET) == "2026-27"
    assert sorted(f.name for f in (tmp_path / "2026-27").iterdir()) == [
        "bootstrap-static.json", "element-summary_1.json", "element-summary_2.json",
        "element-summary_3.json", "event-live_1.json", "fixtures.json"]   # GW1 finished: filed
    assert ss.available(root=tmp_path) == ["2026-27"]

    n = len(fake_api)
    ss.pull_live(root=tmp_path, **QUIET)                  # fresh player files are kept
    assert not [c for c in fake_api[n:] if c.startswith(("element-summary", "event/"))]


def test_pull_live_refuses_a_frozen_season(tmp_path, fake_api):
    ss.pull_live(root=tmp_path, **QUIET)
    ss.freeze("2026-27", root=tmp_path)
    before = (tmp_path / "2026-27" / "bootstrap-static.json").read_text()
    with pytest.raises(PermissionError):
        ss.pull_live(root=tmp_path, **QUIET)
    assert (tmp_path / "2026-27" / "bootstrap-static.json").read_text() == before


def test_load_round_trip_and_season_check(tmp_path, fake_api):
    ss.pull_live(root=tmp_path, **QUIET)
    s = ss.load("2026-27", root=tmp_path)
    assert len(s.players) == 3 and s.players["code"].tolist() == [1001, 1002, 1003]
    assert sorted(s.history["id"]) == [1, 2, 3]                     # finished matches only
    assert sorted(s.pending["id"]) == [1, 2, 3]                     # GW2 isn't finished
    assert s.live.set_index("id")["total_points"].to_dict() == {1: 1, 2: 2, 3: 3}
    assert s.live["explained"].all() and s.live["gw"].eq(1).all()
    assert set(s.past["element_code"]) == {1001, 1002, 1003}
    assert s.finished_gameweeks == 1

    # a folder that holds a different season's data is refused
    (tmp_path / "2025-26").mkdir()
    (tmp_path / "2026-27" / "bootstrap-static.json").replace(
        tmp_path / "2025-26" / "bootstrap-static.json")
    with pytest.raises(ValueError):
        ss.load("2025-26", root=tmp_path)


def test_only_finished_matches_count_as_played():
    """Mid-gameweek FPL lists every match, played or not: only finished ones are kept."""
    import pandas as pd
    history = pd.DataFrame({"id": [1, 1, 1], "fixture": [10, 11, 12], "round": [7, 8, 8],
                            "minutes": [90, 90, 0]})
    fixtures = pd.DataFrame({"fixture_id": [10, 11, 12], "finished": [True, True, False]})
    kept = ss.played_only(history, fixtures)
    assert list(kept["fixture"]) == [10, 11]
    assert ss.played_only(history.drop(columns="fixture"), fixtures).equals(
        history.drop(columns="fixture"))                       # nothing to check against
