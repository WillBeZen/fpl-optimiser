"""Archive tests — an earlier season's match records from a public archive, merged
into the season store and checked against FPL's own data (mock files, no network)."""
import json

import pandas as pd
import pytest

from fpl_optimiser.ingest import archive, fpl_api
from fpl_optimiser.ingest import season_store as ss

FIXTURES = [{"id": 1, "event": 1, "team_h": 1, "team_a": 2, "finished": True, "team_h_score": 2, "team_a_score": 0},
            {"id": 2, "event": 2, "team_h": 2, "team_a": 1, "finished": True, "team_h_score": 1, "team_a_score": 1}]


def _bootstrap(points):
    return {"events": [{"id": 1, "deadline_time": "2025-08-15T17:30:00Z", "finished": True},
                       {"id": 2, "deadline_time": "2025-08-22T17:30:00Z", "finished": True}],
            "teams": [{"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
                      {"id": 2, "code": 43, "name": "Man City", "short_name": "MCI"}],
            "elements": [{"id": i, "code": 1000 + i, "web_name": f"P{i}", "first_name": "A", "second_name": f"B{i}",
                          "team": 1 if i < 3 else 2, "element_type": 2, "now_cost": 50, "total_points": p,
                          "minutes": 180} for i, p in points.items()]}


def _match(pid, fixture, pts):
    """One player's match record, as FPL's element-summary history serves it."""
    home = (pid < 3) == (fixture == 1)                    # ids 1-2 play for club 1, 3 for club 2
    f = FIXTURES[fixture - 1]
    return {"element": pid, "fixture": fixture, "round": f["event"], "opponent_team": 2 if pid < 3 else 1,
            "was_home": home, "kickoff_time": f"2025-08-{15 + 7 * (fixture - 1)}T14:00:00Z",
            "team_h_score": f["team_h_score"], "team_a_score": f["team_a_score"], "minutes": 90,
            "total_points": pts, "value": 50}


def _archive(rows, repeat=0):
    """The archive's file: FPL's fields plus its own name / position / team / xP / GW."""
    df = pd.DataFrame(rows).assign(name="A B", position="DEF", team="Arsenal", xP=2.0)
    df["GW"] = df["round"]
    return pd.concat([df, df.head(repeat)], ignore_index=True)


POINTS = {1: 8, 2: 3, 3: 5}                        # season totals: two matches each
MATCHES = [_match(1, 1, 6), _match(1, 2, 2), _match(2, 1, 1), _match(2, 2, 2), _match(3, 1, 0), _match(3, 2, 5)]


def test_blob_id_is_gits():
    assert archive.blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"   # git hash-object


def test_parse_keeps_fpls_fields_and_drops_repeats():
    a = archive.parse_merged_gw(_archive(MATCHES, repeat=2))
    assert len(a) == len(MATCHES) and a.attrs["repeats"] == 2
    assert not set(archive.EXTRA) & set(a.columns) and list(a["id"]) == list(a["element"])


def _filed(tmp_path):
    """A filed 2025/26 with FPL's file for player 1 only, and an archive for all three
    (player 1's archive points differ, so it shows whose record is used)."""
    store, arc = tmp_path / "fpl", tmp_path / "archive"
    d = store / "2025-26"
    d.mkdir(parents=True)
    (d / "bootstrap-static.json").write_text(json.dumps(_bootstrap(POINTS)))
    (d / "fixtures.json").write_text(json.dumps(FIXTURES))
    fpl_rows = [m for m in MATCHES if m["element"] == 1]
    (d / "element-summary_1.json").write_text(json.dumps({"history": fpl_rows}))
    rows = [dict(m, total_points=99) if m["element"] == 1 else m for m in MATCHES]
    (arc / "2025-26").mkdir(parents=True)
    _archive(rows, repeat=1).to_csv(arc / "2025-26" / archive.FILE, index=False)
    return store, arc


def test_the_archive_fills_in_players_fpls_files_dont_cover(tmp_path):
    store, arc = _filed(tmp_path)
    s = ss.load("2025-26", store, archive=arc)
    assert sorted(s.history["id"].unique()) == [1, 2, 3] and len(s.history) == 6
    assert s.history.loc[s.history["id"] == 1, "total_points"].tolist() == [6, 2]
    assert s.history.loc[s.history["id"] == 3, "total_points"].tolist() == [0, 5]
    # a test store reads no archive unless told to; False reads FPL's files only
    assert ss.load("2025-26", store).history["id"].nunique() == 1
    assert ss.load("2025-26", store, archive=False).history["id"].nunique() == 1


def test_a_test_store_never_reads_the_real_archive(tmp_path, monkeypatch):
    store, arc = _filed(tmp_path)
    monkeypatch.setattr(ss, "STORE", store)                # a test's store, patched in
    monkeypatch.setattr(archive, "ARCHIVE", arc)           # standing in for the real archive
    assert ss.load("2025-26").history["id"].nunique() == 1
    assert ss.load("2025-26", archive=True).history["id"].nunique() == 3


def test_checks_pass_on_agreeing_data_and_catch_problems():
    boot = _bootstrap(POINTS)
    teams = fpl_api.parse_teams(boot)
    players, fixtures = fpl_api.parse_players(boot), fpl_api.parse_fixtures(FIXTURES, teams)
    fpl = pd.DataFrame([m for m in MATCHES if m["element"] == 1]).assign(id=1)

    def run(rows, repeat=0):
        return {c["check"].split(": ", 1)[1]: c for c in
                archive.checks(archive.parse_merged_gw(_archive(rows, repeat)), fpl, players, fixtures, "2025-26")}

    good = run(MATCHES, repeat=1)
    assert all(c["ok"] for c in good.values()) and len(good) == 3
    assert "repeated rows dropped first: 1" in good["adds up to FPL's season totals"]["detail"]
    assert good["agrees with FPL's own match files"]["detail"].endswith("identical")

    changed = [dict(m, total_points=7) if (m["element"], m["fixture"]) == (1, 1) else m for m in MATCHES]
    bad = run(changed)
    assert not bad["agrees with FPL's own match files"]["ok"] and not bad["adds up to FPL's season totals"]["ok"]
    assert not run(MATCHES[:-1])["adds up to FPL's season totals"]["ok"]                 # a match missing
    wrong_side = [dict(m, was_home=not m["was_home"]) if m["element"] == 3 else m for m in MATCHES]
    assert not run(wrong_side)["matches the fixture list"]["ok"]


class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def test_pull_keeps_only_the_pinned_file(tmp_path, monkeypatch):
    calls = []

    class Session:
        def get(self, url, timeout):
            calls.append(url)
            return _Response(served)

    monkeypatch.setattr(fpl_api, "_http", lambda: Session())
    content = b"element,fixture\n1,1\n"
    monkeypatch.setitem(archive.SOURCES, "2025-26", {"url": "https://example.test/merged_gw.csv",
                                                     "blob": archive.blob_id(content), "bytes": len(content)})
    served = b"something else"
    with pytest.raises(ValueError):
        archive.pull("2025-26", tmp_path)
    assert not archive.path("2025-26", tmp_path).exists()
    served = content
    p = archive.pull("2025-26", tmp_path)
    assert p.read_bytes() == content and len(calls) == 2
    archive.pull("2025-26", tmp_path)                                    # already there: no download
    assert len(calls) == 2
