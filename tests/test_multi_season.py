"""End to end across seasons: store -> link -> prior -> training table (no network).

Two tiny filed seasons. Club C is relegated and D promoted; player 'Star' (code
1001) plays both seasons under different ids; 'Gone' leaves, 'New' arrives.
"""
import json

import pytest

import config
from fpl_optimiser.features import multi_season
from fpl_optimiser.ingest import season_store


def _teams(spec):
    return [{"id": i, "code": code, "name": name, "short_name": name, "strength": 3,
             "strength_defence_home": 1100 + 10 * i, "strength_defence_away": 1050 + 10 * i}
            for i, code, name in spec]


def _player(pid, code, team, element_type, second, dob, **totals):
    p = {"id": pid, "code": code, "team": team, "element_type": element_type,
         "web_name": second, "first_name": "Jo", "second_name": second, "birth_date": dob,
         "region": 1, "now_cost": 50, "total_points": 0, "minutes": 0, "starts": 0}
    p.update(totals)
    return p


def _fixture(fid, gw, home, away, home_goals, away_goals):
    return {"id": fid, "event": gw, "team_h": home, "team_a": away, "finished": True,
            "team_h_score": home_goals, "team_a_score": away_goals}


def _match(rnd, opponent, home, minutes, points):
    return {"round": rnd, "opponent_team": opponent, "was_home": home, "minutes": minutes,
            "total_points": points, "expected_goals": "0.10", "expected_assists": "0.10",
            "expected_goal_involvements": "0.20", "defensive_contribution": 3,
            "bonus": 0, "bps": 10}


def _file(root, season, finished_gws, teams, players, fixtures, histories):
    start = int(season[:4])
    events = [{"id": gw, "deadline_time": f"{start}-08-15T17:30:00Z" if gw == 1
               else f"{start + 1}-05-01T17:30:00Z", "finished": gw <= finished_gws}
              for gw in range(1, 39)]
    d = root / season
    d.mkdir(parents=True)
    (d / "bootstrap-static.json").write_text(
        json.dumps({"events": events, "teams": _teams(teams), "elements": players}))
    (d / "fixtures.json").write_text(json.dumps(fixtures))
    for pid, rows in histories.items():
        (d / f"element-summary_{pid}.json").write_text(json.dumps({"history": rows}))


@pytest.fixture
def two_seasons(tmp_path, monkeypatch):
    monkeypatch.setattr(season_store, "STORE", tmp_path)
    _file(tmp_path, "2025-26", 38, [(1, 3, "A"), (2, 43, "B"), (3, 90, "C")],
          [_player(1, 1001, 1, 3, "Star", "2000-01-01", total_points=190, minutes=3000, starts=34),
           _player(2, 1002, 2, 4, "Gone", "1990-01-01", total_points=50, minutes=900, starts=10)],
          [_fixture(1, 1, 1, 2, 2, 0), _fixture(2, 2, 2, 3, 1, 1)],
          {1: [_match(1, 2, True, 90, 8)],
           2: [_match(1, 1, False, 90, 2), _match(2, 3, True, 90, 5)]})
    _file(tmp_path, "2026-27", 2, [(1, 3, "A"), (2, 43, "B"), (3, 56, "D")],
          [_player(5, 1001, 1, 3, "Star", "2000-01-01", total_points=11, minutes=180, starts=2),
           _player(6, 2001, 3, 4, "New", "2005-05-05", total_points=6, minutes=90, starts=1)],
          [_fixture(1, 1, 3, 1, 1, 3), _fixture(2, 2, 1, 2, 0, 0)],
          {5: [_match(1, 3, False, 90, 9), _match(2, 2, True, 90, 2)],
           6: [_match(1, 1, True, 90, 6)]})


def test_build_all_stacks_seasons_with_weights_and_prior(two_seasons):
    tbl = multi_season.build_all()
    assert tbl.groupby("season").size().to_dict() == {"2025-26": 3, "2026-27": 3}
    assert tbl.groupby("season")["sample_weight"].first().to_dict() == {
        "2025-26": config.PREV_SEASON_WEIGHT, "2026-27": 1.0}

    star = tbl[tbl["code"] == 1001].set_index(["season", "round"])
    assert set(star["id"]) == {1, 5}                          # same person, two season ids
    gw1 = star.loc[("2026-27", 1)]
    assert gw1["has_prev_season"]
    assert gw1["form_points_pg"] == pytest.approx(190 / 38)   # GW1 form = last season's average
    assert gw1["prev_points_per_90"] == pytest.approx(190 / 3000 * 90)

    new = tbl[tbl["code"] == 2001].iloc[0]
    assert not new["has_prev_season"] and new["is_cold"]
    assert not tbl.loc[tbl["season"] == "2025-26", "has_prev_season"].any()   # 2024-25 not filed
