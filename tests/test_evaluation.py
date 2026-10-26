"""The honest evaluation harness: each past deadline rebuilt as it stood (no network)."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

import config
from fpl_optimiser.model import evaluation, inputs
from tests.test_multi_season import two_seasons  # noqa: F401  (pytest fixture)


def _season():
    """Three clubs; player 1 moves from club 1 to club 2 after GW2 (price 5.0 -> 5.5);
    player 3 is first listed in GW3."""
    fixtures = pd.DataFrame({"fixture_id": [1, 2, 3, 4], "gameweek": [1, 2, 3, 4], "finished": True,
                             "home_id": [1, 1, 2, 2], "away_id": [3, 3, 3, 3],
                             "home_score": [1, 2, 3, 0], "away_score": [0, 0, 1, 1]})
    history = pd.DataFrame({"id": [1, 1, 1, 3], "round": [1, 2, 3, 3], "fixture": [1, 2, 3, 3],
                            "was_home": [True, True, True, True], "value": [50, 50, 55, 45],
                            "minutes": [90, 90, 90, 90], "total_points": [2, 2, 2, 2]})
    players = pd.DataFrame({"id": [1, 2, 3], "web_name": ["A", "B", "C"], "position": ["MID", "DEF", "FWD"],
                            "team": [2, 3, 2], "price": [6.0, 4.5, 4.6]})
    teams = pd.DataFrame({"id": [1, 2, 3], "name": ["One", "Two", "Three"], "short_name": ["ONE", "TWO", "THR"]})
    return NS(history=history, fixtures=fixtures, players=players, teams=teams)


def test_players_as_they_stood_at_a_deadline():
    cur = _season()
    at2 = evaluation.players_at(cur, 2).set_index("id")
    assert set(at2.index) == {1}                        # C wasn't listed yet; B never played a match
    assert at2.at[1, "team"] == 1 and at2.at[1, "price"] == 5.0 and at2.at[1, "team_name"] == "One"
    at3 = evaluation.players_at(cur, 3).set_index("id")
    assert at3.at[1, "team"] == 2 and at3.at[1, "price"] == 5.5 and at3.at[3, "price"] == 4.5


def test_later_results_are_hidden_and_training_stops_before_the_deadline():
    f = evaluation.fixtures_at(_season().fixtures, 3)
    assert f.loc[f["gameweek"] >= 3, "finished"].eq(False).all() and f.loc[f["gameweek"] >= 3, "home_score"].isna().all()
    assert f.loc[f["gameweek"] < 3, "finished"].all() and f.loc[f["gameweek"] < 3, "home_score"].notna().all()
    table = pd.DataFrame({"season": ["2025-26", "2026-27", "2026-27", "2026-27"], "round": [30, 1, 2, 3],
                          "target_points": [1, 2, 3, 4], "minutes": [90] * 4})
    assert evaluation.training_rows(table, "2026-27", 3)["round"].tolist() == [30, 1, 2]


def test_the_blend_weight_is_chosen_from_out_of_sample_rows():
    y = np.array([1.0, 4.0, 2.0, 7.0])
    rows = pd.DataFrame({"target_points": y, "xp": y + 1.5, "xp_events": y})
    assert evaluation.choose_weight(rows) == 1.0            # the event model was exactly right
    assert evaluation.choose_weight(rows.assign(xp=y, xp_events=y - 2)) == 0.0
    mid = evaluation.choose_weight(rows.assign(xp=y + 1, xp_events=y - 1))
    assert mid == pytest.approx(0.5)


def test_baselines_come_from_the_training_rows_only():
    train = pd.DataFrame({"season": "2026-27", "id": [1, 1, 2, 2], "position": "MID",
                          "price": [5.0, 5.0, 5.2, 5.2], "target_points": [2, 0, 6, 0],
                          "minutes": [90, 0, 90, 0]})
    rows = pd.DataFrame({"id": [1, 9], "position": ["MID", "MID"], "price": [5.1, 5.0]})
    per_match, per_app = evaluation.position_price(train, rows)
    # bands are £4.6-5.1m and £5.1-5.6m: player 1 (5.1) looks like player 2's band, 9 (5.0) like player 1's
    assert per_match.tolist() == pytest.approx([3.0, 1.0])
    assert per_app.tolist() == pytest.approx([6.0, 2.0])    # points per appearance only
    avg = evaluation.season_average(train, "2026-27", rows)
    assert avg.tolist() == pytest.approx([1.0, 2.0])        # player 9 has no match: everyone's average


def _first_week():
    """Two deadlines, three players; the reference is exactly right in GW2 only."""
    rows = []
    for r, ref_err in ((2, 0.0), (3, 2.0)):
        for pid, act in ((1, 6), (2, 2), (3, 0)):
            rows.append({"id": pid, "round": r, "fixture_id": 10 * r + pid, "target_points": act,
                         "minutes": 90 if act else 0, "p_play": 0.9, "trained_to": r - 1,
                         "event_weight": 0.5, "in_pool": pid != 3,
                         "xp_blend": act + ref_err, "xp": act + 1.0, "xp_events": act - 1.0 + 2 * ref_err,
                         "blend_wf": act + ref_err, "pos_price_p": 3.0, "pos_price": 2.0, "ewma": act / 2,
                         "season_avg": 2.0})
    return pd.DataFrame(rows)


def test_scoring_and_paired_uncertainty():
    ev = evaluation.Evaluation(season="2026-27", rounds=[2, 3], first_week=_first_week(), horizons={},
                               pools={2: {1, 2}, 3: {1, 2}}, weights={2: 0.5, 3: 0.5})
    pool = evaluation.player_gameweeks(ev, "pool")
    assert set(pool["id"]) == {1, 2} and len(evaluation.player_gameweeks(ev, "all")) == 6
    gw = evaluation.by_gameweek(pool)
    ref = gw[gw["method"] == "xp_blend"].set_index("round")
    assert ref.at[2, "rmse"] == 0 and ref.at[3, "rmse"] == pytest.approx(2.0)
    assert ref.at[2, "top11"] == ref.at[2, "best11"] == 8
    s = evaluation.summary(gw, pool).set_index("method")
    assert s.at["xp_blend", "gameweeks"] == 2 and s.at["xp_blend", "mae"] == pytest.approx(1.0)
    pr = evaluation.paired(gw).set_index(["method", "metric"])
    # the points model is off by 1 every week; the reference by 0 then 2: mean RMSE difference 0
    assert pr.at[("xp", "rmse"), "mean_diff"] == pytest.approx(0.0)
    assert pr.at[("xp", "rmse"), "reference_better"] == 1 and pr.at[("xp", "rmse"), "gameweeks"] == 2
    assert pr.at[("xp", "rmse"), "low"] < 0 < pr.at[("xp", "rmse"), "high"]
    bt = evaluation.backtest_table(ev)
    assert list(bt.columns) == ["method", "MAE", "RMSE", "rank_corr", "top11_points", "gameweeks", "predictions"]


def test_a_replay_on_a_filed_store(two_seasons):
    from fpl_optimiser.features import multi_season

    table = multi_season.build_all()
    ev = evaluation.run(table=table, horizon=2)
    assert ev.rounds == [2] and ev.weights == {2: config.EVENT_MODEL_WEIGHT}   # nothing earlier to learn from
    fw = ev.first_week
    assert len(fw) and (fw["trained_to"] < fw["round"]).all() and ev.unmatched == 0
    assert set(evaluation.METHODS) <= set(fw.columns) and fw[list(evaluation.METHODS)].notna().all().all()
    assert ev.pools[2] and fw["in_pool"].any() and set(ev.horizons[2]["round"]) == {2, 3}
    assert (fw["half_life"] == config.RECENCY_HALF_LIFE_GW).all()
    with pytest.raises(ValueError, match="half-life"):
        evaluation.run(table=table.assign(half_life=config.RECENCY_HALF_LIFE_GW * 2), horizon=1)


def test_model_inputs_are_all_knowable_at_the_deadline():
    assert "opp_archetype" not in inputs.FEATURES and "opp_str_def" not in inputs.FEATURES
    assert "earlier_season" in inputs.FEATURES
    rows = pd.DataFrame({c: [1.0000000001] for c in inputs.NUMERIC} |
                        {c: [False] for c in inputs.FLAGS if c != "earlier_season"} | {"position": ["MID"]})
    X = inputs.design_matrix(rows)
    assert X.at[0, "earlier_season"] == 0.0                  # absent when predicting: this season
    assert X.at[0, "price"] == 1.0                            # rounded: two code paths can't diverge
