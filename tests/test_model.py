"""Prediction model (Section 3): minutes + points models, backtest, forecast. No network."""
import numpy as np
import pandas as pd
import pytest

import config
from fpl_optimiser.model import backtest, inputs, minutes, points, predict
from tests.test_multi_season import two_seasons  # noqa: F401  (pytest fixture)


def _table(n_players: int = 60, rounds: int = 8, seed: int = 0) -> pd.DataFrame:
    """A synthetic training table with an obvious signal: 'regulars' (high recent
    minutes) always play and score ~6; 'fringe' players rarely get on."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        regular = pid % 2 == 0
        for r in range(1, rounds + 1):
            plays = regular or rng.random() < 0.1
            rows.append({
                "season": "2026-27", "id": pid, "round": r,
                "position": ["GK", "DEF", "MID", "FWD"][pid % 4], "price": 8.0 if regular else 4.5,
                "form_minutes_pg": 85.0 if regular else 8.0, "form_points_pg": 5.5 if regular else 0.4,
                "form_starts_rate": 0.95 if regular else 0.05,
                "target_points": float(rng.normal(6, 1)) if (plays and regular) else (1.0 if plays else 0.0),
                "minutes": 90.0 if plays and regular else (20.0 if plays else 0.0),
                "was_home": r % 2 == 0, "is_cold": r < 3, "has_prev_season": False,
                "prev_changed_team": False, "sample_weight": 1.0,
            })
    t = pd.DataFrame(rows)
    for col in inputs.NUMERIC:
        if col not in t:
            t[col] = np.nan
    return t


def test_points_model_learns_who_plays_and_what_they_score():
    t = _table()
    model = points.fit(t)
    out = model.predict(t)
    assert (out["p_60"] <= out["p_play"] + 1e-12).all()
    assert np.allclose(out["xp"], out["p_play"] * out["pts_if_play"])
    regular = t["form_minutes_pg"] > 50
    assert out.loc[regular, "p_play"].mean() > 0.9 > 0.3 > out.loc[~regular, "p_play"].mean()
    assert out.loc[regular, "xp"].mean() > 4 > 1 > out.loc[~regular, "xp"].mean()


def test_minutes_model_copes_with_one_class():
    t = _table().assign(minutes=90.0)                    # everyone always plays
    p = minutes.fit(t).predict(t)
    assert np.allclose(p["p_play"], 1.0) and np.allclose(p["p_60"], 1.0)


def test_backtest_only_ever_trains_on_the_past(monkeypatch):
    seen = []
    real_fit = points.fit

    def spy(train):
        seen.append(train["round"].max())
        return real_fit(train)

    monkeypatch.setattr(points, "fit", spy)
    t = _table(rounds=6)
    preds = backtest.walk_forward(t, "2026-27", rounds=[4, 5, 6])
    assert seen == [3, 4, 5]                             # predicting r used rounds < r only
    assert sorted(preds["round"].unique()) == [4, 5, 6]

    summary = backtest.summarise(preds).set_index("method")
    needs_events = {"Event model", "Blend (what the planner uses)"}            # need events_data
    assert set(summary.index) == set(backtest.METHODS.values()) - needs_events
    # with this obvious signal the model's top 11 are all regulars
    assert summary.loc["Points model", "top11_points"] > 50


def test_availability_flags():
    players = pd.DataFrame({"id": [1, 2, 3, 4, 5],
                            "status": ["a", "d", "i", "u", "s"],
                            "chance_of_playing_next_round": [None, 75, None, None, 0]})
    nxt = predict.availability(players, gw=9, next_gw=9)
    later = predict.availability(players, gw=10, next_gw=9)
    assert nxt.to_dict() == {1: 1.0, 2: 0.75, 3: 0.0, 4: 0.0, 5: 0.0}
    assert later.to_dict() == {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0, 5: 1.0}


def test_forecast_from_a_filed_store(two_seasons):
    fc = predict.forecast(horizon=3)
    assert fc.next_gw == 3 and fc.gameweeks == [3, 4, 5]
    assert len(fc.xp) == 2 * 3                           # every player x every gameweek
    gw3 = fc.xp[fc.xp["gw"] == 3].set_index("web_name")
    assert gw3.loc["Star", "fixtures"] == "D (H)" and gw3.loc["New", "fixtures"] == "A (A)"
    blank = fc.xp[fc.xp["gw"] == 4]
    assert (blank["xp"] == 0).all() and (blank["fixtures"] == "-").all()
    assert list(fc.wide().columns[-4:]) == ["GW3", "GW4", "GW5", "total"]


def test_forecast_blends_the_points_and_event_models(two_seasons):
    fc = predict.forecast(horizon=3)
    fx = fc.per_fixture
    w = config.EVENT_MODEL_WEIGHT
    assert fx["xp"].tolist() == pytest.approx(((1 - w) * fx["xp_model"] + w * fx["xp_events"]).tolist())
    assert {"xg", "p_cs", "Goals", "Clean sheet"} <= set(fx.columns)
    assert fc.ratings is not None and set(fc.ratings.teams["short_name"]) == {"A", "B", "D"}
    gw3 = fc.xp[fc.xp["gw"] == 3]
    assert gw3["xp"].tolist() == pytest.approx(((1 - w) * gw3["xp_model"] + w * gw3["xp_events"]).tolist())
    only_model = predict.forecast(horizon=3, event_weight=0.0)
    assert only_model.xp["xp"].tolist() == pytest.approx(only_model.xp["xp_model"].tolist())
