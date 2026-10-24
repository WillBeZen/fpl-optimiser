"""Squad optimiser (Section 4) and multi-week planner (Section 5) — synthetic players."""
import numpy as np
import pandas as pd
import pytest

import config
from fpl_optimiser.optimise import squad


def _candidates(seed: int = 0, n_clubs: int = 8) -> pd.DataFrame:
    """A small pool: per club 1 GK, 3 DEF, 3 MID, 2 FWD with random prices/xp."""
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for club in range(n_clubs):
        for pos, n in (("GK", 1), ("DEF", 3), ("MID", 3), ("FWD", 2)):
            for _ in range(n):
                pid += 1
                price = round(float(rng.uniform(4.0, 12.0)), 1)
                rows.append({"id": pid, "web_name": f"P{pid}", "team_name": f"Club{club}",
                             "position": pos, "price": price,
                             "xp": round(price * 0.5 + float(rng.normal(0, 1)), 2)})
    return pd.DataFrame(rows)


def _assert_legal(s: squad.Squad, budget: float = config.BUDGET) -> None:
    p = s.players
    assert len(p) == config.SQUAD_SIZE
    assert p["position"].value_counts().to_dict() == config.POSITION_QUOTA
    assert p["team_name"].value_counts().max() <= config.MAX_PER_CLUB
    assert p["price"].sum() <= budget + 1e-9
    xi = s.xi
    assert len(xi) == config.XI_SIZE
    for pos, (lo, hi) in config.FORMATION_BOUNDS.items():
        assert lo <= (xi["position"] == pos).sum() <= hi
    assert p["captain"].sum() == 1 and p["vice"].sum() == 1
    assert xi["captain"].any() and xi["vice"].any()
    assert not (p["captain"] & p["vice"]).any()


def test_best_squad_obeys_every_rule_and_captains_the_best_starter():
    s = squad.pick_squad(_candidates())
    _assert_legal(s)
    assert s.xi.loc[s.xi["captain"], "xp"].iloc[0] == s.xi["xp"].max()
    assert s.formation.count("-") == 2 and s.status == "Optimal"
    assert s.xp == pytest.approx(s.xi["xp"].sum() + s.xi["xp"].max())


def test_more_budget_never_hurts_and_forced_choices_are_honoured():
    c = _candidates()
    tight = squad.pick_squad(c, budget=90.0)
    _assert_legal(tight, budget=90.0)
    assert squad.pick_squad(c, budget=120.0).xp >= tight.xp

    best = squad.pick_squad(c)
    star = best.xi.sort_values("xp").iloc[-1]["id"]                 # drop the best starter
    without = squad.pick_squad(c, exclude=(star,))
    assert star not in set(without.players["id"]) and without.xp <= best.xp
    cheap = c[c["position"] == "FWD"].nsmallest(1, "price")["id"].iloc[0]
    assert cheap in set(squad.pick_squad(c, must_include=(cheap,)).players["id"])


def test_club_limit_binds():
    """Club0's attackers are the best by far; only 3 of them may be picked."""
    c = _candidates()
    c.loc[(c["team_name"] == "Club0") & (c["position"] != "GK"), "xp"] += 20
    picked = squad.pick_squad(c).players
    assert (picked["team_name"] == "Club0").sum() == config.MAX_PER_CLUB


def test_impossible_budget_is_reported():
    with pytest.raises(ValueError, match="no legal squad"):
        squad.pick_squad(_candidates(), budget=20.0)


# --------------------------------------------------------------------------- #
# multi-week planner (Section 5)
# --------------------------------------------------------------------------- #
from fpl_optimiser.optimise import planner  # noqa: E402


def _weeks(c: pd.DataFrame, gws=(1, 2, 3)) -> pd.DataFrame:
    """The same xp every gameweek (tests then change single players)."""
    return pd.concat([c[["id", "xp"]].assign(gw=g) for g in gws], ignore_index=True)


def _my_team(c: pd.DataFrame):
    s = squad.pick_squad(c)
    return list(s.players["id"]), round(config.BUDGET - s.cost, 1)


def test_selling_price_keeps_half_the_profit():
    assert planner.selling_price(7.0, 7.3) == 7.1      # +0.3 -> keep 0.1 (rounded down)
    assert planner.selling_price(7.0, 7.4) == 7.2
    assert planner.selling_price(5.5, 5.6) == 5.5      # +0.1 -> keep nothing
    assert planner.selling_price(7.0, 6.8) == 6.8      # a fall is taken in full


def test_no_transfers_keeps_the_squad_every_week():
    c = _candidates()
    mine, bank = _my_team(c)
    p = planner.plan(_weeks(c), c, mine, bank, free_transfers=1, max_transfers_per_gw=0)
    for w in p.weeks:
        assert set(w.team["id"]) == set(mine) and w.bought.empty and w.hits == 0
    assert [w.free_transfers for w in p.weeks] == [1, 2, 3]   # unused transfers roll over


def test_injured_starter_is_replaced_with_the_free_transfer():
    c = _candidates()
    mine, bank = _my_team(c)
    hurt = squad.pick_squad(c).xi.sort_values("xp").iloc[-1]["id"]   # the best starter
    xp = _weeks(c)
    xp.loc[xp["id"] == hurt, "xp"] = 0.0                               # out for every week
    p = planner.plan(xp, c, mine, bank, free_transfers=1)
    first = p.weeks[0]
    assert hurt in set(first.sold["id"]) and len(first.bought) == 1 and first.hits == 0
    assert hurt not in set(first.team["id"]) and first.bank_after >= 0
    assert p.hit_cost == 0 and p.net == pytest.approx(p.xp)


@pytest.mark.parametrize("gain, takes_hit", [(5.0, False), (8.0, True)])
def test_a_hit_is_taken_only_when_it_beats_the_threshold(gain, takes_hit):
    """A controlled market: my 15 (a worthless bench, two injured starting
    midfielders) plus exactly two spare midfielders worth `gain`. The free transfer
    brings one in; the second costs a -4 hit, worth it only above
    HIT_THRESHOLD_POINTS (6)."""
    layout = (["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)
    bench = {1, 6, 11, 14}                       # one GK, DEF, MID, FWD: worth nothing
    injured = {9, 10}                            # two starting midfielders
    mine = pd.DataFrame({"id": range(15), "position": layout,
                         "team_name": [f"Club{i % 6}" for i in range(15)], "price": 5.0})
    mine["xp"] = [0.0 if i in bench | injured else 5.0 for i in range(15)]
    spares = pd.DataFrame({"id": [100, 101], "position": "MID", "team_name": ["Club6", "Club7"],
                           "price": 5.0, "xp": gain})
    market = pd.concat([mine, spares], ignore_index=True).assign(web_name=lambda d: d["id"].astype(str))
    p = planner.plan(market[["id", "xp"]].assign(gw=1), market, list(range(15)), bank=0.0,
                     free_transfers=1, ft_value=0.0)
    week = p.weeks[0]
    assert len(week.bought) == (2 if takes_hit else 1)       # the free one always happens
    assert set(week.bought["id"]) <= {100, 101}
    assert (week.hits == 1) is takes_hit
    assert p.hit_cost == (config.TRANSFER_HIT_COST if takes_hit else 0)
