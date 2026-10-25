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
from fpl_optimiser.ingest import my_team  # noqa: E402
from fpl_optimiser.optimise import planner  # noqa: E402


def _weeks(c: pd.DataFrame, gws=(1, 2, 3)) -> pd.DataFrame:
    """The same xp every gameweek (tests then change single players)."""
    return pd.concat([c[["id", "xp"]].assign(gw=g) for g in gws], ignore_index=True)


def _my_team(c: pd.DataFrame):
    s = squad.pick_squad(c)
    return list(s.players["id"]), round(config.BUDGET - s.cost, 1)


def test_selling_price_keeps_half_the_profit():
    assert my_team.selling_price(7.0, 7.3) == 7.1      # +0.3 -> keep 0.1 (rounded down)
    assert my_team.selling_price(7.0, 7.4) == 7.2
    assert my_team.selling_price(5.5, 5.6) == 5.5      # +0.1 -> keep nothing
    assert my_team.selling_price(7.0, 6.8) == 6.8      # a fall is taken in full


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


# --------------------------------------------------------------------------- #
# candidate pool, plan diagnostics and what-ifs
# --------------------------------------------------------------------------- #
from fpl_optimiser.optimise import pool, whatif  # noqa: E402


def _injured_market(n_clubs=20, seed=3):
    """A bigger market (enough players to prune), my squad, and its best starter out."""
    c = _candidates(seed=seed, n_clubs=n_clubs)
    mine, bank = _my_team(c)
    xp = _weeks(c)
    hurt = squad.pick_squad(c).xi.sort_values("xp").iloc[-1]["id"]
    xp.loc[xp["id"] == hurt, "xp"] = 0.0
    xp.loc[xp["gw"] == 2, "xp"] *= 1.1                       # weeks differ a little
    return c, mine, bank, xp


def test_dominated_players_are_pruned_but_owned_ones_kept():
    pts = np.array([[5.0, 5.0], [6.0, 6.0], [4.0, 7.0]])
    price = np.array([6.0, 5.0, 5.0])
    # player 0 is beaten by 1 (cheaper, more points every week); 2 beats nobody everywhere
    assert list(pool.dominated_counts(pts, price)) == [1, 0, 0]

    c, mine, bank, xp = _injured_market()
    dud = pd.DataFrame({"id": [999], "web_name": ["Dud"], "team_name": ["Club0"], "position": ["MID"],
                        "price": [12.0], "xp": [0.0]})
    c, xp = pd.concat([c, dud], ignore_index=True), pd.concat([xp, _weeks(dud)], ignore_index=True)
    kept = pool.prune(xp, c, keep=mine)
    assert set(mine) <= kept and len(kept) < len(c)
    assert 999 not in kept                                    # dear and pointless: never optimal
    assert 999 in pool.prune(xp, c, keep=mine + [999])        # ...unless you own him


def test_pruning_gives_the_same_plan_and_stats_are_recorded():
    c, mine, bank, xp = _injured_market()
    fast = planner.plan(xp, c, mine, bank, free_transfers=2)
    full = planner.plan(xp, c, mine, bank, free_transfers=2, prune=False)
    assert fast.objective == pytest.approx(full.objective, abs=1e-6)
    assert [set(w.team["id"]) for w in fast.weeks] == [set(w.team["id"]) for w in full.weeks]
    s = fast.stats
    assert s["players"] == len(c) and s["candidates"] < full.stats["candidates"] == len(c)
    assert s["variables"] > s["binary"] > 0 and s["constraints"] > 0 and s["solve_seconds"] >= 0


def test_free_transfers_carried_are_reported_week_by_week():
    c, mine, bank, xp = _injured_market()
    p = planner.plan(xp, c, mine, bank, free_transfers=2)
    for w, nxt in zip(p.weeks, p.weeks[1:]):
        assert w.free_after == nxt.free_transfers
        assert w.free_after == min(config.MAX_BANKED_TRANSFERS,
                                   w.free_transfers - len(w.bought) + w.hits + 1)
    assert p.free_end == p.weeks[-1].free_after


def test_transfer_ladder_and_budget_value():
    c, mine, bank, xp = _injured_market()
    inputs = whatif.plan_inputs(xp, c, mine, bank, free_transfers=1)
    best = planner.plan(**inputs)
    ladder = whatif.transfer_ladder(inputs)
    assert list(ladder["transfers"]) == [0, 1, 2]              # free transfers + 1
    assert ladder.loc[0, "out"] == "-" and pd.isna(ladder.loc[0, "gain"])
    assert ladder.loc[2, "hits"] == 1 and ladder.loc[2, "hit_cost"] >= config.TRANSFER_HIT_COST
    assert (ladder["objective"] <= best.objective + 1e-6).all()   # the plan is the best rung
    assert ladder["objective"].max() == pytest.approx(best.objective, abs=1e-6)
    assert ladder.loc[1, "gain"] > 0                           # replacing the injured starter pays

    money = whatif.budget_value(inputs, extras=(1.0, 3.0), base=best)
    assert list(money["extra"]) == [0.0, 1.0, 3.0]
    assert money["gain"].is_monotonic_increasing and money.loc[0, "gain"] == 0.0


def test_alternatives_reasons_constraints_and_objective_parts():
    c, mine, bank, xp = _injured_market()
    p = planner.plan(xp, c, mine, bank, free_transfers=1)
    week = p.weeks[0]
    assert len(week.bought) == 1
    alt = whatif.alternatives(week, xp, c, n=4)
    chosen = alt[alt["chosen"]]
    assert len(chosen) == 1 and chosen["player"].iloc[0] == week.bought["web_name"].iloc[0]
    pos = week.bought["position"].iloc[0]
    assert set(c.set_index("web_name").loc[alt["player"], "position"]) == {pos}
    assert (alt["price"] <= alt["budget"] + 1e-9).all()
    assert not set(alt["player"]) & (set(week.team["web_name"]) - set(chosen["player"]))

    why = whatif.transfer_reasons(week, xp)
    assert list(why["Out"]) == list(week.sold["web_name"]) and why["Gain over horizon"].iloc[0] > 0

    use = whatif.constraint_use(week)
    assert use["clubs"]["players"].sum() == config.SQUAD_SIZE and use["bank_after"] >= 0
    assert all(lo <= n <= hi for n, lo, hi in use["formation"].values())

    parts = whatif.objective_parts(p)
    assert parts["value"].sum() == pytest.approx(p.objective, abs=0.05)   # tie-breaks are tiny


# --------------------------------------------------------------------------- #
# chips: wildcard, free hit, bench boost, triple captain
# --------------------------------------------------------------------------- #
ALL_CHIPS = {(1, 19): config.CHIPS}          # every chip, first set (not expiring in GW1-3)


def test_chips_available_per_set_from_your_history():
    used = [{"name": "3xc", "event": 3}, {"name": "bboost", "event": 21}]
    assert my_team.chip_windows() == [(1, config.CHIP_SPLIT_GW), (config.CHIP_SPLIT_GW + 1, 38)]
    assert my_team.chips_available(used[:1], next_gw=9) == {
        (1, 19): ("wildcard", "free_hit", "bench_boost"), (20, 38): config.CHIPS}
    # after the split the first set has expired, chips unused or not
    assert my_team.chips_available(used, next_gw=25) == {(20, 38): ("wildcard", "free_hit", "triple_captain")}
    assert my_team.chips_available([], next_gw=19)[(1, 19)] == config.CHIPS   # GW19 is still the first half


def test_a_big_double_gameweek_triggers_the_bench_boost():
    c = _candidates()
    mine, bank = _my_team(c)
    xp = _weeks(c)
    flat = planner.plan(xp, c, mine, bank, free_transfers=1, chips={(1, 19): ("bench_boost",)})
    assert flat.chips == {}                                   # an ordinary bench isn't worth it
    xp.loc[xp["gw"] == 2, "xp"] *= 2                          # everyone plays twice in GW2
    p = planner.plan(xp, c, mine, bank, free_transfers=1, chips={(1, 19): ("bench_boost",)})
    assert p.chips == {"bench_boost": 2}
    week = p.weeks[1]
    team = week.team
    bench = team.loc[team["role"] == "bench", "xp"].sum()
    assert week.chip_gain == pytest.approx(bench) and bench * (1 - config.BENCH_WEIGHT) > 12
    assert week.xp == pytest.approx(team.loc[team["role"] == "XI", "xp"].sum()
                                    + team.loc[team["captain"], "xp"].sum() + bench)


def test_a_blank_gameweek_triggers_the_free_hit_and_the_squad_comes_back():
    c = _candidates()
    mine, bank = _my_team(c)
    xp = _weeks(c)
    blank = c.loc[c["id"].isin(mine), "team_name"].value_counts().index[:4]   # most of my clubs
    xp.loc[(xp["gw"] == 2) & xp["id"].isin(c.loc[c["team_name"].isin(blank), "id"]), "xp"] = 0.0
    p = planner.plan(xp, c, mine, bank, free_transfers=1, chips=ALL_CHIPS)
    assert p.chips == {"free_hit": 2}
    hit = p.weeks[1]
    assert len(hit.bought) >= 5 and hit.hits == 0 and hit.bank_after >= 0
    assert not set(hit.team.loc[hit.team["role"] == "XI", "team_name"]) & set(blank)
    assert set(p.weeks[2].team["id"]) == set(p.weeks[0].team["id"]) == set(mine)   # back, untouched
    assert [w.free_transfers for w in p.weeks] == [1, 2, 3]   # free transfers kept through it


def test_a_wildcard_rebuilds_a_broken_squad_without_hits():
    c = _candidates()
    mine, bank = _my_team(c)
    xp = _weeks(c)
    xp.loc[xp["id"].isin(mine[:10]), "xp"] = 0.0             # ten of my players out for good
    p = planner.plan(xp, c, mine, bank, free_transfers=1, chips={(1, 19): ("wildcard",)})
    assert p.chips == {"wildcard": 1}
    first = p.weeks[0]
    assert len(first.bought) >= 8 and first.hits == 0 and p.hit_cost == 0
    assert first.free_after == 2                             # the free transfer is kept, plus one


def test_one_chip_a_week_each_once_and_the_triple_captain_on_a_huge_week():
    c = _candidates()
    mine, bank = _my_team(c)
    xp = _weeks(c)
    star = squad.pick_squad(c).xi.sort_values("xp").iloc[-1]["id"]
    xp.loc[xp["gw"] == 2, "xp"] *= 2                          # a double: bench boost and triple captain both want GW2
    xp.loc[(xp["gw"] == 3) & (xp["id"] == star), "xp"] = 15.0   # ... and the star has a huge GW3
    p = planner.plan(xp, c, mine, bank, free_transfers=1,
                     chips={(1, 19): ("bench_boost", "triple_captain")})
    assert p.chips == {"bench_boost": 2, "triple_captain": 3}
    assert p.weeks[2].captain == c.set_index("id").at[star, "web_name"]
    assert p.weeks[2].chip_gain == pytest.approx(15.0 * (config.TRIPLE_CAPTAIN_MULTIPLIER - config.CAPTAIN_MULTIPLIER))


def test_the_split_rule_first_set_by_gw19_second_after():
    c = _candidates()
    mine, bank = _my_team(c)
    xp = _weeks(c, gws=(18, 19, 20, 21))
    xp.loc[xp["gw"] == 19, "xp"] *= 2.5                       # the biggest double, before the split
    xp.loc[xp["gw"] == 21, "xp"] *= 2.0                       # a smaller one after it
    second_only = {(1, 19): (), (20, 38): ("bench_boost",)}   # first-half bench boost already used
    p = planner.plan(xp, c, mine, bank, free_transfers=1, chips=second_only)
    assert p.chips == {"bench_boost": 21}                     # not GW19: that set's chip is gone
    both = {(1, 19): ("bench_boost",), (20, 38): ("bench_boost",)}
    p = planner.plan(xp, c, mine, bank, free_transfers=1, chips=both)
    assert [w.chip for w in p.weeks] == [None, "bench_boost", None, "bench_boost"]   # one per set


def test_a_chip_about_to_expire_is_played_for_any_gain():
    c = _candidates()
    mine, bank = _my_team(c)
    xp = _weeks(c, gws=(17, 18, 19))                          # an ordinary bench: below the threshold
    assert planner.plan(xp, c, mine, bank, 1, chips={(1, 38): ("bench_boost",)}).chips == {}
    expiring = planner.plan(xp, c, mine, bank, 1, chips={(1, 19): ("bench_boost",)})
    assert list(expiring.chips) == ["bench_boost"] and expiring.chip_charge == 0
    assert planner.chip_charge("bench_boost", 1, (1, 19), [17, 18, 19]) == 0
    assert planner.chip_charge("bench_boost", 1, (1, 38), [17, 18, 19]) == pytest.approx(
        config.HORIZON_DECAY * config.CHIP_THRESHOLD_POINTS["bench_boost"])
    with pytest.raises(ValueError, match="can't be played"):
        planner.plan(xp, c, mine, bank, 1, chips={(20, 38): ("bench_boost",)}, play_chips={"bench_boost": 18})
    with pytest.raises(ValueError, match="unknown chips"):
        planner.plan(xp, c, mine, bank, 1, chips={(1, 19): ("bboost",)})        # FPL's name, not ours


def test_chip_what_ifs_and_held_chips():
    c = _candidates()
    mine, bank = _my_team(c)
    xp = _weeks(c)
    xp.loc[xp["gw"] == 2, "xp"] *= 2
    inputs = whatif.plan_inputs(xp, c, mine, bank, free_transfers=1, chips={(1, 19): ("bench_boost",)})
    best = planner.plan(**inputs)
    grid = whatif.chip_weeks(inputs, chosen=best)
    assert list(grid["gw"]) == [1, 2, 3] and set(grid["chip"]) == {"bench_boost"}
    top = grid.loc[grid["gain"].idxmax()]
    assert top["gw"] == 2 and top["worth"] and top["planned"] and grid["planned"].sum() == 1
    assert top["gain"] - top["must_beat"] == pytest.approx(best.objective - planner.plan(
        **{**inputs, "chips": None}).objective, abs=1e-6)
    held = whatif.hold_chips(inputs, best)
    assert held["play_chips"] == {"bench_boost": 2} and planner.plan(**held).chips == {"bench_boost": 2}
    parts = whatif.objective_parts(best)
    assert "GW2 · bench boost" in set(parts["part"]) and "Chip thresholds" in set(parts["part"])
    assert parts["value"].sum() == pytest.approx(best.objective, abs=0.05)
