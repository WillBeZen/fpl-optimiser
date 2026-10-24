"""Your team from FPL's public endpoints — mock payloads, no network."""
from fpl_optimiser.ingest import fpl_api, my_team


def _row(gw, transfers=0, cost=0):
    return {"event": gw, "event_transfers": transfers, "event_transfers_cost": cost}


def test_free_transfers_replay_the_season():
    history = {
        "current": [_row(1, 0), _row(2, 0), _row(3, 1), _row(4, 3, cost=4),
                    _row(5, 10), _row(6), _row(7), _row(8), _row(9)],
        "chips": [{"name": "wildcard", "event": 5}],
    }
    expected = {2: 1,    # after picking the GW1 squad
                3: 2,    # GW2: none used -> banked
                4: 2,    # GW3: one used, one gained
                5: 1,    # GW4: three made, one paid (-4) -> both free ones spent
                6: 2,    # GW5: wildcard keeps them, and one more arrives
                7: 3, 8: 4, 9: 5,
                10: 5}   # capped at 5
    for next_gw, ft in expected.items():
        assert my_team.free_transfers(history, next_gw) == ft, next_gw


def test_free_transfers_for_a_late_joiner():
    assert my_team.free_transfers({"current": [_row(5), _row(6)]}, next_gw=7) == 2
    assert my_team.free_transfers({"current": []}, next_gw=1) == 1


BOOTSTRAP = {"elements": [
    {"id": 1, "now_cost": 74, "cost_change_start": 4},   # started at 7.0, now 7.4
    {"id": 2, "now_cost": 55, "cost_change_start": -5},  # started at 6.0, now 5.5
    {"id": 3, "now_cost": 102, "cost_change_start": 2},  # bought at 9.8 in GW4
]}
TRANSFERS = [{"element_in": 3, "element_in_cost": 98, "element_out": 9, "event": 4,
              "time": "2026-09-20T10:00:00Z"}]
PICKS = {"picks": [{"element": i} for i in (1, 2, 3)],
         "entry_history": {"event": 8, "bank": 15}, "active_chip": None}


def test_purchase_and_selling_prices():
    paid = my_team.purchase_prices([1, 2, 3], TRANSFERS, BOOTSTRAP)
    assert paid == {1: 7.0, 2: 6.0, 3: 9.8}
    team = my_team.assemble(123, {"current": [_row(1)]}, TRANSFERS, PICKS, BOOTSTRAP, next_gw=9)
    assert team.squad_ids == [1, 2, 3] and team.bank == 1.5 and team.gw == 8
    assert team.sell_prices == {1: 7.2, 2: 5.5, 3: 10.0}   # keep half of rises; falls in full


def test_load_uses_the_squad_a_free_hit_reverts_to(monkeypatch):
    calls = []

    def fake_get(path, ttl_hours=0):
        calls.append(path)
        if path.endswith("/history"):
            return {"current": [_row(g) for g in range(1, 9)]}
        if path.endswith("/transfers"):
            return []
        if path.endswith("/event/8/picks"):
            return {**PICKS, "active_chip": "freehit", "picks": [{"element": 99}]}
        return PICKS

    monkeypatch.setattr(fpl_api, "_cached_get", fake_get)
    team = my_team.load(123, BOOTSTRAP | {"elements": BOOTSTRAP["elements"]
                                          + [{"id": 99, "now_cost": 50}]}, next_gw=9)
    assert "entry/123/event/7/picks" in calls
    assert team.squad_ids == [1, 2, 3] and "free hit" in team.note
