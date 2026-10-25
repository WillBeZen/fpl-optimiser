"""Your FPL team, from FPL's public endpoints (no login): squad, bank, free
transfers and selling prices — what the transfer planner (Section 5) starts from.

Endpoints, by manager (entry) id, the number in your team's URL:
  entry/{id}/history/          each gameweek's transfers and hits, plus chips played
  entry/{id}/transfers/        every transfer, with what you paid
  entry/{id}/event/{gw}/picks/ your 15 in that gameweek, and your bank

Two things FPL doesn't publish are reconstructed:
  free transfers  replayed over your season: 1 after your first gameweek, +1 a week,
                  capped at config.MAX_BANKED_TRANSFERS; free transfers used are spent;
                  a wildcard or free hit keeps what you had (and you still gain one).
  selling prices  what you paid (your last transfer in, or his start-of-season price
                  for someone in your original squad) -> FPL's rule: you keep half of
                  any rise, rounded down to £0.1m (selling_price).
Both are best estimates: one-off FPL top-ups (e.g. extra transfers around AFCON)
aren't visible, so the runner lets you override free transfers.

Chips still available come from the chips you've played (history): each chip set
covers a range of gameweeks (chip_windows: GW1 to config.CHIP_SPLIT_GW, then the
rest), a chip played in that range uses the set's copy, and a set whose last
gameweek has passed has expired (chips_available).

Fetching is thin and cached briefly; `assemble()` is pure and tested on mock payloads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from fpl_optimiser.ingest import fpl_api

_TTL_H = 0.25      # your team changes when you make transfers: keep the cache short

# FPL's chip names (history, picks) -> ours (config.CHIPS)
FPL_CHIPS = {"wildcard": "wildcard", "freehit": "free_hit", "bboost": "bench_boost",
             "3xc": "triple_captain"}


def selling_price(purchase: float, now: float) -> float:
    """FPL's rule: keep half of any rise (rounded down to £0.1m); take any fall in full."""
    p, n = round(purchase * 10), round(now * 10)
    return (p + (n - p) // 2) / 10 if n > p else n / 10


@dataclass
class MyTeam:
    manager_id: int
    gw: int                         # the gameweek these picks are from
    squad_ids: list[int]
    bank: float                     # £m
    free_transfers: int             # for the next gameweek (estimated)
    sell_prices: dict[int, float]   # player id -> what he sells for now (£m)
    purchase_prices: dict[int, float]
    chips_used: list[dict]          # [{'name': 'wildcard', 'event': 5}, ...]
    note: str = ""
    # chips you can still play, per set: {(first gw, last gw): ('wildcard', ...)}
    chips_available: dict[tuple[int, int], tuple[str, ...]] = field(default_factory=dict)


def chip_windows() -> list[tuple[int, int]]:
    """The gameweeks each chip set can be played in, as (first, last): with two sets,
    GW1 to config.CHIP_SPLIT_GW and the gameweeks after it."""
    if config.CHIP_SETS_PER_SEASON == 1:
        return [(1, config.GAMEWEEKS_IN_SEASON)]
    return [(1, config.CHIP_SPLIT_GW), (config.CHIP_SPLIT_GW + 1, config.GAMEWEEKS_IN_SEASON)]


def chips_available(chips_used: list[dict], next_gw: int) -> dict[tuple[int, int], tuple[str, ...]]:
    """The chips you can still play from `next_gw`, per set: {(first gw, last gw): chips}
    (config.CHIPS names, in that order). A set that ended before `next_gw` is left out:
    its unused chips expired. `chips_used` is FPL's history list ({'name', 'event'})."""
    out = {}
    for first, last in chip_windows():
        if last < next_gw:
            continue
        used = {FPL_CHIPS.get(c["name"], c["name"]) for c in chips_used if first <= c["event"] <= last}
        out[(first, last)] = tuple(c for c in config.CHIPS if c not in used)
    return out


def free_transfers(history: dict, next_gw: int) -> int:
    """Free transfers available for `next_gw`, by replaying the season so far."""
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    ft = None
    for row in sorted(history.get("current", []), key=lambda r: r["event"]):
        gw = row["event"]
        if gw >= next_gw:
            break
        if ft is None:                       # your first gameweek: squad picked for free
            ft = config.FREE_TRANSFERS_PER_GW
            continue
        if chips.get(gw) in ("wildcard", "freehit"):
            ft = min(config.MAX_BANKED_TRANSFERS, ft + config.FREE_TRANSFERS_PER_GW)
            continue
        paid = (row.get("event_transfers_cost") or 0) // config.TRANSFER_HIT_COST
        used_free = max((row.get("event_transfers") or 0) - paid, 0)
        ft = min(config.MAX_BANKED_TRANSFERS, max(ft - used_free, 0) + config.FREE_TRANSFERS_PER_GW)
    return ft if ft is not None else config.FREE_TRANSFERS_PER_GW


def purchase_prices(squad_ids: list[int], transfers: list[dict],
                    bootstrap: dict) -> dict[int, float]:
    """What you paid for each current player (£m)."""
    paid: dict[int, float] = {}
    for t in sorted(transfers, key=lambda t: (t["event"], t.get("time", ""))):
        paid[t["element_in"]] = t["element_in_cost"] / 10     # the latest purchase wins
    start = {e["id"]: (e["now_cost"] - (e.get("cost_change_start") or 0)) / 10
             for e in bootstrap["elements"]}
    return {i: paid.get(i, start.get(i)) for i in squad_ids}


def assemble(manager_id: int, history: dict, transfers: list[dict], picks: dict,
             bootstrap: dict, next_gw: int, note: str = "") -> MyTeam:
    """Pure: FPL payloads -> MyTeam."""
    squad = [p["element"] for p in picks["picks"]]
    now = {e["id"]: e["now_cost"] / 10 for e in bootstrap["elements"]}
    paid = purchase_prices(squad, transfers, bootstrap)
    sell = {i: selling_price(paid[i], now[i]) if paid.get(i) is not None else now[i] for i in squad}
    return MyTeam(manager_id=manager_id, gw=picks.get("entry_history", {}).get("event", next_gw - 1),
                  squad_ids=squad, bank=picks.get("entry_history", {}).get("bank", 0) / 10,
                  free_transfers=free_transfers(history, next_gw), sell_prices=sell,
                  purchase_prices=paid, chips_used=history.get("chips", []), note=note,
                  chips_available=chips_available(history.get("chips", []), next_gw))


def load(manager_id: int, bootstrap: dict, next_gw: int) -> MyTeam:
    """Fetch your team as it stands for `next_gw` (network; cached for 15 minutes).

    Uses the picks from the gameweek before `next_gw`. If that was a free hit, your
    squad reverts, so the picks from the gameweek before it are used instead.
    Transfers you've already made for `next_gw` aren't public until its deadline.
    """
    get = lambda path: fpl_api._cached_get(path, ttl_hours=_TTL_H)  # noqa: E731
    history = get(f"entry/{manager_id}/history")
    transfers = get(f"entry/{manager_id}/transfers")
    gw = next_gw - 1
    picks = get(f"entry/{manager_id}/event/{gw}/picks")
    note = ""
    if picks.get("active_chip") == "freehit" and gw > 1:
        picks = get(f"entry/{manager_id}/event/{gw - 1}/picks")
        note = f"GW{gw} was a free hit, so your squad reverts to GW{gw - 1}'s."
    return assemble(manager_id, history, transfers, picks, bootstrap, next_gw, note)
