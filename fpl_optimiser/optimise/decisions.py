"""Decision backtest — score the teams, captains and transfers each method would have
picked, not just its point predictions.

Everything is driven by the honest replay (model/evaluation.py): at each past
deadline every method's predictions use only what was known then, and players carry
their price at that deadline. Two tests:

  weekly picks  each gameweek, the best legal squad (15 players, £100m, 3 per club),
                starting XI, captain and vice from that method's predictions, as if
                with a free hit every week. Scored on FPL's real points with its real
                rules: automatic substitutes from the bench, the vice-captain's
                double if the captain didn't play.
  season        one squad carried through the gameweeks: picked from scratch at the
                first deadline, then each week the planner (optimise/planner.py, no
                chips) decides transfers, banking and hits from that method's
                multi-week predictions, with FPL's selling prices. Scored on real
                points minus 4 per hit; "transfer gain" compares it with keeping the
                first squad all along.

The hindsight-best team (the best legal squad picked knowing the real points) is an
upper bound for reference, not a target.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from fpl_optimiser.ingest.my_team import selling_price
from fpl_optimiser.model import evaluation
from fpl_optimiser.optimise import planner, squad

METHODS = ["xp_blend", "xp", "xp_events", "pos_price_p", "season_avg", "ewma"]   # simulated
HINDSIGHT = "hindsight"
FORMATION_MIN = {pos: lo for pos, (lo, _) in config.FORMATION_BOUNDS.items()}
WORKERS = 4


# --------------------------------------------------------------------------- #
# FPL's scoring of a picked team
# --------------------------------------------------------------------------- #
def realised(team: pd.DataFrame, points: dict, minutes: dict) -> dict:
    """A team's real gameweek score with FPL's rules. `team` is a team sheet (role,
    bench_order, position, captain, vice, id); points / minutes: id -> real value.

    Starters who didn't play are replaced from the bench in order, a goalkeeper only
    by the reserve goalkeeper, an outfielder only if the formation stays legal (at
    least 3 DEF, 2 MID, 1 FWD) and the substitute played. The captain's points count
    twice; if the captain didn't play, the vice-captain's do."""
    xi = team[team["role"] == "XI"].copy()
    bench = team[team["role"] == "bench"].sort_values("bench_order")
    played = {i: minutes.get(i, 0) > 0 for i in team["id"]}
    on = list(xi["id"])
    pos = dict(zip(team["id"], team["position"]))
    subs = []
    for out in [i for i in on if not played[i]]:
        for _, b in bench.iterrows():
            if b["id"] in on or b["id"] in [s for _, s in subs] or not played[b["id"]]:
                continue
            if (pos[out] == "GK") != (b["position"] == "GK"):
                continue
            trial = [x for x in on if x != out] + [b["id"]]
            counts = pd.Series([pos[x] for x in trial]).value_counts()
            if all(counts.get(p, 0) >= lo for p, lo in FORMATION_MIN.items()):
                on = trial
                subs.append((out, b["id"]))
                break
    cap = team.loc[team["captain"], "id"]
    vice = team.loc[team["vice"], "id"]
    cap_id = int(cap.iloc[0]) if len(cap) else None
    vice_id = int(vice.iloc[0]) if len(vice) else None
    doubled = cap_id if cap_id is not None and played.get(cap_id) else (
        vice_id if vice_id is not None and played.get(vice_id) else None)
    base = float(sum(points.get(i, 0) for i in on))
    extra = float(points.get(doubled, 0)) * (config.CAPTAIN_MULTIPLIER - 1) if doubled is not None else 0.0
    best_cap = max((points.get(i, 0) for i in on), default=0)
    return {"points": base + extra, "captain_points": float(points.get(doubled, 0)) if doubled is not None else 0.0,
            "captain_id": doubled, "best_captain_points": float(best_cap), "subs": len(subs)}


# --------------------------------------------------------------------------- #
# Inputs at each deadline
# --------------------------------------------------------------------------- #
def week_points(ev, r: int) -> tuple[dict, dict]:
    """Real points and minutes of gameweek r per player (doubles added up)."""
    fw = ev.first_week[ev.first_week["round"] == r]
    g = fw.groupby("id")[["target_points", "minutes"]].sum()
    return g["target_points"].to_dict(), g["minutes"].to_dict()


def horizon_xp(ev, r: int, method: str) -> pd.DataFrame:
    """One row per (id, gw) of method's predictions made at deadline r, over its horizon."""
    fx = ev.horizons[r]
    return (fx.groupby(["id", "round"])[method].sum().reset_index()
            .rename(columns={"round": "gw", method: "xp"}))


def candidates(players_r: pd.DataFrame, xp: pd.Series) -> pd.DataFrame:
    """Players at r with the given expected points (index = id); no prediction -> 0."""
    c = players_r[["id", "web_name", "team_name", "position", "price"]].copy()
    c["xp"] = c["id"].map(xp).fillna(0.0)
    return c


# --------------------------------------------------------------------------- #
# Test 1: a free hit every week
# --------------------------------------------------------------------------- #
def weekly_picks(ev, cur, methods=METHODS) -> pd.DataFrame:
    """Each gameweek and method: the best legal 15 / XI / captain from its own
    predictions, scored on real points; plus the hindsight-best team (its bench is
    worth nothing, so it is the most an XI and captain could have scored)."""
    rows = []
    for r in ev.rounds:
        players_r = evaluation.players_at(cur, r)
        pts, mins = week_points(ev, r)
        first = ev.first_week[ev.first_week["round"] == r]
        for m in list(methods) + [HINDSIGHT]:
            xp = (pd.Series(pts) if m == HINDSIGHT else first.groupby("id")[m].sum())
            s = squad.pick_squad(candidates(players_r, xp), bench_weight=0.0 if m == HINDSIGHT else config.BENCH_WEIGHT)
            res = realised(s.players, pts, mins)
            rows.append({"method": m, "round": r, "predicted": s.xp, "cost": s.cost, **res})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Test 2: a season with transfers
# --------------------------------------------------------------------------- #
@dataclass
class Season:
    method: str
    weeks: pd.DataFrame          # one row per gameweek: points, hits, transfers, captain, bank, ...
    hold: pd.DataFrame           # the same first squad kept all along (best XI / captain each week)


def _xi_for(team_ids: list[int], players_r: pd.DataFrame, xp: pd.Series) -> squad.Squad:
    """The best XI, captain and vice from a fixed 15. Neither the budget nor the club
    limit applies: a player's move between clubs can leave a kept squad with four."""
    c = candidates(players_r[players_r["id"].isin(team_ids)], xp)
    return squad.pick_squad(c, budget=1e9, max_per_club=config.SQUAD_SIZE)


def simulate(ev, cur, method: str, horizon_decay: float = config.HORIZON_DECAY) -> Season:
    """Carry one squad through the replayed gameweeks, the planner deciding transfers
    from `method`'s predictions at each deadline (see module docstring)."""
    rounds = ev.rounds
    r0 = rounds[0]
    p0 = evaluation.players_at(cur, r0)
    xp0 = horizon_xp(ev, r0, method)
    weighted = xp0.assign(w=horizon_decay ** (xp0["gw"] - r0)).assign(v=lambda d: d["xp"] * d["w"])
    start = squad.pick_squad(candidates(p0, weighted.groupby("id")["v"].sum()))
    team = list(start.players["id"])
    bank = round(config.BUDGET - start.cost, 1)
    bought_for = dict(zip(start.players["id"], start.players["price"]))
    free = config.FREE_TRANSFERS_PER_GW
    rows, hold_rows = [], []
    first_squad = None
    for r in rounds:
        players_r = evaluation.players_at(cur, r)
        missing = [i for i in set(team) | set(first_squad or []) if i not in set(players_r["id"])]
        if missing:                      # a squad player missing from r's records: today's listing
            players_r = pd.concat([players_r, cur.players[cur.players["id"].isin(missing)]], ignore_index=True)
        pts, mins = week_points(ev, r)
        xp_r = horizon_xp(ev, r, method)
        now = xp_r[xp_r["gw"] == r].set_index("id")["xp"]
        if r == r0:
            sheet = _xi_for(team, players_r, now).players
            # a squad picked from scratch, as at the start of a season: one free transfer next week
            moves_out, moves_in, hits, bank_after, free_after = [], [], 0, bank, config.FREE_TRANSFERS_PER_GW
            first_squad = team[:]
        else:
            price_now = players_r.set_index("id")["price"]
            sell = {i: selling_price(bought_for[i], float(price_now.get(i, bought_for[i]))) for i in team}
            p = planner.plan(xp_r, players_r[["id", "web_name", "team_name", "position", "price"]], team, bank,
                             free, sell, chips=None)
            wk = p.weeks[0]
            sheet = wk.team
            moves_out, moves_in = list(wk.sold["id"]), list(wk.bought["id"])
            hits, bank_after, free_after = wk.hits, wk.bank_after, wk.free_after
            for i in moves_out:
                bought_for.pop(i, None)
            for i in moves_in:
                bought_for[i] = float(price_now[i])
            team = list(sheet["id"])
        res = realised(sheet, pts, mins)
        rows.append({"method": method, "round": r, "transfers": len(moves_in), "hits": hits,
                     "hit_cost": hits * config.TRANSFER_HIT_COST, "net": res["points"] - hits * config.TRANSFER_HIT_COST,
                     "bank": bank_after, "free_after": free_after, **res})
        bank, free = bank_after, free_after
        held = _xi_for(first_squad, players_r, now).players
        hold_rows.append({"method": method, "round": r, **realised(held, pts, mins)})
    return Season(method=method, weeks=pd.DataFrame(rows), hold=pd.DataFrame(hold_rows))


def season_test(ev, cur, methods=METHODS) -> dict[str, Season]:
    """simulate() for each method, side by side (the solver runs as its own process)."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return dict(zip(methods, pool.map(lambda m: simulate(ev, cur, m), methods)))


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def weekly_summary(picks: pd.DataFrame) -> pd.DataFrame:
    """Per method: average real points a gameweek (and spread), captain points, how
    often the captain was the XI's best scorer, and the share of hindsight's points."""
    best = picks[picks["method"] == HINDSIGHT].set_index("round")["points"]
    rows = []
    for m, d in picks.groupby("method", sort=False):
        d = d.set_index("round")
        rows.append({"method": m, "points": d["points"].mean(), "points_sd": d["points"].std(ddof=1),
                     "captain_points": d["captain_points"].mean(),
                     "captain_best": float((d["captain_points"] >= d["best_captain_points"]).mean()),
                     "of_hindsight": float(d["points"].sum() / best.reindex(d.index).sum()),
                     "gameweeks": len(d)})
    return pd.DataFrame(rows)


def season_summary(seasons: dict[str, Season]) -> pd.DataFrame:
    """Per method: total points after hits, transfers, hits, captain points, and the
    gain from transferring over keeping the first squad."""
    rows = []
    for m, s in seasons.items():
        w, h = s.weeks, s.hold
        rows.append({"method": m, "net": w["net"].sum(), "points": w["points"].sum(), "hit_cost": w["hit_cost"].sum(),
                     "transfers": int(w["transfers"].sum()), "hits": int(w["hits"].sum()),
                     "captain_points": w["captain_points"].sum(), "held": h["points"].sum(),
                     "transfer_gain": w["net"].sum() - h["points"].sum(), "gameweeks": len(w)})
    return pd.DataFrame(rows)


def paired_weeks(per_gw: pd.DataFrame, value: str, reference: str = evaluation.REFERENCE) -> pd.DataFrame:
    """Reference minus each method, gameweek by gameweek: mean, SE, 90% interval, wins."""
    from scipy import stats

    wide = per_gw.pivot(index="round", columns="method", values=value)
    rows = []
    for m in wide.columns:
        if m in (reference, HINDSIGHT):
            continue
        d = (wide[reference] - wide[m]).dropna()
        k = len(d)
        se = float(d.std(ddof=1) / np.sqrt(k)) if k > 1 else np.nan
        half = float(stats.t.ppf(0.95, k - 1) * se) if k > 1 else np.nan
        rows.append({"method": m, "mean_diff": float(d.mean()), "se": se, "low": float(d.mean()) - half,
                     "high": float(d.mean()) + half, "reference_better": int((d > 0).sum()), "gameweeks": k})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Running it (cached: the results only change when the replay's predictions do)
# --------------------------------------------------------------------------- #
VERSION = 1                                  # bump when the decision logic changes
CACHE_DIR = config.PROCESSED_DIR / "decisions"


def fingerprint(ev, methods=METHODS) -> str:
    """Identifies everything the decision test depends on: the replay's predictions
    and real points, the deadlines, the planner's dials and this module's version."""
    import hashlib

    fw = ev.first_week[["id", "round", "target_points", "minutes"] + list(methods)]
    hz = pd.concat([ev.horizons[r][["id", "round"] + list(methods)].assign(deadline=r) for r in ev.rounds])
    h = hashlib.sha256()
    for frame in (fw, hz):
        h.update(pd.util.hash_pandas_object(frame.round(9), index=False).values.tobytes())
    dials = (VERSION, ev.season, tuple(ev.rounds), config.BUDGET, config.HORIZON_DECAY, config.HIT_THRESHOLD_POINTS,
             config.FREE_TRANSFER_VALUE, config.BENCH_WEIGHT, config.MAX_BANKED_TRANSFERS, config.FREE_TRANSFERS_PER_GW,
             config.TRANSFER_HIT_COST, config.CAPTAIN_MULTIPLIER, tuple(methods))
    h.update(repr(dials).encode())
    return h.hexdigest()[:16]


def run(ev, cur, methods=METHODS, cache: bool = True, cache_dir=None) -> dict:
    """Both tests, reusing the last results if nothing they depend on has changed.
    Returns {"picks", "seasons", "fingerprint", "cached"}."""
    import pickle

    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    key = fingerprint(ev, methods)
    path = cache_dir / f"{ev.season}.pkl"
    if cache and path.exists():
        try:
            saved = pickle.loads(path.read_bytes())
            if saved.get("fingerprint") == key:
                return {**saved, "cached": True}
        except Exception:                   # unreadable cache: recompute
            pass
    out = {"picks": weekly_picks(ev, cur, methods), "seasons": season_test(ev, cur, methods), "fingerprint": key}
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(out))
    return {**out, "cached": False}
