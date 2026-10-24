"""What-ifs around the transfer plan — the planner's reasoning made visible.

The planner (planner.py) returns one answer. A scout wants to know how close the
alternatives were:

  transfer_ladder   the best plan with exactly 0, 1, 2, ... transfers this gameweek:
                    what each extra move is worth, and whether a -4 hit pays
  budget_value      the same plan with £0.5m / £1m / £2m more in the bank: what money
                    is worth to this squad right now
  alternatives      for each transfer this week, the next-best players who could
                    have come in instead (same position, affordable, club limit kept)
  transfer_reasons  each move side by side: who goes, who comes in, their expected
                    points week by week
  constraint_use    which rules are binding: clubs at their 3-player limit, money left
  objective_parts   the number the solver maximised, split into its pieces

Every what-if re-solves the same integer program with one thing changed, so the
answers are exact for the model, not estimates. The solver (CBC) runs as its own
process, so the re-solves run side by side in threads.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import config
from fpl_optimiser.optimise import planner


WORKERS = 4


def _plan(inputs: dict, **change) -> planner.Plan | None:
    """planner.plan with one thing changed; None if no legal plan exists."""
    try:
        return planner.plan(**{**inputs, **change})
    except ValueError:
        return None


def _plans(inputs: dict, changes: list[dict]) -> list[planner.Plan | None]:
    """Several what-ifs at once (each a separate solver process)."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(lambda change: _plan(inputs, **change), changes))


def plan_inputs(xp: pd.DataFrame, players: pd.DataFrame, squad_ids: list[int], bank: float,
                free_transfers: int, sell_prices: dict[int, float] | None = None) -> dict:
    """The planner's arguments, bundled so each what-if can change just one."""
    return dict(xp=xp, players=players, squad_ids=list(squad_ids), bank=bank,
                free_transfers=free_transfers, sell_prices=sell_prices or {})


def _moves(week: planner.Week) -> tuple[str, str]:
    return ", ".join(week.sold["web_name"]) or "-", ", ".join(week.bought["web_name"]) or "-"


def transfer_ladder(inputs: dict, max_moves: int | None = None) -> pd.DataFrame:
    """The best plan with exactly k transfers this gameweek, k = 0 .. max_moves
    (default: your free transfers + 1, so the first hit is shown too).

    `gain` is the change in the solver's objective from one more transfer: what
    the planner weighs (decayed points, hit threshold, value of banked transfers)."""
    top = max_moves if max_moves is not None else min(inputs["free_transfers"] + 1, config.SQUAD_SIZE)
    rows, prev = [], None
    for k, p in enumerate(_plans(inputs, [{"moves_now": k} for k in range(top + 1)])):
        if p is None:
            break
        first = p.weeks[0]
        out, inn = _moves(first)
        rows.append({"transfers": k, "hits": first.hits, "out": out, "in": inn,
                     "captain": first.captain, "xp_next": first.xp, "xp_horizon": p.xp,
                     "hit_cost": p.hit_cost, "net": p.net, "objective": p.objective,
                     "gain": None if prev is None else p.objective - prev})
        prev = p.objective
    return pd.DataFrame(rows)


def budget_value(inputs: dict, extras=(0.5, 1.0, 2.0), base: planner.Plan | None = None) -> pd.DataFrame:
    """How much better the plan gets with more money in the bank today."""
    base = base or _plan(inputs)
    rows = [{"extra": 0.0, "net": base.net, "objective": base.objective, "gain": 0.0,
             "in": _moves(base.weeks[0])[1]}]
    for extra, p in zip(extras, _plans(inputs, [{"bank": round(inputs["bank"] + e, 1)} for e in extras])):
        if p is not None:
            rows.append({"extra": extra, "net": p.net, "objective": p.objective,
                         "gain": p.objective - base.objective, "in": _moves(p.weeks[0])[1]})
    return pd.DataFrame(rows)


def _pairs(week: planner.Week) -> list[tuple[pd.Series, pd.Series]]:
    """Match each player out with the player in at the same position (dearest first)."""
    out = []
    for pos in config.POSITION_QUOTA:
        sold = week.sold[week.sold["position"] == pos].sort_values("sell_price", ascending=False)
        bought = week.bought[week.bought["position"] == pos].sort_values("price", ascending=False)
        out += list(zip((r for _, r in sold.iterrows()), (r for _, r in bought.iterrows())))
    return out


def horizon_points(xp: pd.DataFrame, decay: float = config.HORIZON_DECAY) -> pd.DataFrame:
    """Per player: expected points each gameweek, their plain total and the decayed
    total the planner weighs (index = player id)."""
    gws = sorted(xp["gw"].unique())
    grid = xp.pivot_table(index="id", columns="gw", values="xp", aggfunc="sum").reindex(columns=gws).fillna(0.0)
    out = grid.copy()
    out.columns = [f"GW{g}" for g in gws]
    out["total"] = grid.sum(axis=1)
    out["weighted"] = sum(grid[g] * decay ** k for k, g in enumerate(gws))
    return out


def transfer_reasons(week: planner.Week, xp: pd.DataFrame) -> pd.DataFrame:
    """Each move this gameweek: out vs in, expected points per gameweek and in total."""
    pts = horizon_points(xp)
    rows = []
    for o, i in _pairs(week):
        po, pi = pts.loc[o["id"]] if o["id"] in pts.index else None, pts.loc[i["id"]]
        row = {"Pos": o["position"], "Out": o["web_name"], "Sells for £m": o["sell_price"],
               "In": i["web_name"], "Costs £m": i["price"],
               "Money freed £m": round(o["sell_price"] - i["price"], 1)}
        for c in pts.columns:
            if c.startswith("GW"):
                row[c] = round((pi[c] - (po[c] if po is not None else 0.0)), 2)
        row["Gain over horizon"] = round(pi["total"] - (po["total"] if po is not None else 0.0), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def alternatives(week: planner.Week, xp: pd.DataFrame, players: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """For each transfer in `week`: the best players who could have come in instead,
    keeping every other move. Affordable means within the money that slot had (the
    bank after the week plus what the chosen player cost); the club limit is checked
    against the rest of the squad. Ranked by the planner's decayed horizon points."""
    pts = horizon_points(xp)
    info = players.set_index("id")
    squad = set(week.team["id"])
    clubs = week.team["team_name"].value_counts()
    rows = []
    for o, chosen in _pairs(week):
        budget = round(week.bank_after + chosen["price"], 1)
        club_count = clubs.copy()
        club_count[chosen["team_name"]] -= 1
        pool = info[(info["position"] == chosen["position"]) & ~info.index.isin(squad - {chosen["id"]})
                    & (info["price"] <= budget + 1e-9) & info.index.isin(pts.index)]
        ok = pool["team_name"].map(lambda c: club_count.get(c, 0) < config.MAX_PER_CLUB)
        pool = pool[ok].join(pts[["weighted", "total"]]).sort_values("weighted", ascending=False)
        best = pool.head(n)
        if chosen["id"] not in best.index and chosen["id"] in pool.index:
            best = pd.concat([best, pool.loc[[chosen["id"]]]])
        ref = pts.at[chosen["id"], "weighted"] if chosen["id"] in pts.index else 0.0
        for pid, r in best.iterrows():
            rows.append({"slot": f"{o['web_name']} out", "budget": budget, "player": r["web_name"],
                         "club": r["team_name"], "price": r["price"], "xp_horizon": r["total"],
                         "weighted": r["weighted"], "vs_chosen": r["weighted"] - ref,
                         "chosen": pid == chosen["id"]})
    return pd.DataFrame(rows)


def constraint_use(week: planner.Week) -> dict:
    """Which squad rules are binding after this week's moves."""
    team = week.team
    clubs = team["team_name"].value_counts().rename_axis("club").reset_index(name="players")
    clubs["at_limit"] = clubs["players"] >= config.MAX_PER_CLUB
    xi = team[team["role"] == "XI"]["position"].value_counts()
    formation = {pos: (int(xi.get(pos, 0)), lo, hi) for pos, (lo, hi) in config.FORMATION_BOUNDS.items()}
    return {"clubs": clubs, "bank_after": week.bank_after, "formation": formation,
            "free_before": week.free_transfers, "free_after": week.free_after, "hits": week.hits}


def objective_parts(p: planner.Plan, decay: float = config.HORIZON_DECAY,
                    bench_weight: float = config.BENCH_WEIGHT,
                    hit_penalty: float = config.HIT_THRESHOLD_POINTS,
                    ft_value: float = config.FREE_TRANSFER_VALUE) -> pd.DataFrame:
    """The solver's objective, piece by piece: each gameweek's XI, captain and bench
    (weighted by horizon decay), minus the hit penalty, plus banked transfers."""
    rows = []
    for k, w in enumerate(p.weeks):
        team = w.team
        xi = team.loc[team["role"] == "XI", "xp"].sum()
        cap = team.loc[team["captain"], "xp"].sum()
        bench = team.loc[team["role"] == "bench", "xp"].sum()
        weight = decay ** k
        rows.append({"part": f"GW{w.gw}", "weight": weight, "xi": xi, "captain": cap,
                     "bench": bench, "value": weight * (xi + cap + bench_weight * bench)})
    hits = sum(w.hits for w in p.weeks)
    rows.append({"part": "Hits", "weight": None, "xi": None, "captain": None, "bench": None,
                 "value": -hit_penalty * hits})
    rows.append({"part": "Banked transfers", "weight": None, "xi": None, "captain": None, "bench": None,
                 "value": ft_value * p.free_end})
    return pd.DataFrame(rows)
