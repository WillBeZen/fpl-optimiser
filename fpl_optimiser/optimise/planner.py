"""Multi-week transfer planner — one mixed-integer program over the whole horizon.

From your current squad, bank and free transfers, decide for each of the next few
gameweeks who to buy and sell, the XI and the captain, maximising expected points
over the horizon. Because every week is planned together, it can see that banking
a transfer now buys two next week, or that a -4 hit pays off over five weeks.

Each gameweek t has the single-gameweek squad rules (optimise/squad.py), plus:
  squad[t] = squad[t-1] + bought[t] - sold[t]     (squad[0] = your squad today)
  bank[t]  = bank[t-1] + sales - purchases >= 0   (your players sell at the prices
                                                   passed in: ingest/my_team.py)
  free transfers: hits[t] >= transfers[t] - free[t];
                  free[t+1] <= free[t] - transfers[t] + hits[t] + 1, and <= the cap

Objective, each gameweek weighted config.HORIZON_DECAY ** (weeks ahead):
  XI expected points + captain again + bench at config.BENCH_WEIGHT
  - config.HIT_THRESHOLD_POINTS per hit   (a -4 is taken only if it gains more than
                                           this: the margin for prediction error)
  + config.FREE_TRANSFER_VALUE per transfer still banked after the last gameweek
Chips (wildcard, free hit, bench boost, triple captain) are not planned yet.

Before solving, players who can't be optimal are dropped (optimise/pool.py): the
same answer, several times faster. `Plan.stats` records the problem's size and
solve time for the report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
import pulp

import config
from fpl_optimiser.optimise import pool
from fpl_optimiser.optimise.squad import _present, solve

_FT_TIEBREAK = 1e-4


@dataclass
class Week:
    gw: int
    sold: pd.DataFrame          # players out (with the price you'd get)
    bought: pd.DataFrame        # players in
    free_transfers: int         # available before this gameweek's transfers
    hits: int                   # paid transfers (each costs config.TRANSFER_HIT_COST)
    bank_after: float
    team: pd.DataFrame          # the 15, XI/bench, captain, ordered like a team sheet
    xp: float                   # XI + captain's double, this gameweek
    formation: str
    free_after: int = 0         # free transfers carried into the next gameweek

    @property
    def captain(self) -> str:
        return self.team.loc[self.team["captain"], "web_name"].iloc[0]


@dataclass
class Plan:
    weeks: list[Week] = field(default_factory=list)
    xp: float = 0.0             # expected points over the horizon (undiscounted, before hits)
    hit_cost: int = 0           # points spent on hits
    status: str = ""
    free_end: int = 0           # free transfers still banked after the last gameweek
    objective: float = 0.0      # the solver's objective (decayed, with penalties and bonus)
    stats: dict = field(default_factory=dict)   # problem size and solve time

    @property
    def net(self) -> float:
        return self.xp - self.hit_cost


def plan(xp: pd.DataFrame, players: pd.DataFrame, squad_ids: list[int], bank: float,
         free_transfers: int, sell_prices: dict[int, float] | None = None,
         max_transfers_per_gw: int | None = None, moves_now: int | None = None,
         decay: float = config.HORIZON_DECAY, hit_penalty: float = config.HIT_THRESHOLD_POINTS,
         ft_value: float = config.FREE_TRANSFER_VALUE,
         bench_weight: float = config.BENCH_WEIGHT, prune: bool = True) -> Plan:
    """The best transfer plan over the gameweeks in `xp`.

    xp            one row per (player id, gw) with column `xp` (predict.Forecast.xp)
    players       id, web_name, team_name, position, price (today's price)
    squad_ids     your 15 player ids now; bank in £m; free_transfers available now
    sell_prices   what each of your players sells for (default: today's price)
    max_transfers_per_gw  0 = "no transfers" (the baseline to compare a plan with)
    moves_now     exactly this many transfers in the first gameweek (for what-ifs)
    prune         drop players who can't be optimal first (pool.py); same answer, faster
    """
    started = time.perf_counter()
    gws = sorted(xp["gw"].unique())
    n_players = xp["id"].nunique()
    if prune:
        xp = xp[xp["id"].isin(pool.prune(xp, players, keep=squad_ids))]
    pts = xp.pivot_table(index="id", columns="gw", values="xp", aggfunc="sum").reindex(columns=gws)
    c = players.set_index("id").loc[lambda d: d.index.isin(pts.index) | d.index.isin(squad_ids)]
    pts = pts.reindex(c.index).fillna(0.0)
    ids = list(c.index)
    owned = {i: int(i in set(squad_ids)) for i in ids}
    buy_price = c["price"].to_dict()
    sell_price = {i: (sell_prices or {}).get(i, buy_price[i]) if owned[i] else buy_price[i] for i in ids}

    prob = pulp.LpProblem("plan", pulp.LpMaximize)
    T = range(len(gws))
    key = [(i, t) for i in ids for t in T]
    sq = prob.add_variable_dicts("squad", key, cat="Binary")
    st = prob.add_variable_dicts("start", key, cat="Binary")
    cp = prob.add_variable_dicts("captain", key, cat="Binary")
    vc = prob.add_variable_dicts("vice", key, cat="Binary")
    buy = prob.add_variable_dicts("buy", key, cat="Binary")
    sell = prob.add_variable_dicts("sell", key, cat="Binary")
    ft = prob.add_variable_dicts("free", list(range(len(gws) + 1)), lowBound=0,
                                 upBound=config.MAX_BANKED_TRANSFERS, cat="Integer")
    hits = prob.add_variable_dicts("hits", list(T), lowBound=0, cat="Integer")
    money = prob.add_variable_dicts("bank", list(T), lowBound=0)

    prob += (pulp.lpSum(decay ** t * pts.at[i, gws[t]] * (st[i, t] + cp[i, t] + 1e-3 * vc[i, t]
                                                          + bench_weight * (sq[i, t] - st[i, t]))
                        for i, t in key)
             - hit_penalty * pulp.lpSum(hits[t] for t in T)
             + ft_value * ft[len(gws)]
             # tie-break only: report the free transfers you'd really have each week
             # (otherwise any value up to the true count is equally "optimal")
             + _FT_TIEBREAK * pulp.lpSum(ft[t] for t in range(1, len(gws))))

    prob += ft[0] == min(max(int(free_transfers), 0), config.MAX_BANKED_TRANSFERS)
    for t in T:
        n_moves = pulp.lpSum(buy[i, t] for i in ids)
        prob += n_moves == pulp.lpSum(sell[i, t] for i in ids)
        prob += hits[t] >= n_moves - ft[t]
        prob += ft[t + 1] <= ft[t] - n_moves + hits[t] + config.FREE_TRANSFERS_PER_GW
        if max_transfers_per_gw is not None:
            prob += n_moves <= max_transfers_per_gw
        if moves_now is not None and t == 0:
            prob += n_moves == moves_now
        prob += money[t] == ((bank if t == 0 else money[t - 1])
                             + pulp.lpSum(sell_price[i] * sell[i, t] - buy_price[i] * buy[i, t] for i in ids))
        for i in ids:
            prev = owned[i] if t == 0 else sq[i, t - 1]
            prob += sq[i, t] == prev + buy[i, t] - sell[i, t]
            prob += buy[i, t] + sell[i, t] <= 1
            prob += st[i, t] <= sq[i, t]
            prob += cp[i, t] <= st[i, t]
            prob += vc[i, t] <= st[i, t]
            prob += cp[i, t] + vc[i, t] <= 1
        prob += pulp.lpSum(sq[i, t] for i in ids) == config.SQUAD_SIZE
        prob += pulp.lpSum(st[i, t] for i in ids) == config.XI_SIZE
        prob += pulp.lpSum(cp[i, t] for i in ids) == 1
        prob += pulp.lpSum(vc[i, t] for i in ids) == 1
        for pos, quota in config.POSITION_QUOTA.items():
            in_pos = [i for i in ids if c.at[i, "position"] == pos]
            lo, hi = config.FORMATION_BOUNDS[pos]
            prob += pulp.lpSum(sq[i, t] for i in in_pos) == quota
            prob += pulp.lpSum(st[i, t] for i in in_pos) >= lo
            prob += pulp.lpSum(st[i, t] for i in in_pos) <= hi
        for club, members in c.groupby("team_name").groups.items():
            prob += pulp.lpSum(sq[i, t] for i in members) <= config.MAX_PER_CLUB

    built = time.perf_counter()
    status = solve(prob)
    if status != "Optimal":
        raise ValueError(f"no legal plan found (solver: {status}) — check the squad and bank")

    out = Plan(status=status, free_end=int(round(ft[len(gws)].value())),
               objective=float(pulp.value(prob.objective)),
               stats={"players": n_players, "candidates": len(ids), "gameweeks": len(gws),
                      "variables": len(prob.variables()), "constraints": prob.numConstraints(),
                      "binary": sum(v.cat == pulp.LpInteger and v.upBound == 1 for v in prob.variables()),
                      "build_seconds": built - started, "solve_seconds": time.perf_counter() - built})
    info = c.reset_index()[["id", "web_name", "team_name", "position", "price"]]
    for t, gw in enumerate(gws):
        chosen = [i for i in ids if sq[i, t].value() > 0.5]
        team = info.set_index("id").loc[chosen].reset_index()
        team["xp"] = [pts.at[i, gw] for i in chosen]
        team["role"] = ["XI" if st[i, t].value() > 0.5 else "bench" for i in chosen]
        team["captain"] = [cp[i, t].value() > 0.5 for i in chosen]
        team["vice"] = [vc[i, t].value() > 0.5 for i in chosen]
        sheet = _present(team, "xp")
        moved_out = [i for i in ids if sell[i, t].value() > 0.5]
        moved_in = [i for i in ids if buy[i, t].value() > 0.5]
        week = Week(
            gw=int(gw),
            sold=info.set_index("id").loc[moved_out].reset_index().assign(
                sell_price=[sell_price[i] for i in moved_out], xp=[pts.at[i, gw] for i in moved_out]),
            bought=info.set_index("id").loc[moved_in].reset_index().assign(
                xp=[pts.at[i, gw] for i in moved_in]),
            free_transfers=int(round(ft[t].value())), hits=int(round(hits[t].value())),
            bank_after=round(money[t].value(), 1), team=sheet.players, xp=sheet.xp,
            formation=sheet.formation, free_after=int(round(ft[t + 1].value())))
        out.weeks.append(week)
    out.xp = sum(w.xp for w in out.weeks)
    out.hit_cost = config.TRANSFER_HIT_COST * sum(w.hits for w in out.weeks)
    return out
