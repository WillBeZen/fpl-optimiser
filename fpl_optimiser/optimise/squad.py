"""Single-gameweek squad optimiser — a mixed-integer linear program (PuLP + CBC).

Given each player's expected points, choose the 15-man squad, the starting XI and
the captain that maximise expected points under every FPL rule in config.py:

  - exactly 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD   (config.POSITION_QUOTA)
  - total price within the budget                    (config.BUDGET)
  - at most 3 players from any one club              (config.MAX_PER_CLUB)
  - an XI of 11 in a legal formation, 1 GK           (config.FORMATION_BOUNDS)
  - a captain (scores double) and a vice-captain, both in the XI

Objective: the XI's expected points, plus the captain's again, plus the bench at
config.BENCH_WEIGHT (bench players only score when a starter doesn't play).

Each player is a yes/no decision (a binary variable): in the squad? in the XI?
captain? The rules are linear constraints on those decisions, so the solver finds
the true best squad, not just a good-looking one.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
import pulp

import config

_VICE_TIEBREAK = 1e-3      # makes the best non-captain starter the vice-captain


def solve(prob: pulp.LpProblem) -> str:
    """Solve with the CBC solver bundled in PuLP 3; returns e.g. 'Optimal'.

    PuLP warns that 4.0 will stop bundling CBC. requirements.txt keeps PuLP < 4, so
    the notice is silenced here rather than printed on every run.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PULP_CBC_CMD is deprecated", category=DeprecationWarning)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[prob.status]


@dataclass
class Squad:
    players: pd.DataFrame     # the 15: role ("XI"/"bench"), captain/vice flags, bench order
    xp: float                 # expected points: XI + captain bonus
    cost: float
    formation: str            # e.g. "3-5-2"
    status: str               # the solver's verdict ("Optimal")

    @property
    def xi(self) -> pd.DataFrame:
        return self.players[self.players["role"] == "XI"]

    @property
    def captain(self) -> str:
        return self.players.loc[self.players["captain"], "web_name"].iloc[0]


def pick_squad(candidates: pd.DataFrame, xp: str = "xp", budget: float = config.BUDGET,
               must_include: tuple = (), exclude: tuple = (),
               bench_weight: float = config.BENCH_WEIGHT, max_per_club: int = config.MAX_PER_CLUB) -> Squad:
    """The best legal squad for one gameweek.

    candidates : one row per player: id, web_name, team_name, position, price and
                 an expected-points column named by `xp`
    must_include / exclude : player ids to force in or keep out
    max_per_club : the club limit (lifted to pick the XI of a squad you already own,
                 which a player's move between clubs can take past it)
    """
    c = candidates.dropna(subset=[xp, "price"]).reset_index(drop=True)
    ids = list(c.index)
    points = c[xp].to_dict()

    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    squad = prob.add_variable_dicts("squad", ids, cat="Binary")
    start = prob.add_variable_dicts("start", ids, cat="Binary")
    cap = prob.add_variable_dicts("captain", ids, cat="Binary")
    vice = prob.add_variable_dicts("vice", ids, cat="Binary")

    prob += (pulp.lpSum(points[i] * (start[i] + cap[i] + bench_weight * (squad[i] - start[i])
                                     + _VICE_TIEBREAK * vice[i]) for i in ids))

    prob += pulp.lpSum(squad[i] for i in ids) == config.SQUAD_SIZE
    prob += pulp.lpSum(start[i] for i in ids) == config.XI_SIZE
    prob += pulp.lpSum(c.at[i, "price"] * squad[i] for i in ids) <= budget
    for pos, quota in config.POSITION_QUOTA.items():
        in_pos = [i for i in ids if c.at[i, "position"] == pos]
        lo, hi = config.FORMATION_BOUNDS[pos]
        prob += pulp.lpSum(squad[i] for i in in_pos) == quota
        prob += pulp.lpSum(start[i] for i in in_pos) >= lo
        prob += pulp.lpSum(start[i] for i in in_pos) <= hi
    for club, rows in c.groupby("team_name").groups.items():
        prob += pulp.lpSum(squad[i] for i in rows) <= max_per_club
    prob += pulp.lpSum(cap[i] for i in ids) == 1
    prob += pulp.lpSum(vice[i] for i in ids) == 1
    for i in ids:
        prob += start[i] <= squad[i]
        prob += cap[i] <= start[i]
        prob += vice[i] <= start[i]
        prob += cap[i] + vice[i] <= 1
        if c.at[i, "id"] in must_include:
            prob += squad[i] == 1
        if c.at[i, "id"] in exclude:
            prob += squad[i] == 0

    status = solve(prob)
    if status != "Optimal":
        raise ValueError(f"no legal squad found (solver: {status}) — check budget and exclusions")

    chosen = [i for i in ids if squad[i].value() > 0.5]
    out = c.loc[chosen].copy()
    out["role"] = ["XI" if start[i].value() > 0.5 else "bench" for i in chosen]
    out["captain"] = [cap[i].value() > 0.5 for i in chosen]
    out["vice"] = [vice[i].value() > 0.5 for i in chosen]
    return _present(out, xp)


def _present(out: pd.DataFrame, xp: str) -> Squad:
    """Order the 15 like an FPL team sheet and summarise it."""
    order = {pos: n for n, pos in enumerate(config.POSITION_QUOTA)}
    out["_pos"] = out["position"].map(order)
    xi = out[out["role"] == "XI"].sort_values(["_pos", xp], ascending=[True, False])
    # bench: the reserve keeper first (FPL keeps it separate), then outfielders by xp
    bench = out[out["role"] == "bench"].assign(_gk=lambda d: d["position"] != "GK")
    bench = bench.sort_values(["_gk", xp], ascending=[True, False])
    bench["bench_order"] = range(len(bench))
    team = pd.concat([xi, bench]).drop(columns=["_pos", "_gk"])
    counts = xi["position"].value_counts()
    formation = "-".join(str(counts.get(p, 0)) for p in ["DEF", "MID", "FWD"])
    total = float(xi[xp].sum() + xi.loc[xi["captain"], xp].sum())
    return Squad(players=team.reset_index(drop=True), xp=total, cost=float(out["price"].sum()),
                 formation=formation, status="Optimal")
