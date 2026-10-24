"""Candidate pool — which players are worth giving to the solver at all.

Most of FPL's ~700 players can never be in an optimal squad: for each of them
there are plenty of others at the same position who cost no more AND are expected
to score at least as much in every gameweek of the horizon. Dropping those
"dominated" players before solving shrinks the integer program to a third of its
size, and it solves several times faster, with the same answer.

A player is dropped only when at least `quota + slack` same-position players
dominate him (quota = how many of that position a squad holds, e.g. 5 midfielders).
The slack is the safety margin: even if some dominating players are already in the
squad or blocked by the 3-per-club rule, enough are left to take his place. Players
you own are always kept (they sell at their own selling price, so price alone
doesn't decide for them). The planner's tests check the pruned and full solves agree.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

import config

SLACK = 5


def dominated_counts(pts: np.ndarray, price: np.ndarray) -> np.ndarray:
    """For each player j: how many others are no dearer and score at least as much in
    every gameweek, strictly better somewhere (more points or a lower price).

    pts   (n players x n gameweeks) expected points;  price (n,)"""
    ge = (pts[:, None, :] >= pts[None, :, :]).all(axis=2)      # ge[q, j]: q >= j every week
    gt = (pts[:, None, :] > pts[None, :, :]).any(axis=2)
    cheaper_or_equal = price[:, None] <= price[None, :]
    strictly_cheaper = price[:, None] < price[None, :]
    dom = ge & cheaper_or_equal & (gt | strictly_cheaper)
    np.fill_diagonal(dom, False)
    return dom.sum(axis=0)


def prune(xp: pd.DataFrame, players: pd.DataFrame, keep: Iterable[int] = (),
          slack: int = SLACK) -> set[int]:
    """Ids worth solving over: every player in `keep`, plus each player dominated by
    fewer than quota + slack others at his position.

    xp       one row per (id, gw) with column `xp`;  players: id, position, price"""
    pts = xp.pivot_table(index="id", columns="gw", values="xp", aggfunc="sum").fillna(0.0)
    info = players.set_index("id")
    kept = set(int(i) for i in keep)
    for pos, quota in config.POSITION_QUOTA.items():
        ids = info.index[(info["position"] == pos) & info.index.isin(pts.index)]
        if len(ids) == 0:
            continue
        counts = dominated_counts(pts.loc[ids].to_numpy(dtype=float),
                                  info.loc[ids, "price"].to_numpy(dtype=float))
        kept |= {int(i) for i in ids[counts < quota + slack]}
    return kept
