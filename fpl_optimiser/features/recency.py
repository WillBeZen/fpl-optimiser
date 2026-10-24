"""Exponential recency weighting.

A match played `g` gameweeks before the one we're predicting is weighted by

    weight = 0.5 ** (g / half_life)

So at g = half_life it counts half as much as the most recent match; at
2*half_life, a quarter. A smaller half_life reacts faster to current form; a
larger one trusts the longer record. The default lives in config.RECENCY_HALF_LIFE_GW.

These are pure functions over numbers/arrays — the "which matches count" logic
(strictly-prior, no leakage) lives in features.build.
"""
from __future__ import annotations

import numpy as np

import config


def decay_weight(gameweeks_ago, half_life: float = config.RECENCY_HALF_LIFE_GW):
    """Weight(s) for observation(s) `gameweeks_ago` in the past."""
    g = np.asarray(gameweeks_ago, dtype=float)
    return 0.5 ** (g / float(half_life))


def weighted_mean(values, gameweeks_ago,
                  half_life: float = config.RECENCY_HALF_LIFE_GW) -> float:
    """Recency-weighted mean of `values`, ignoring NaNs. NaN if nothing usable."""
    v = np.asarray(values, dtype=float)
    w = decay_weight(gameweeks_ago, half_life)
    mask = ~np.isnan(v)
    if not mask.any() or w[mask].sum() == 0:
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))