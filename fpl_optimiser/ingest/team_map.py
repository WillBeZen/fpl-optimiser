"""Reconcile team names between FPL and FBref.

Team-level mapping is the v1 form of "ID reconciliation": opponent characteristics
are team-level, and the FPL API already carries per-player stats, so we don't yet
need fragile player-name matching across sources.

`normalise` collapses common spelling variants to a canonical key; `reconcile`
matches two name lists and REPORTS anything unmatched — so a promoted or renamed
team surfaces loudly rather than silently dropping out of the join.
"""
from __future__ import annotations

import re

# Canonical short keys for 2025/26 Premier League clubs. Update on promotion /
# relegation (for 2026/27, confirm the three promoted clubs once teams are known).
CANONICAL = {
    "arsenal", "aston villa", "bournemouth", "brentford", "brighton",
    "burnley", "chelsea", "crystal palace", "everton", "fulham", "leeds",
    "liverpool", "man city", "man utd", "newcastle", "nott'm forest",
    "sunderland", "tottenham", "west ham", "wolves",
}

# Spelling variants (as seen from FPL, FBref, Understat, etc.) -> canonical key.
ALIASES = {
    "manchester city": "man city",
    "manchester utd": "man utd", "manchester united": "man utd",
    "newcastle utd": "newcastle", "newcastle united": "newcastle",
    "nottingham forest": "nott'm forest", "nottm forest": "nott'm forest",
    "nott'ham forest": "nott'm forest",
    "tottenham hotspur": "tottenham", "spurs": "tottenham",
    "wolverhampton": "wolves", "wolverhampton wanderers": "wolves",
    "brighton & hove albion": "brighton", "brighton and hove albion": "brighton",
    "west ham united": "west ham",
    "leeds united": "leeds",
    "afc bournemouth": "bournemouth",
}


def normalise(name: str) -> str:
    s = name.strip().lower().replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    return ALIASES.get(s, s)


def reconcile(fpl_names, fbref_names) -> dict:
    """Match two name lists.

    Returns {'matched': {fpl_name: fbref_name}, 'fpl_only': [...], 'fbref_only': [...]}.
    An empty 'fpl_only' and 'fbref_only' means every team lined up.
    """
    fpl_norm = {normalise(n): n for n in fpl_names}
    fbref_norm = {normalise(n): n for n in fbref_names}
    matched, fpl_only, fbref_only = {}, [], []
    for key in sorted(set(fpl_norm) | set(fbref_norm)):
        in_fpl, in_fbref = key in fpl_norm, key in fbref_norm
        if in_fpl and in_fbref:
            matched[fpl_norm[key]] = fbref_norm[key]
        elif in_fpl:
            fpl_only.append(fpl_norm[key])
        else:
            fbref_only.append(fbref_norm[key])
    return {"matched": matched, "fpl_only": fpl_only, "fbref_only": fbref_only}
