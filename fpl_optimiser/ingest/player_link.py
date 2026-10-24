"""Link players (and clubs) across seasons — the bridge from 2025/26 to 2026/27.

FPL re-assigns every player `id` each season, so last season's id 1 is not this
season's id 1. Players are matched in three passes, strictest first. Each pass only
sees players the earlier passes left unmatched, and only accepts ONE-TO-ONE matches:
an ambiguous candidate is set aside and reported, never guessed.

  1. "code"                FPL's permanent player code (Opta's id): exact. A code
                           match whose birth dates disagree is rejected, not trusted.
  2. "name+dob"            same full name (accents/case ignored) and date of birth —
                           catches a player whose code changed (e.g. a temporary one).
  3. "dob+region+surname"  same birth date, nationality and surname — catches a
                           player whose listed name changed between seasons.

Whoever is left is a genuine arrival (new signing, promoted club, academy) or
departure. Clubs are linked on their permanent club `code`.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

_COLS = ["id", "code", "web_name", "first_name", "second_name", "birth_date", "region",
         "position", "team_name", "price", "total_points", "minutes"]


@dataclass
class PlayerLinks:
    links: pd.DataFrame        # one row per linked player: both seasons' ids, names, teams
    arrivals: pd.DataFrame     # this season's players with no match last season
    departures: pd.DataFrame   # last season's players with no match this season
    rejected: pd.DataFrame     # candidate matches refused (birth-date conflict / ambiguous)

    def summary(self) -> dict:
        return {
            "linked": len(self.links),
            "by_method": self.links["method"].value_counts().to_dict(),
            "arrivals": len(self.arrivals),
            "departures": len(self.departures),
            "rejected": len(self.rejected),
            "changed_team": int(self.links["changed_team"].sum()),
            "changed_position": int(self.links["changed_position"].sum()),
        }


# Letters Unicode can't split into "base letter + accent", so stripping accents
# alone would DELETE them (Ødegaard -> degaard). Spell them out first.
_TRANSLITERATE = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe",
                                "Œ": "OE", "ß": "ss", "ł": "l", "Ł": "L", "đ": "d",
                                "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ı": "i"})


def normalise_name(*parts) -> str:
    """'Raya Martín' -> 'raya martin': accents, case and punctuation ignored."""
    s = " ".join(p for p in parts if isinstance(p, str)).translate(_TRANSLITERATE)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]", " ", s.lower())).strip()


def _prepare(players: pd.DataFrame) -> pd.DataFrame:
    p = players[[c for c in _COLS if c in players.columns]].copy()
    p["name_key"] = [normalise_name(f, s) for f, s in zip(p["first_name"], p["second_name"])]
    p["surname"] = [(normalise_name(s).split() or [""])[-1] for s in p["second_name"]]
    return p


def _match(a: pd.DataFrame, b: pd.DataFrame, keys: list[str],
           method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join unmatched players on `keys`; keep one-to-one pairs, return the rest as rejected."""
    usable = lambda df: df.dropna(subset=keys)[lambda d: (d[keys].astype(str) != "").all(axis=1)]  # noqa: E731
    m = usable(a).merge(usable(b), on=keys, suffixes=("_prev", "_curr"))
    if "code" in keys:                       # keep both seasons' code columns uniformly
        m["code_prev"] = m["code_curr"] = m["code"]
    dup = m["id_prev"].duplicated(keep=False) | m["id_curr"].duplicated(keep=False)
    rejected = m[dup].assign(reason=f"ambiguous {method}")
    m = m[~dup]
    if method == "code":                     # a code match must also agree on birth date
        both = m["birth_date_prev"].notna() & m["birth_date_curr"].notna()
        clash = both & (m["birth_date_prev"] != m["birth_date_curr"])
        rejected = pd.concat([rejected, m[clash].assign(reason="code match, birth dates differ")])
        m = m[~clash]
    return m.assign(method=method), rejected


def link_players(prev: pd.DataFrame, curr: pd.DataFrame) -> PlayerLinks:
    """Link last season's players (`prev`) to this season's (`curr`).

    Both are fpl_api.parse_players tables (need id, code, first/second name,
    birth_date, region; position/team_name/price are carried along for context).
    """
    a, b = _prepare(prev), _prepare(curr)
    found, rejected = [], []
    for keys, method in ([["code"], "code"],
                         [["name_key", "birth_date"], "name+dob"],
                         [["birth_date", "region", "surname"], "dob+region+surname"]):
        taken_a = {i for f in found for i in f["id_prev"]}
        taken_b = {i for f in found for i in f["id_curr"]}
        m, r = _match(a[~a["id"].isin(taken_a)], b[~b["id"].isin(taken_b)], keys, method)
        found.append(m)
        rejected.append(r)

    links = pd.concat(found, ignore_index=True)
    for col in ("birth_date", "region"):       # a join key in some passes, suffixed in others
        links[col] = links[col].fillna(links[f"{col}_curr"]).fillna(links[f"{col}_prev"])
    links["changed_team"] = links["team_name_prev"] != links["team_name_curr"]
    links["changed_position"] = links["position_prev"] != links["position_curr"]
    keep = ["id_prev", "id_curr", "code_prev", "code_curr", "web_name_prev", "web_name_curr",
            "team_name_prev", "team_name_curr", "position_prev", "position_curr",
            "price_prev", "price_curr", "birth_date", "region", "method",
            "changed_team", "changed_position"]
    links = links[keep].sort_values(["method", "web_name_curr"]).reset_index(drop=True)

    cols = ["id", "code", "web_name", "position", "team_name", "price", "total_points", "minutes"]
    cols_a, cols_b = [c for c in cols if c in prev], [c for c in cols if c in curr]
    return PlayerLinks(
        links=links,
        arrivals=curr.loc[~curr["id"].isin(links["id_curr"]), cols_b].reset_index(drop=True),
        departures=prev.loc[~prev["id"].isin(links["id_prev"]), cols_a].reset_index(drop=True),
        rejected=pd.concat(rejected, ignore_index=True),
    )


def link_teams(prev_teams: pd.DataFrame, curr_teams: pd.DataFrame) -> pd.DataFrame:
    """Clubs matched on permanent `code`: each season's id and name, plus a
    `status` of stayed / promoted / relegated."""
    m = prev_teams[["id", "code", "name"]].merge(
        curr_teams[["id", "code", "name"]], on="code", how="outer",
        suffixes=("_prev", "_curr"), indicator=True)
    m["status"] = m["_merge"].astype(str).map(
        {"both": "stayed", "left_only": "relegated", "right_only": "promoted"})
    return m.drop(columns="_merge").sort_values(["status", "code"]).reset_index(drop=True)


def check_against_history_past(links: pd.DataFrame, past: pd.DataFrame,
                               prev_players: pd.DataFrame, prev_season: str) -> dict:
    """Independent check of the links using FPL's own record.

    Each player's element-summary this season carries `history_past`: his totals
    for earlier seasons, tagged with his permanent `element_code`. For every linked
    player that FPL reports a `prev_season` entry for, check that the code and the
    season's total points both agree with last season's snapshot.
    """
    label = prev_season.replace("-", "/")                  # '2025-26' -> '2025/26'
    if past.empty:
        return {"checked": 0, "code_agrees": 0, "points_agree": 0}
    p = past[past["season_name"] == label][["id", "element_code", "total_points"]]
    m = (links.merge(p, left_on="id_curr", right_on="id")
              .merge(prev_players[["id", "total_points"]], left_on="id_prev", right_on="id",
                     suffixes=("_fpl", "_snapshot")))
    return {
        "checked": len(m),
        "code_agrees": int((m["element_code"] == m["code_prev"]).sum()),
        "points_agree": int((m["total_points_fpl"] == m["total_points_snapshot"]).sum()),
    }
