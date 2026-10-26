"""Choosing settings on last season, then checking them on this one.

Tuning a setting on the weeks then quoted as evidence is circular (docs/08). So the
recency half-life and the blend's event-model weight are chosen on the last full
season, replayed honestly deadline by deadline (model/evaluation.py), and then
checked on this season's finished gameweeks. The rules were fixed before looking:

  half-life     each candidate in HALF_LIVES rebuilds the whole replay (the form
                features, and the event model's team ratings and player rates).
                Chosen: the lowest RMSE of the blend whose weight is chosen from
                earlier weeks (RULE_METHOD) over every player-gameweek; the half-life in
                config.py when the study runs stays unless another beats it by more
                than TIE.
  blend weight  at that half-life, the fixed event-model weight with the lowest RMSE
                over the same player-gameweeks (evaluation.WEIGHT_GRID).

The full replay is then run at the chosen settings, so last season's tables and its
decision test describe the planner's blend as it now is.

scripts/study_last_season.py runs the study (about 20 minutes; once a season, or
after a model change) and files it in data/processed/last_season/<season>.pkl. The
weekly run only reads it; the Accuracy and Decision test tabs show it.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from fpl_optimiser.model import evaluation

HALF_LIVES = (4.0, 6.0, 8.0, 12.0, 16.0)
TIE = 0.002
RULE_METHOD = "blend_wf"         # the half-life is chosen before the weight, so not on a fixed one
STUDY_DIR = config.PROCESSED_DIR / "last_season"
VERSION = 2


def _rmse(e: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(e.to_numpy(dtype=float)))))


def weight_curve(ev, grid=evaluation.WEIGHT_GRID) -> pd.Series:
    """RMSE over every player-gameweek of the replay, per fixed event-model weight."""
    pgw = evaluation.player_gameweeks(ev, "all").dropna(subset=["xp", "xp_events", "actual"])
    return pd.Series([_rmse((1 - w) * pgw["xp"] + w * pgw["xp_events"] - pgw["actual"]) for w in grid],
                     index=pd.Index([float(w) for w in grid], name="weight"), name="rmse")


def choose_half_life(rmse: dict[float, float], current: float, tie: float = TIE) -> float:
    """The lowest RMSE; the current half-life unless another beats it by more than `tie`."""
    best = min(rmse, key=rmse.get)
    if current in rmse and rmse[current] - rmse[best] <= tie:
        return float(current)
    return float(best)


def half_life_scores(ev) -> dict:
    """What the study keeps of one replay: every method's summary and per-gameweek
    scores over every player (and the planner's pool, if the replay built pools)."""
    out = {"weights": dict(ev.weights)}
    for pop in ("all", "pool"):
        if pop == "pool" and not any(ev.pools.values()):
            continue
        pgw = evaluation.player_gameweeks(ev, pop)
        gw = evaluation.by_gameweek(pgw)
        out[pop] = {"summary": evaluation.summary(gw, pgw), "gameweeks": gw}
    return out


def blend_rows(ev, weight: float, population: str = "all") -> pd.DataFrame:
    """Player-gameweeks with the blend at a fixed weight (`blend`) and the actual points."""
    pgw = evaluation.player_gameweeks(ev, population)
    return pgw.assign(blend=(1 - weight) * pgw["xp"] + weight * pgw["xp_events"])


def compare_blends(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """Two blends on the same player-gameweeks (blend_rows), gameweek by gameweek: RMSE
    and the real points of each one's top 11; new minus old, with a 90% interval."""
    from scipy import stats

    both = (new[["id", "round", "blend", "actual"]].merge(old[["id", "round", "blend"]], on=["id", "round"],
                                                         suffixes=("_new", "_old"))
            .dropna(subset=["blend_new", "blend_old", "actual"]))
    rows = []
    for r, d in both.groupby("round"):
        rows.append({"round": int(r), "rmse_new": _rmse(d["blend_new"] - d["actual"]),
                     "rmse_old": _rmse(d["blend_old"] - d["actual"]),
                     "top11_new": float(d.nlargest(11, "blend_new")["actual"].sum()),
                     "top11_old": float(d.nlargest(11, "blend_old")["actual"].sum())})
    gw = pd.DataFrame(rows)
    out = {"gameweeks": gw, "n": len(both)}
    for m in ("rmse", "top11"):
        diff = gw[f"{m}_new"] - gw[f"{m}_old"]
        k = len(diff)
        half = float(stats.t.ppf(0.95, k - 1) * diff.std(ddof=1) / np.sqrt(k)) if k > 1 else np.nan
        out[m] = {"new": _rmse(both["blend_new"] - both["actual"]) if m == "rmse" else float(gw["top11_new"].mean()),
                  "old": _rmse(both["blend_old"] - both["actual"]) if m == "rmse" else float(gw["top11_old"].mean()),
                  "mean_diff": float(diff.mean()), "low": float(diff.mean()) - half, "high": float(diff.mean()) + half,
                  "new_better": int(((diff < 0) if m == "rmse" else (diff > 0)).sum()), "gameweeks": k}
    return out


def run(season: str, half_lives=HALF_LIVES, before: tuple[float, float] | None = None,
        log=print) -> tuple[dict, object]:
    """The study on a finished season: a quick replay (first weeks only) at every
    candidate half-life, the half-life and then the weight chosen, and a full replay
    at the chosen settings (pools, the planning horizon) for the methods and the
    decision test. `before`: the (half-life, weight) the choice is checked against on
    this season, config.py's by default; the tie rule always protects config.py's
    half-life. Returns (study, the full replay)."""
    from fpl_optimiser.features import multi_season

    incumbent = float(config.RECENCY_HALF_LIFE_GW)
    before_h, before_w = (float(x) for x in (before or (config.RECENCY_HALF_LIFE_GW, config.EVENT_MODEL_WEIGHT)))
    tables, by_h, curves = {}, {}, {}
    for h in sorted({float(x) for x in half_lives} | {incumbent, before_h}):
        log(f"    {season} at half-life {h:g} ...")
        tables[h] = multi_season.build_all([season], half_life=h)
        quick = evaluation.run(season, tables[h], half_life=h, event_weight=before_w, horizon=1, with_pools=False)
        by_h[h], curves[h] = half_life_scores(quick), weight_curve(quick)
    rmse = {h: float(s["all"]["summary"].set_index("method").at[RULE_METHOD, "rmse"]) for h, s in by_h.items()}
    h_star = choose_half_life(rmse, incumbent)
    w_star = float(curves[h_star].idxmin())
    log(f"    {season}: the full replay at half-life {h_star:g}, weight {w_star:g} ...")
    ev = evaluation.run(season, tables[h_star], half_life=h_star, event_weight=w_star)
    study = {"season": season, "version": VERSION, "created": datetime.now(timezone.utc).isoformat(timespec="minutes"),
             "rounds": list(ev.rounds), "before": {"half_life": before_h, "weight": before_w}, "incumbent": incumbent,
             "half_lives": by_h, "rmse": rmse, "rule_method": RULE_METHOD, "weight_curve": curves[h_star],
             "methods": half_life_scores(ev), "chosen": {"half_life": h_star, "weight": w_star},
             "rules": (f"half-life: the lowest RMSE over every player-gameweek of the blend whose weight is chosen "
                       f"from earlier weeks, config.py's ({incumbent:g}) unless another beats it by more than {TIE}; "
                       f"weight: the lowest RMSE at that half-life")}
    return study, ev


def check_this_season(season: str, study: dict, log=print) -> dict:
    """The chosen half-life and weight against the settings before the study, on this
    season's finished gameweeks (every player, first weeks only)."""
    from fpl_optimiser.features import multi_season

    def replay(h):
        log(f"    {season} at half-life {h:g} ...")
        return evaluation.run(season, multi_season.build_all(half_life=h), half_life=h, horizon=1, with_pools=False)

    (h1, w1), (h0, w0) = ((study[k]["half_life"], study[k]["weight"]) for k in ("chosen", "before"))
    new = replay(h1)
    old = new if h0 == h1 else replay(h0)
    return {"season": season, "rounds": list(new.rounds), "new": {"half_life": h1, "weight": w1},
            "old": {"half_life": h0, "weight": w0}, **compare_blends(blend_rows(new, w1), blend_rows(old, w0))}


def path(season: str):
    return STUDY_DIR / f"{season}.pkl"


def save(study: dict) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    path(study["season"]).write_bytes(pickle.dumps(study))


def load(season: str) -> dict | None:
    """The filed study of `season`, or None (not run, unreadable or an older version)."""
    p = path(season)
    if not p.exists():
        return None
    try:
        study = pickle.loads(p.read_bytes())
    except Exception:
        return None
    return study if study.get("version") == VERSION else None
