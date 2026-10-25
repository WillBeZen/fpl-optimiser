"""The forecast log — what the model actually predicted before each deadline.

The accuracy check (model/accuracy.py) replays past gameweeks, but it can't replay
FPL's injury flags as they stood at the time (FPL doesn't archive them). So every
weekly run also files its forecast for the next gameweek, flags applied, as

    data/processed/forecast_log/<season>/gw<NN>.csv

and the last run before a deadline is what stays filed for that gameweek (later
runs predict the following gameweek). Once a logged gameweek is played, the
Accuracy tab compares those as-made forecasts with what happened too.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

LOG_DIR = config.PROCESSED_DIR / "forecast_log"
COLUMNS = ["id", "code", "web_name", "team_name", "position", "price", "gw", "fixtures",
           "xp", "xp_raw", "xp_model", "xp_events", "p_play", "p_60", "status", "news"]


def path(season: str, gw: int, root: Path | None = None) -> Path:
    return (root or LOG_DIR) / season / f"gw{gw:02d}.csv"


def save(fc, root: Path | None = None) -> Path:
    """File `fc`'s (predict.Forecast) expected points for its next gameweek."""
    rows = fc.xp[fc.xp["gw"] == fc.next_gw]
    out = rows[[c for c in COLUMNS if c in rows.columns]].copy()
    out["logged_at"] = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}"
    p = path(fc.season, fc.next_gw, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(p)
    return p


def load(season: str, gameweeks, root: Path | None = None) -> pd.DataFrame:
    """The logged forecasts for those gameweeks (only the ones that were logged)."""
    frames = [pd.read_csv(path(season, g, root)) for g in gameweeks if path(season, g, root).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS + ["logged_at"])
