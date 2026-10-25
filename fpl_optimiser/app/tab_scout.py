"""Tab 4 · Scout — the whole player market, the way a scout reads it.

Captain candidates, a filterable table of every player worth a look (expected
points per gameweek, value, ownership, set pieces), the best options per position
and price band, differentials, price-change watch, set-piece takers, where this
model and FPL's own estimate disagree, and who is over- or under-performing his xG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_optimiser.app import charts
from fpl_optimiser.app.common import Context, WeekResults, has_events, money, own, pct, player_info
from fpl_optimiser.app.html import Report, Section, card, data_table, esc

TAB = "scout"

# Price bands per position (upper bounds, £m): budget / mid-price / premium.
BANDS = {"GK": (4.5, 5.0), "DEF": (4.6, 5.5), "MID": (6.0, 8.0), "FWD": (6.0, 8.0)}


def _band(pos: str, price: float) -> str:
    lo, hi = BANDS[pos]
    return "Budget" if price <= lo else ("Mid-price" if price <= hi else "Premium")


def _duties(p: pd.Series) -> str:
    out = [lbl for col, lbl in (("penalties_order", "Pens"), ("corners_and_indirect_freekicks_order", "Corners"),
                                ("direct_freekicks_order", "FKs")) if p.get(col) == 1]
    return ", ".join(out)


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    fc = w.forecast
    info = player_info(w, ctx)
    wide = fc.wide().set_index("id")
    gw_cols = [c for c in wide.columns if c.startswith("GW")]
    mine = set(w.team.squad_ids) if w.team is not None else set()

    # --- captaincy ------------------------------------------------------------------
    nxt = fc.xp[fc.xp["gw"] == fc.next_gw].nlargest(12, "xp")
    labels = [f"{n} · {c} {p} ({f})" for n, c, p, f in
              zip(nxt["web_name"], nxt["team_name"], nxt["position"], nxt["fixtures"])]
    blocks = [card(r.chart("xp-next", charts.top_bars, labels, list(nxt["xp"]),
                           f"Highest expected points, GW{fc.next_gw}", "Expected points"))]
    if has_events(w):
        fx = fc.per_fixture[fc.per_fixture["round"] == fc.next_gw]
        g = fx.groupby("id").agg(xg=("xg", "sum"), p_goal=("p_goal", lambda s: 1 - (1 - s).prod()),
                                 p_assist=("p_assist", lambda s: 1 - (1 - s).prod()))
        cap = pd.DataFrame({"Player": nxt["web_name"].to_numpy(), "Club": nxt["team_name"].to_numpy(),
                            "Fixture": nxt["fixtures"].to_numpy(), "Expected pts": nxt["xp"].to_numpy(),
                            "xG": nxt["id"].map(g["xg"]).to_numpy(), "P(goal)": nxt["id"].map(g["p_goal"]).to_numpy(),
                            "P(assist)": nxt["id"].map(g["p_assist"]).to_numpy(),
                            "Owned by": nxt["id"].map(info["selected_by_percent"]).to_numpy(),
                            "FPL's estimate": nxt["id"].map(info["ep_next"]).to_numpy()})
        blocks.append(table_block(cap, {"Expected pts": "{:.2f}", "xG": "{:.2f}", "P(goal)": pct,
                                        "P(assist)": pct, "Owned by": own, "FPL's estimate": "{:.1f}"}))
    r.sections.append(Section(
        "captaincy", f"Captaincy and top picks, GW{fc.next_gw}",
        "The highest expected scorers next gameweek across the whole game, with their chance of a goal "
        "or an assist and how many managers own them.", blocks, tab=TAB))

    # --- the market -------------------------------------------------------------------
    top = wide[wide["total"] > 0].nlargest(300, "total")
    rates = fc.rates if fc.rates is not None else pd.DataFrame(index=top.index)
    nxt_all = fc.xp[fc.xp["gw"] == fc.next_gw].set_index("id")
    rows = []
    for pid, p in top.iterrows():
        q = info.loc[pid] if pid in info.index else pd.Series(dtype=object)
        row = {"Player": p["web_name"], "Club": p["team_name"], "Pos": p["position"], "Price": p["price"],
               "Owned": q.get("selected_by_percent", np.nan)}
        for c in gw_cols:
            row[c] = p[c]
        row["Total"] = p["total"]
        row["Per £m"] = p["total"] / p["price"] if p["price"] else np.nan
        row["P(plays) next"] = nxt_all["p_play"].get(pid, np.nan)
        row["xG/90"] = rates["xg90"].get(pid, np.nan) if "xg90" in rates else np.nan
        row["xA/90"] = rates["xa90"].get(pid, np.nan) if "xa90" in rates else np.nan
        row["Set pieces"] = _duties(q) if len(q) else ""
        row["Status"] = p["news"] if isinstance(p["news"], str) and p["news"] else ""
        row["_id"] = pid
        rows.append(row)
    market = pd.DataFrame(rows)
    hi = max(float(top[gw_cols].quantile(0.98).max()), 1.0)
    clubs = sorted(market["Club"].dropna().unique())
    filters = ('<div class="filters" data-table="market-table">'
               '<input type="search" data-filter="text" placeholder="Search player or club" aria-label="Search">'
               '<label>Position <select data-filter="pos"><option value="">All</option>'
               + "".join(f'<option value="{p}">{p}</option>' for p in ("GK", "DEF", "MID", "FWD"))
               + '</select></label><label>Club <select data-filter="club"><option value="">All</option>'
               + "".join(f'<option value="{esc(c.lower())}">{esc(c)}</option>' for c in clubs)
               + '</select></label><label>Max price £m <input type="number" step="0.5" min="3.5" '
                 'data-filter="maxprice"></label><label>Max owned % <input type="number" step="5" min="0" '
                 'data-filter="maxown"></label><span class="count"></span></div>')
    tbl = data_table(
        market, id="market-table", hidden=("_id",),
        fmt={"Price": money, "Owned": own, "Total": "{:.1f}", "Per £m": "{:.2f}", "P(plays) next": pct,
             "xG/90": "{:.2f}", "xA/90": "{:.2f}", **{c: "{:.1f}" for c in gw_cols}},
        heat={c: (1, hi) for c in gw_cols}, tall=True,
        row_attrs=lambda row: {"pos": row["Pos"], "club": str(row["Club"]).lower(), "price": row["Price"],
                               "own": 0 if pd.isna(row["Owned"]) else row["Owned"],
                               **({"class": "mine"} if row["_id"] in mine else {})})
    r.sections.append(Section(
        "market", "Player market",
        f"The top {len(market)} players by expected points over GW{fc.gameweeks[0]}-{fc.gameweeks[-1]}. "
        "Filter by position, club, price or ownership; click a column to sort. Your players are marked "
        "with a bar on the left. Per £m = expected points over the horizon per £m of price.",
        [card(filters, tbl)], tab=TAB))

    # --- best by position and price band ----------------------------------------------------
    t = top.assign(band=[_band(pos, pr) for pos, pr in zip(top["position"], top["price"])])
    rows = []
    for pos in ("GK", "DEF", "MID", "FWD"):
        for band in ("Premium", "Mid-price", "Budget"):
            for pid, p in t[(t["position"] == pos) & (t["band"] == band)].nlargest(3, "total").iterrows():
                rows.append({"Pos": pos, "Band": band, "Player": p["web_name"], "Club": p["team_name"],
                             "Price": p["price"], "Total": p["total"], "Per £m": p["total"] / p["price"],
                             "Owned": info["selected_by_percent"].get(pid, np.nan),
                             "Yours": "yes" if pid in mine else ""})
    lims = ", ".join(f"{pos} budget ≤ £{lo}m, premium > £{hi}m" for pos, (lo, hi) in BANDS.items())
    r.sections.append(Section(
        "targets", "Best by position and price",
        f"The three best per position and price band over the horizon ({lims}).",
        [card(data_table(pd.DataFrame(rows), fmt={"Price": money, "Total": "{:.1f}", "Per £m": "{:.2f}",
                                                  "Owned": own}, sortable=False))], tab=TAB))

    # --- differentials ------------------------------------------------------------------------
    owned = top.index.map(info["selected_by_percent"])
    diff = top[(pd.Series(owned, index=top.index) < 10)].nlargest(15, "total")
    ddf = pd.DataFrame({"Player": diff["web_name"], "Club": diff["team_name"], "Pos": diff["position"],
                        "Price": diff["price"], "Owned": diff.index.map(info["selected_by_percent"]),
                        "Total": diff["total"], f"GW{fc.next_gw}": diff[f"GW{fc.next_gw}"]})
    r.sections.append(Section(
        "differentials", "Differentials",
        "The best expected scorers owned by fewer than 10% of managers: points from these move your "
        "rank, because most rivals don't have them.",
        [card(data_table(ddf, fmt={"Price": money, "Owned": own, "Total": "{:.1f}", f"GW{fc.next_gw}": "{:.1f}"}))],
        tab=TAB))

    _market_moves(r, w, info, mine)


def table_block(df: pd.DataFrame, fmt: dict) -> str:
    return card(data_table(df, fmt=fmt))


def _market_moves(r: Report, w: WeekResults, info: pd.DataFrame, mine: set) -> None:
    fc = w.forecast
    i = info[info["minutes"].fillna(0) > 0].copy()

    # price watch: this gameweek's net transfers and FPL's price-change progress
    if i["transfers_in_event"].notna().any():
        i["net"] = i["transfers_in_event"] - i["transfers_out_event"]
        cols = {"web_name": "Player", "team_name": "Club", "position": "Pos", "price": "Price",
                "net": "Net transfers this GW", "price_change_percent": "Progress to a change"}
        rising = i.nlargest(10, "net")[list(cols)].rename(columns=cols)
        falling = i.nsmallest(10, "net")[list(cols)].rename(columns=cols)
        fmt = {"Price": money, "Net transfers this GW": "{:+,.0f}",
               "Progress to a change": lambda v: f"{round(v):+d}%"}
        r.sections.append(Section(
            "price-watch", "Price watch",
            "Prices move with net transfers. 'Progress to a change' is FPL's own measure: near +100% a "
            "rise is close, near -100% a fall. Buy risers before they rise; sell fallers before they fall.",
            [card(f"<h3>Most bought this gameweek</h3>{data_table(rising, fmt=fmt)}",
                  f"<h3>Most sold this gameweek</h3>{data_table(falling, fmt=fmt)}", pair=True)], tab=TAB))

    # set pieces
    rows = []
    for club, d in info.groupby("team_name"):
        def first(col):
            who = d[d[col] == 1]["web_name"]
            return ", ".join(who) if len(who) else "–"
        rows.append({"Club": club, "Penalties": first("penalties_order"),
                     "Corners and indirect FKs": first("corners_and_indirect_freekicks_order"),
                     "Direct free kicks": first("direct_freekicks_order")})
    if rows and info["penalties_order"].notna().any():
        r.sections.append(Section(
            "set-pieces", "Set-piece takers",
            "FPL's first-choice takers. A penalty is worth about 0.76 xG, so a penalty taker's xG is "
            "a little higher than his open play suggests.",
            [card(data_table(pd.DataFrame(rows), sortable=True))], tab=TAB))

    # model vs FPL's own estimate
    nxt = fc.xp[fc.xp["gw"] == fc.next_gw].set_index("id")
    ep = info["ep_next"].reindex(nxt.index)
    if ep.notna().any():
        cmp = nxt.assign(ep=ep, gap=nxt["xp"] - ep)
        cmp = cmp[cmp["ep"].notna() & ((cmp["xp"] > 2) | (cmp["ep"] > 2))]
        cols = {"web_name": "Player", "team_name": "Club", "fixtures": "Fixture", "xp": "This model",
                "ep": "FPL's estimate", "gap": "Difference"}
        fmt = {"This model": "{:.2f}", "FPL's estimate": "{:.1f}", "Difference": "{:+.2f}"}
        r.sections.append(Section(
            "vs-fpl", "Where we disagree with FPL",
            f"GW{fc.next_gw}: this model's expected points against FPL's own estimate (ep_next). Big gaps "
            "are worth a second look: either the model sees something FPL's simple form-based number "
            "misses, or it's missing news.",
            [card(f"<h3>We rate higher</h3>{data_table(cmp.nlargest(10, 'gap')[list(cols)].rename(columns=cols), fmt=fmt)}",
                  f"<h3>FPL rates higher</h3>{data_table(cmp.nsmallest(10, 'gap')[list(cols)].rename(columns=cols), fmt=fmt)}",
                  pair=True)], tab=TAB))

    # finishing: goals and assists vs expected
    f = i[i["minutes"] >= 450].copy()
    if f["expected_goals"].notna().any():
        f["g_minus_xg"] = f["goals_scored"] - f["expected_goals"]
        f["a_minus_xa"] = f["assists"] - f["expected_assists"]
        cols = {"web_name": "Player", "team_name": "Club", "goals_scored": "Goals", "expected_goals": "xG",
                "g_minus_xg": "Goals - xG", "assists": "Assists", "expected_assists": "xA",
                "a_minus_xa": "Assists - xA"}
        fmt = {"xG": "{:.2f}", "Goals - xG": "{:+.2f}", "xA": "{:.2f}", "Assists - xA": "{:+.2f}"}
        r.sections.append(Section(
            "finishing", "Finishing: lucky or good?",
            "Goals and assists this season against their expected values (450+ minutes). Big positive "
            "gaps tend to shrink (hot finishing rarely lasts); big negative gaps suggest returns are due. "
            "The model works from xG, so it already looks through the luck.",
            [card(f"<h3>Scoring above xG</h3>{data_table(f.nlargest(10, 'g_minus_xg')[list(cols)].rename(columns=cols), fmt=fmt)}",
                  f"<h3>Scoring below xG</h3>{data_table(f.nsmallest(10, 'g_minus_xg')[list(cols)].rename(columns=cols), fmt=fmt)}",
                  pair=True)], tab=TAB))
