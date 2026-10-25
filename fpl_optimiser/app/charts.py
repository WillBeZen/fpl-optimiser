"""Plotly chart builders for the HTML report — pure: DataFrames in, figures out.

Every builder takes `theme` ("light" or "dark") and draws with that mode's own
palette steps (the report renders whichever matches the viewer's setting). The
palette is a validated colour-blind-safe set: scatters use at most three
categorical colours (more cannot stay distinguishable for every reader), single
series use slot 1, and z-scores use a blue <-> red diverging scale with a neutral
grey midpoint. Marks are thin, grids are hairlines, and only a few points carry
direct labels — the hover tooltips and the tables under each chart carry the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fpl_optimiser.features.player_profiles import ARCHETYPE_FEATURES, LABELS

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"],
              "stack": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
              "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
              "low": "#2a78d6", "mid": "#f0efec", "high": "#e34948"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70"],
             "stack": ["#3987e5", "#d95926", "#199e70", "#c98500"],
             "seq": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#3987e5", "#6da7ec", "#9ec5f4"],
             "low": "#3987e5", "mid": "#383835", "high": "#e66767"},
}

# Up to four series that sit side by side (stacked bars, a few lines) use "stack":
# the palette's first four slots, which pass the colour-blind checks for adjacent
# pairs. Heatmaps use "seq": one blue ramp, light -> dark (dark mode: dark -> light).

# The two stats each position's player-type map is drawn on (all types are
# clustered on more — the fingerprint beside each map shows the full picture).
MAP_AXES = {"GK": ("saves_90", "xgc_90"), "DEF": ("defcon_90", "xgi_90"),
            "MID": ("defcon_90", "xgi_90"), "FWD": ("xg_90", "xa_90")}

POSITION_NAMES = {"GK": "Goalkeepers", "DEF": "Defenders", "MID": "Midfielders", "FWD": "Forwards"}

# Compact column heads for the fingerprint heatmap (its title says "per 90").
SHORT = {"xg_90": "xG", "xa_90": "xA", "xgi_90": "xGI", "defcon_90": "Def.<br>contrib.",
         "saves_90": "Saves", "xgc_90": "xG<br>conceded", "bonus_90": "Bonus",
         "minutes_share": "Minutes<br>share"}


def _layout(t: dict, title: str = "", x: str = "", y: str = "", height: int = 440,
            legend_rows: int = 0) -> dict:
    """Shared chrome. The legend is one entry per row under the title, and the top
    margin grows with it, so long type names never wrap into the title."""
    axis = dict(gridcolor=t["grid"], gridwidth=1, linecolor=t["axis"], zeroline=False,
                tickfont=dict(color=t["muted"], size=12), title_font=dict(color=t["ink2"], size=13))
    return dict(
        height=height + 22 * legend_rows, paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(family=FONT, color=t["ink2"], size=13),
        title=dict(text=title, font=dict(color=t["ink"], size=15), x=0.01, xanchor="left",
                   y=1, yref="container", yanchor="top", pad=dict(t=14)),
        margin=dict(l=60, r=20, t=(52 if title else 16) + 22 * legend_rows, b=56),
        xaxis={**axis, "title_text": x}, yaxis={**axis, "title_text": y},
        legend=dict(orientation="v", x=0, xanchor="left", y=1.02, yanchor="bottom",
                    font=dict(color=t["ink2"], size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"],
                        font=dict(family=FONT, color=t["ink"], size=12)),
        hovermode="closest", hoverdistance=24,
    )


def _label(fig: go.Figure, t: dict, x, y, text: str) -> None:
    """A selective direct label with a thin leader line (never one on every point)."""
    fig.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=0, arrowwidth=1,
                       arrowcolor=t["muted"], ax=18, ay=-22,
                       font=dict(color=t["ink2"], size=11), bgcolor=t["surface"])


def type_map(players: pd.DataFrame, position: str, theme: str = "light",
             n_labels: int = 4) -> go.Figure:
    """One position's profiled players on two telling stats, coloured by type."""
    t = THEMES[theme]
    xcol, ycol = MAP_AXES[position]
    grp = players[players["position"] == position]
    fig = go.Figure()
    for a, part in grp.groupby("archetype"):
        fig.add_trace(go.Scatter(
            x=part[xcol], y=part[ycol], mode="markers", name=part["archetype_name"].iloc[0],
            marker=dict(size=10, color=t["series"][int(a)], line=dict(width=2, color=t["surface"])),
            customdata=part[["web_name", "team_name", "price", "pts_90", "minutes",
                             "archetype_name"]].to_numpy(),
            hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]} · £%{customdata[2]:.1f}m"
                           f"<br>{LABELS[xcol]} %{{x:.2f}} · {LABELS[ycol]} %{{y:.2f}}"
                           "<br>Points/90 %{customdata[3]:.2f} · %{customdata[4]:,.0f} min this season"
                           "<br>%{customdata[5]}<extra></extra>")))
    for _, r in grp.nlargest(n_labels, "pts_90").iterrows():
        _label(fig, t, r[xcol], r[ycol], r["web_name"])
    fig.update_layout(**_layout(t, f"{POSITION_NAMES[position]}: player types",
                                LABELS[xcol], LABELS[ycol], legend_rows=grp["archetype"].nunique()))
    return fig


def _ink_on(fill_hex: str, t: dict) -> str:
    """Primary ink or white, whichever reads on a heatmap cell's fill."""
    r, g, b = (int(fill_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 0.5 else "#0b0b0b"


def _blend(c1: str, c2: str, f: float) -> str:
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)])
    return "#%02x%02x%02x" % tuple(np.round(a + (b - a) * f).astype(int))


def type_fingerprint(types: pd.DataFrame, position: str, theme: str = "light") -> go.Figure:
    """Each type's centre vs the position average, in standard deviations."""
    t = THEMES[theme]
    feats = ARCHETYPE_FEATURES[position]
    grp = types[types["position"] == position].sort_values("archetype")
    z = grp[[f"z_{f}" for f in feats]].to_numpy(dtype=float)
    lim = max(1.5, float(np.ceil(np.abs(z).max() * 2) / 2))
    rows = [f"{name} ({n})" for name, n in zip(grp["archetype_name"], grp["n_players"])]
    cols = [SHORT[f] for f in feats]
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=rows, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorscale=[[0, t["low"]], [0.5, t["mid"]], [1, t["high"]]],
        colorbar=dict(title=dict(text="SD vs avg", font=dict(color=t["ink2"], size=12)),
                      tickfont=dict(color=t["muted"], size=11), outlinewidth=0, thickness=10),
        hovertemplate="<b>%{z:+.2f} SD</b> vs the position average<br>%{x} · %{y}<extra></extra>"))
    notes = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            v = round(float(z[i, j]), 1)
            end = t["high"] if v > 0 else t["low"]
            fill = _blend(t["mid"], end, min(abs(v) / lim, 1))
            notes.append(dict(x=col, y=row, text=f"{v:+.1f}" if v else "0.0", showarrow=False,
                              font=dict(color=_ink_on(fill, t), size=12)))
    layout = _layout(t, "What sets each type apart (per 90, vs the position average)", height=320)
    layout["annotations"] = notes
    layout["yaxis"].update(autorange="reversed", showgrid=False, tickfont=dict(color=t["ink2"], size=12))
    layout["xaxis"].update(showgrid=False, title_text="", tickangle=0,
                           tickfont=dict(color=t["ink2"], size=12))
    layout["margin"].update(l=10, b=64)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    fig.update_yaxes(automargin=True)
    return fig


def attack_scatter(players: pd.DataFrame, theme: str = "light", n_labels: int = 6) -> go.Figure:
    """Midfielders and forwards: expected goals vs expected assists per 90."""
    t = THEMES[theme]
    att = players[players["position"].isin(["MID", "FWD"])]
    fig = go.Figure()
    for slot, pos in enumerate(["MID", "FWD"]):
        part = att[att["position"] == pos]
        fig.add_trace(go.Scatter(
            x=part["xg_90"], y=part["xa_90"], mode="markers", name=POSITION_NAMES[pos],
            marker=dict(size=10, color=t["series"][slot], line=dict(width=2, color=t["surface"])),
            customdata=part[["web_name", "team_name", "price", "xgi_90", "pts_90"]].to_numpy(),
            hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]} · £%{customdata[2]:.1f}m"
                           "<br>xG/90 %{x:.2f} · xA/90 %{y:.2f} · xGI/90 %{customdata[3]:.2f}"
                           "<br>Points/90 %{customdata[4]:.2f}<extra></extra>")))
    for _, r in att.nlargest(n_labels, "xgi_90").iterrows():
        _label(fig, t, r["xg_90"], r["xa_90"], r["web_name"])
    fig.update_layout(**_layout(t, "Attacking threat per 90 (midfielders and forwards)",
                                "xG/90", "xA/90", legend_rows=2))
    return fig


def risers_fallers(change: pd.DataFrame, prev_label: str, curr_label: str,
                   theme: str = "light", n: int = 5) -> go.Figure:
    """Points per 90 last season vs this season, one dot per linked player.

    change : columns web_name, team_name, position, pts_90_prev, pts_90_curr, delta
    """
    t = THEMES[theme]
    picked = pd.concat([change.nlargest(n, "delta"), change.nsmallest(n, "delta")])
    rest = change.drop(picked.index)
    hover = ("<b>%{customdata[0]}</b> · %{customdata[1]} · %{customdata[2]}"
             "<br>Points/90: %{x:.2f} last season → %{y:.2f} now (%{customdata[3]:+.2f})<extra></extra>")
    cd = ["web_name", "team_name", "position", "delta"]
    lo = 0.0
    hi = float(np.ceil(max(change["pts_90_prev"].max(), change["pts_90_curr"].max())))
    fig = go.Figure([
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="No change",
                   line=dict(color=t["axis"], width=1), hoverinfo="skip"),
        go.Scatter(x=rest["pts_90_prev"], y=rest["pts_90_curr"], mode="markers", name="Other players",
                   marker=dict(size=8, color=t["muted"], opacity=0.45, line=dict(width=0)),
                   customdata=rest[cd].to_numpy(), hovertemplate=hover),
        go.Scatter(x=picked["pts_90_prev"], y=picked["pts_90_curr"], mode="markers",
                   name=f"Biggest {n} risers and fallers",
                   marker=dict(size=10, color=t["series"][0], line=dict(width=2, color=t["surface"])),
                   customdata=picked[cd].to_numpy(), hovertemplate=hover),
    ])
    for _, r in picked.iterrows():
        _label(fig, t, r["pts_90_prev"], r["pts_90_curr"], r["web_name"])
    fig.update_layout(**_layout(t, f"Risers and fallers: points per 90, {prev_label} → {curr_label}",
                                f"Points/90 {prev_label}", f"Points/90 {curr_label}",
                                height=500, legend_rows=3))
    return fig


def value_bars(players: pd.DataFrame, theme: str = "light", n: int = 15) -> go.Figure:
    """The best points per £m so far this season (single series)."""
    t = THEMES[theme]
    top = players.nlargest(n, "points_per_m").iloc[::-1]       # best at the top
    names = [f"{w} · {team} {pos}" for w, team, pos in
             zip(top["web_name"], top["team_name"], top["position"])]
    fig = go.Figure(go.Bar(
        x=top["points_per_m"], y=names, orientation="h",
        marker=dict(color=t["series"][0], cornerradius=4),
        text=[f"{v:.1f}" for v in top["points_per_m"]], textposition="outside",
        textfont=dict(color=t["ink2"], size=12), cliponaxis=False,
        customdata=top[["points", "price"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>%{x:.2f} pts per £m · %{customdata[0]} pts at £%{customdata[1]:.1f}m<extra></extra>"))
    layout = _layout(t, "Best value so far: points per £m", "Points per £m", height=150 + 28 * n)
    layout["yaxis"].update(showgrid=False, tickfont=dict(color=t["ink2"], size=12), automargin=True)
    layout["margin"].update(r=48)
    layout["bargap"] = 0.35
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig


def count_bars(counts: pd.Series, title: str, x_title: str, theme: str = "light") -> go.Figure:
    """A small horizontal bar chart of counts (single series), largest at the top."""
    t = THEMES[theme]
    c = counts.sort_values()
    fig = go.Figure(go.Bar(
        x=c.to_numpy(), y=c.index.astype(str), orientation="h",
        marker=dict(color=t["series"][0], cornerradius=4),
        text=c.to_numpy(), textposition="outside", textfont=dict(color=t["ink2"], size=12),
        cliponaxis=False, hovertemplate="<b>%{y}</b>: %{x}<extra></extra>"))
    layout = _layout(t, title, x_title, height=150 + 28 * len(c))
    layout["yaxis"].update(showgrid=False, tickfont=dict(color=t["ink2"], size=12), automargin=True)
    layout["margin"].update(r=40)
    layout["bargap"] = 0.35
    fig.update_layout(**layout)
    return fig


def top_bars(labels: list[str], values: list[float], title: str, x_title: str,
             theme: str = "light", fmt: str = "{:.1f}") -> go.Figure:
    """A ranked horizontal bar chart (single series), first item at the top."""
    t = THEMES[theme]
    labels, values = list(labels)[::-1], list(values)[::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=t["series"][0], cornerradius=4),
        text=[fmt.format(v) for v in values], textposition="outside",
        textfont=dict(color=t["ink2"], size=12), cliponaxis=False,
        hovertemplate="<b>%{y}</b>: %{x:.2f}<extra></extra>"))
    layout = _layout(t, title, x_title, height=150 + 28 * len(values))
    layout["yaxis"].update(showgrid=False, tickfont=dict(color=t["ink2"], size=12), automargin=True)
    layout["margin"].update(r=48)
    layout["bargap"] = 0.35
    fig.update_layout(**layout)
    return fig


def plan_vs_baseline(gws: list[int], plan_xp: list[float], base_xp: list[float],
                     theme: str = "light") -> go.Figure:
    """Expected points per gameweek with the transfer plan vs making no transfers.
    The plan is the point (accent colour, labelled); the baseline is context (grey)."""
    t = THEMES[theme]
    x = [f"GW{g}" for g in gws]
    fig = go.Figure([
        go.Bar(x=x, y=base_xp, name="No transfers", marker=dict(color=t["muted"], cornerradius=4),
               hovertemplate="%{x}, no transfers: %{y:.1f} pts<extra></extra>"),
        go.Bar(x=x, y=plan_xp, name="With the plan", marker=dict(color=t["series"][0], cornerradius=4),
               text=[f"{v:.1f}" for v in plan_xp], textposition="outside",
               textfont=dict(color=t["ink2"], size=12), cliponaxis=False,
               hovertemplate="%{x}, with the plan: %{y:.1f} pts<extra></extra>"),
    ])
    layout = _layout(t, "Expected points per gameweek (XI + captain, before hits)",
                     "", "Expected points", height=380, legend_rows=2)
    layout["xaxis"].update(showgrid=False)
    layout.update(barmode="group", bargap=0.35, bargroupgap=0.12)
    fig.update_layout(**layout)
    return fig


# --------------------------------------------------------------------------- #
# The model and planner walk-throughs, fixtures and the squad
# --------------------------------------------------------------------------- #
def _hbars_layout(t: dict, title: str, x_title: str, n: int) -> dict:
    layout = _layout(t, title, x_title, height=150 + 28 * n)
    layout["yaxis"].update(showgrid=False, tickfont=dict(color=t["ink2"], size=12), automargin=True)
    layout["margin"].update(r=56)
    layout["bargap"] = 0.35
    return layout


def weight_bars(curve: pd.DataFrame, theme: str = "light") -> go.Figure:
    """How much each past gameweek (and last season, as one block) counts in the form
    used to predict the next gameweek. Two series: last season, this season."""
    t = THEMES[theme]
    last, this = curve[curve["round"] == 0], curve[curve["round"] > 0]
    fig = go.Figure([
        go.Bar(x=last["label"], y=last["share"] * 100, name="Last season (all 38 matches together)",
               marker=dict(color=t["series"][1], cornerradius=4),
               text=[f"{v:.0%}" for v in last["share"]], textposition="outside",
               textfont=dict(color=t["ink2"], size=12), cliponaxis=False,
               hovertemplate="%{x}: %{y:.1f}% of the form<extra></extra>"),
        go.Bar(x=this["label"], y=this["share"] * 100, name="This season, match by match",
               marker=dict(color=t["series"][0], cornerradius=4),
               text=[f"{v:.0%}" for v in this["share"]], textposition="outside",
               textfont=dict(color=t["ink2"], size=12), cliponaxis=False,
               customdata=this[["gameweeks_ago", "weight"]].to_numpy(),
               hovertemplate="%{x} (%{customdata[0]:.0f} gameweeks ago): weight %{customdata[1]:.2f}, "
                             "%{y:.1f}% of the form<extra></extra>"),
    ])
    layout = _layout(t, "How much each match counts towards form", "", "Share of the form (%)",
                     height=380, legend_rows=2)
    layout["xaxis"].update(showgrid=False)
    layout.update(bargap=0.3)
    fig.update_layout(**layout)
    return fig


def share_line(share: pd.DataFrame, current_gw: int, theme: str = "light") -> go.Figure:
    """Last season's share of a regular's form, gameweek by gameweek (single series)."""
    t = THEMES[theme]
    now = share[share["gw"] == current_gw]
    fig = go.Figure([
        go.Scatter(x=share["gw"], y=share["share"] * 100, mode="lines", name="Last season's share",
                   line=dict(color=t["series"][1], width=2),
                   hovertemplate="GW%{x}: last season is %{y:.0f}% of form<extra></extra>"),
        go.Scatter(x=now["gw"], y=now["share"] * 100, mode="markers+text", name="Now",
                   marker=dict(size=10, color=t["series"][1], line=dict(width=2, color=t["surface"])),
                   text=[f"GW{current_gw}: {v:.0%}" for v in now["share"]], textposition="top right",
                   textfont=dict(color=t["ink2"], size=12), hoverinfo="skip", showlegend=False),
    ])
    layout = _layout(t, "Last season fades as this season's matches arrive", "Gameweek",
                     "Last season's share of form (%)", height=340)
    layout["yaxis"].update(range=[0, 105])
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def importance_bars(fi: pd.DataFrame, col: str, title: str, x_title: str, theme: str = "light",
                    n: int = 12) -> go.Figure:
    """The inputs a model leans on most (permutation importance), single series."""
    t = THEMES[theme]
    top = fi.nlargest(n, col).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top[col], y=top["label"], orientation="h", marker=dict(color=t["series"][0], cornerradius=4),
        text=[f"{v:.3f}" for v in top[col]], textposition="outside",
        textfont=dict(color=t["ink2"], size=12), cliponaxis=False,
        hovertemplate="<b>%{y}</b>: shuffling it makes the model worse by %{x:.4f}<extra></extra>"))
    fig.update_layout(**_hbars_layout(t, title, x_title, len(top)))
    return fig


def response_lines(resp: pd.DataFrame, theme: str = "light") -> go.Figure:
    """The points model's points-if-he-plays against the opponent's goals conceded per
    game, one line per position (direct labels at the line ends)."""
    t = THEMES[theme]
    fig = go.Figure()
    for k, pos in enumerate(["GK", "DEF", "MID", "FWD"]):
        d = resp[resp["position"] == pos].sort_values("opp_gc_pg")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["opp_gc_pg"], y=d["pts_if_play"], mode="lines+markers", name=POSITION_NAMES[pos],
            line=dict(color=t["stack"][k], width=2), marker=dict(size=8, line=dict(width=2, color=t["surface"])),
            hovertemplate=f"{POSITION_NAMES[pos]}: %{{y:.2f}} pts if he plays, vs a defence conceding "
                          "%{x:.1f} a game<extra></extra>"))
        end = d.iloc[-1]
        fig.add_annotation(x=end["opp_gc_pg"], y=end["pts_if_play"], text=POSITION_NAMES[pos],
                           showarrow=False, xanchor="left", xshift=8, font=dict(color=t["ink2"], size=11))
    layout = _layout(t, "Points model: points if he plays vs the opponent's defence",
                     "Opponent's goals conceded a game", "Points if he plays (average)",
                     height=400, legend_rows=4)
    layout["margin"].update(r=110)
    fig.update_layout(**layout)
    return fig


def ratings_scatter(teams: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Each club's attack (xG created per game) vs defence (xG conceded per game,
    better at the top), labelled with its short name."""
    t = THEMES[theme]
    fig = go.Figure(go.Scatter(
        x=teams["attack"], y=teams["defence"], mode="markers+text", text=teams["short_name"],
        textposition="top center", textfont=dict(color=t["ink2"], size=11),
        marker=dict(size=11, color=t["series"][0], line=dict(width=2, color=t["surface"])),
        customdata=teams[["name", "att_index", "def_index", "prior_share"]].to_numpy(),
        hovertemplate=("<b>%{customdata[0]}</b><br>Attack %{x:.2f} xG a game (%{customdata[1]:.2f}x average)"
                       "<br>Defence %{y:.2f} xG conceded a game (%{customdata[2]:.2f}x average)"
                       "<br>Last season still %{customdata[3]:.0%} of the rating<extra></extra>")))
    mean = float(np.mean(np.concatenate([teams["attack"], teams["defence"]])))
    fig.add_vline(x=mean, line=dict(color=t["axis"], width=1, dash="dot"))
    fig.add_hline(y=mean, line=dict(color=t["axis"], width=1, dash="dot"))
    layout = _layout(t, "Team ratings: attack vs defence (top right = strong at both)",
                     "Attack: xG created per game", "Defence: xG conceded per game (lower is better)", height=520)
    layout["yaxis"].update(autorange="reversed")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def ladder_bars(ladder: pd.DataFrame, chosen: int, theme: str = "light") -> go.Figure:
    """What each extra transfer this gameweek adds to the planner's objective. Moves
    up to the chosen number in the accent colour; beyond it, grey."""
    t = THEMES[theme]
    d = ladder.dropna(subset=["gain"])
    labels = [f"{k} transfer{'s' if k > 1 else ''}" + (f" ({h} hit)" if h else "")
              for k, h in zip(d["transfers"], d["hits"])]
    colours = [t["series"][0] if k <= chosen else t["muted"] for k in d["transfers"]]
    fig = go.Figure(go.Bar(
        x=labels, y=d["gain"], marker=dict(color=colours, cornerradius=4),
        text=[f"{v:+.1f}" for v in d["gain"]], textposition="outside",
        textfont=dict(color=t["ink2"], size=12), cliponaxis=False,
        customdata=d[["in", "out"]].to_numpy(),
        hovertemplate="<b>%{x}</b>: %{y:+.2f} vs one fewer<br>Out: %{customdata[1]}<br>In: %{customdata[0]}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    layout = _layout(t, "What each extra transfer this week is worth", "",
                     "Gain in the planner's objective (points)", height=380)
    layout["xaxis"].update(showgrid=False)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def budget_bars(budget: pd.DataFrame, theme: str = "light") -> go.Figure:
    """How much the plan improves with more money in the bank (single series)."""
    t = THEMES[theme]
    d = budget[budget["extra"] > 0]
    fig = go.Figure(go.Bar(
        x=[f"+£{e:g}m" for e in d["extra"]], y=d["gain"], marker=dict(color=t["series"][0], cornerradius=4),
        text=[f"{v:+.1f}" for v in d["gain"]], textposition="outside",
        textfont=dict(color=t["ink2"], size=12), cliponaxis=False, customdata=d[["in"]].to_numpy(),
        hovertemplate="<b>%{x}</b>: %{y:+.2f}<br>Would buy: %{customdata[0]}<extra></extra>"))
    layout = _layout(t, "What more money would be worth", "", "Gain in the planner's objective (points)",
                     height=320)
    layout["xaxis"].update(showgrid=False)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


SOURCE_GROUPS = {"Appearance": ["Appearance"], "Attack": ["Goals", "Assists"],
                 "Defence": ["Clean sheet", "Conceded", "Saves", "Def. contributions"],
                 "Bonus and cards": ["Bonus", "Cards"]}


def sources_bars(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Where each player's expected points come from (event model), stacked in four
    groups. df: one row per player with `label` and the events.SOURCES columns."""
    t = THEMES[theme]
    d = df.iloc[::-1]
    fig = go.Figure()
    for k, (group, cols) in enumerate(SOURCE_GROUPS.items()):
        v = d[cols].sum(axis=1)
        fig.add_trace(go.Bar(
            x=v, y=d["label"], orientation="h", name=group,
            marker=dict(color=t["stack"][k], line=dict(width=2, color=t["surface"])),
            hovertemplate=f"<b>%{{y}}</b><br>{group}: %{{x:.2f}} pts<extra></extra>"))
    layout = _hbars_layout(t, "Where the expected points come from (event model)", "Expected points", len(d))
    layout["height"] += 22 * 4
    layout["margin"].update(t=52 + 22 * 4)
    layout["barmode"] = "relative"
    fig.update_layout(**layout)
    return fig


def heatmap(z: pd.DataFrame, title: str, theme: str = "light", fmt: str = "{:.1f}",
            colorbar: str = "", height: int | None = None) -> go.Figure:
    """A sequential heatmap (one blue ramp) with each cell's value written in it."""
    t = THEMES[theme]
    vals = z.to_numpy(dtype=float)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    hi = hi if hi > lo else lo + 1
    scale = [[i / (len(t["seq"]) - 1), c] for i, c in enumerate(t["seq"])]
    rows, cols = [str(r) for r in z.index], [str(c) for c in z.columns]
    fig = go.Figure(go.Heatmap(
        z=vals, x=cols, y=rows, zmin=lo, zmax=hi, colorscale=scale, xgap=2, ygap=2,
        colorbar=dict(title=dict(text=colorbar, font=dict(color=t["ink2"], size=12)),
                      tickfont=dict(color=t["muted"], size=11), outlinewidth=0, thickness=10),
        hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    # every cell's label in one list: add_annotation one at a time re-validates all the
    # earlier ones, which is quadratic (minutes for a 15 x 20 grid)
    notes = []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = vals[i, j]
            if np.isnan(v):
                continue
            step = int(np.clip((v - lo) / (hi - lo) * (len(t["seq"]) - 1), 0, len(t["seq"]) - 1))
            notes.append(dict(x=c, y=r, text=fmt.format(v), showarrow=False,
                              font=dict(color=_ink_on(t["seq"][step], t), size=11)))
    layout = _layout(t, title, height=height or (120 + 34 * len(rows)))
    layout["annotations"] = notes
    layout["yaxis"].update(autorange="reversed", showgrid=False, tickfont=dict(color=t["ink2"], size=12))
    layout["xaxis"].update(showgrid=False, side="top", tickfont=dict(color=t["ink2"], size=11), tickangle=0)
    layout["margin"].update(l=10, t=90, b=16)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    fig.update_yaxes(automargin=True)
    return fig
