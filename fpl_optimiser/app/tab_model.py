"""Tab 5 · The model — the prediction engine, stage by stage.

  1 the data            what's filed and what the models learn from
  2 recency weighting   the half-life in action, with a worked example
  3 teams               what the models see about each opponent
  4 players             the inputs that describe a player
  5 matchups            player types against defensive styles; how the points
                        model responds to the opponent
  6 minutes model       who plays, and what it leans on
  7 points model        what he scores if he plays, and what it leans on
  8 event model         team ratings -> fixture goals -> goals, assists, clean sheets
  9 blend and flags     the two models combined; FPL's injury flags applied
 10 validation          the walk-forward backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from fpl_optimiser.app import charts
from fpl_optimiser.app.common import Context, WeekResults, has_events, pct, player_info
from fpl_optimiser.app.html import Report, Section, bars, card, data_table, kv, note, stage, table
from fpl_optimiser.model import events, explain, inputs

TAB = "model"

INPUT_NOTES = {
    "price": "today's price: the market's view of quality",
    "form_points_pg": "recency-weighted points per gameweek (with last season blended in)",
    "form_minutes_pg": "recency-weighted minutes per gameweek",
    "form_xg_pg": "recency-weighted expected goals per gameweek",
    "form_xa_pg": "recency-weighted expected assists per gameweek",
    "form_xgi_pg": "recency-weighted xG + xA per gameweek",
    "form_defcon_pg": "recency-weighted defensive contributions per gameweek",
    "form_bonus_pg": "recency-weighted bonus points per gameweek",
    "form_bps_pg": "recency-weighted BPS per gameweek",
    "form_starts_rate": "recency-weighted share of matches with 60+ minutes",
    "games_played": "this season's appearances so far",
    "last_minutes": "minutes in the latest match (unweighted: who is playing right now)",
    "starts_last3": "share of the last three matches with 60+ minutes",
    "zero_run": "matches missed in a row (0 minutes) going into this one",
    "opp_gc_pg_asof": "the opponent's recency-weighted goals conceded per game",
    "x_xgi_oppgc": "attacking form x opponent leakiness (an interaction)",
    "prev_points_per_gw": "last season's points per gameweek",
    "prev_minutes_per_gw": "last season's minutes per gameweek",
    "prev_points_per_90": "last season's points per 90 (270+ minutes)",
    "prev_xgi_per_90": "last season's xGI per 90",
    "prev_defcon_per_90": "last season's defensive contributions per 90",
    "was_home": "playing at home",
    "is_cold": "fewer than 3 games yet (form filled from the position's typical value)",
    "has_prev_season": "has an FPL record last season",
    "prev_changed_team": "changed club since last season",
    "earlier_season": "a training row from last season, never when predicting",
    "position": "GK / DEF / MID / FWD",
}


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    fc = w.forecast
    gw = fc.next_gw
    table_all = w.table
    this = table_all[table_all["season"] == fc.season] if table_all is not None and "season" in table_all else None

    # 1 · the data --------------------------------------------------------------------
    blocks = [stage(1, "The data", "Every filed season becomes one row per player per match played (only "
                    "matches FPL has marked finished count). Last season's rows count for "
                    f"{config.PREV_SEASON_WEIGHT:g}x when the models are fitted.")]
    if table_all is not None and len(table_all):
        sizes = (table_all.groupby("season")
                 .agg(rows=("id", "size"), players=("id", "nunique"), played=("minutes", lambda m: int((m > 0).sum())),
                      weight=("sample_weight", "first") if "sample_weight" in table_all else ("id", "size"))
                 .reset_index().rename(columns={"season": "Season", "rows": "Player-matches",
                                                "players": "Players", "played": "With minutes",
                                                "weight": "Fitting weight"}))
        blocks.append(card(data_table(sizes, fmt={"Fitting weight": "{:g}"}, sortable=False)))
    r.sections.append(Section("m-data", "1 · Data", "", blocks, tab=TAB))

    # 2 · recency weighting --------------------------------------------------------------
    curve = explain.weight_curve(gw)
    share = explain.prior_share()
    blocks = [stage(2, "Recency weighting (the half-life)",
                    f"Form is a weighted average of past matches: a match g gameweeks ago counts "
                    f"0.5^(g / {config.RECENCY_HALF_LIFE_GW:g}). So a match {config.RECENCY_HALF_LIFE_GW:g} "
                    f"gameweeks ago counts half as much as last week's. Last season enters as 38 extra "
                    f"matches that ended {config.PREV_SEASON_GAP_GW:g} gameweeks before GW1, so it fades "
                    f"as this season goes on."),
              card(r.chart("m-weights", charts.weight_bars, curve),
                   r.chart("m-prior-share", charts.share_line, share, gw), pair=True)]
    if ctx is not None and ctx.season is not None:
        roles = w.plan.weeks[0].team if w.plan is not None else w.wildcard.players
        star = roles.loc[roles["captain"], "id"].iloc[0]
        info = player_info(w, ctx)
        m, s = explain.form_story(ctx.season.history, star, gw, ctx.prior, ctx.season.teams)
        if len(m):
            name = info.at[star, "web_name"]
            parts = [f"<h3>Worked example: {name}, your captain</h3>",
                     bars([(f"GW{rd}", float(pts), f"GW{rd} {opp}: {pts:g} pts, weight {wt:.2f} ({sh:.0%})")
                           for rd, opp, pts, wt, sh in zip(m["round"], m["opponent"], m["points"], m["weight"], m["share"])],
                          label=f"{name}'s points per match",
                          sub=[(float(sh / max(m["share"].max(), 1e-9)), f"{sh:.0%}") for sh in m["share"]]),
                     kv([("Form the model sees", f"{s['form']:.2f} pts/game"),
                         ("Plain average this season", f"{s['plain_average']:.2f}"),
                         ("Last season", "–" if np.isnan(s["last_season"]) else f"{s['last_season']:.2f} pts/gameweek"),
                         ("Last season's share of form", pct(s["last_season_share"]))])]
            blocks.append(card("".join(parts)))
            ids = list(w.team.squad_ids) if w.team is not None else list(w.wildcard.players["id"])
            hl = explain.half_life_table(ctx.season.history, ids, gw, ctx.prior)
            hl.insert(0, "Player", hl.index.map(info["web_name"]))
            hl["Swing"] = hl.iloc[:, 1:].max(axis=1) - hl.iloc[:, 1:].min(axis=1)
            blocks.append(table(hl.sort_values("Swing", ascending=False).round(2),
                                "Table: your squad's form under a shorter and longer half-life", open_=True))
            blocks.append(note("A shorter half-life chases recent form (bigger swings for streaky players); a "
                               "longer one trusts the longer record. Players with a big swing are the ones "
                               "where the choice matters. Change RECENCY_HALF_LIFE_GW in config.py and re-run "
                               "the backtest to test it."))
    r.sections.append(Section("m-recency", "2 · Recency", "", blocks, tab=TAB))

    # 3 · teams ----------------------------------------------------------------------------
    blocks = [stage(3, "Team characteristics",
                    "The points model sees each opponent through three numbers: its recency-weighted goals "
                    "conceded per game so far, FPL's defence rating for the venue, and its defensive style. "
                    "The event model (stage 8) rates attack and defence from xG instead.")]
    if ctx is not None and ctx.profiles is not None and fc.ratings is not None and ctx.season is not None:
        from fpl_optimiser.features import build
        names = explain.style_names(ctx.profiles)
        prof = ctx.profiles.set_index("team_id")
        gc = build.team_defence_at(ctx.season.fixtures, gw)
        rt = fc.ratings.teams.set_index("team_id")
        t = pd.DataFrame({"Club": prof["team"], "Goals conceded/game (as of now)": gc.reindex(prof.index),
                          "FPL defence rating, home": prof["str_def_home"],
                          "FPL defence rating, away": prof["str_def_away"],
                          "Style": prof["archetype"].map(names),
                          "Event model: attack": rt["attack"].reindex(prof.index),
                          "Event model: defence": rt["defence"].reindex(prof.index)}).sort_values(
            "Goals conceded/game (as of now)")
        blocks.append(card(data_table(t, fmt={"Goals conceded/game (as of now)": "{:.2f}",
                                               "Event model: attack": "{:.2f}", "Event model: defence": "{:.2f}"},
                                      heat={"Goals conceded/game (as of now)": (0.5, 2.5)})))
    r.sections.append(Section("m-teams", "3 · Teams", "", blocks, tab=TAB))

    # 4 · players --------------------------------------------------------------------------
    feats = pd.DataFrame({"Input": [explain.FEATURE_LABELS.get(f, f) for f in inputs.FEATURES],
                          "What it is": [INPUT_NOTES.get(f, "") for f in inputs.FEATURES]})
    r.sections.append(Section("m-players", "4 · Players", "", [
        stage(4, "Player characteristics",
              f"Both tree models read the same {len(inputs.FEATURES)} inputs for every player and fixture, "
              "all computed only from matches before it (no peeking). The Player analysis tab groups players "
              "into types on their per-90 profiles."),
        card(data_table(feats, sortable=False))], tab=TAB))

    # 5 · matchups -------------------------------------------------------------------------
    blocks = [stage(5, "Player types against defensive styles",
                    "What each type of player has actually scored per appearance this season against each "
                    "defensive style, then how the points model's estimate moves with the opponent.")]
    if this is not None and ctx is not None and ctx.types is not None and ctx.profiles is not None:
        mh = explain.matchup_history(this, ctx.types.set_index("id")["archetype_name"], ctx.profiles)
        if len(mh):
            styles = list(explain.style_names(ctx.profiles).values())
            mh["row"] = mh["position"] + " · " + mh["type"]
            order = {p: n for n, p in enumerate(config.POSITION_QUOTA)}
            z = (mh.pivot(index="row", columns="style", values="points").reindex(columns=styles))
            z = z.loc[sorted(z.index, key=lambda s: (order.get(s.split(" · ")[0], 9), s))]
            n = mh.pivot(index="row", columns="style", values="matches").reindex(index=z.index, columns=styles)
            blocks.append(card(r.chart("m-matchups", charts.heatmap, z, "Points per appearance: player type vs "
                                       "opponent style", colorbar="Pts/app"),
                               table(n.fillna(0).astype(int).reset_index(names="Type"),
                                     "Table: appearances behind each cell (small numbers are noisy)")))
    if has_events(w):
        rows = fc.per_fixture[fc.per_fixture["round"] == gw]
        # where the trees have plenty of data: rounds 4+ (a defence's record means something
        # by then), middle 80% of matches
        seen = (this.loc[this["round"] >= 4, "opp_gc_pg_asof"].dropna() if this is not None
                else pd.Series(dtype=float))
        lo, hi = (seen.quantile(0.1), seen.quantile(0.9)) if len(seen) > 20 else (0.6, 2.4)
        grid = np.round(np.linspace(lo, hi, 9), 2)
        blocks.append(card(r.chart("m-response", charts.response_lines, explain.opponent_response(fc.model, rows, grid)),
                           note(f"Each line holds every player's own form fixed and moves only the opponent's "
                                f"goals conceded, across the range where the trees have most data ({lo:.1f}-{hi:.1f}: "
                                "the middle 80% of this season's matches from GW4 on). Flat lines, or a dip at "
                                "the leaky end, mean the trees haven't yet learned a reliable opponent effect from "
                                "a few weeks of data. That is why the event model (stage 8), which prices every "
                                "fixture explicitly, is blended in.")))
    r.sections.append(Section("m-matchups", "5 · Matchups", "", blocks, tab=TAB))

    # 6-7 · the tree models ----------------------------------------------------------------
    fi = explain.feature_importance(fc.model, this) if this is not None and len(this) else None
    blocks = [stage(6, "The minutes model: will he play?",
                    "Two classifiers (gradient-boosted trees) give P(plays) and P(60+ minutes). Minutes are "
                    "the biggest driver of FPL points. Importance = how much worse the model gets when one "
                    "input is shuffled.")]
    if fi is not None:
        blocks.append(card(r.chart("m-imp-minutes", charts.importance_bars, fi, "minutes",
                                   "What the minutes model leans on", "Rise in error when shuffled")))
    if w.team is not None and has_events(w):
        nxt = fc.per_fixture[(fc.per_fixture["round"] == gw) & fc.per_fixture["id"].isin(w.team.squad_ids)]
        info = player_info(w, ctx)
        mins = pd.DataFrame({"Player": nxt["id"].map(info["web_name"]), "Fixture": nxt["fixture"],
                             "P(plays)": nxt["p_play"], "P(60+)": nxt["p_60"], "Expected minutes": nxt["xmins"]}
                            ).sort_values("P(60+)")
        blocks.append(table(mins, "Table: your squad's minutes next gameweek (after FPL's flags)", open_=True))
    r.sections.append(Section("m-minutes", "6 · Minutes", "", blocks, tab=TAB))

    blocks = [stage(7, "The points model: what he scores if he plays",
                    "A regressor (gradient-boosted trees) trained only on matches the player played, so "
                    "injury news scales exactly the right part: expected points = P(plays) x points if he plays.")]
    if fi is not None:
        blocks.append(card(r.chart("m-imp-points", charts.importance_bars, fi, "points",
                                   "What the points model leans on", "Rise in RMSE (points) when shuffled"),
                           table(fi[["label", "points", "minutes"]].rename(
                               columns={"label": "Input", "points": "Points model", "minutes": "Minutes model"}).round(4),
                               "Table: every input's importance")))
    r.sections.append(Section("m-points", "7 · Points", "", blocks, tab=TAB))

    # 8 · the event model ------------------------------------------------------------------
    blocks = [stage(8, "The event model: goals, assists, clean sheets",
                    "A transparent football model built in three steps, each as of the next deadline.")]
    if has_events(w):
        rt = fc.ratings
        blocks.append(card(
            '<div class="formula">'
            "team rating   attack  = recency-weighted xG created per game (+ last season's goals as a prior)\n"
            "              defence = recency-weighted xG conceded per game\n"
            f"fixture       home goals = {rt.mean_goals:.2f} x {rt.home:.2f} (home) x home attack index x away defence index\n"
            f"              away goals = {rt.mean_goals:.2f} x {rt.away:.2f} (away) x away attack index x home defence index\n"
            "              P(clean sheet) = exp(-opponent's expected goals)\n"
            "player        xG = his xG per 90 (shrunk to the position average) x expected minutes / 90\n"
            "                   x (this fixture's team goals / his team's usual)\n"
            "              points = appearance + xG x goal pts + xA x 3 + P(60+) x P(clean sheet) x CS pts\n"
            "                       - E[goals conceded / 2] + saves / 3 + P(def. contr.) x 2 + bonus + cards"
            "</div>",
            note("Expected minutes come from the minutes model, so both models agree on who plays. "
                 "The Fixtures tab shows the team ratings and every club's fixture run.")))
        nxt = fc.per_fixture[fc.per_fixture["round"] == gw]
        played = nxt[nxt["p_play"] > 0.5]
        avg = played.groupby("position")[events.SOURCES + ["xp_events"]].mean()
        avg = avg.reindex([p for p in config.POSITION_QUOTA if p in avg.index])
        blocks.append(table(avg.round(2).reset_index(names="Position").rename(columns={"xp_events": "Total"}),
                            "Table: average expected points by source, players likely to play, by position",
                            open_=True))
    r.sections.append(Section("m-events", "8 · Event model", "", blocks, tab=TAB))

    # 9 · blend and flags ------------------------------------------------------------------
    w_ev = fc.event_weight
    blocks = [stage(9, "Blending and injury flags",
                    f"The planner uses (1 - {w_ev:g}) x points model + {w_ev:g} x event model "
                    "(EVENT_MODEL_WEIGHT in config.py; the Accuracy tab compares it with a weight chosen only from earlier weeks). FPL's chance of playing "
                    "then scales next gameweek; a return date in the news ('Expected back 8 Nov') keeps a "
                    "player out until then, and 'Unknown return date' brings him back gradually "
                    f"({', '.join(f'{x:.0%}' for x in config.UNKNOWN_RETURN_RAMP)}).")]
    if has_events(w):
        top = fc.xp[fc.xp["gw"] == gw].nlargest(15, "xp")
        cmp = pd.DataFrame({"Player": top["web_name"], "Fixture": top["fixtures"], "Points model": top["xp_model"],
                            "Event model": top["xp_events"], "Blend (used)": top["xp"],
                            "Gap": top["xp_model"] - top["xp_events"]})
        blocks.append(table(cmp, f"Table: the two models for the top 15, GW{gw}", open_=True))
    flags = explain.flag_impact(fc)
    if len(flags):
        f = flags.rename(columns={"web_name": "Player", "team_name": "Club", "position": "Pos",
                                  "status": "Status", "news": "FPL news", "xp_raw": "If fit", "xp": "After flags",
                                  "cut": "Cut"}).drop(columns="id")
        blocks.append(table(f.round(2), f"Table: who FPL's flags cut most in GW{gw}", open_=True))
    r.sections.append(Section("m-blend", "9 · Blend", "", blocks, tab=TAB))

    # 10 · validation ----------------------------------------------------------------------
    blocks = [stage(10, "Validation: every past deadline replayed",
                    "Each past gameweek is predicted as it stood at its deadline (models, features, players and "
                    "prices from before it only), then compared with what happened, alongside simple methods "
                    "on the same players. Top-11 = the real points of the 11 players each method rated "
                    "highest; RMSE = typical miss per player-match; rank correlation = how well it orders "
                    "players. Every player here; the Accuracy tab has the planner's pool, gameweek-by-gameweek "
                    "uncertainty and the data checks.")]
    if w.backtest is not None:
        bt = w.backtest
        blocks.append(card(r.chart("backtest", charts.top_bars, list(bt["method"]), list(bt["top11_points"]),
                                   "Real points of each method's top 11, per gameweek", "Points"),
                           table(bt.round(3), "Table: all scores", open_=True)))
    else:
        blocks.append(note("Not run this time (--quick). Run python scripts/run_week.py to include it."))
    r.sections.append(Section("backtest", "10 · Validation", "", blocks, tab=TAB))
