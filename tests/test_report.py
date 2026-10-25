"""Charts + the HTML report — synthetic data, no network."""
import json
import re

import pandas as pd
import pytest

import config
from fpl_optimiser.app import charts, report
from fpl_optimiser.features import player_profiles as pp
from tests.test_multi_season import two_seasons  # noqa: F401  (pytest fixture)
from tests.test_player_profiles import _midfielders


@pytest.fixture(scope="module")
def typed():
    prof = pp.build_profiles(_midfielders(), curr_gameweeks=30)
    return pp.assign_archetypes(prof, k={"GK": 2, "DEF": 2, "MID": 3, "FWD": 2}, min_minutes=450)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_charts_draw_with_the_themes_validated_palette(typed, theme):
    players, types = typed
    t = charts.THEMES[theme]
    fig = charts.type_map(players, "MID", theme=theme)
    assert 2 <= len(fig.data) <= 3                         # a scatter never shows > 3 colours
    assert {tr.marker.color for tr in fig.data} <= set(t["series"])
    assert fig.layout.paper_bgcolor == t["surface"]

    heat = charts.type_fingerprint(types, "MID", theme=theme).data[0]
    assert heat.zmin == -heat.zmax                         # diverging scale centred on 0

    change = pd.DataFrame({"web_name": list("ABCDEFGHIJKL"), "team_name": "X", "position": "MID",
                           "pts_90_prev": range(12), "pts_90_curr": range(12, 0, -1)})
    change["delta"] = change["pts_90_curr"] - change["pts_90_prev"]
    assert len(charts.risers_fallers(change, "2025/26", "2026/27", theme=theme).data) == 3
    assert len(charts.value_bars(players, theme=theme, n=5).data[0].x) == 5


def test_report_is_one_self_contained_offline_page(typed, tmp_path):
    players, _ = typed
    evil = players.assign(web_name="</script><img src=x onerror=alert(1)>")
    r = report.Report(title="T", subtitle="S")
    r.sections.append(report.Section("types", "Player types", "intro <b>", [
        report.card(r.chart("types", charts.type_map, evil, "MID")),   # same name as the section
        report.table(evil[["web_name", "pts_90"]], "Table")]))
    page = report.render(r)
    assert r.write(tmp_path / "r.html").read_text(encoding="utf-8") == page

    # offline: Plotly is inlined, nothing is fetched
    assert not re.search(r"<script[^>]*\ssrc=|<link[^>]*\shref=", page)
    assert "Plotly" in page
    # a chart's placeholder can never share its section's id
    assert 'id="fig-types"' in page and page.count('id="types"') == 1
    # names are data: the JSON block stays intact and both themes are there ...
    data = page.split('<script type="application/json" id="fig-data">')[1].split("</script>")[0]
    assert set(json.loads(data)["types"]) == {"light", "dark"}
    # ... and everywhere else they are escaped, never live markup
    rest = page.replace(data, "")
    assert "<img src=x" not in rest and "&lt;img src=x" in rest
    assert "intro &lt;b&gt;" in rest


def test_build_report_from_a_filed_store(two_seasons):
    r = report.build_report()
    assert r.title.endswith("2026/27")
    assert {"Linked from last season", "New arrivals"} <= {label for label, _, _ in r.tiles}
    assert {"link", "next"} <= {s.id for s in r.sections}
    assert r.sections[-1].id == "glossary"                  # the glossary closes the page
    assert "arrivals" in r.figures
    page = report.render(r)
    assert '<a href="#tab-glossary" data-tab="glossary" role="tab">Glossary</a>' in page
    assert 'id="gloss-filter"' in page and 'id="tab-analysis"' in page


def test_glossary_terms_are_unique_and_quote_the_live_settings(monkeypatch):
    from fpl_optimiser.app import glossary
    terms = [term for _, ts in glossary.groups() for term, _ in ts]
    assert len(terms) == len(set(terms)) >= 40
    assert len({glossary.slug(t) for t in terms}) == len(terms)          # every anchor is unique
    assert all(text.strip() for _, ts in glossary.groups() for _, text in ts)
    assert glossary.slug("Hit (-4)") == "hit-4"

    monkeypatch.setattr(config, "HIT_THRESHOLD_POINTS", 7.5)            # read at build time
    monkeypatch.setattr(config, "RECENCY_HALF_LIFE_GW", 6.0)
    text = dict(t for _, ts in glossary.groups() for t in ts)
    assert "more than 7.5 expected points" in text["Hit threshold"]
    assert "6 gameweeks ago counts half" in text["Form (recency weighting)"]

    page = report.render(report.Report("T", "S", sections=[report.glossary_section()]))
    assert 'id="g-hit-threshold"' in page and "<dt>xG (expected goals)</dt>" in page


# --------------------------------------------------------------------------- #
# this week's sections (plan, news check, from-scratch squad, predictions)
# --------------------------------------------------------------------------- #
from fpl_optimiser.claude.advisor import Advice  # noqa: E402
from fpl_optimiser.ingest.my_team import MyTeam  # noqa: E402
from fpl_optimiser.model.predict import Forecast  # noqa: E402
from fpl_optimiser.optimise import planner, squad  # noqa: E402
from tests.test_optimise import _candidates  # noqa: E402

HOSTILE = "<img src=x onerror=alert(1)>"


def _week(with_team=True, advice=None, backtest=None):
    c = _candidates().assign(code=lambda d: d["id"] + 1000, status="a", news="")
    xp = pd.concat([c.assign(gw=g, fixtures="ABC (H)", xp_raw=c["xp"], p_play=1.0, p_60=1.0)
                    for g in (9, 10)], ignore_index=True)
    fc = Forecast(season="2026-27", next_gw=9, gameweeks=[9, 10], xp=xp,
                  per_fixture=pd.DataFrame(), model=None, deadline="2026-10-31 11:00")
    wc = squad.pick_squad(xp[xp["gw"] == 9])
    plan = base = team = None
    if with_team:
        ids, bank = list(wc.players["id"]), round(100 - wc.cost, 1)
        plan = planner.plan(xp, fc.players, ids, bank, 1)
        base = planner.plan(xp, fc.players, ids, bank, 1, max_transfers_per_gw=0)
        team = MyTeam(manager_id=1, gw=8, squad_ids=ids, bank=bank, free_transfers=1,
                      sell_prices={}, purchase_prices={}, chips_used=[])
    return report.WeekResults(fc, wc, plan, base, team, advice, backtest)


def _render(week):
    r = report.Report(title="T", subtitle="S")
    report._add_week(r, week)
    return r, report.render(r)


def test_week_sections_with_a_plan_and_a_news_check():
    advice = Advice(action_score=82, verdict="adjust", summary=HOSTILE, captain_note="fine",
                    player_flags=[{"player": HOSTILE, "status": "doubt", "note": "knock",
                                   "source": "javascript:alert(1)"},
                                  {"player": "Ok", "status": "fit", "note": "-",
                                   "source": "https://example.com/a"}],
                    adjustments=["swap"], risks=[], model="claude-sonnet-5-5", searches=2,
                    sources=[{"title": "A", "url": "https://example.com/a"}],
                    input_tokens=12_000, output_tokens=3_000)
    r, page = _render(_week(advice=advice))
    tabs = {}
    for s in r.sections:
        tabs.setdefault(s.tab, []).append(s.id)
    assert tabs["accuracy"] == ["acc-summary"]                 # no replay this run: a note, not a crash
    assert tabs["week"] == ["week", "alerts", "advice", "captain"]
    assert {"squad-grid"} <= set(tabs["squad"]) and {"captaincy", "market", "targets"} <= set(tabs["scout"])
    assert tabs["model"][0] == "m-data" and tabs["model"][-1] == "backtest"
    assert tabs["planner"] == ["p-inputs", "p-pool", "p-problem", "p-baseline", "p-ladder", "p-plan",
                               "p-chips", "p-why", "p-limits", "p-objective", "p-captain", "wildcard"]
    assert len({s.id for s in r.sections}) == len(r.sections)          # ids are unique
    labels = {label: value for label, value, _ in r.tiles}
    assert labels["Next deadline"] == "GW9" and labels["News check"] == "82/100"
    assert "Plan vs no transfers" in labels and {"plan", "xp-next"} <= set(r.figures)
    rest = page.split('<script type="application/json"')[0]
    assert HOSTILE not in rest and "&lt;img src=x" in rest              # escaped everywhere
    assert 'href="javascript' not in page                                # unsafe link dropped
    assert '<a href="https://example.com/a" target="_blank" rel="noopener noreferrer">' in page
    assert "12,000 input and 3,000 output tokens, about $0.07" in page   # 0.024 + 0.03 + 2 searches


def test_the_chip_stage_shows_sets_left_the_plan_and_each_chip_in_each_week():
    from fpl_optimiser.optimise import whatif
    week = _week()
    fc, ids, bank = week.forecast, list(week.team.squad_ids), week.team.bank
    xp = fc.xp.copy()
    xp.loc[xp["gw"] == 10, "xp"] *= 2                          # a double in GW10: bench boost
    fc.xp = xp
    chips = {(1, 19): ("free_hit", "bench_boost"), (20, 38): config.CHIPS}
    week.team.chips_available = chips
    inputs = whatif.plan_inputs(xp, fc.players, ids, bank, 1, chips=chips)
    week.plan = planner.plan(**inputs)
    week.chips = whatif.chip_weeks(inputs, chosen=week.plan)
    assert week.plan.chips == {"bench_boost": 10}
    r, page = _render(week)
    sec = next(s for s in r.sections if s.id == "p-chips")
    text = "".join(sec.blocks)
    assert "This plan plays the bench boost in GW10." in text and "planned for GW10" in text
    assert "GW1-19" in text and "free hit, bench boost" in text and "not yet" in text   # second set
    assert "✓" in text and "Chip this week" not in {label for label, _, _ in r.tiles}   # GW9 has none
    plan_rows = next(s for s in r.sections if s.id == "p-plan")
    assert "bench boost" in "".join(plan_rows.blocks)


def test_week_sections_without_a_team_or_the_advisor():
    r, page = _render(_week(with_team=False))
    assert "FPL_TEAM_ID" in page and "Skipped this run" in page
    assert "Best squad from scratch" in {label for label, _, _ in r.tiles}
    failed = Advice(action_score=0, verdict="wait", summary="", error="no key")
    assert "Advisor unavailable: no key" in _render(_week(with_team=False, advice=failed))[1]


def test_backtest_section_when_given():
    bt = pd.DataFrame({"method": ["Model", "Form"], "MAE": [1.3, 1.3], "RMSE": [2.2, 2.3],
                       "rank_corr": [0.6, 0.6], "top11_points": [49.6, 38.9],
                       "gameweeks": [7, 7], "predictions": [100, 100]})
    r, _ = _render(_week(backtest=bt))
    assert next(s for s in r.sections if s.id == "backtest").tab == "model" and "backtest" in r.figures


def test_the_full_dashboard_from_a_filed_store(two_seasons):
    """Forecast, the honest replay and its checks from a filed store: every tab builds and renders."""
    from fpl_optimiser.features import multi_season
    from fpl_optimiser.model import accuracy, evaluation, predict

    fc = predict.forecast(horizon=3)
    table = multi_season.build_all()
    ev = evaluation.run(fc.season, table)
    bt = evaluation.backtest_table(ev)
    halves = accuracy.half_life_check(ev, (4.0, 8.0))
    wc = squad.pick_squad(_candidates())                       # ids from elsewhere: skipped, not a crash
    week = report.WeekResults(fc, wc, backtest=bt, table=table, evaluation=ev, half_lives=halves,
                              forecast_check=accuracy.forecast_path_check(table, ev))
    r = report.build_report(week=week)
    ids = {s.id for s in r.sections}
    assert {"ratings", "ticker-attack", "ticker-defence", "styles"} <= ids          # fixtures tab
    assert {"m-recency", "m-matchups", "m-events", "m-blend", "backtest"} <= ids    # model tab
    assert {"explorer", "market", "glossary", "link"} <= ids
    assert {"acc-summary", "acc-gameweeks", "acc-players", "acc-calibration", "acc-halflife",
            "acc-data"} <= ids                                                      # accuracy tab
    assert {"acc-methods", "acc-gameweeks", "acc-scatter", "acc-calibration"} <= set(r.figures)
    checks = next(s for s in r.sections if s.id == "acc-data").blocks[0]
    assert "✗ fail" not in checks and "Two code paths give the same predictions" in checks
    summary = "".join(next(s for s in r.sections if s.id == "acc-summary").blocks)
    for label in evaluation.METHODS.values():                  # every method, side by side
        assert label in summary
    assert "Is the blend better than each simple method?" in summary
    assert {"ratings", "m-weights", "m-prior-share", "m-response"} <= set(r.figures)
    assert evaluation.METHODS["xp_blend"] in set(bt["method"])
    page = report.render(r)
    for tab in ("week", "squad", "fixtures", "scout", "model", "accuracy", "planner", "analysis", "glossary"):
        assert f'id="tab-{tab}"' in page
    assert page.count('class="explore"') >= 2 and 'data-explorer="explorer-panels"' in page
    import re
    ids = re.findall(r'\sid="([^"]+)"', page.split('<script type="application/json"')[0])
    assert len(ids) == len(set(ids)), "duplicate DOM ids break the page's script"


def test_data_table_cells_sort_shade_and_escape():
    from fpl_optimiser.app import html
    df = pd.DataFrame({"Player": [HOSTILE, "B"], "Pts": [7.25, 1.0], "GW9": [(6.0, "6.0<small>ARS</small>"), (0.5, "x")],
                       "_id": [1, 2]})
    out = html.data_table(df, fmt={"Pts": "{:.1f}"}, heat={"GW9": (0, 7)}, html_cols=("GW9",), hidden=("_id",),
                          row_attrs=lambda r: {"pos": "MID", **({"class": "mine"} if r["_id"] == 1 else {})})
    assert HOSTILE not in out and "&lt;img" in out
    assert '<td data-v="7.25">7.2</td>' in out or '<td data-v="7.25">7.3</td>' in out
    assert '<td data-v="6" class="h6">6.0<small>ARS</small></td>' in out
    assert '<tr data-pos="MID" class="mine">' in out and "_id" not in out
    assert html.heat_class(0, 0, 7) == "h0" and html.heat_class(99, 0, 7) == "h6" and html.heat_class(None, 0, 1) == ""
    mixed = html.data_table(pd.DataFrame({"GW": [6], "Players": [240], "Bias": [-0.25]}))
    assert '<td data-v="6">6</td>' in mixed and '<td data-v="240">240</td>' in mixed     # ints stay ints
    assert html.error_class(-5) == "e0" and html.error_class(0.1) == "e3" and html.error_class(9) == "e6"
    assert "el.style.height" in html.JS            # charts keep their height when the window resizes
