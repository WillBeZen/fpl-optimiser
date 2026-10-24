"""Charts + the HTML report — synthetic data, no network."""
import json
import re

import pandas as pd
import pytest

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
    assert "arrivals" in r.figures
    report.render(r)                                        # renders without error
