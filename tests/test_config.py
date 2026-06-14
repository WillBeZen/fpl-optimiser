"""Config integrity tests.  Run:  python -m pytest -q"""
import config


def test_validate_passes():
    config.validate()  # raises if the rule set is internally inconsistent


def test_quota_sums_to_squad():
    assert sum(config.POSITION_QUOTA.values()) == config.SQUAD_SIZE == 15


def test_budget_and_club_limit():
    assert config.BUDGET == 100.0
    assert config.MAX_PER_CLUB == 3


def test_hit_threshold_never_breaks_even():
    # we must never take a -4 that only breaks even on paper
    assert config.HIT_THRESHOLD_POINTS >= config.TRANSFER_HIT_COST


def test_formation_can_field_eleven():
    lo = sum(l for l, _ in config.FORMATION_BOUNDS.values())
    hi = sum(h for _, h in config.FORMATION_BOUNDS.values())
    assert lo <= config.XI_SIZE <= hi


def test_scoring_has_defensive_contributions():
    # the 2025/26 rule the model must not forget
    assert config.SCORING["defensive_contribution_threshold"]["DEF"] == 10
    assert config.SCORING["defensive_contribution_threshold"]["MID"] == 12
