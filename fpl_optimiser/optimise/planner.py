"""Multi-week planner — roll the squad model over config.PLANNING_HORIZON_GW
gameweeks, deciding when to bank a free transfer, spend it, or take a -4 hit
(only when the gain over the horizon clears config.HIT_THRESHOLD_POINTS), plus
chip timing. Implemented in Section 5."""
