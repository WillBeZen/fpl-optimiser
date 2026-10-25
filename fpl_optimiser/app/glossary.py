"""The report's glossary: every term the page uses, in plain English.

Numbers in the definitions (the hit threshold, the half-life, the bench weight, ...)
are read from config.py, so the glossary always describes the settings the report
was built with. Each term gets an anchor (`#g-<slug>`) so other text can link to it.
"""
from __future__ import annotations

import re

import config


def slug(term: str) -> str:
    """'Hit (-4)' -> 'hit-4': the term's anchor, without the 'g-' prefix."""
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


def groups() -> list[tuple[str, list[tuple[str, str]]]]:
    """[(group title, [(term, definition), ...]), ...] in reading order."""
    s, hit, d = config.SCORING, config.TRANSFER_HIT_COST, config.HORIZON_DECAY
    cs, dc = s["clean_sheet"], s["defensive_contribution_threshold"]
    half = config.RECENCY_HALF_LIFE_GW
    return [
        ("The game", [
            ("Gameweek (GW)", "One round of Premier League fixtures. Your team is locked at the "
             "round's deadline and scores from every match in it."),
            ("Deadline", "The time teams lock for a gameweek, set by FPL shortly before the round's "
             "first kick-off. The report shows FPL's own time, in UTC."),
            ("Squad", f"Your {config.SQUAD_SIZE} players: "
             + ", ".join(f"{n} {p}" for p, n in config.POSITION_QUOTA.items())
             + f", within £{config.BUDGET:g}m and at most {config.MAX_PER_CLUB} from one club."),
            ("Starting XI and bench", f"The {config.XI_SIZE} who score each gameweek, and the "
             f"{config.SQUAD_SIZE - config.XI_SIZE} who don't. If a starter doesn't play, a bench "
             "player comes on automatically (keeping the formation legal)."),
            ("Formation", "How the XI is split: "
             + ", ".join(f"{lo}-{hi} {p}" if lo != hi else f"{lo} {p}"
                         for p, (lo, hi) in config.FORMATION_BOUNDS.items())
             + ". Written DEF-MID-FWD, e.g. 3-4-3."),
            ("Captain (C) and vice-captain (V)", f"The captain's points count "
             f"{config.CAPTAIN_MULTIPLIER}x. If the captain doesn't play, the vice-captain's count "
             f"{config.CAPTAIN_MULTIPLIER}x instead."),
            ("Clean sheet", f"No goals conceded while the player is on the pitch, for 60+ minutes: "
             f"{cs['GK']} points for a goalkeeper or defender, {cs['MID']} for a midfielder."),
            ("Bonus points and BPS", f"1-{s['bonus_max']} extra points for the three best "
             "performers in each match, ranked by BPS (the Bonus Points System, FPL's own "
             "match-performance score)."),
            ("Defensive contributions", f"{s['defensive_contribution_points']} points for "
             f"reaching {dc['DEF']} clearances, blocks, interceptions and tackles (CBIT) in a match "
             f"as a defender, or {dc['MID']} including ball recoveries (CBIRT) as a midfielder or "
             "forward."),
            ("Status flag", "FPL's availability flag: a available, d doubtful (with a % chance of "
             "playing), i injured, s suspended, u unavailable (e.g. left the club), n not available "
             "(e.g. on loan). Next gameweek's expected points are scaled by FPL's chance of playing."),
            ("Blank gameweek", "A gameweek in which a club has no fixture (often a cup clash), so "
             "its players score nothing."),
            ("Double gameweek", "A gameweek in which a club plays twice, so its players' expected "
             "points add up over both fixtures."),
        ]),
        ("Transfers and chips", [
            ("Free transfer", f"You get {config.FREE_TRANSFERS_PER_GW} each gameweek; unused ones "
             f"carry over, up to {config.MAX_BANKED_TRANSFERS}. FPL doesn't publish how many you "
             "have, so the run counts them from your transfer history (override with "
             "--free-transfers)."),
            ("Banking a transfer", "Making no transfer this week, so the free one carries over and "
             "you have more next week."),
            (f"Hit (-{hit})", f"Each transfer beyond your free ones costs {hit} points, taken off "
             "that gameweek's score."),
            ("Bank", "Money you haven't spent, in £m. A purchase must fit within the bank plus "
             "the selling prices of the players going out."),
            ("Selling price", "What FPL pays when you sell: the price you paid plus half of any "
             "rise since, rounded down to £0.1m; a fall is taken in full. Bought at £5.5m, now "
             "£6.2m: sells for £5.8m."),
            ("Chips", f"One-off boosts: wildcard, free hit, bench boost and triple captain. You get "
             f"{config.CHIP_SETS_PER_SEASON} sets a season: the first must be used by the "
             f"GW{config.CHIP_SPLIT_GW} deadline, the second after it. The planner decides which "
             "gameweek (if any) to play each one in, one chip a week at most."),
            ("Wildcard", "Unlimited free transfers for one gameweek, and you keep the new squad. "
             "'Best squad from scratch' is what a wildcard would build."),
            ("Free hit", "Unlimited free transfers for one gameweek only; your squad goes back to "
             "the previous one afterwards."),
            ("Bench boost", "Your bench's points count too, for one gameweek."),
            ("Triple captain", f"Your captain's points count {config.TRIPLE_CAPTAIN_MULTIPLIER}x "
             f"instead of {config.CAPTAIN_MULTIPLIER}x, for one gameweek."),
        ]),
        ("Stats", [
            ("xG (expected goals)", "How likely each of a player's shots was to go in (0 to 1, "
             "judged on where and how it was taken), added up. It measures chances, which predict "
             "future goals better than goals themselves do."),
            ("xA (expected assists)", "The xG of the shots a player set up with his passes."),
            ("xGI (expected goal involvement)", "xG + xA."),
            ("xG conceded", "The xG of the shots faced while the player was on the pitch. Lower is "
             "better for goalkeepers and defenders."),
            ("Per 90", "Per 90 minutes played, so part-timers and regulars compare fairly. Rates "
             f"from few minutes are noisy: player types need {config.PROFILE_MIN_MINUTES}+ minutes."),
            ("Points/90", "FPL points per 90 minutes played."),
            ("Minutes share", "The share of the available minutes a player actually played "
             "(1.0 = every minute)."),
            ("Points per £m", "Points this season divided by the player's current price: value "
             "for money."),
            ("Ownership (Owned by)", "The share of FPL managers who have the player. Points from a "
             "player most rivals own barely move your rank; points from one they don't, do."),
            ("Differential", "A good player few managers own (under 10% here): a chance to gain rank."),
            ("Set-piece order", "FPL's first, second... choice for penalties, corners and direct free "
             "kicks. A penalty is worth about 0.76 xG, so first-choice takers score more."),
            ("Price-change progress", "FPL's own measure of how close a price change is, from net "
             "transfers: near +100% a rise is close, near -100% a fall."),
            ("FPL's estimate (ep_next)", "FPL's own expected points for next gameweek, a simple "
             "form-based number, shown for comparison."),
            ("Finishing above xG", "Scoring more goals than the chances were worth. It rarely lasts; "
             "the models work from xG, so they look through it."),
        ]),
        ("Predictions", [
            ("Expected points (xP)", "The average number of points a player is expected to score in a "
             f"gameweek, summed over its fixtures: the blend of the points model and the event model "
             f"({1 - config.EVENT_MODEL_WEIGHT:g} / {config.EVENT_MODEL_WEIGHT:g}). An average over many "
             "possible outcomes, not a guess at the exact score."),
            ("Points model", "Gradient-boosted trees that predict P(plays) and the points a player scores "
             "if he plays, straight from his form, price, last season and the opponent."),
            ("Event model", "A transparent model that builds expected points from football events: team "
             "ratings give each fixture's expected goals, and each player's per-90 rates give his xG, xA, "
             "clean-sheet chance and so on, which FPL's scoring turns into points."),
            ("Blend", f"What the planner uses: {1 - config.EVENT_MODEL_WEIGHT:g} x the points model + "
             f"{config.EVENT_MODEL_WEIGHT:g} x the event model. The backtest found it better than either "
             "alone."),
            ("Expected minutes (xMins)", "P(60+) x his usual minutes in a full appearance + P(plays but "
             "under 60) x his usual minutes off the bench."),
            ("Points by source", "The event model's expected points split into appearance, goals, "
             "assists, clean sheet, goals conceded, saves, defensive contributions, bonus and cards."),
            ("Team ratings", "Each club's attack (xG created per game) and defence (xG conceded per game), "
             "recent matches counting more, with last season as a fading prior. 'Index' = vs the league "
             "average (1.0)."),
            ("Fixture ticker", "Each club's expected goals, or clean-sheet chance, gameweek by gameweek: "
             "which clubs to pick attackers and defenders from."),
            ("Shrinkage", "Mixing a player's own rate with his position's average when he has played little, "
             "so a lucky 90 minutes doesn't make him a star."),
            ("P(plays) and P(60+)", "The model's probability that a player gets on the pitch, and "
             "that he plays 60+ minutes (worth an extra point, and needed for a clean sheet)."),
            ("Form (recency weighting)", f"Recent matches count more: a match {half:g} gameweeks "
             f"ago counts half as much as this week's, {2 * half:g} gameweeks ago a quarter."),
            ("Half-life", f"How many gameweeks it takes for a match to count half as much "
             f"({half:g} here). Smaller = reacts faster to form."),
            ("Last season (prior)", f"A player's last-season averages enter his form as if that "
             f"season ended {config.PREV_SEASON_GAP_GW:g} gameweeks before GW1, so they matter early "
             f"on and fade as this season's matches pile up. When the model is fitted, a last-season "
             f"row counts {config.PREV_SEASON_WEIGHT:g}x a row from this season."),
            ("New arrival", "A player with no FPL record last season (promoted club, signing from "
             "abroad, academy). Until he has played, his form starts from the typical value for his "
             "position."),
            ("Opponent strength", "The model sees each opponent by its characteristics, not its "
             "name: goals conceded per game so far, FPL's defensive strength rating, and its "
             f"defensive style (the 20 clubs grouped into {config.ARCHETYPE_K} styles)."),
            ("Player type", "Players grouped position by position by their per-90 profile and "
             "minutes share (k-means clustering). A type is named after what sets it apart, e.g. "
             "'High xA · Low minutes'."),
            ("SD vs average (High / Low)", "How far a type's centre sits from the position "
             "average, in standard deviations (SD). +1 SD is well above average; 'High' or 'Low' in "
             "a type's name means at least 0.5 SD."),
        ]),
        ("The plan", [
            ("Planning horizon", f"How many gameweeks ahead the planner looks: "
             f"{config.PLANNING_HORIZON_GW} by default (--horizon)."),
            ("Horizon decay", f"Each gameweek further ahead counts {d:g}x the one before (1, "
             f"{d:g}, {d * d:.2f}, ...), because forecasts get less certain with distance."),
            ("Hit threshold", f"A -{hit} hit is only taken if it gains more than "
             f"{config.HIT_THRESHOLD_POINTS:g} expected points over the horizon. The extra over "
             f"{hit} is a safety margin for forecast error."),
            ("Free-transfer value", f"A free transfer still banked at the end of the plan is worth "
             f"{config.FREE_TRANSFER_VALUE:g} expected points, so the planner doesn't spend one on "
             "a tiny gain."),
            ("Bench weight", f"Bench players' expected points count at {config.BENCH_WEIGHT:g}x, "
             "since they only score when a starter doesn't play. Keeps the bench cheap, not useless."),
            ("No transfers (baseline)", "Your current squad kept for the whole horizon, with its "
             "best XI and captain each week. The plan is measured against it."),
            ("Plan vs no transfers", "The extra expected points the plan earns over the horizon, "
             "after hits, compared with making no transfers."),
            ("Best squad from scratch", f"The best {config.SQUAD_SIZE} that £{config.BUDGET:g}m "
             "buys for next gameweek, ignoring your current squad."),
            ("Optimiser (integer program)", "The solver that picks the squad, XI, captain and "
             "transfers. It finds the highest expected points that obey every FPL rule, exactly, "
             "rather than by rule of thumb."),
            ("Candidate pool", "The players given to the solver: anyone who could be optimal. A player is "
             "left out when several others at his position cost no more and score at least as much "
             "every week."),
            ("Chip threshold", "A chip is only played if it gains more than its threshold, in that "
             "gameweek's expected points: " + ", ".join(
                 f"{name.replace('_', ' ')} {pts:g}" for name, pts in config.CHIP_THRESHOLD_POINTS.items())
             + ". It's what the chip should be worth if saved for a better week. A chip about to "
             "expire is worth nothing saved, so it's used in its best remaining week."),
            ("Objective", "The single number the planner maximises: decayed XI, captain and bench points, "
             "minus the hit penalty and chip thresholds, plus the value of transfers still banked."),
            ("Transfer ladder", "The best plan with exactly 0, 1, 2, ... transfers this week: what each "
             "extra move is worth, and whether a hit pays."),
            ("Money freed", "What a sale raises minus what its replacement costs: money another move can use."),
            ("Binding constraint", "A rule the plan is pressed against, e.g. a club already at 3 players "
             "or no money left; loosening it would improve the plan."),
        ]),
        ("Model check", [
            ("Backtest (walk-forward)", "Each past gameweek predicted using only what was known "
             "before it, then compared with what happened: how the model would really have done."),
            ("Form and Season average (baselines)", "Two simple rules the model has to beat: "
             "recency-weighted points per gameweek, and the plain average so far this season."),
            ("MAE", "Mean absolute error: the average miss, in points per fixture. Lower is better."),
            ("RMSE", "Root mean squared error: like MAE, but big misses count more. Lower is better."),
            ("Rank correlation", "How well a method puts players in the right order within a "
             "gameweek (Spearman, -1 to 1). Ordering is what picking needs. Higher is better."),
            ("Top-11 points", "The real points of the 11 players a method rated highest, per "
             "gameweek: the most direct test of picking. Higher is better."),
            ("Permutation importance", "How much worse a model gets when one input is shuffled (so it "
             "carries no information): the bigger, the more the model leans on it."),
            ("Opponent style", "The 20 clubs grouped into tight, middling and leaky defences (k-means on "
             "goals conceded and FPL's defence ratings)."),
        ]),
        ("News check", [
            ("Action score", "Claude's 0-100 score for how strongly to act on this week's moves "
             "after reading the news: 0 = wait for news, 100 = act now with confidence. News "
             "adjusts, never overrides, the maths."),
            ("Verdict", "act (go ahead), adjust (go ahead with the change Claude names) or wait "
             "(hold the transfers until the news is clearer)."),
            ("Player flag", "fit, doubt, out or unknown, from news Claude found, with the source "
             "it rests on. 'unknown' means it found nothing reliable."),
        ]),
        ("Seasons", [
            ("Player code", "FPL gives every player a new id each season but a permanent code. "
             "Seasons are linked on the code, then on name + date of birth for anyone left."),
            ("Link check", "How many of the links agree with FPL's own record of each player's "
             "past seasons."),
            ("Risers and fallers", "The change in points per 90 between last season and this, "
             "for the same players."),
        ]),
    ]
