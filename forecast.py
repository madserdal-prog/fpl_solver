"""
FPL Forecast Model -- Part 2 (fpl-auto-pipeline-guide.md)
=========================================================
Computes expected points (xP) per player, 5 gameweeks ahead, using the
formula from the guide:

  xP(p, g) = P(player) * (
      xG90 * game_factor(g) * xG_points
    + xA90 * game_factor(g) * xA_points
    + baseline_points(position)
    + expected_bonus(ICT index)
    - FDR_adjustment(opponent)
  )

Input (in practice, once Part 1/the collector is built):
  - master_data.json  -- bootstrap-static dump (elements: xG90, xA90,
    ICT, chance_of_playing_next_round, status, now_cost, element_type, team)
  - fixtures.json      -- dump of the fixtures endpoint (FDR per team/round)

Synthetic versions of both are generated here, since this environment has
no network access to the real FPL API. Swap out make_synthetic_master()
and make_synthetic_fixtures() with real collector output once Part 1 is
built -- the rest of the model is unchanged regardless of data source.

Output: forecast_gw{N}.json, in the exact format solver_general.py's
load_pool_from_forecast() expects.
"""

import json
import numpy as np

ELEMENT_TYPES = ["GK", "DEF", "MID", "FWD"]

# Points per goal depend on position (FPL rules); assists give the same points regardless of position.
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3
APPEARANCE_POINTS = 2.0   # baseline_points(position) -- simplified as equal across positions
BONUS_SCALE = 1.2         # scales the ICT index into expected bonus points
FDR_WEIGHT = 0.5          # how much opponent difficulty pulls xP down/up


# ---------------------------------------------------------------------------
# Synthetic test data (replaced by real collector dumps later)
# ---------------------------------------------------------------------------

def make_synthetic_master(seed: int = 42, n_teams: int = 10):
    """
    Synthetic version of bootstrap-static's "elements" list. Field names
    mirror the real FPL fields (expected_goals_per_90, ict_index, etc.) so
    this is a straight swap once real data is wired in.
    """
    rng = np.random.default_rng(seed)
    counts = {"GK": 6, "DEF": 15, "MID": 15, "FWD": 14}
    elements = []
    pid = 1
    for pos, n in counts.items():
        for _ in range(n):
            quality = float(rng.beta(2.0, 4.0))
            now_cost = int(round(max(39, 39 + quality * 100 + rng.normal(0, 2))))  # tenths, i.e. 3.9-13.9m

            xg90 = round(max(0.0, {"GK": 0.0, "DEF": 0.05, "MID": 0.25, "FWD": 0.45}[pos]
                              * (0.4 + quality) + rng.normal(0, 0.03)), 3)
            xa90 = round(max(0.0, {"GK": 0.0, "DEF": 0.08, "MID": 0.22, "FWD": 0.15}[pos]
                              * (0.4 + quality) + rng.normal(0, 0.03)), 3)
            ict = round(max(0.0, quality * 12 + rng.normal(0, 1.0)), 2)

            # Injury/rotation status: most available, some doubtful/injured
            status_roll = rng.random()
            if status_roll < 0.05:
                status, chance = "i", 0        # injured
            elif status_roll < 0.12:
                status, chance = "d", 50       # doubtful
            else:
                status, chance = "a", 100      # available

            elements.append({
                "id": pid,
                "web_name": f"{pos}{pid}",
                "team": f"Team{(pid % n_teams) + 1}",
                "element_type": pos,   # real API: 1-4 (int) -- map to GK/DEF/MID/FWD on real integration
                "now_cost": now_cost,
                "expected_goals_per_90": xg90,
                "expected_assists_per_90": xa90,
                "ict_index_per_90": ict,
                "chance_of_playing_next_round": chance,
                "status": status,
                "minutes": int(rng.integers(300, 1000)),  # trustworthy sample by default in this demo
                "form": round(max(0.0, quality * 6 + rng.normal(0, 0.5)), 1),  # roughly consistent with quality
                # Minutes probability -- in practice derived from the last 5 matches' minutes pattern
                "start_probability": float(np.clip(0.5 + quality * 0.4 + rng.normal(0, 0.05), 0.03, 0.98)),
            })
            pid += 1
    return {"elements": elements}


def make_synthetic_fixtures(teams, seed: int = 7, start_gw: int = 4, horizon: int = 5):
    """
    Synthetic fixtures list with FDR (1=easy, 5=hard), including some
    double and blank gameweeks to show that game_factor(g) is handled correctly.
    """
    rng = np.random.default_rng(seed)
    fixtures = []
    for gw in range(start_gw, start_gw + horizon):
        shuffled = list(teams)
        rng.shuffle(shuffled)
        pairs = list(zip(shuffled[0::2], shuffled[1::2]))

        # ~15% chance that a round is a double/blank round, for illustration
        if rng.random() < 0.15 and len(pairs) >= 2:
            # Create a double gameweek: the first pair's away team gets an extra
            # match, effectively removing that team from another pair (a blank for that opponent).
            extra_home, extra_away = pairs[0]
            fixtures.append({
                "event": gw, "team_h": extra_home, "team_a": extra_away,
                "team_h_difficulty": int(rng.integers(2, 5)),
                "team_a_difficulty": int(rng.integers(2, 5)),
            })

        for h, a in pairs:
            fixtures.append({
                "event": gw, "team_h": h, "team_a": a,
                "team_h_difficulty": int(rng.integers(1, 6)),
                "team_a_difficulty": int(rng.integers(1, 6)),
            })
    return fixtures


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

def build_fixture_lookup(fixtures):
    """
    Returns dict[(team, gameweek)] -> list of FDR values the team faces in
    that round (empty list = blank, two values = double gameweek).
    """
    lookup = {}
    for fx in fixtures:
        gw = fx["event"]
        lookup.setdefault((fx["team_h"], gw), []).append(fx["team_a_difficulty"])
        lookup.setdefault((fx["team_a"], gw), []).append(fx["team_h_difficulty"])
    return lookup


MINUTES_TRUST_THRESHOLD = 300  # matches collector.py's MIN_SAMPLE_MINUTES -- below this,
                                # the underlying-stats projection is blended with real "form"
FORM_INFLUENCE = 0.35  # minimum weight given to recent actual form, REGARDLESS of sample size --
                        # this is what catches an established player whose season-cumulative
                        # underlying stats are still good from an early hot patch, but whose
                        # recent actual output has genuinely collapsed. Without this, a player
                        # with plenty of minutes gets full "trust" and the small-sample blend
                        # above does nothing to catch a pure recency decline.

# Rough conservative "typical" points-per-match priors, used to shrink the
# outcome anchor (form/points_per_game) itself when it's based on very few
# minutes. Without this, a single freak match (e.g. a defender's one 63-minute
# appearance producing a goal + clean sheet + max bonus = 14 points) gets
# trusted at FULL face value as if it were a demonstrated ability, not a
# one-off outlier -- this caused a real bug: a cheap, injury-doubtful
# defender with a single huge match projected at 9.45 xP, well above what
# any of his actual underlying stats supported, purely because "form: 14.0"
# was taken at face value with no regard for the sample size behind it.
BASELINE_PPG_BY_POSITION = {"GK": 2.5, "DEF": 3.0, "MID": 3.0, "FWD": 3.0}


def p_play(element):
    """P(player) -- probability of meaningful playing time this round."""
    if element["status"] == "i":
        return 0.0
    chance = element.get("chance_of_playing_next_round")
    if chance is not None and chance < 100:
        return chance / 100.0
    return element.get("start_probability", 0.75)


def compute_xp_series(element, fixture_lookup, start_gw, horizon):
    """
    Computes xP for one player for each round in the horizon, following
    the guide's formula. Returns a list of length `horizon`.

    IMPORTANT -- form-blending safeguard: expected_goals_per_90 etc. are
    derived from underlying "process" stats (chance quality), which can
    diverge sharply from FPL's own "form" field (actual recent returns),
    especially in a small minutes sample (e.g. early season, or right
    after a player returns from injury). A player can look great on
    process stats while genuinely producing nothing in real matches --
    exactly the situation that caused an unproven, zero-form player to get
    picked as captain. Below MINUTES_TRUST_THRESHOLD minutes, the raw
    process-based estimate is blended with FPL's own form figure,
    proportional to how little we should trust the small sample.

    SECOND safeguard (the outcome anchor's own reliability): the raw form/
    points_per_game figure ITSELF gets shrunk toward a conservative
    position-average baseline when minutes are low -- not just down-
    weighted in the blend above. A single-match outlier shouldn't be
    trusted at face value just because it's the only data point available;
    it should be treated as noisy, not definitive.
    """
    pos = element["element_type"]
    goal_pts = GOAL_POINTS[pos]
    play_prob = p_play(element)
    minutes = element.get("minutes", MINUTES_TRUST_THRESHOLD)  # assume trustworthy if absent (synthetic data)
    trust = min(1.0, minutes / MINUTES_TRUST_THRESHOLD) if minutes else 0.0

    # Use whichever real outcome-signal is actually informative: "form" resets
    # to 0 for everyone at the start of a new season (no games played yet this
    # season), so it can't discriminate between players at that point. Before
    # any real form exists, fall back to points_per_game -- last season's
    # (or season-to-date's) ACTUAL converted points rate, which is what's
    # displayed as "Pts/Match" on FPL's own site. This is what catches a
    # player whose underlying process stats (expected_goals_per_90 etc.)
    # look good, but who has a genuine history of NOT converting that
    # process into real points.
    form = element.get("form", 0.0)
    raw_outcome_anchor = form if form > 0 else element.get("points_per_game", 0.0)
    baseline_ppg = BASELINE_PPG_BY_POSITION.get(pos, 3.0)
    outcome_anchor = trust * raw_outcome_anchor + (1 - trust) * baseline_ppg

    series = []
    for i in range(horizon):
        gw = start_gw + i
        difficulties = fixture_lookup.get((element["team"], gw), [])
        game_factor = len(difficulties)  # 0 = blank, 1 = normal, 2 = double

        if game_factor == 0:
            series.append(0.0)
            continue

        avg_fdr = sum(difficulties) / len(difficulties)
        fdr_adjustment = (avg_fdr - 3.0) * FDR_WEIGHT  # >0 => tough opponent, pulls xP down

        attack_value = (
            element["expected_goals_per_90"] * game_factor * goal_pts
            + element["expected_assists_per_90"] * game_factor * ASSIST_POINTS
        )
        expected_bonus = (element["ict_index_per_90"] / 10.0) * BONUS_SCALE * game_factor

        process_based_xp = play_prob * (
            attack_value
            + APPEARANCE_POINTS * game_factor
            + expected_bonus
            - fdr_adjustment * game_factor
        )

        # Blend: trust the process-based estimate proportional to sample size,
        # fall back toward the real outcome-anchor (form, or points_per_game if
        # form isn't informative yet) for the rest. This directly prevents a
        # small, lucky (or unlucky) sample of underlying stats from dominating
        # the projection when actual results say otherwise.
        outcome_based_xp = play_prob * outcome_anchor * game_factor
        sample_blend = trust * process_based_xp + (1 - trust) * outcome_based_xp

        # Second, SEPARATE safeguard: give the outcome-anchor a minimum say
        # regardless of how many minutes the player has -- catches an
        # established player whose season-cumulative rates are still propped
        # up by an early hot patch, even though recent actual form (or
        # historical points_per_game) says otherwise. Without this,
        # sample_blend above trusts the process estimate completely once
        # minutes are high, no matter how badly the outcome-anchor disagrees.
        xp = (1 - FORM_INFLUENCE) * sample_blend + FORM_INFLUENCE * outcome_based_xp

        series.append(round(max(0.0, xp), 2))

    return series



def build_forecast(master, fixtures, start_gw, horizon=5):
    teams = sorted(set(e["team"] for e in master["elements"]))
    fixture_lookup = build_fixture_lookup(fixtures)

    players = []
    for element in master["elements"]:
        xp_series = compute_xp_series(element, fixture_lookup, start_gw, horizon)
        players.append({
            "id": element["id"],
            "name": element["web_name"],
            "xP": xp_series,
        })

    return {
        "gameweek": start_gw,
        "horizon": horizon,
        "players": players,
    }


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    master = make_synthetic_master()
    teams = sorted(set(e["team"] for e in master["elements"]))
    fixtures = make_synthetic_fixtures(teams, start_gw=4, horizon=5)

    forecast = build_forecast(master, fixtures, start_gw=4, horizon=5)

    with open("master_data.json", "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False)
    with open("forecast_gw4.json", "w", encoding="utf-8") as f:
        json.dump(forecast, f, ensure_ascii=False, indent=2)

    top10 = sorted(forecast["players"], key=lambda p: -p["xP"][0])[:10]
    print("Top 10 players by xP this round (GW4):")
    for p in top10:
        print(f"  {p['name']:>8}  xP(GW4..GW8) = {p['xP']}")

    blanks = [p for p in forecast["players"] if p["xP"][0] == 0.0]
    print(f"\n{len(blanks)} players have a blank gameweek in GW4 (illustrates game_factor=0 handling).")
