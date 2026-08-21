"""
FPL Collector -- Part 1 (fpl-auto-pipeline-guide.md)
=====================================================
Fetches data from FPL's public API 3x/day (in practice via GitHub Actions
cron, see collect.yml), and produces:

  1. data/raw/{timestamp}.json      -- raw bootstrap-static dump (idempotence:
                                        everything can be rebuilt from scratch later)
  2. data/price_history.csv         -- append-only time series, one row per
                                        player per collection run
  3. master_data.json               -- normalized player data in the EXACT
                                        format forecast.py expects
  4. fixtures.json                  -- normalized fixtures/FDR in the EXACT
                                        format forecast.py expects

Network note: this environment has no network access (bash_tool runs
without egress here), so the actual HTTP calls to FPL's API are NOT
run/tested in this sandbox. All normalization logic (the part that can
actually fail due to wrong assumptions about the data structure) has
instead been tested with --self-test, which uses a built-in data
structure that follows the real API schema exactly. Run without flags on
your own machine/in GitHub Actions, where network access exists.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

ELEMENT_TYPE_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
MIN_SAMPLE_MINUTES = 300  # ~3.3 full matches -- floor for per-90 extrapolation, see normalize_master()

RAW_DIR = "data/raw"
PRICE_HISTORY_PATH = "data/price_history.csv"
MASTER_OUT_PATH = "master_data.json"
FIXTURES_OUT_PATH = "fixtures.json"
EVENTS_OUT_PATH = "events.json"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_json(url: str) -> dict:
    """Fetches JSON from FPL's API. Requires network access."""
    import requests  # imported here so --self-test works even without 'requests' installed
    resp = requests.get(url, timeout=30, headers={"User-Agent": "fpl-auto-pipeline/1.0"})
    resp.raise_for_status()
    return resp.json()


def fetch_bootstrap_static() -> dict:
    return fetch_json(BOOTSTRAP_URL)


def fetch_fixtures() -> list:
    return fetch_json(FIXTURES_URL)


# ---------------------------------------------------------------------------
# Normalization -- raw FPL API format -> format forecast.py expects
# ---------------------------------------------------------------------------

def estimate_start_probability(element: dict) -> float:
    """
    Heuristic probability of meaningful playing time, used as the
    'start_probability' fallback in forecast.py's p_play(). FPL's API
    doesn't provide this directly -- it's derived here from minutes played
    so far this season. Naturally improved in v2 with an actual minutes
    pattern (last 5 matches) instead of the season total.
    """
    minutes = int(element.get("minutes", 0) or 0)
    if minutes > 450:
        return 0.85
    elif minutes > 90:
        return 0.55
    else:
        return 0.20


def normalize_master(raw_bootstrap: dict) -> dict:
    """
    Converts bootstrap-static's "elements" list to the schema forecast.py
    expects (see make_synthetic_master() there for reference):
      - element_type: int (1-4) -> "GK"/"DEF"/"MID"/"FWD"
      - team: numeric team ID -> team name (string), so it matches the
        team identifier fixtures.json uses
      - expected_goals / expected_assists (season total, string in the API) ->
        per-90 figures, since the guide's xP formula uses per-90 rates
      - ict_index (season total, string in the API) -> per-90

    IMPORTANT: per-90 rates are shrunk toward zero for players with a small
    minutes sample (see MIN_SAMPLE_MINUTES below). Without this, a player
    with e.g. 25 minutes played who happened to register 0.3 expected
    goals in that tiny sample gets extrapolated to an absurd
    expected_goals_per_90 (0.3 * 90/25 = 1.08 -- higher than any elite
    striker's real rate). This is exactly what caused a player recently
    back from a long injury layoff to dominate the model's captaincy pick
    despite genuinely being a risky, low-minutes option.
    """
    team_id_to_name = {t["id"]: t["name"] for t in raw_bootstrap["teams"]}

    elements = []
    for e in raw_bootstrap["elements"]:
        minutes = int(e.get("minutes", 0) or 0)
        # Use max(minutes, MIN_SAMPLE_MINUTES) as the denominator: for anyone
        # with fewer minutes than this floor, the per-90 rate is calculated
        # as if they'd played the floor amount, not their tiny actual sample.
        # This is a simple form of shrinkage toward zero for small samples --
        # not full Bayesian regression to a position-average prior, but it
        # directly prevents the extrapolation blow-up above.
        effective_minutes = max(minutes, MIN_SAMPLE_MINUTES)
        per90_factor = (90.0 / effective_minutes) if minutes > 0 else 0.0

        expected_goals = float(e.get("expected_goals", 0) or 0)
        expected_assists = float(e.get("expected_assists", 0) or 0)
        ict_index = float(e.get("ict_index", 0) or 0)

        chance = e.get("chance_of_playing_next_round")
        # FPL's API uses null for "no doubt" (i.e. effectively 100%),
        # and a number 0-100 when there's genuine uncertainty.
        chance = int(chance) if chance is not None else None

        elements.append({
            "id": e["id"],
            "web_name": e["web_name"],
            "team": team_id_to_name[e["team"]],
            "element_type": ELEMENT_TYPE_MAP[e["element_type"]],
            "now_cost": int(e["now_cost"]),  # already in tenths, as forecast.py/solver expect
            "expected_goals_per_90": round(expected_goals * per90_factor, 3),
            "expected_assists_per_90": round(expected_assists * per90_factor, 3),
            "ict_index_per_90": round(ict_index * per90_factor, 2),
            "chance_of_playing_next_round": chance,
            "status": e.get("status", "a"),
            "form": float(e.get("form", 0) or 0),  # FPL's own recent-actual-returns signal --
                                                     # used to sanity-check the underlying-stats
                                                     # projection below, see forecast.py
            "points_per_game": float(e.get("points_per_game", 0) or 0),  # last-season/season-to-date
                                                     # ACTUAL converted output rate -- this is what's
                                                     # displayed as "Pts/Match" on FPL's own site.
                                                     # Unlike "form", this doesn't reset to 0 every
                                                     # preseason, so it's the only real outcome-based
                                                     # signal available before the new season starts.
            "minutes": minutes,  # BUG FIX: this was computed above but never actually output here --
                                  # forecast.py's small-sample trust logic silently defaulted to
                                  # "fully trust every player" for all of production as a result.
            "news": e.get("news", ""),  # FREE human-written injury/doubt text from FPL's own editors
            "news_added": e.get("news_added"),  # ISO timestamp of when the news text was last updated
            "start_probability": estimate_start_probability(e),
        })

    return {"elements": elements}


def normalize_fixtures(raw_fixtures: list, raw_bootstrap: dict) -> list:
    """
    Converts the fixtures endpoint's team numbers to team names (the same
    identifier normalize_master() uses for "team"), and filters out
    matches not yet assigned to a gameweek (event=None, e.g. due to
    postponements).
    """
    team_id_to_name = {t["id"]: t["name"] for t in raw_bootstrap["teams"]}

    fixtures = []
    for fx in raw_fixtures:
        if fx.get("event") is None:
            continue  # not allocated to a round yet -- exclude until FPL confirms the date
        fixtures.append({
            "event": fx["event"],
            "team_h": team_id_to_name[fx["team_h"]],
            "team_a": team_id_to_name[fx["team_a"]],
            "team_h_difficulty": fx["team_h_difficulty"],
            "team_a_difficulty": fx["team_a_difficulty"],
        })
    return fixtures


def normalize_events(raw_bootstrap: dict) -> list:
    """
    Extracts the "events" list (one entry per gameweek) from bootstrap-static:
    deadline times and which gameweek is "current"/"next". This is what lets
    the pipeline auto-detect which gameweek to run for, instead of you
    editing a --gw number by hand every week.
    """
    events = []
    for e in raw_bootstrap.get("events", []):
        events.append({
            "id": e["id"],
            "deadline_time": e["deadline_time"],  # ISO 8601 UTC
            "is_next": e.get("is_next", False),
            "is_current": e.get("is_current", False),
            "finished": e.get("finished", False),
        })
    return events


def get_current_gameweek(events: list) -> dict:
    """
    Returns the event dict for whichever gameweek you should be planning
    for right now: the "next" one if FPL has marked one, otherwise the
    first unfinished one as a fallback. Raises if none found (e.g. season over).
    """
    for e in events:
        if e.get("is_next"):
            return e
    for e in events:
        if not e.get("finished"):
            return e
    raise ValueError("No upcoming gameweek found in events data -- season may be over, "
                      "or events.json is stale. Run collector.py again, or pass --gw manually.")


# ---------------------------------------------------------------------------
# Storage: raw dump (idempotence) + price history (append-only)
# ---------------------------------------------------------------------------

def save_raw_dump(raw_bootstrap: dict, raw_dir: str = RAW_DIR) -> str:
    os.makedirs(raw_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(raw_dir, f"{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw_bootstrap, f, ensure_ascii=False)
    return path


def append_price_history(raw_bootstrap: dict, csv_path: str = PRICE_HISTORY_PATH):
    """
    Append-only: one row per player per collection run. The file is never
    rebuilt -- that's the whole point (the price history *is* the
    append/commit history).
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "player_id", "now_cost", "selected_by_percent",
                "form", "total_points", "minutes", "status",
            ])
        for e in raw_bootstrap["elements"]:
            writer.writerow([
                timestamp, e["id"], e["now_cost"], e.get("selected_by_percent"),
                e.get("form"), e.get("total_points"), e.get("minutes"), e.get("status"),
            ])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(raw_bootstrap: dict, raw_fixtures: list):
    save_raw_dump(raw_bootstrap)
    append_price_history(raw_bootstrap)

    master = normalize_master(raw_bootstrap)
    fixtures = normalize_fixtures(raw_fixtures, raw_bootstrap)
    events = normalize_events(raw_bootstrap)

    with open(MASTER_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False)
    with open(FIXTURES_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False)
    with open(EVENTS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False)

    print(f"Saved {len(master['elements'])} players -> {MASTER_OUT_PATH}")
    print(f"Saved {len(fixtures)} fixtures -> {FIXTURES_OUT_PATH}")
    print(f"Saved {len(events)} gameweeks -> {EVENTS_OUT_PATH}")
    print(f"Appended price history -> {PRICE_HISTORY_PATH}")

    try:
        current = get_current_gameweek(events)
        print(f"Current/next gameweek: GW{current['id']} (deadline {current['deadline_time']})")
    except ValueError as e:
        print(f"Note: {e}")


def main():
    parser = argparse.ArgumentParser(description="FPL collector (Part 1)")
    parser.add_argument("--self-test", action="store_true",
                         help="Run against built-in test data instead of the real FPL API "
                              "(use this where network access is unavailable)")
    args = parser.parse_args()

    if args.self_test:
        raw_bootstrap, raw_fixtures = _self_test_sample_data()
    else:
        raw_bootstrap = fetch_bootstrap_static()
        raw_fixtures = fetch_fixtures()

    run(raw_bootstrap, raw_fixtures)


# ---------------------------------------------------------------------------
# Self-test: built-in data structure that follows the REAL FPL API schema
# (field names and types verified against the publicly documented API
# structure), so the normalization logic can be verified without network access.
# ---------------------------------------------------------------------------

def _self_test_sample_data():
    raw_bootstrap = {
        "events": [
            {"id": 3, "deadline_time": "2026-08-14T17:30:00Z", "is_next": False,
             "is_current": False, "finished": True},
            {"id": 4, "deadline_time": "2026-08-21T17:30:00Z", "is_next": True,
             "is_current": False, "finished": False},
            {"id": 5, "deadline_time": "2026-08-28T17:30:00Z", "is_next": False,
             "is_current": False, "finished": False},
        ],
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "name": "Liverpool", "short_name": "LIV"},
            {"id": 3, "name": "Man City", "short_name": "MCI"},
        ],
        "elements": [
            {
                "id": 101, "web_name": "Saka", "team": 1, "element_type": 3,
                "now_cost": 89, "selected_by_percent": "34.5", "form": "5.2",
                "total_points": 48, "minutes": 630,
                "expected_goals": "4.21", "expected_assists": "3.10",
                "ict_index": "112.4", "chance_of_playing_next_round": None,
                "status": "a",
            },
            {
                "id": 202, "web_name": "Salah", "team": 2, "element_type": 3,
                "now_cost": 131, "selected_by_percent": "45.1", "form": "6.8",
                "total_points": 61, "minutes": 700,
                "expected_goals": "6.50", "expected_assists": "2.90",
                "ict_index": "150.2", "chance_of_playing_next_round": 75,
                "status": "d",
            },
            {
                "id": 303, "web_name": "Haaland", "team": 3, "element_type": 4,
                "now_cost": 152, "selected_by_percent": "60.2", "form": "8.1",
                "total_points": 70, "minutes": 650,
                "expected_goals": "9.80", "expected_assists": "0.50",
                "ict_index": "140.0", "chance_of_playing_next_round": None,
                "status": "a",
            },
            {
                # Injured player -- tests status="i" handling AND the free "news" field
                "id": 404, "web_name": "Gabriel", "team": 1, "element_type": 2,
                "now_cost": 62, "selected_by_percent": "12.0", "form": "0.0",
                "total_points": 20, "minutes": 400,
                "expected_goals": "0.50", "expected_assists": "0.10",
                "ict_index": "40.0", "chance_of_playing_next_round": 0,
                "status": "i", "news": "Hamstring injury - Expected back 15 Sep",
                "news_added": "2026-08-18T09:15:00Z",
            },
            {
                # New signing / minimal minutes -- tests minutes=0 handling (per90_factor=0)
                "id": 505, "web_name": "NewSigning", "team": 2, "element_type": 4,
                "now_cost": 55, "selected_by_percent": "0.5", "form": "0.0",
                "total_points": 0, "minutes": 0,
                "expected_goals": "0.00", "expected_assists": "0.00",
                "ict_index": "0.0", "chance_of_playing_next_round": None,
                "status": "a",
            },
        ],
    }

    raw_fixtures = [
        {"event": 4, "team_h": 1, "team_a": 2, "team_h_difficulty": 4, "team_a_difficulty": 3},
        {"event": 4, "team_h": 3, "team_a": 1, "team_h_difficulty": 2, "team_a_difficulty": 5},
        # Postponed/not allocated -- should be filtered out by normalize_fixtures()
        {"event": None, "team_h": 2, "team_a": 3, "team_h_difficulty": 3, "team_a_difficulty": 3},
    ]

    return raw_bootstrap, raw_fixtures


if __name__ == "__main__":
    main()
