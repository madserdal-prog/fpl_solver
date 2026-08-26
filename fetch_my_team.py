"""
Fetch your actual team from FPL -- fills in my_team.json automatically
=====================================================================
FPL's entry/{team_id}/event/{gw}/picks/ endpoint is public (no login
required) and gives your actual squad, bank, and transfer history for a
given round. This removes the need to type in the player IDs manually.

You still need to supply chips_available and locked_players yourself,
since those are strategic choices the endpoint can't know anything about.

Find your team_id: go to the "Points" page for your team on
fantasy.premierleague.com while logged in -- the URL looks like this:
  https://fantasy.premierleague.com/entry/{team_id}/event/{gw}
team_id is the number in the URL.
"""

import argparse
import json
import os

import requests

import my_team as my_team_module

ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
ENTRY_HISTORY_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/history/"

CHIPS_GIVING_UNLIMITED_TRANSFERS = {"wildcard", "freehit"}
MAX_BANKED_FREE_TRANSFERS = 5


def fetch_entry_picks(team_id: int, gw: int) -> dict:
    import requests
    url = ENTRY_PICKS_URL.format(team_id=team_id, gw=gw)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "fpl-auto-pipeline/1.0"})
    resp.raise_for_status()
    return resp.json()


def fetch_entry_history(team_id: int) -> dict:
    """
    FPL's /entry/{team_id}/history/ endpoint (public, no login needed)
    returns the FULL season history in one call:
      history["current"] -> list of {"event": gw_number, "event_transfers": N, ...}
                             for every gameweek played so far
      history["chips"]   -> list of {"name": "wildcard"/"freehit"/"bboost"/"3xc", "event": gw_number, ...}
                             for every chip played so far

    This is what lets us COMPUTE free_transfers correctly instead of
    guessing or requiring a manual correction every week -- FPL already
    tracks the exact history needed; we don't need to maintain our own.
    """
    url = ENTRY_HISTORY_URL.format(team_id=team_id)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "fpl-auto-pipeline/1.0"})
    resp.raise_for_status()
    return resp.json()


def compute_free_transfers(history_data: dict, upcoming_gw: int) -> int:
    """
    Simulates FPL's actual free-transfer rollover rule forward from GW1 to
    determine how many free transfers are available going into `upcoming_gw`:

      - Start with 1 free transfer for GW1.
      - Each gameweek you don't use your free transfer, it banks (+1),
        up to a maximum of 5.
      - If you use more transfers than you have banked, you take a hit,
        but you can't go negative -- your banked count just drops to 0.
      - Playing a Wildcard or Free Hit does NOT consume your banked
        transfers at all -- that gameweek is treated as if you made zero
        transfers for banking purposes (confirmed directly by the
        Premier League's own chip guide: "you will KEEP your banked
        transfers" when playing a Wildcard), even though the chip itself
        gives you unlimited free changes that week.

    Returns 1 (the safe default) if history is empty (e.g. before GW1).
    """
    chip_by_event = {c["event"]: c["name"] for c in history_data.get("chips", [])}
    gameweeks = sorted(history_data.get("current", []), key=lambda g: g["event"])

    free_transfers = 1
    for gw_stats in gameweeks:
        event = gw_stats["event"]
        if event >= upcoming_gw:
            break

        transfers_used = gw_stats.get("event_transfers", 0)
        if chip_by_event.get(event) in CHIPS_GIVING_UNLIMITED_TRANSFERS:
            transfers_used = 0  # chip transfers don't touch the banked count

        banked_after = max(0, free_transfers - transfers_used)
        free_transfers = min(MAX_BANKED_FREE_TRANSFERS, banked_after + 1)

    return free_transfers


def build_my_team_from_entry(entry_data: dict, chips_available=None, locked_players=None,
                              existing: dict = None, computed_free_transfers: int = None) -> dict:
    """
    entry_data is the response from fetch_entry_picks(). Structure
    (publicly documented FPL API format):
      entry_data["picks"]          -> list of {"element": player_id, ...} (15 of them)
      entry_data["entry_history"]["bank"]            -> bank in tenths (e.g. 3 = 0.3m)
      entry_data["entry_history"]["event_transfers"] -> transfers already used this round

    existing: a previously-saved my_team.json (if any). chips_available and
    locked_players are carried over unchanged from it when present -- those
    remain judgment calls automation shouldn't silently overwrite (e.g. a
    planned wildcard week).

    computed_free_transfers: the result of compute_free_transfers() against
    real FPL history, if available. When given, this REPLACES any manually-set
    or carried-over free_transfers value, since it's now derived correctly
    from FPL's own data rather than guessed. Falls back to the old
    guess-or-carry-over behavior only if history couldn't be fetched.
    """
    squad = [p["element"] for p in entry_data["picks"]]
    bank = entry_data["entry_history"]["bank"] / 10.0

    if computed_free_transfers is not None:
        free_transfers = computed_free_transfers
        ft_note = "free_transfers computed from your real FPL transfer history (see compute_free_transfers())."
    elif existing:
        free_transfers = existing.get("free_transfers", 1)
        ft_note = "free_transfers carried over from the previous file (history lookup unavailable this run)."
    else:
        free_transfers = 1
        ft_note = "free_transfers defaulted to 1 (no history available, no previous file to carry over)."

    if existing:
        chips_available = chips_available if chips_available is not None else existing.get("chips_available", [])
        locked_players = locked_players if locked_players is not None else existing.get("locked_players", [])
        note = f"current_squad/bank auto-refreshed from FPL. {ft_note} chips_available/locked_players " \
               f"carried over from the previous file -- edit them manually if they need correcting."
    else:
        chips_available = chips_available or []
        locked_players = locked_players or []
        note = f"Fetched automatically from the FPL entry endpoint. {ft_note} Fill in chips_available yourself."

    my_team = {
        "current_squad": squad,
        "bank": round(bank, 1),
        "free_transfers": free_transfers,
        "chips_available": chips_available,
        "locked_players": locked_players,
        "notes": note,
    }
    my_team_module.validate_my_team(my_team)
    return my_team


def main():
    parser = argparse.ArgumentParser(description="Fetch your FPL team and build my_team.json")
    parser.add_argument("--team-id", type=int, help="Your FPL team_id (from the URL on fantasy.premierleague.com)")
    parser.add_argument("--gw", type=int, help="Current/last completed gameweek")
    parser.add_argument("--out", default="my_team.json")
    parser.add_argument("--self-test", action="store_true",
                         help="Run against built-in test data instead of the real API")
    parser.add_argument("--allow-missing", action="store_true",
                         help="If the picks endpoint 404s (e.g. this gameweek hasn't locked yet), "
                              "exit quietly instead of crashing -- useful when this runs unattended "
                              "in a scheduled workflow, before knowing for certain whether a given "
                              "gameweek has actually locked.")
    args = parser.parse_args()

    existing = None
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            existing = json.load(f)

    if args.self_test:
        entry_data = _self_test_entry_data()
        computed_ft = compute_free_transfers(_self_test_history_data(), upcoming_gw=args.gw + 1)
    else:
        if args.team_id is None or args.gw is None:
            parser.error("--team-id and --gw are required (or use --self-test)")
        try:
            entry_data = fetch_entry_picks(args.team_id, args.gw)
        except requests.exceptions.HTTPError as e:
            if args.allow_missing:
                print(f"GW{args.gw} picks not available yet ({e}) -- leaving {args.out} unchanged.")
                return
            raise

        computed_ft = None
        try:
            history = fetch_entry_history(args.team_id)
            computed_ft = compute_free_transfers(history, upcoming_gw=args.gw + 1)
        except Exception as e:
            print(f"Could not fetch/compute free_transfers from history ({e}) -- "
                  f"falling back to carried-over/default value.")

    my_team = build_my_team_from_entry(entry_data, existing=existing, computed_free_transfers=computed_ft)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(my_team, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out}:")
    print(json.dumps(my_team, indent=2, ensure_ascii=False))
    if existing:
        print("\ncurrent_squad and bank refreshed from FPL. chips_available/free_transfers/"
              "locked_players carried over unchanged -- edit manually if they need correcting.")
    else:
        print("\nREMEMBER: fill in 'chips_available' and double-check 'free_transfers' manually.")


def _self_test_entry_data():
    return {
        "picks": [{"element": i} for i in [101, 202, 303, 404, 505, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        "entry_history": {"bank": 4, "event_transfers": 1},  # bank=4 -> 0.4m
    }


def _self_test_history_data():
    """
    Realistic 5-gameweek scenario to test compute_free_transfers() against:
      GW1: 0 transfers (initial squad)      -> banks to 2 for GW2
      GW2: 0 transfers (banked again)       -> banks to 3 for GW3
      GW3: 2 transfers, no hit (had 3 FT)   -> drops to 1 banked, +1 = 2 for GW4
      GW4: wildcard played, 8 transfers     -> doesn't touch banked count, +1 = 3 for GW5
      GW5: 0 transfers                      -> banks to 4 for GW6
    Expected free_transfers going into GW6: 4
    """
    return {
        "current": [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},
            {"event": 3, "event_transfers": 2},
            {"event": 4, "event_transfers": 8},
            {"event": 5, "event_transfers": 0},
        ],
        "chips": [
            {"name": "wildcard", "event": 4},
        ],
    }


if __name__ == "__main__":
    main()
