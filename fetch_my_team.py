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

import my_team as my_team_module

ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"


def fetch_entry_picks(team_id: int, gw: int) -> dict:
    import requests
    url = ENTRY_PICKS_URL.format(team_id=team_id, gw=gw)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "fpl-auto-pipeline/1.0"})
    resp.raise_for_status()
    return resp.json()


def build_my_team_from_entry(entry_data: dict, chips_available=None, locked_players=None) -> dict:
    """
    entry_data is the response from fetch_entry_picks(). Structure
    (publicly documented FPL API format):
      entry_data["picks"]          -> list of {"element": player_id, ...} (15 of them)
      entry_data["entry_history"]["bank"]            -> bank in tenths (e.g. 3 = 0.3m)
      entry_data["entry_history"]["event_transfers"] -> transfers already used this round
    """
    squad = [p["element"] for p in entry_data["picks"]]
    bank = entry_data["entry_history"]["bank"] / 10.0

    # FPL's API doesn't state the number of REMAINING free transfers directly
    # in this endpoint -- it depends on the previous round's rollover. As a
    # v1, 1 free transfer is assumed as a conservative default; adjust
    # manually in my_team.json afterward if you know you've banked transfers (up to 5).
    free_transfers = 1

    my_team = {
        "current_squad": squad,
        "bank": round(bank, 1),
        "free_transfers": free_transfers,
        "chips_available": chips_available or [],
        "locked_players": locked_players or [],
        "notes": "Fetched automatically from the FPL entry endpoint. Check free_transfers manually "
                 "if you've banked transfers, and fill in chips_available yourself.",
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
    args = parser.parse_args()

    if args.self_test:
        entry_data = _self_test_entry_data()
    else:
        if args.team_id is None or args.gw is None:
            parser.error("--team-id and --gw are required (or use --self-test)")
        entry_data = fetch_entry_picks(args.team_id, args.gw)

    my_team = build_my_team_from_entry(entry_data)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(my_team, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out}:")
    print(json.dumps(my_team, indent=2, ensure_ascii=False))
    print("\nREMEMBER: fill in 'chips_available' and double-check 'free_transfers' manually.")


def _self_test_entry_data():
    return {
        "picks": [{"element": i} for i in [101, 202, 303, 404, 505, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        "entry_history": {"bank": 4, "event_transfers": 1},  # bank=4 -> 0.4m
    }


if __name__ == "__main__":
    main()
