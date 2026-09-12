"""
record_transfer.py -- tell the pipeline about a transfer you already made
============================================================================
fetch_my_team.py can only see your squad AFTER a gameweek's deadline has
locked it in -- it has no way to see a transfer you've just made but that
hasn't locked yet. Without this, any re-run before the deadline works from
a stale current_squad and can suggest something that conflicts with what
you actually did.

This updates my_team.json directly: swaps the player out/in, adjusts bank
using the REAL sell-value rule (not raw market price -- see my_team.py's
compute_sell_value), decrements free_transfers, and keeps purchase_prices
in sync for future sell-value calculations.

Usage:
  python3 record_transfer.py --in Rogers --out Okafor
  python3 record_transfer.py --in Rogers Barry --out Okafor Nmecha   (multiple at once)
"""

import argparse
import json

import my_team as my_team_module


def resolve_player_id(name: str, master_data: dict) -> int:
    """Exact (case-insensitive) match on web_name. Errors out clearly on
    zero or multiple matches, rather than guessing -- same principle as
    find_player_ids.py."""
    matches = [e for e in master_data["elements"] if e["web_name"].lower() == name.lower()]
    if not matches:
        # Fall back to substring match, in case of a slightly different spelling
        matches = [e for e in master_data["elements"] if name.lower() in e["web_name"].lower()]
    if not matches:
        raise SystemExit(f"No player found matching '{name}'. Check spelling, or use "
                          f"find_player_ids.py to look up the exact web_name.")
    if len(matches) > 1:
        options = "\n".join(f"  id={e['id']}  {e['web_name']} ({e['team']}, {e['element_type']})"
                             for e in matches)
        raise SystemExit(f"'{name}' matches multiple players:\n{options}\n"
                          f"Re-run with the exact web_name shown above.")
    return matches[0]["id"]


def record_transfers(my_team: dict, master_data: dict, in_names: list, out_names: list) -> dict:
    if len(in_names) != len(out_names):
        raise SystemExit(f"--in has {len(in_names)} name(s) but --out has {len(out_names)} -- "
                          f"they must be given in matching pairs.")

    master_by_id = {e["id"]: e for e in master_data["elements"]}
    current_squad = set(my_team["current_squad"])
    purchase_prices = dict(my_team.get("purchase_prices", {}))
    bank = my_team["bank"]
    free_transfers = my_team["free_transfers"]

    summary_lines = []

    for in_name, out_name in zip(in_names, out_names):
        in_id = resolve_player_id(in_name, master_data)
        out_id = resolve_player_id(out_name, master_data)

        if out_id not in current_squad:
            raise SystemExit(f"'{out_name}' (id={out_id}) is not in your current_squad -- "
                              f"can't sell a player you don't own. Check my_team.json.")
        if in_id in current_squad:
            raise SystemExit(f"'{in_name}' (id={in_id}) is already in your current_squad.")

        in_player = master_by_id[in_id]
        out_player = master_by_id[out_id]
        if in_player["element_type"] != out_player["element_type"]:
            print(f"WARNING: {in_name} ({in_player['element_type']}) and {out_name} "
                  f"({out_player['element_type']}) are different positions -- this may leave "
                  f"your squad with an invalid position count. The next solver run will fail "
                  f"clearly if so (MILP infeasible), rather than silently producing something wrong.")

        # Real sell value, using tracked purchase price if we have one
        out_purchase_price = purchase_prices.get(str(out_id))
        if out_purchase_price is not None:
            sell_value = my_team_module.compute_sell_value(out_purchase_price, out_player["now_cost"] / 10.0)
        else:
            sell_value = out_player["now_cost"] / 10.0  # not tracked -- assume no profit/loss

        in_price = in_player["now_cost"] / 10.0
        bank = round(bank + sell_value - in_price, 1)

        current_squad.discard(out_id)
        current_squad.add(in_id)
        purchase_prices.pop(str(out_id), None)
        purchase_prices[str(in_id)] = in_price  # best available estimate, same convention as
                                                  # update_purchase_prices() for a newly-acquired player

        free_transfers = max(0, free_transfers - 1)

        summary_lines.append(
            f"  IN: {in_player['web_name']} (£{in_price}m)   "
            f"OUT: {out_player['web_name']} (sold for £{sell_value}m)"
        )

    if bank < 0:
        raise SystemExit(f"This would leave bank at £{bank}m, which is negative -- "
                          f"you can't actually afford this in real FPL. Nothing was saved.")

    updated = dict(my_team)
    updated["current_squad"] = sorted(current_squad)
    updated["bank"] = bank
    updated["free_transfers"] = free_transfers
    updated["purchase_prices"] = purchase_prices
    updated["notes"] = (my_team.get("notes", "") + " | Manually recorded transfer(s) via "
                         "record_transfer.py, ahead of fetch_my_team.py seeing them.").strip(" |")

    my_team_module.validate_my_team(updated)
    return updated, summary_lines


def main():
    parser = argparse.ArgumentParser(description="Record a transfer you already made, before the deadline locks it in")
    parser.add_argument("--in", dest="in_names", nargs="+", required=True, help="Player name(s) coming IN")
    parser.add_argument("--out", dest="out_names", nargs="+", required=True, help="Player name(s) going OUT")
    parser.add_argument("--my-team", default="my_team.json")
    parser.add_argument("--master", default="master_data.json")
    args = parser.parse_args()

    with open(args.my_team, encoding="utf-8") as f:
        my_team = json.load(f)
    with open(args.master, encoding="utf-8") as f:
        master_data = json.load(f)

    updated, summary_lines = record_transfers(my_team, master_data, args.in_names, args.out_names)

    with open(args.my_team, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print("Recorded:")
    for line in summary_lines:
        print(line)
    print(f"\nNew bank: £{updated['bank']}m  |  Free transfers remaining: {updated['free_transfers']}")
    print(f"Wrote {args.my_team}")


if __name__ == "__main__":
    main()
