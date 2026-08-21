"""
Free critic -- zero-cost alternative to critic.py
=====================================================
critic.py calls the Anthropic API (paid, per-token) with live web search.
This module instead uses data FPL already gives away for free in
bootstrap-static and that collector.py now captures:

  - "status": "i" (injured) / "d" (doubtful) / "s" (suspended) / "a" (available)
  - "chance_of_playing_next_round": FPL's own percentage estimate
  - "news": free-text injury/doubt description, WRITTEN BY FPL'S OWN
    EDITORS -- e.g. "Hamstring injury - Expected back 15 Sep". This is the
    single most useful free signal available; it's the same information
    commercial services build businesses around, and it costs nothing
    because it's already part of the API call collector.py makes anyway.

What this WILL catch: acute injuries, suspensions, and anything FPL's own
editors have flagged with doubt text.

What this will NOT catch, and critic.py (or a manual chat with Claude)
would: rotation risk for a technically-fit player recently back from a
long injury lay-off, active transfer sagas, and anything from press
conferences that hasn't yet been reflected in FPL's own status field.
FPL's status/news fields lag real news by anywhere from hours to a day or
two. See the "manual weekly check" workflow for covering that gap for free.

Output format is IDENTICAL to critic.py's critic_flags_*.json, so it's a
drop-in replacement anywhere run_real.py expects critic output.

Usage:
  python3 critic_free.py --solution solver_solution_gw4.json --out critic_flags_gw4.json
"""

import argparse
import json


def assess_player(element: dict):
    """
    Returns a flag dict (same shape as critic.py's flags) if this player's
    free FPL data suggests risk, or None if there's nothing to flag.
    """
    status = element.get("status", "a")
    chance = element.get("chance_of_playing_next_round")
    news = (element.get("news") or "").strip()

    if status == "i":
        return {
            "player_name": element["web_name"],
            "risk_level": "high",
            "reason": news or "Marked injured (status='i') in FPL's own data",
            "source": "FPL API status field" + (f" + news: \"{news}\"" if news else ""),
            "recommendation": "avoid_captain",
        }

    if status == "s":
        return {
            "player_name": element["web_name"],
            "risk_level": "high",
            "reason": news or "Marked suspended (status='s') in FPL's own data",
            "source": "FPL API status field",
            "recommendation": "avoid_start",
        }

    if status == "d" or (chance is not None and chance < 75):
        return {
            "player_name": element["web_name"],
            "risk_level": "medium",
            "reason": news or f"Marked doubtful, chance_of_playing_next_round={chance}",
            "source": "FPL API status/chance_of_playing_next_round fields",
            "recommendation": "monitor",
        }

    if chance is not None and chance < 100:
        return {
            "player_name": element["web_name"],
            "risk_level": "low",
            "reason": news or f"chance_of_playing_next_round={chance} (minor doubt)",
            "source": "FPL API chance_of_playing_next_round field",
            "recommendation": "monitor",
        }

    if news:
        # Available (status="a", chance=100) but FPL's editors still left a note --
        # e.g. an illness that's since cleared but the text hasn't been wiped yet.
        return {
            "player_name": element["web_name"],
            "risk_level": "low",
            "reason": news,
            "source": "FPL API news field",
            "recommendation": "monitor",
        }

    return None


def flags_to_solver_inputs(flags: list, master_data: dict, high_penalty=8.0, medium_penalty=3.0):
    """Same translation logic as critic.py's flags_to_solver_inputs(), kept separate
    to avoid a dependency between the free and paid critics."""
    name_to_id = {e["web_name"]: e["id"] for e in master_data["elements"]}

    risk_penalty = {}
    blocked_captain_ids = set()
    unmatched = []

    for flag in flags:
        pid = name_to_id.get(flag["player_name"])
        if pid is None:
            unmatched.append(flag["player_name"])
            continue
        if flag["risk_level"] == "high":
            risk_penalty[pid] = high_penalty
            blocked_captain_ids.add(pid)
        elif flag["risk_level"] == "medium":
            risk_penalty[pid] = medium_penalty

    return risk_penalty, blocked_captain_ids, unmatched


def run_free_critic(solution_path: str, master_data_path: str, out_path: str):
    with open(solution_path, encoding="utf-8") as f:
        solution = json.load(f)
    with open(master_data_path, encoding="utf-8") as f:
        master_data = json.load(f)

    squad_names = set(solution["squad"])
    elements_by_name = {e["web_name"]: e for e in master_data["elements"]}

    flags = []
    for name in squad_names:
        element = elements_by_name.get(name)
        if element is None:
            continue
        flag = assess_player(element)
        if flag:
            flags.append(flag)

    if flags:
        high = [f["player_name"] for f in flags if f["risk_level"] == "high"]
        medium = [f["player_name"] for f in flags if f["risk_level"] == "medium"]
        parts = []
        if high:
            parts.append(f"{len(high)} high-risk ({', '.join(high)})")
        if medium:
            parts.append(f"{len(medium)} medium-risk ({', '.join(medium)})")
        summary = f"Found {' and '.join(parts)} based on FPL's own status/news data." if parts else \
            "Minor notes found, no major concerns."
    else:
        summary = "No status/news flags found in FPL's own data for this squad."

    critic_output = {"flags": flags, "summary": summary}
    risk_penalty, blocked_captain_ids, unmatched = flags_to_solver_inputs(flags, master_data)

    result = {
        "critic_output": critic_output,
        "risk_penalty": risk_penalty,
        "blocked_captain_ids": sorted(blocked_captain_ids),
        "unmatched_names": unmatched,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== Free critic assessment (FPL's own status/news data, no API cost) ===")
    print(summary)
    for flag in flags:
        print(f"  [{flag['risk_level'].upper()}] {flag['player_name']}: {flag['reason']} "
              f"({flag['recommendation']})")
    print(f"\nReminder: this only catches what FPL's own editors have already flagged. "
          f"It will NOT catch rotation risk for a technically-fit returning player, or "
          f"an active transfer saga -- consider a quick manual check for those (see the "
          f"weekly workflow notes).")
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Free critic -- zero-cost news check using FPL's own status/news fields")
    parser.add_argument("--solution", required=True, help="Path to solver_solution_gw{N}.json")
    parser.add_argument("--master", default="master_data.json")
    parser.add_argument("--out", default="critic_flags.json")
    args = parser.parse_args()
    run_free_critic(args.solution, args.master, args.out)


if __name__ == "__main__":
    main()
