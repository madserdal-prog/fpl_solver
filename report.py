"""
Slack reporter -- Part 5 (fpl-auto-pipeline-guide.md)
========================================================
Posts the solver's recommendation to Slack so you get notified before the
deadline instead of having to remember to check a terminal.

Uses a Slack Incoming Webhook, which is FREE -- no Slack API billing, no
per-message cost, just a URL. Set it up once:

  1. Go to https://api.slack.com/apps -> "Create New App" -> "From scratch"
  2. Name it (e.g. "FPL Bot"), pick your workspace
  3. Left sidebar -> "Incoming Webhooks" -> toggle it on
  4. "Add New Webhook to Workspace" -> pick a channel (or your own DM)
  5. Copy the webhook URL (looks like https://hooks.slack.com/services/T.../B.../xxx)
  6. Set it as an environment variable:
       Windows (PowerShell): $env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/..."
       Mac/Linux:             export SLACK_WEBHOOK_URL=https://hooks.slack.com/...

Usage:
  python3 report.py --solution solver_solution_gw4.json --gw 4
"""

import argparse
import json
import os
import sys

import requests


def format_message(solution: dict, gw: int, critic_summary: str = None,
                    news_lookup_errors: list = None, news_systematically_blocked: bool = False) -> str:
    lines = [f"*GW{gw} solver recommendation*", ""]

    # Transfers -- explicit IN/OUT labels instead of an arrow, since "A <- B"
    # reads ambiguously (which one are you actually buying?). One bullet per
    # swap is also easier to scan than a single long comma-separated line,
    # especially with 5+ transfers on an initial-team build.
    lines.append("*Transfers:*")
    if solution.get("transfers_in"):
        for player_in, player_out in zip(solution["transfers_in"], solution["transfers_out"]):
            lines.append(f"  • IN: *{player_in}*   OUT: {player_out}")
        if solution.get("hit_cost"):
            lines.append(f"  _(hit taken: -{solution['hit_cost']} points)_")
    else:
        lines.append("  none")
    lines.append("")

    # Full squad -- starting XI first (captain marked with (C)), bench listed
    # separately, so you can see the whole 15 the solver is actually working
    # with, not just what changed from last week.
    starting_xi = solution.get("starting_xi", [])
    squad = solution.get("squad", [])
    captain = solution.get("captain")
    bench = [p for p in squad if p not in starting_xi]

    xi_display = [f"{p} (C)" if p == captain else p for p in starting_xi]
    lines.append(f"*Starting XI:* {', '.join(xi_display)}")
    if bench:
        lines.append(f"*Bench:* {', '.join(bench)}")
    lines.append("")

    lines.append(f"*Captain:* {captain}")
    lines.append(f"*Expected points:* {solution['expected_points']}")

    if solution.get("risk_adjusted_players"):
        lines.append(f":warning: Risk-adjusted (news flags found): {', '.join(solution['risk_adjusted_players'])}")

    if critic_summary:
        lines.append(f"\n_Critic summary: {critic_summary}_")

    # IMPORTANT: without this, a fully-failed Google News check (e.g. blocked
    # from GitHub Actions' shared IPs) and a genuinely clean squad look IDENTICAL
    # in Slack -- both just say "no flags found". Surface the distinction explicitly.
    if news_systematically_blocked:
        lines.append(
            f":rotating_light: Google News lookup failed for ALL players this run -- looks like a "
            f"systematic block (e.g. Google rejecting GitHub Actions' IP range), not random flakiness. "
            f"Treat \"no flags\" above with real caution: only FPL's own status data was actually checked."
        )
    elif news_lookup_errors:
        lines.append(
            f":warning: Google News lookup failed for {len(news_lookup_errors)} player(s) this run "
            f"(isolated hiccup) -- FPL's own status data was still checked normally for them."
        )

    lines.append("\nReview and make your transfers on fantasy.premierleague.com before the deadline.")
    return "\n".join(lines)


def send_to_slack(message: str, webhook_url: str):
    resp = requests.post(webhook_url, json={"text": message}, timeout=15)
    resp.raise_for_status()


def main():
    parser = argparse.ArgumentParser(description="Post the solver's recommendation to Slack")
    parser.add_argument("--solution", required=True, help="Path to solver_solution_gw{N}.json")
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument("--critic-flags", default=None,
                         help="Path to critic_flags_gw{N}.json, to include the critic's summary")
    args = parser.parse_args()

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("ERROR: SLACK_WEBHOOK_URL environment variable is not set.", file=sys.stderr)
        print("See the top of report.py for free setup instructions.", file=sys.stderr)
        sys.exit(1)

    with open(args.solution, encoding="utf-8") as f:
        solution = json.load(f)

    critic_summary = None
    news_lookup_errors = None
    news_systematically_blocked = False
    if args.critic_flags and os.path.exists(args.critic_flags):
        with open(args.critic_flags, encoding="utf-8") as f:
            critic_flags_data = json.load(f)
        critic_summary = critic_flags_data["critic_output"].get("summary")
        news_lookup_errors = critic_flags_data.get("news_lookup_errors")
        news_systematically_blocked = critic_flags_data.get("news_systematically_blocked", False)

    message = format_message(solution, args.gw, critic_summary, news_lookup_errors, news_systematically_blocked)
    send_to_slack(message, webhook_url)
    print("Posted to Slack.")
    print("\n--- Message sent ---")
    print(message)


if __name__ == "__main__":
    main()
