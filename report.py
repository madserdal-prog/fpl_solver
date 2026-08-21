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


def format_message(solution: dict, gw: int, critic_summary: str = None) -> str:
    lines = [f"*GW{gw} solver recommendation*"]

    if solution.get("transfers_in"):
        transfers = ", ".join(
            f"{i} <- {o}" for i, o in zip(solution["transfers_in"], solution["transfers_out"])
        )
        hit_note = f" (hit taken: -{solution['hit_cost']})" if solution.get("hit_cost") else ""
        lines.append(f"Transfers: {transfers}{hit_note}")
    else:
        lines.append("Transfers: none")

    lines.append(f"Captain: *{solution['captain']}*")
    lines.append(f"Expected points: {solution['expected_points']}")

    if solution.get("risk_adjusted_players"):
        lines.append(f":warning: Risk-adjusted (news flags found): {', '.join(solution['risk_adjusted_players'])}")

    if critic_summary:
        lines.append(f"\n_Critic summary: {critic_summary}_")

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
    if args.critic_flags and os.path.exists(args.critic_flags):
        with open(args.critic_flags, encoding="utf-8") as f:
            critic_summary = json.load(f)["critic_output"].get("summary")

    message = format_message(solution, args.gw, critic_summary)
    send_to_slack(message, webhook_url)
    print("Posted to Slack.")
    print("\n--- Message sent ---")
    print(message)


if __name__ == "__main__":
    main()
