"""
Critic -- Part 4 (fpl-auto-pipeline-guide.md)
===============================================
Takes the solver's proposed squad and challenges it with information the
solver doesn't have: injury news, transfer sagas, press conferences,
rotation signals.

EXPLICIT PROHIBITION: the critic must NEVER compute its own xP numbers. It
finds and assesses QUALITATIVE risk (high/medium/low) per player, with a
brief justification and source. The solver (quantitative) and the critic
(qualitative) are deliberately kept separate -- see the guide's
division-of-labor principle.

Requires an Anthropic API key (ANTHROPIC_API_KEY environment variable),
since this script runs independently of claude.ai and needs its own
access to web search. This is NOT free -- API calls cost per token,
separate from any claude.ai/Claude Pro subscription. Check current
pricing at https://docs.claude.com if you're unsure about cost.

Usage:
  python3 critic.py --solution solver_solution_gw4.json --out critic_flags_gw4.json
"""

import argparse
import json
import os
import re
import sys

import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"  # see product-self-knowledge for the current model name at time of use

SYSTEM_PROMPT = """\
You are a qualitative risk assessor for a Fantasy Premier League squad. You are given a proposed \
squad from a mathematical solver. Your ONLY job is to search for fresh information (last 1-2 \
weeks) about each player and assess the risk that they will NOT play/perform as expected this round.

Check specifically for:
- Injuries (ongoing, or recently returned from a long-term injury -- "eased back in",
  "managed minutes", "rotational role" are all red flags even if the player is technically "available")
- Active transfer sagas (the player could be sold/distracted/benched before the transfer window closes)
- Suspensions/bans
- Press conference statements about rotation/resting

You are EXPLICITLY PROHIBITED from computing, estimating, or implying your own point values (xP, \
expected points, etc.) for any player. You assess QUALITATIVE risk, not quantitative value. If you \
find yourself about to say something like "this should get him roughly X points", stop -- that is not your job.

Respond with ONLY valid JSON in this exact format, no other text before or after:
{
  "flags": [
    {
      "player_name": "...",
      "risk_level": "high" | "medium" | "low",
      "reason": "brief, concrete justification with what was found",
      "source": "brief source reference (site/date if possible)",
      "recommendation": "avoid_captain" | "avoid_start" | "monitor" | "no_concern"
    }
  ],
  "summary": "2-3 sentences summarizing the most important findings across the squad"
}

Include ONLY players where you found something real to say -- do not create entries for players \
with no news of significance. If you find nothing concerning for a player, do not include them.
"""


def call_anthropic_with_search(squad_names: list, api_key: str) -> dict:
    """
    Calls the Anthropic API with the web_search tool enabled. Handles the
    fact that the response may consist of multiple content blocks
    (tool_use, tool_result, text) -- see the anthropic_api_in_artifacts
    documentation for why this can't assume the answer is in content[0].
    """
    user_message = (
        "Proposed squad this round: " + ", ".join(squad_names) + ". "
        "Search for fresh information about these players and assess risk per your instructions."
    )

    payload = {
        "model": MODEL,
        "max_tokens": 4000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    resp = requests.post(ANTHROPIC_API_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def extract_text_blocks(api_response: dict) -> str:
    """
    Extracts all text blocks from an Anthropic API response and joins them
    together. Ignores tool_use/tool_result blocks (the web search's
    "work"), since we only want the critic's final text answer.
    """
    texts = [block["text"] for block in api_response.get("content", [])
             if block.get("type") == "text"]
    return "\n".join(texts)


def parse_critic_response(raw_text: str) -> dict:
    """
    Parses the critic's JSON response. Two robustness measures, since the
    model can deliver multiple text blocks (e.g. a short lead-in sentence
    BEFORE the actual JSON, even when the system prompt asks for "JSON only"):
      1. Strip markdown fences (```json ... ```) wherever they appear in the text.
      2. If the whole string still isn't valid JSON, extract just the
         substring from the FIRST '{' to the LAST '}' and try that instead --
         this ignores any explanatory text outside the JSON object itself.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError(f"Found no JSON object in the critic's response:\n{cleaned[:500]}")

    candidate = cleaned[first:last + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse the critic's response as JSON: {e}\n\nAttempted substring:\n{candidate[:500]}"
        )


def flags_to_solver_inputs(critic_output: dict, master_data: dict,
                            high_penalty=8.0, medium_penalty=3.0):
    """
    Translates the critic's qualitative flags into the quantitative inputs
    solver_general.solve() expects:
      - risk_level="high"   -> large risk_penalty AND blocked from captaincy
      - risk_level="medium" -> moderate risk_penalty
      - risk_level="low"    -> small/no risk_penalty (visible in the log only)

    Name matching is a case-insensitive substring search against
    master_data's web_name -- works for most cases, but MULTIPLE MATCHES
    are resolved by taking the lowest id (same "pick one even if uncertain"
    principle as find_player_ids.py). Double-check manually if in doubt.
    """
    name_to_id = {}
    for e in master_data["elements"]:
        name_to_id.setdefault(e["web_name"].lower(), e["id"])

    risk_penalty = {}
    blocked_captain_ids = set()
    unmatched = []

    for flag in critic_output.get("flags", []):
        player_name = flag["player_name"]
        matches = [pid for name, pid in name_to_id.items() if player_name.lower() in name]
        if not matches:
            unmatched.append(player_name)
            continue
        pid = matches[0]

        if flag["risk_level"] == "high":
            risk_penalty[pid] = high_penalty
            blocked_captain_ids.add(pid)
        elif flag["risk_level"] == "medium":
            risk_penalty[pid] = medium_penalty

    return risk_penalty, blocked_captain_ids, unmatched


def run_critic(solution_path: str, master_data_path: str, out_path: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: the ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with e.g.:", file=sys.stderr)
        print('  Windows (PowerShell): $env:ANTHROPIC_API_KEY = "your-key"', file=sys.stderr)
        print("  Mac/Linux:            export ANTHROPIC_API_KEY=your-key", file=sys.stderr)
        sys.exit(1)

    with open(solution_path, encoding="utf-8") as f:
        solution = json.load(f)
    with open(master_data_path, encoding="utf-8") as f:
        master_data = json.load(f)

    squad_names = solution["squad"]
    print(f"Sending {len(squad_names)} players to the critic for a news check...")

    api_response = call_anthropic_with_search(squad_names, api_key)
    raw_text = extract_text_blocks(api_response)
    critic_output = parse_critic_response(raw_text)

    risk_penalty, blocked_captain_ids, unmatched = flags_to_solver_inputs(
        critic_output, master_data)

    result = {
        "critic_output": critic_output,
        "risk_penalty": risk_penalty,
        "blocked_captain_ids": sorted(blocked_captain_ids),
        "unmatched_names": unmatched,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== Critic assessment ===")
    print(critic_output.get("summary", "(no summary returned)"))
    for flag in critic_output.get("flags", []):
        print(f"  [{flag['risk_level'].upper()}] {flag['player_name']}: {flag['reason']} "
              f"({flag.get('recommendation', '?')})")
    if unmatched:
        print(f"\nWARNING: could not find a player ID for: {unmatched} -- check spelling manually.")

    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Critic -- news check on the solver's squad")
    parser.add_argument("--solution", required=True, help="Path to solver_solution_gw{N}.json")
    parser.add_argument("--master", default="master_data.json")
    parser.add_argument("--out", default="critic_flags.json")
    args = parser.parse_args()
    run_critic(args.solution, args.master, args.out)


if __name__ == "__main__":
    main()
