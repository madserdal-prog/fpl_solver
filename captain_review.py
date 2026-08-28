"""
Captain review -- xAI Grok API
=================================
Reviews the solver's own top-5 captain shortlist and either confirms the
top pick or recommends a DIFFERENT candidate from that SAME shortlist,
with a qualitative reason (small sample size, position risk, etc.).

GUARDRAILS (same principle as critic.py's solver/critic separation):
  - Can ONLY recommend a player already in the shortlist -- never invents
    a new name.
  - Can NEVER compute or state its own point projection -- it may only
    reference the xp values it was given.
  - Recommendation-only: this does NOT change solver_solution_gw{N}.json's
    actual captain. It's surfaced in the report for you to act on or not.

Requires XAI_API_KEY (a personal xAI/Grok account, separate from any
Anthropic billing). Costs money per call -- check current pricing at
console.x.ai if unsure.

Usage (called from run_real.py, not typically run standalone):
  review_captain(final_solution, pool, master_data, api_key)
"""

import json
import os
import re

import requests

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4.1-fast"  # cheaper tier -- this is a reasoning-over-given-data task,
                                  # not a task needing the flagship model's extra capability
SHORTLIST_SIZE = 5

SYSTEM_PROMPT = """\
You are reviewing a Fantasy Premier League captain shortlist. You will be given up to 5 \
candidates that a mathematical solver already ranked highest by expected points (xp), each \
with supporting context: minutes played this season, recent form, points per game, price, \
position, and injury/doubt status.

Your job is to either CONFIRM the solver's top-ranked candidate, or recommend a DIFFERENT \
candidate FROM THIS SAME SHORTLIST if there is a clear, specific qualitative reason -- for \
example: the top pick's number rests on a very small sample of minutes (a likely one-match \
fluke, not a repeatable rate), or the top pick is a defender with no known penalty/set-piece \
duties (defenders have a structurally low captaincy ceiling in real FPL -- a goal+assist+clean \
sheet combination is rare), when a similarly-ranked alternative doesn't carry that same risk.

You are EXPLICITLY FORBIDDEN from:
- Recommending any player NOT in the shortlist given to you.
- Computing, estimating, or stating your own point projection for any player. Only ever refer \
to the xp values you were given, never invent or adjust a number.

If you have no clear, specific qualitative objection to the top pick, confirm it -- do not \
override just to have an opinion.

Respond with ONLY valid JSON in this exact format, no other text before or after:
{
  "recommended_captain": "<name, exactly as given in the shortlist>",
  "same_as_solver_top_pick": true or false,
  "reasoning": "1-3 sentences explaining the recommendation"
}
"""


def build_candidate_shortlist(final_solution: dict, pool: list, master_data: dict,
                               top_n: int = SHORTLIST_SIZE) -> list:
    """
    Builds the top-N starting XI candidates by xp, each enriched with the
    context needed for a qualitative review (minutes, form, price, status).
    """
    pool_by_id = {p["id"]: p for p in pool}
    master_by_id = {e["id"]: e for e in master_data["elements"]}

    starters = [pool_by_id[i] for i in final_solution["starting_xi_ids"] if i in pool_by_id]
    starters_sorted = sorted(starters, key=lambda p: -p["xp"])[:top_n]

    shortlist = []
    for p in starters_sorted:
        master_entry = master_by_id.get(p["id"], {})
        shortlist.append({
            "name": p["name"],
            "xp": p["xp"],
            "price": p["price"],
            "position": p["position"],
            "minutes_this_season": master_entry.get("minutes"),
            "form": master_entry.get("form"),
            "points_per_game": master_entry.get("points_per_game"),
            "status": master_entry.get("status"),
            "chance_of_playing_next_round": master_entry.get("chance_of_playing_next_round"),
        })
    return shortlist


def call_grok(shortlist: list, solver_top_pick: str, api_key: str, model: str = DEFAULT_MODEL) -> dict:
    user_message = (
        f"Solver's top-ranked pick (highest xp): {solver_top_pick}\n\n"
        f"Shortlist:\n{json.dumps(shortlist, indent=2, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,  # low -- this is a judgment call that should be consistent, not creative
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    resp = requests.post(XAI_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_review_response(raw_text: str) -> dict:
    """Same robust-parsing approach as critic.py -- strip fences, fall back
    to extracting the {...} substring if the model added any stray text."""
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError(f"Found no JSON object in the review response:\n{cleaned[:500]}")
    return json.loads(cleaned[first:last + 1])


def review_captain(final_solution: dict, pool: list, master_data: dict, api_key: str,
                    model: str = DEFAULT_MODEL) -> dict:
    """
    Returns:
      {"recommended_captain": str, "same_as_solver_top_pick": bool,
       "reasoning": str, "shortlist": list}
    or a dict with "error" set if the review couldn't be completed --
    callers should treat that as "skip the review, don't block the run".
    """
    shortlist = build_candidate_shortlist(final_solution, pool, master_data)
    if not shortlist:
        return {"error": "Empty shortlist -- nothing to review."}

    solver_top_pick = shortlist[0]["name"]  # shortlist is already sorted by xp descending

    try:
        api_response = call_grok(shortlist, solver_top_pick, api_key, model)
        raw_text = api_response["choices"][0]["message"]["content"]
        result = parse_review_response(raw_text)
    except Exception as e:
        return {"error": str(e)}

    valid_names = {c["name"] for c in shortlist}
    if result.get("recommended_captain") not in valid_names:
        return {"error": f"Review recommended a player outside the shortlist "
                          f"({result.get('recommended_captain')!r}) -- ignoring."}

    result["shortlist"] = shortlist
    return result


if __name__ == "__main__":
    # Minimal smoke test using synthetic data -- does NOT call the real API.
    pool = [
        {"id": 1, "name": "PlayerA", "xp": 6.2, "price": 8.0, "position": "MID"},
        {"id": 2, "name": "PlayerB", "xp": 5.9, "price": 7.5, "position": "FWD"},
    ]
    master_data = {"elements": [
        {"id": 1, "minutes": 77, "form": 17.0, "points_per_game": 17.0, "status": "a",
         "chance_of_playing_next_round": None},
        {"id": 2, "minutes": 900, "form": 5.5, "points_per_game": 5.0, "status": "a",
         "chance_of_playing_next_round": None},
    ]}
    solution = {"starting_xi_ids": [1, 2]}
    shortlist = build_candidate_shortlist(solution, pool, master_data, top_n=5)
    print(json.dumps(shortlist, indent=2, ensure_ascii=False))
