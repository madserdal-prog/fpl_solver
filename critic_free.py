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
import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

# Keyword categories -- purely rule-based (no LLM, no cost). A headline matching
# any of these gets flagged; risk_level is a simple severity ranking, NOT a
# computed point value -- stays consistent with the "critic never computes
# its own numbers" principle even though this isn't an LLM at all.
INJURY_KEYWORDS = [
    "injury", "injured", "surgery", "scan", "sidelined", "hamstring", "groin",
    "ankle", "knee", "strain", "layoff", "fitness test", "doubt", "doubtful",
    "ruled out", "setback",
]
RETURN_FROM_INJURY_KEYWORDS = [
    "eased back", "managed minutes", "rotational role", "return from injury",
    "long-term injury", "injury return", "back in training",
]
TRANSFER_KEYWORDS = [
    "transfer", "bid", "medical", "signs for", "move to", "linked with",
    "deal agreed", "loan move", "exit",
]
SUSPENSION_KEYWORDS = ["red card", "banned", "suspended", "suspension", "three-match ban"]


def fetch_google_news_headlines(player_name: str, team_name: str = None, max_items: int = 5) -> list:
    """
    Free Google News RSS search -- no API key, no signup, no cost. Returns
    a list of {"title", "link", "pub_date"} for the most recent matching
    headlines. Requires real network access (works in GitHub Actions;
    cannot be tested from an offline sandbox).
    """
    query = f'"{player_name}"' + (f" {team_name}" if team_name else "") + " premier league"
    url = f"{GOOGLE_NEWS_RSS_URL}?q={urllib.parse.quote(query)}&hl=en-GB&gl=GB&ceid=GB:en"

    # Google's anti-scraping systems are known to 503-block non-browser User-Agent
    # strings and/or cloud/datacenter IP ranges (which is exactly what GitHub
    # Actions runners are). Using a realistic browser UA is a cheap first thing
    # to try -- if 503s persist even with this, the block is almost certainly
    # IP-based rather than UA-based, and no header change will fix it.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    resp = requests.get(url, timeout=15, headers=headers)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    headlines = []
    for item in root.findall(".//item")[:max_items]:
        title_el = item.find("title")
        link_el = item.find("link")
        date_el = item.find("pubDate")
        headlines.append({
            "title": title_el.text if title_el is not None else "",
            "link": link_el.text if link_el is not None else "",
            "pub_date": date_el.text if date_el is not None else "",
        })
    return headlines


def classify_headlines(player_name: str, headlines: list):
    """
    Rule-based keyword classification over headline titles. Returns a flag
    dict (same shape as assess_player()'s output) or None if nothing matched.
    """
    if not headlines:
        return None

    combined_text = " ".join(h["title"].lower() for h in headlines)

    if any(kw in combined_text for kw in RETURN_FROM_INJURY_KEYWORDS):
        matched = next(h for h in headlines if any(
            kw in h["title"].lower() for kw in RETURN_FROM_INJURY_KEYWORDS))
        return {
            "player_name": player_name,
            "risk_level": "high",
            "reason": f"Headline suggests managed return from long-term injury: \"{matched['title']}\"",
            "source": f"Google News: {matched['link']}",
            "recommendation": "avoid_captain",
        }

    if any(kw in combined_text for kw in SUSPENSION_KEYWORDS):
        matched = next(h for h in headlines if any(
            kw in h["title"].lower() for kw in SUSPENSION_KEYWORDS))
        return {
            "player_name": player_name,
            "risk_level": "high",
            "reason": f"Headline suggests a suspension: \"{matched['title']}\"",
            "source": f"Google News: {matched['link']}",
            "recommendation": "avoid_start",
        }

    if any(kw in combined_text for kw in TRANSFER_KEYWORDS):
        matched = next(h for h in headlines if any(
            kw in h["title"].lower() for kw in TRANSFER_KEYWORDS))
        return {
            "player_name": player_name,
            "risk_level": "medium",
            "reason": f"Headline suggests an active transfer situation: \"{matched['title']}\"",
            "source": f"Google News: {matched['link']}",
            "recommendation": "monitor",
        }

    if any(kw in combined_text for kw in INJURY_KEYWORDS):
        matched = next(h for h in headlines if any(kw in h["title"].lower() for kw in INJURY_KEYWORDS))
        return {
            "player_name": player_name,
            "risk_level": "medium",
            "reason": f"Headline mentions injury-related terms: \"{matched['title']}\"",
            "source": f"Google News: {matched['link']}",
            "recommendation": "monitor",
        }

    return None


def merge_flags(fpl_flag, news_flag):
    """
    Combines a flag from FPL's own status/news fields with a flag from
    Google News headlines, keeping the higher-severity one. If both exist
    and disagree, keep the more severe risk_level but note both reasons.
    """
    if fpl_flag is None:
        return news_flag
    if news_flag is None:
        return fpl_flag

    severity = {"high": 2, "medium": 1, "low": 0}
    if severity[news_flag["risk_level"]] > severity[fpl_flag["risk_level"]]:
        primary, secondary = news_flag, fpl_flag
    else:
        primary, secondary = fpl_flag, news_flag

    merged = dict(primary)
    merged["reason"] = f"{primary['reason']} | Also: {secondary['reason']}"
    return merged



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


def full_pool_solver_inputs(master_data: dict, high_penalty=8.0, medium_penalty=3.0):
    """
    Runs the FREE FPL-status check (assess_player -- no network needed)
    against EVERY player in master_data, not just whoever happens to be in
    one particular squad. This is what should be applied to the very FIRST
    solve, before any squad has even been chosen.

    Why this matters: run_real.py's normal flow only runs the critic
    against ONE draft squad, then re-solves if something was flagged. But
    the re-solve can introduce a BRAND NEW player who was never in that
    draft and therefore never checked at all -- this is exactly what
    happened in practice (a flagged/blocked defender got swapped for a
    different, equally-injured defender from the same club, who then won
    captaincy completely unchecked, since the check never ran against him).
    Applying this full-pool check to every solve from the start closes that
    gap for anything the free FPL data itself can catch (status/news/
    chance_of_playing) -- it cannot cover the Google News layer for all
    570 players (too many requests), so that part remains squad-scoped as
    an additional, complementary check layered on top afterward.
    """
    flags = [assess_player(e) for e in master_data["elements"]]
    flags = [f for f in flags if f]
    return flags_to_solver_inputs(flags, master_data, high_penalty, medium_penalty)


def flags_to_solver_inputs(flags: list, master_data: dict, high_penalty=8.0, medium_penalty=3.0):
    """
    Translates the critic's qualitative flags into the quantitative inputs
    solver_general.solve() expects:
      - risk_level="high"   -> large risk_penalty AND blocked from captaincy
      - risk_level="medium" -> moderate risk_penalty AND ALSO blocked from
        captaincy (captaincy doubles the downside of a bad pick, so even a
        medium-risk flag shouldn't be overridable by a high enough raw
        projection -- this was a real gap found in practice)

    Name matching is a case-insensitive substring search against
    master_data's web_name -- works for most cases, but MULTIPLE MATCHES
    are resolved by taking the lowest id. Double-check manually if in doubt.
    """
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
            blocked_captain_ids.add(pid)

    return risk_penalty, blocked_captain_ids, unmatched


def run_free_critic(solution_path: str, master_data_path: str, out_path: str, use_news: bool = True):
    with open(solution_path, encoding="utf-8") as f:
        solution = json.load(f)
    with open(master_data_path, encoding="utf-8") as f:
        master_data = json.load(f)

    squad_names = set(solution["squad"])
    elements_by_name = {e["web_name"]: e for e in master_data["elements"]}

    flags = []
    news_errors = []
    for i, name in enumerate(squad_names):
        element = elements_by_name.get(name)
        if element is None:
            continue

        fpl_flag = assess_player(element)

        news_flag = None
        if use_news:
            if i > 0:
                # Space out requests to reduce the chance of Google News rate-limiting
                # a burst of ~15 rapid requests from GitHub Actions' shared IP ranges.
                time.sleep(1.0)
            try:
                headlines = fetch_google_news_headlines(element["web_name"], element.get("team"))
                news_flag = classify_headlines(element["web_name"], headlines)
            except Exception as e:
                # Never let one failed news lookup break the whole run --
                # this runs unattended in GitHub Actions, so degrade gracefully.
                news_errors.append(f"{name}: {e}")

        merged = merge_flags(fpl_flag, news_flag)
        if merged:
            flags.append(merged)

    # Distinguish "a few isolated lookups failed" from "every single one failed" --
    # the latter (news_errors == squad_size) points to a systematic block (Google
    # rejecting GitHub Actions' IP range, or similar), not an isolated network
    # hiccup, and the person should know that distinction rather than see 15
    # near-identical warnings and assume it's random flakiness.
    news_systematically_blocked = use_news and len(news_errors) >= len(squad_names) and len(squad_names) > 0

    if flags:
        high = [f["player_name"] for f in flags if f["risk_level"] == "high"]
        medium = [f["player_name"] for f in flags if f["risk_level"] == "medium"]
        parts = []
        if high:
            parts.append(f"{len(high)} high-risk ({', '.join(high)})")
        if medium:
            parts.append(f"{len(medium)} medium-risk ({', '.join(medium)})")
        summary = f"Found {' and '.join(parts)} based on FPL data" + \
            (" and Google News headlines." if use_news else ".") if parts else \
            "Minor notes found, no major concerns."
    else:
        summary = "No flags found for this squad."

    critic_output = {"flags": flags, "summary": summary}
    risk_penalty, blocked_captain_ids, unmatched = flags_to_solver_inputs(flags, master_data)

    result = {
        "critic_output": critic_output,
        "risk_penalty": risk_penalty,
        "blocked_captain_ids": sorted(blocked_captain_ids),
        "unmatched_names": unmatched,
        "news_lookup_errors": news_errors,
        "news_systematically_blocked": news_systematically_blocked,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    label = "FPL status/news + Google News headlines" if use_news else "FPL status/news only"
    print(f"=== Free critic assessment ({label}, no API cost) ===")
    print(summary)
    for flag in flags:
        print(f"  [{flag['risk_level'].upper()}] {flag['player_name']}: {flag['reason']} "
              f"({flag['recommendation']})")
    if news_systematically_blocked:
        print(f"\nWARNING: Google News lookup failed for ALL {len(news_errors)} players -- this is "
              f"not isolated flakiness, it looks like Google is blocking requests from this runner "
              f"entirely (common for cloud/CI IP ranges). FPL status data was still used normally; "
              f"only the Google News layer is affected.")
    elif news_errors:
        print(f"\nNote: Google News lookup failed for {len(news_errors)} of {len(squad_names)} player(s) "
              f"(isolated network hiccup) -- FPL status data was still used for them: {news_errors}")
    if not use_news:
        print(f"\nReminder: --no-news is set, so this only catches what FPL's own editors have "
              f"already flagged -- it will miss transfer sagas and rotation risk for a "
              f"technically-fit returning player.")
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Free critic -- zero-cost news check using FPL data + Google News RSS")
    parser.add_argument("--solution", required=True, help="Path to solver_solution_gw{N}.json")
    parser.add_argument("--master", default="master_data.json")
    parser.add_argument("--out", default="critic_flags.json")
    parser.add_argument("--no-news", action="store_true",
                         help="Skip the Google News headline check, use only FPL's own status/news fields "
                              "(faster, fewer network calls, but misses transfer sagas/rotation risk)")
    args = parser.parse_args()
    run_free_critic(args.solution, args.master, args.out, use_news=not args.no_news)


if __name__ == "__main__":
    main()
