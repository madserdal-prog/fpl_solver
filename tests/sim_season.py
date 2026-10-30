"""
Simulated FPL season in the exact raw formats the collector and backtest use.

Purpose: exercise backtest.py end-to-end (snapshots -> replay of two code
versions -> scoring with actual points) without network access. Results on
simulated data validate the MACHINERY and give a sanity check; they are NOT
evidence of real-world performance -- the simulator's scoring rules mirror
FPL, but its hidden "truth" is synthetic.
"""
import json
import math
import os
import random
from datetime import datetime, timedelta, timezone

POS_CODE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
GOAL_PTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
DEFCON_TH = {"DEF": 10, "MID": 12, "FWD": 12}


def _pois(rnd, lam):
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rnd.random()
        if p < l:
            return k
        k += 1


def simulate(out_dir, n_gws=9, seed=11):
    rnd = random.Random(seed)
    raw_dir = os.path.join(out_dir, "data", "raw"); bt_dir = os.path.join(out_dir, "data", "backtest")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(bt_dir, exist_ok=True)

    teams = []
    for t in range(20):
        q = rnd.gauss(0, 0.28)
        teams.append({"id": t + 1, "name": f"Club{t + 1:02d}", "att": 1.35 * math.exp(q + rnd.gauss(0, 0.1)),
                      "dfc": 1.35 * math.exp(-q + rnd.gauss(0, 0.1))})
    avg = 1.35
    strength = sorted(teams, key=lambda t: t["att"] / t["dfc"])
    fdr = {t["id"]: 1 + (i * 5) // 20 for i, t in enumerate(strength)}   # 1 weakest .. 5 strongest

    players, pid = [], 1
    roles = ([("GK", 0.97), ("GK", 0.03)] + [("DEF", 0.92)] * 4 + [("DEF", 0.3)]
             + [("MID", 0.9)] * 4 + [("MID", 0.35)] + [("FWD", 0.9), ("FWD", 0.5), ("FWD", 0.15)])
    for t in teams:
        for pos, sp in roles:
            g_share = {"GK": 0.0, "DEF": rnd.uniform(0.02, 0.06), "MID": rnd.uniform(0.06, 0.2),
                       "FWD": rnd.uniform(0.2, 0.45)}[pos]
            a_share = {"GK": 0.01, "DEF": rnd.uniform(0.04, 0.12), "MID": rnd.uniform(0.1, 0.25),
                       "FWD": rnd.uniform(0.06, 0.15)}[pos]
            quality = (g_share * GOAL_PTS[pos] + a_share * 3) * t["att"]
            base = {"GK": 42, "DEF": 42, "MID": 48, "FWD": 50}[pos]
            players.append({"id": pid, "team": t["id"], "pos": pos, "start_p": sp, "g": g_share, "a": a_share,
                            "dc": {"GK": 0, "DEF": rnd.uniform(6.5, 12), "MID": rnd.uniform(5, 13),
                                   "FWD": rnd.uniform(2.5, 6)}[pos],
                            "now_cost": int(base + 40 * quality * (0.6 + sp * 0.5)),
                            "minutes": 0, "starts": 0, "xg": 0.0, "xa": 0.0, "xgc": 0.0, "saves": 0, "dcon": 0,
                            "ict": 0.0, "total": 0, "apps": 0, "hist": [], "inj": 0, "doubt": False})
            pid += 1
    by_team = {t["id"]: [p for p in players if p["team"] == t["id"]] for t in teams}
    # penalty takers: highest goal share per club
    for t in teams:
        for order, p in enumerate(sorted(by_team[t["id"]], key=lambda p: -p["g"])[:2], start=1):
            p["pen"] = order

    start = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)
    deadlines = {g: start + timedelta(days=7 * (g - 1)) for g in range(1, n_gws + 2)}
    fixtures_raw, ids = [], [t["id"] for t in teams]
    for g in range(1, n_gws + 2):
        rnd.shuffle(ids)
        for h, a in zip(ids[0::2], ids[1::2]):
            fixtures_raw.append({"event": g, "team_h": h, "team_a": a,
                                 "team_h_difficulty": fdr[a], "team_a_difficulty": fdr[h]})

    def snapshot(g):
        """Raw bootstrap as of just before GW g's deadline (stats through GW g-1)."""
        els = []
        for p in players:
            form_pts = p["hist"][-4:]
            els.append({
                "id": p["id"], "web_name": f"P{p['id']}", "team": p["team"], "element_type": POS_CODE[p["pos"]],
                "now_cost": p["now_cost"], "minutes": p["minutes"], "starts": p["starts"],
                "expected_goals": f"{p['xg']:.2f}", "expected_assists": f"{p['xa']:.2f}",
                "ict_index": f"{p['ict']:.1f}", "expected_goals_conceded": f"{p['xgc']:.2f}",
                "saves": p["saves"], "defensive_contribution": p["dcon"], "total_points": p["total"],
                "form": f"{(sum(form_pts) / len(form_pts)) if form_pts else 0:.1f}",
                "points_per_game": f"{(p['total'] / p['apps']) if p['apps'] else 0:.1f}",
                "status": "i" if p["inj"] > 0 else ("d" if p["doubt"] else "a"),
                "chance_of_playing_next_round": 0 if p["inj"] > 0 else (75 if p["doubt"] else None),
                "news": "Injured" if p["inj"] > 0 else ("Knock - 75% chance of playing" if p["doubt"] else ""),
                "news_added": None, "penalties_order": p.get("pen"), "transfers_in_event": 0,
                "transfers_out_event": 0, "selected_by_percent": "1.0"})
        events = [{"id": e, "deadline_time": deadlines[e].strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "is_next": e == g, "is_current": e == g - 1, "finished": e < g} for e in range(1, n_gws + 2)]
        return {"events": events, "teams": [{"id": t["id"], "name": t["name"], "short_name": t["name"][:3]} for t in teams],
                "elements": els, "total_players": 9_000_000}

    live = {}
    for g in range(1, n_gws + 1):
        # availability news for THIS gameweek, published before its deadline
        for p in players:
            p["doubt"] = p["inj"] == 0 and rnd.random() < 0.03
            if p["inj"] == 0 and rnd.random() < 0.015:
                p["inj"] = rnd.randint(1, 4)
        snap = snapshot(g)
        ts = (deadlines[g] - timedelta(hours=3)).strftime("%Y%m%dT%H%M%SZ")
        json.dump(snap, open(os.path.join(raw_dir, f"{ts}.json"), "w"))

        gw_pts = {}
        for fx in [f for f in fixtures_raw if f["event"] == g]:
            th, ta = teams[fx["team_h"] - 1], teams[fx["team_a"] - 1]
            lam = {fx["team_h"]: th["att"] * ta["dfc"] / avg * 1.1, fx["team_a"]: ta["att"] * th["dfc"] / avg * 0.9}
            goals = {k: _pois(rnd, v) for k, v in lam.items()}
            opp = {fx["team_h"]: fx["team_a"], fx["team_a"]: fx["team_h"]}
            match_bps = []
            for team_id in (fx["team_h"], fx["team_a"]):
                squad = by_team[team_id]
                mins = {}
                for p in squad:
                    if p["inj"] > 0 or (p["doubt"] and rnd.random() < 0.25):
                        mins[p["id"]] = 0
                    elif rnd.random() < p["start_p"]:
                        mins[p["id"]] = 90 if rnd.random() < 0.75 or p["pos"] in ("GK", "DEF") else rnd.randint(60, 85)
                    else:
                        mins[p["id"]] = rnd.randint(10, 30) if rnd.random() < 0.3 else 0
                on = [p for p in squad if mins[p["id"]] > 0]
                scorers = {p["id"]: 0 for p in squad}; assists = {p["id"]: 0 for p in squad}
                for _ in range(goals[team_id]):
                    cand = [p for p in on if p["pos"] != "GK"] or on
                    s = rnd.choices(cand, weights=[p["g"] * mins[p["id"]] + 1e-6 for p in cand])[0]
                    scorers[s["id"]] += 1
                    if rnd.random() < 0.75:
                        ac = [p for p in on if p["id"] != s["id"]] or on
                        assists[rnd.choices(ac, weights=[p["a"] * mins[p["id"]] + 1e-6 for p in ac])[0]["id"]] += 1
                conceded = goals[opp[team_id]]; lam_against = lam[opp[team_id]]
                for p in squad:
                    m = mins[p["id"]]
                    if m == 0:
                        continue
                    pos = p["pos"]
                    dc = _pois(rnd, p["dc"] * m / 90) if pos != "GK" else 0
                    sv = _pois(rnd, lam_against * 2.2) if pos == "GK" else 0
                    pts = 1 + (1 if m >= 60 else 0) + scorers[p["id"]] * GOAL_PTS[pos] + assists[p["id"]] * 3
                    if m >= 60 and conceded == 0:
                        pts += CS_PTS[pos]
                    if pos in ("GK", "DEF") and m >= 60:
                        pts -= conceded // 2
                    pts += sv // 3
                    if pos in DEFCON_TH and dc >= DEFCON_TH[pos]:
                        pts += 2
                    bps = scorers[p["id"]] * 24 + assists[p["id"]] * 9 + (12 if m >= 60 and conceded == 0 and pos in ("GK", "DEF") else 0) + dc + sv * 2 + rnd.randint(0, 6)
                    match_bps.append([bps, p["id"], pts])
                    p["minutes"] += m; p["starts"] += 1 if m >= 60 or (m > 0 and rnd.random() < 0.0) else 0
                    p["xg"] += lam[team_id] * p["g"] * m / 90; p["xa"] += lam[team_id] * p["a"] * 0.75 * m / 90
                    if pos in ("GK", "DEF"):
                        p["xgc"] += lam_against * m / 90
                    p["saves"] += sv; p["dcon"] += dc
                    p["ict"] += 10 * (lam[team_id] * (p["g"] + p["a"]) * 3) * m / 90 + 2 * m / 90
                    gw_pts[p["id"]] = [pts, m]
            for rank, (_, pid_, _) in enumerate(sorted(match_bps, reverse=True)[:3]):
                gw_pts[pid_][0] += 3 - rank
        for p in players:
            pts, m = gw_pts.get(p["id"], [0, 0])
            gw_pts.setdefault(p["id"], [0, 0])
            p["total"] += pts
            if m > 0:
                p["apps"] += 1
                p["hist"].append(pts)
            if p["inj"] > 0:
                p["inj"] -= 1
        live[g] = gw_pts

    # The "user": a sensible fixed squad (top xP-ish per price), never transfers
    picks, per_club = [], {}
    for pos, n in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        cand = sorted([p for p in players if p["pos"] == pos],
                      key=lambda p: -(p["start_p"] * (p["g"] + p["a"] + 0.05)) / p["now_cost"])
        for p in cand:
            if len([x for x in picks if x["pos"] == pos]) == n:
                break
            if per_club.get(p["team"], 0) < 3:
                picks.append(p); per_club[p["team"]] = per_club.get(p["team"], 0) + 1
    squad = [p["id"] for p in picks]
    bank = round(100.0 - sum(p["now_cost"] for p in picks) / 10.0, 1)

    gws = []
    for g in range(2, n_gws + 1):
        snap_file = sorted(f for f in os.listdir(raw_dir))[g - 1]
        gws.append({"gw": g, "snapshot": os.path.join(raw_dir, snap_file), "squad": squad,
                    "bank": max(0.0, bank), "free_transfers": 1,
                    "actual": {str(k): v for k, v in live[g].items()},
                    "user": {"points": sum(sorted((live[g][i][0] for i in squad), reverse=True)[:11]),
                             "transfers_cost": 0, "active_chip": None}})
    json.dump({"gws": gws, "skipped": [{"gw": 1, "reason": "GW1 is an initial squad build"}]},
              open(os.path.join(bt_dir, "inputs.json"), "w"))
    json.dump(fixtures_raw, open(os.path.join(bt_dir, "fixtures_raw.json"), "w"))
    return os.path.join(bt_dir, "inputs.json"), os.path.join(bt_dir, "fixtures_raw.json")


if __name__ == "__main__":
    import sys
    print(simulate(sys.argv[1] if len(sys.argv) > 1 else "/tmp/simseason"))
