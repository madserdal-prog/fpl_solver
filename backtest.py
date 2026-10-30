"""
backtest.py -- replay past gameweeks with any version of the pipeline
=======================================================================
Answers "how would this code have done?" honestly:

  * Each past gameweek is replayed ONE STEP from your REAL squad going into
    that week (bank + free transfers too), using the collector snapshot taken
    just before that week's deadline -- i.e. only information available at
    the time. This avoids compounding divergence: every version gets the
    same starting point every week.
  * The recommendation (transfers, XI, captain, bench order) is scored with
    the players' ACTUAL points, including FPL-style autosubs and the
    vice-captain rule, minus real hit costs.
  * Forecast accuracy (xP vs actual points) is measured separately.

Not replayed: chips (path-dependent across weeks -- tested by scenario tests
instead) and the news critics (historical news isn't reproducible). Both
versions are scored identically, so the comparison stays fair.

Steps (run by .github/workflows/backtest.yml; needs network + data/raw):
  python backtest.py fetch   --team-id 7755120
  python backtest.py replay  --code-dir .              --out data/backtest/results_new.json
  python backtest.py replay  --code-dir /tmp/baseline  --out data/backtest/results_old.json
  python backtest.py compare --baseline data/backtest/results_old.json \\
                             --candidate data/backtest/results_new.json --out data/backtest/report.md
"""

import argparse
import glob
import importlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

BT_DIR = "data/backtest"
RAW_DIR = "data/raw"
API = "https://fantasy.premierleague.com/api"
HORIZON = 5


# ---------------------------------------------------------------------------
# 1) fetch: gather code-independent inputs (needs network)
# ---------------------------------------------------------------------------

def _get(url):
    import requests
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "fpl-auto-pipeline/1.0"})
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def _parse_snapshot_time(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def snapshot_before(deadline, raw_dir=RAW_DIR):
    """Latest collector snapshot taken strictly before `deadline`, else None."""
    best = None
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        try:
            ts = _parse_snapshot_time(path)
        except ValueError:
            continue
        if ts < deadline and (best is None or ts > best[0]):
            best = (ts, path)
    return best[1] if best else None


def fetch(team_id, raw_dir=RAW_DIR, out_dir=BT_DIR, getter=_get):
    import fetch_my_team
    os.makedirs(out_dir, exist_ok=True)
    bootstrap = getter(f"{API}/bootstrap-static/")
    fixtures_raw = getter(f"{API}/fixtures/")
    history = getter(f"{API}/entry/{team_id}/history/")
    with open(os.path.join(out_dir, "fixtures_raw.json"), "w", encoding="utf-8") as f:
        json.dump(fixtures_raw, f)

    gws, skipped = [], []
    finished = [e for e in bootstrap["events"] if e.get("finished")]
    for ev in finished:
        g = ev["id"]
        if g < 2:
            skipped.append({"gw": g, "reason": "GW1 is an initial squad build, not a transfer decision"})
            continue
        deadline = datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00"))
        snap = snapshot_before(deadline, raw_dir)
        if not snap:
            skipped.append({"gw": g, "reason": "no collector snapshot before this deadline"})
            continue
        prev = getter(f"{API}/entry/{team_id}/event/{g - 1}/picks/")
        this = getter(f"{API}/entry/{team_id}/event/{g}/picks/")
        live = getter(f"{API}/event/{g}/live/")
        gws.append({
            "gw": g,
            "snapshot": snap,
            "squad": [p["element"] for p in prev["picks"]],
            "bank": prev["entry_history"]["bank"] / 10.0,
            "free_transfers": fetch_my_team.compute_free_transfers(history, g),
            "actual": {str(e["id"]): [e["stats"]["total_points"], e["stats"]["minutes"]]
                       for e in live["elements"]},
            "user": {"points": this["entry_history"]["points"],
                     "transfers_cost": this["entry_history"].get("event_transfers_cost", 0),
                     "active_chip": this.get("active_chip")},
        })
        time.sleep(0.5)
    with open(os.path.join(out_dir, "inputs.json"), "w", encoding="utf-8") as f:
        json.dump({"gws": gws, "skipped": skipped}, f)
    print(f"Prepared {len(gws)} gameweek(s); skipped {len(skipped)}: {skipped}")
    return gws, skipped


# ---------------------------------------------------------------------------
# 2) replay: run ONE code version over the prepared inputs
# ---------------------------------------------------------------------------

def _load_code(code_dir):
    """Import collector/forecast/solver from code_dir (fresh, isolated)."""
    sys.path.insert(0, os.path.abspath(code_dir))
    for m in ("collector", "forecast", "solver_general", "my_team", "chips"):
        sys.modules.pop(m, None)
    return (importlib.import_module("collector"), importlib.import_module("forecast"),
            importlib.import_module("solver_general"))


def _apply_overrides(modules, overrides):
    """--set module.CONSTANT=value, e.g. solver_general.HIT_SAFETY_MARGIN=8 --
    replays with a constant changed, to calibrate on real past weeks."""
    for item in overrides or []:
        target, value = item.split("=", 1)
        mod_name, attr = target.rsplit(".", 1)
        mod = modules[mod_name]
        if not hasattr(mod, attr):
            raise SystemExit(f"--set: {mod_name} has no constant {attr}")
        setattr(mod, attr, type(getattr(mod, attr))(value))
        print(f"Override: {target} = {getattr(mod, attr)}")


def replay(code_dir, inputs_path, fixtures_path, out_path, overrides=None):
    collector, forecast, solver = _load_code(code_dir)
    _apply_overrides({"collector": collector, "forecast": forecast, "solver_general": solver}, overrides)
    with open(inputs_path, encoding="utf-8") as f:
        inputs = json.load(f)
    with open(fixtures_path, encoding="utf-8") as f:
        fixtures_raw = json.load(f)

    results = []
    for item in inputs["gws"]:
        g = item["gw"]
        with open(item["snapshot"], encoding="utf-8") as f:
            raw = json.load(f)
        master = collector.normalize_master(raw)
        fixtures = collector.normalize_fixtures(fixtures_raw, raw)
        fc = forecast.build_forecast(master, fixtures, g, HORIZON)
        with tempfile.TemporaryDirectory() as tmp:
            mp, fp = os.path.join(tmp, "master.json"), os.path.join(tmp, "forecast.json")
            json.dump(master, open(mp, "w", encoding="utf-8"))
            json.dump(fc, open(fp, "w", encoding="utf-8"))
            pool = solver.load_pool_from_forecast(fp, mp, horizon_gw=0)
        by_id = {p["id"]: p for p in pool}
        squad = [i for i in item["squad"] if i in by_id]
        budget = round(item["bank"] + sum(by_id[i]["price"] for i in squad), 1)
        entry = {"gw": g, "xp": {str(p["id"]): p["xp"] for p in pool},
                 "positions": {str(p["id"]): p["position"] for p in pool}}
        try:
            sol = solver.solve(pool, current_squad_ids=squad, budget=budget,
                               free_transfers=item["free_transfers"])
            bench = sol.get("bench_ids") or [i for i in sol["squad_ids"] if i not in sol["starting_xi_ids"]]
            vice_name = sol.get("vice_captain")
            vice = next((p["id"] for p in pool if p["name"] == vice_name and p["id"] in sol["starting_xi_ids"]), None)
            if vice is None:   # versions without a vice: best other starter by that version's own xP
                others = [i for i in sol["starting_xi_ids"] if i != sol["captain_id"]]
                vice = max(others, key=lambda i: by_id[i]["xp"]) if others else None
            entry["decision"] = {"squad": sol["squad_ids"], "xi": sol["starting_xi_ids"], "bench": bench,
                                 "captain": sol["captain_id"],
                                 "vice": vice, "hit_cost": sol.get("hit_cost", 0),
                                 "transfers_in": sol.get("transfers_in", []),
                                 "transfers_out": sol.get("transfers_out", [])}
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        entry["hold"] = hold_decision(squad, entry["xp"], entry["positions"])
        if entry["hold"]:
            entry["hold"]["squad"] = squad
        results.append(entry)
        print(f"GW{g}: " + (entry.get("error") or f"captain {entry['decision']['captain']}, "
                                                   f"transfers {entry['decision']['transfers_in']}"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"code_dir": code_dir, "results": results}, f)


def hold_decision(squad, xp, positions):
    """Benchmark: no transfers, best legal XI from the existing squad by xP."""
    xpf = lambda i: xp.get(str(i), 0.0)
    pos = lambda i: positions.get(str(i))
    by = {p: sorted([i for i in squad if pos(i) == p], key=lambda i: -xpf(i)) for p in ("GK", "DEF", "MID", "FWD")}
    best = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if 1 <= f <= 3 and d <= len(by["DEF"]) and m <= len(by["MID"]) and f <= len(by["FWD"]) and by["GK"]:
                xi = by["GK"][:1] + by["DEF"][:d] + by["MID"][:m] + by["FWD"][:f]
                score = sum(xpf(i) for i in xi)
                if best is None or score > best[1]:
                    best = (xi, score)
    if best is None:
        return None
    xi = best[0]
    ranked = sorted(xi, key=lambda i: -xpf(i))
    bench = sorted([i for i in squad if i not in xi], key=lambda i: (pos(i) != "GK", -xpf(i)))
    return {"xi": xi, "bench": bench, "captain": ranked[0], "vice": ranked[1], "hit_cost": 0}


# ---------------------------------------------------------------------------
# Scoring with actual points (FPL autosub + vice rules)
# ---------------------------------------------------------------------------

def _valid(xi, pos):
    c = {p: sum(1 for i in xi if pos(i) == p) for p in ("GK", "DEF", "MID", "FWD")}
    return c["GK"] == 1 and c["DEF"] >= 3 and c["MID"] >= 2 and c["FWD"] >= 1


def score_decision(decision, actual, positions):
    if not decision:
        return None
    pts = lambda i: actual.get(str(i), [0, 0])[0]
    mins = lambda i: actual.get(str(i), [0, 0])[1]
    pos = lambda i: positions.get(str(i))
    xi, used = list(decision["xi"]), set()
    for k, starter in enumerate(list(xi)):
        if mins(starter) > 0:
            continue
        for b in decision["bench"]:
            if b in used or mins(b) == 0:
                continue
            if (pos(starter) == "GK") != (pos(b) == "GK"):
                continue
            trial = xi[:k] + [b] + xi[k + 1:]
            if _valid(trial, pos):
                xi, used = trial, used | {b}
                break
    total = sum(pts(i) for i in xi)
    if mins(decision["captain"]) > 0:
        total += pts(decision["captain"])
    elif decision.get("vice") in xi and mins(decision["vice"]) > 0:
        total += pts(decision["vice"])
    return total - decision.get("hit_cost", 0)


def score_held(decision, start_gw, version_by_gw, inputs_by_gw, weeks=3):
    """Points from HOLDING the decision's squad for `weeks` gameweeks: week 1
    uses the decision's own XI/captain; later weeks pick the best XI from the
    same squad using that version's OWN forecast at the time. The hit is
    charged once. Fair to multi-week planners, whose hits pay off later."""
    if not decision:
        return None
    total = score_decision(decision, inputs_by_gw[start_gw]["actual"],
                           version_by_gw[start_gw]["positions"])
    for g in range(start_gw + 1, start_gw + weeks):
        if g not in version_by_gw or g not in inputs_by_gw:
            return None            # not enough later weeks to evaluate
        later = hold_decision(decision["squad"], version_by_gw[g]["xp"], version_by_gw[g]["positions"])
        total += score_decision(later, inputs_by_gw[g]["actual"], version_by_gw[g]["positions"])
    return total


# ---------------------------------------------------------------------------
# 3) compare: decision quality + forecast accuracy
# ---------------------------------------------------------------------------

def _accuracy(results, inputs_by_gw):
    pairs = {"ALL": []}
    for r in results:
        actual = inputs_by_gw[r["gw"]]["actual"]
        for pid, xp in r["xp"].items():
            a = actual.get(pid)
            if a and a[1] > 0:                      # players who actually played
                pairs["ALL"].append((xp, a[0]))
                pairs.setdefault(r["positions"].get(pid), []).append((xp, a[0]))
    out = {}
    for k, v in pairs.items():
        if len(v) < 2:
            continue
        n = len(v)
        mae = sum(abs(x - y) for x, y in v) / n
        bias = sum(x - y for x, y in v) / n
        mx, my = sum(x for x, _ in v) / n, sum(y for _, y in v) / n
        cov = sum((x - mx) * (y - my) for x, y in v)
        sx = math.sqrt(sum((x - mx) ** 2 for x, _ in v)); sy = math.sqrt(sum((y - my) ** 2 for _, y in v))
        out[k] = {"n": n, "mae": round(mae, 2), "bias": round(bias, 2),
                  "corr": round(cov / (sx * sy), 3) if sx and sy else None}
    return out


def compare(baseline_path, candidate_path, inputs_path, out_path):
    load = lambda p: json.load(open(p, encoding="utf-8"))
    base, cand, inputs = load(baseline_path), load(candidate_path), load(inputs_path)
    inputs_by_gw = {i["gw"]: i for i in inputs["gws"]}
    base_by = {r["gw"]: r for r in base["results"]}
    cand_by = {r["gw"]: r for r in cand["results"]}
    held = {"old": 0, "new": 0, "hold": 0, "n": 0}
    for g in sorted(cand_by):
        o = score_held(base_by.get(g, {}).get("decision"), g, base_by, inputs_by_gw)
        n = score_held(cand_by[g].get("decision"), g, cand_by, inputs_by_gw)
        h = score_held(cand_by[g].get("hold"), g, cand_by, inputs_by_gw)
        if None not in (o, n, h):
            held["old"] += o; held["new"] += n; held["hold"] += h; held["n"] += 1
    rows, tot = [], {"old": 0, "new": 0, "hold": 0, "you": 0}
    for r in cand["results"]:
        g = r["gw"]
        actual = inputs_by_gw[g]["actual"]
        b = base_by.get(g, {})
        new = score_decision(r.get("decision"), actual, r["positions"])
        old = score_decision(b.get("decision"), actual, b.get("positions", r["positions"]))
        hold = score_decision(r.get("hold"), actual, r["positions"])
        you = inputs_by_gw[g]["user"]["points"] - inputs_by_gw[g]["user"]["transfers_cost"]
        chip = inputs_by_gw[g]["user"].get("active_chip")
        rows.append((g, old, new, hold, you, chip, b.get("error"), r.get("error")))
        if None not in (old, new, hold):
            tot["old"] += old; tot["new"] += new; tot["hold"] += hold; tot["you"] += you

    acc_old, acc_new = _accuracy(base["results"], inputs_by_gw), _accuracy(cand["results"], inputs_by_gw)
    md = ["# Backtest: old vs new pipeline", "",
          "One-step replay from your real squad each week, scored with actual points "
          "(autosubs + vice rule, minus hits). No chips or news critics in either version.", "",
          "| GW | Old code | New code | Hold (no transfers) | You (actual, after hits) | Notes |",
          "|---|---|---|---|---|---|"]
    for g, old, new, hold, you, chip, e_old, e_new in rows:
        note = "; ".join(x for x in [f"you played {chip}" if chip else "",
                                     f"old error: {e_old}" if e_old else "", f"new error: {e_new}" if e_new else ""] if x)
        md.append(f"| {g} | {old} | {new} | {hold} | {you} | {note} |")
    md += ["", f"**Totals (weeks where all versions ran):** old {tot['old']}, new {tot['new']}, "
               f"hold {tot['hold']}, you {tot['you']}. New minus old: **{tot['new'] - tot['old']:+}**.", "",
           f"**Decisions held for 3 weeks** ({held['n']} start weeks; fairer to multi-week planning, "
           f"since hits are paid upfront but pay off later): old {held['old']}, new {held['new']}, "
           f"hold {held['hold']}. New minus old: **{held['new'] - held['old']:+}**.", "",
           "Note: 'You' includes chips you actually played, so it isn't directly comparable to the "
           "chip-free replays in weeks marked with a chip.", "",
           "## Forecast accuracy (players who played; lower MAE better, bias ~0 better, higher corr better)", "",
           "| Group | Old MAE | New MAE | Old bias | New bias | Old corr | New corr | n |",
           "|---|---|---|---|---|---|---|---|"]
    for k in ["ALL", "GK", "DEF", "MID", "FWD"]:
        o, n = acc_old.get(k), acc_new.get(k)
        if o and n:
            md.append(f"| {k} | {o['mae']} | {n['mae']} | {o['bias']} | {n['bias']} | {o['corr']} | {n['corr']} | {n['n']} |")
    if inputs.get("skipped"):
        md += ["", "Skipped weeks: " + "; ".join(f"GW{s['gw']} ({s['reason']})" for s in inputs["skipped"])]
    report = "\n".join(md)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return tot, acc_old, acc_new, held


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--team-id", type=int, required=True)
    r = sub.add_parser("replay"); r.add_argument("--code-dir", required=True); r.add_argument("--out", required=True)
    r.add_argument("--inputs", default=f"{BT_DIR}/inputs.json"); r.add_argument("--fixtures", default=f"{BT_DIR}/fixtures_raw.json")
    r.add_argument("--set", action="append", default=[], help="override a constant: module.NAME=value")
    c = sub.add_parser("compare"); c.add_argument("--baseline", required=True); c.add_argument("--candidate", required=True)
    c.add_argument("--inputs", default=f"{BT_DIR}/inputs.json"); c.add_argument("--out", default=f"{BT_DIR}/report.md")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.team_id)
    elif a.cmd == "replay":
        replay(a.code_dir, a.inputs, a.fixtures, a.out, a.set)
    else:
        compare(a.baseline, a.candidate, a.inputs, a.out)


if __name__ == "__main__":
    main()
