"""
Chip planner
=============
FPL 2026/27: two sets of chips (Wildcard, Free Hit, Bench Boost, Triple
Captain), one per half. First-half chips must be played before the GW19
deadline and do NOT carry over; only one chip can be played per gameweek.

Each run this values every unused chip in every remaining week of the
current half, then assigns chips to distinct weeks to maximise total value.
It recommends playing a chip NOW only if the assignment puts it in this
week. The plan is recalculated every week as forecasts (and announced
blank/double gameweeks) change -- treat later weeks as provisional.

Chip values (expected points gained by playing it that week):
  triple_captain -> the captain's xP that week (an extra captain multiple)
  bench_boost    -> the bench's xP that week
  free_hit       -> best one-week squad's lineup xP minus your own that week
  wildcard       -> horizon objective with unlimited transfers now minus the
                    normal plan. Only computable for THIS week, so a future
                    wildcard is valued at WILDCARD_MIN_GAIN: it's played now
                    only if it beats that bar, or if the deadline forces it.
"""

from itertools import permutations

import solver_general as solver

FIRST_HALF_LAST_GW = 18      # first-half chips must be used before the GW19 deadline
SEASON_LAST_GW = 38
WILDCARD_MIN_GAIN = 12.0     # heuristic: a wildcard should earn at least this (discounted
                             # objective points over the horizon) to be worth playing now
CHIP_NAMES = ("wildcard", "free_hit", "bench_boost", "triple_captain")


def last_usable_gw(current_gw: int) -> int:
    return FIRST_HALF_LAST_GW if current_gw <= FIRST_HALF_LAST_GW else SEASON_LAST_GW


def best_lineup(squad_ids, xp_by_id, pos_by_id, blocked_captains=()):
    """Exact best XI for one week from a 15-man squad: tries every legal
    formation. Returns (xi_ids, captain_id, bench_ids, xi_points, captain_xp, bench_points)."""
    by_pos = {pos: sorted([i for i in squad_ids if pos_by_id[i] == pos], key=lambda i: -xp_by_id[i])
              for pos in ("GK", "DEF", "MID", "FWD")}
    best = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not 1 <= f <= 3 or d > len(by_pos["DEF"]) or m > len(by_pos["MID"]) or f > len(by_pos["FWD"]):
                continue
            xi = by_pos["GK"][:1] + by_pos["DEF"][:d] + by_pos["MID"][:m] + by_pos["FWD"][:f]
            score = sum(xp_by_id[i] for i in xi)
            if best is None or score > best[1]:
                best = (xi, score)
    xi, xi_points = best
    eligible = [i for i in xi if i not in set(blocked_captains)] or xi
    captain = max(eligible, key=lambda i: xp_by_id[i])
    bench = [i for i in squad_ids if i not in xi]
    return xi, captain, bench, xi_points, xp_by_id[captain], sum(xp_by_id[i] for i in bench)


def _week_squads(solution, n_weeks):
    """Squad per week offset: this week's, then the planner's future squads,
    then held fixed beyond the solver's horizon."""
    squads = [solution["squad_ids"]] + [w["squad_ids"] for w in solution.get("future_plan", [])]
    while len(squads) < n_weeks:
        squads.append(squads[-1])
    return squads[:n_weeks]


def plan_chips(pool, solution, chips_available, current_gw, solve_kwargs,
               risk_penalty=None, blocked_captain_ids=None):
    """
    pool: from load_pool_from_forecast(), with xp_series long enough to cover
          the window (run_real.py builds an extended forecast for this).
    solution: the main solver result (week-0 squad + future_plan squads).
    solve_kwargs: the kwargs used for the main solve (budget, sell values...).
    Returns a dict describing values, the assignment and any play-now advice.
    """
    chips = [c for c in CHIP_NAMES if c in set(chips_available or [])]
    end_gw = last_usable_gw(current_gw)
    horizon_len = min(len(p.get("xp_series") or [p["xp"]]) for p in pool)
    n_weeks = max(0, min(end_gw - current_gw + 1, horizon_len))
    result = {"deadline_gw": end_gw, "window": [current_gw + w for w in range(n_weeks)],
              "chips": chips, "values": {}, "assignment": {}, "play_now": None,
              "play_now_value": 0.0, "wildcard_solution": None, "free_hit_solution": None,
              "window_shorter_than_deadline": n_weeks < end_gw - current_gw + 1}
    if not chips or n_weeks == 0:
        return result

    risk_penalty = risk_penalty or {}
    pos_by_id = {p["id"]: p["position"] for p in pool}
    xp_week = [{p["id"]: (p.get("xp_series") or [p["xp"]])[w] for p in pool} for w in range(n_weeks)]
    for pid, share in risk_penalty.items():          # near-term risk applies to this week only
        if pid in xp_week[0]:
            xp_week[0][pid] *= (1.0 - min(1.0, max(0.0, share)))
    squads = _week_squads(solution, n_weeks)

    values = {c: {} for c in chips}
    for w in range(n_weeks):
        gw = current_gw + w
        blocked = (blocked_captain_ids or set()) if w == 0 else set()
        _, _, _, xi_pts, cap_xp, bench_pts = best_lineup(squads[w], xp_week[w], pos_by_id, blocked)
        if "triple_captain" in values:
            values["triple_captain"][gw] = round(cap_xp, 2)
        if "bench_boost" in values:
            values["bench_boost"][gw] = round(bench_pts, 2)
        if "free_hit" in values:
            one_week_pool = [dict(p, xp=xp_week[w][p["id"]], xp_series=[xp_week[w][p["id"]]],
                                  xp_horizon_sum=xp_week[w][p["id"]]) for p in pool]
            fh_kwargs = {k: v for k, v in solve_kwargs.items()
                         if k in ("current_squad_ids", "budget", "sell_values", "locked_ids")}
            fh = solver.solve(one_week_pool, unlimited_transfers=True, plan_future_transfers=False,
                              blocked_captain_ids=blocked, **fh_kwargs)
            values["free_hit"][gw] = round(fh["expected_points"] - (xi_pts + cap_xp), 2)
            if w == 0:
                result["free_hit_solution"] = fh

    if "wildcard" in values:
        wc_kwargs = dict(solve_kwargs, unlimited_transfers=True)
        wc = solver.solve(pool_for_horizon(pool, solution["horizon_weeks"]),
                          risk_penalty=risk_penalty, blocked_captain_ids=blocked_captain_ids, **wc_kwargs)
        gain_now = wc["objective_value"] - solution["objective_value"]
        values["wildcard"] = {current_gw + w: (round(gain_now, 2) if w == 0 else WILDCARD_MIN_GAIN)
                              for w in range(n_weeks)}
        result["wildcard_solution"] = wc
    result["values"] = values

    # Assign chips to distinct weeks maximising total value. If there are
    # fewer weeks than chips, the best subset of chips gets the weeks.
    weeks = result["window"]
    k = min(len(chips), len(weeks))
    best_total, best_assign = None, {}
    for chip_subset in permutations(chips, k):
        for week_choice in permutations(weeks, k):
            total = sum(values[c][w] for c, w in zip(chip_subset, week_choice))
            if best_total is None or total > best_total:
                best_total, best_assign = total, dict(zip(chip_subset, week_choice))
    result["assignment"] = best_assign
    now = [c for c, w in best_assign.items() if w == current_gw]
    if now:
        result["play_now"] = now[0]
        result["play_now_value"] = values[now[0]][current_gw]
    return result


def pool_for_horizon(pool, horizon_weeks):
    """Truncate xp_series (and its sum) to the solver's horizon -- the chip
    window can be longer than the transfer-planning horizon."""
    out = []
    for p in pool:
        series = (p.get("xp_series") or [p["xp"]])[:horizon_weeks]
        out.append(dict(p, xp_series=series, xp_horizon_sum=round(sum(series), 3)))
    return out


def summary_line(plan: dict) -> str:
    """One Slack line: play-now advice plus the provisional schedule."""
    if not plan or not plan.get("chips"):
        return ""
    names = {"wildcard": "Wildcard", "free_hit": "Free Hit", "bench_boost": "Bench Boost",
             "triple_captain": "Triple Captain"}
    def label(c, w):
        if c == "wildcard" and w != plan["window"][0]:
            return f"{names[c]} GW{w} (provisional)"   # future wildcard value isn't computable yet
        return f"{names[c]} GW{w} (+{plan['values'][c][w]:.1f})"
    sched = ", ".join(label(c, w) for c, w in sorted(plan["assignment"].items(), key=lambda kv: kv[1]))
    unplaced = [names[c] for c in plan["chips"] if c not in plan["assignment"]]
    expiry = f"chips expire after GW{plan['deadline_gw']}" if plan["deadline_gw"] == FIRST_HALF_LAST_GW \
        else "season's final chips"
    head = (f"PLAY {names[plan['play_now']].upper()} NOW (+{plan['play_now_value']:.1f})"
            if plan["play_now"] else "No chip this week")
    line = f"{head} | Plan ({expiry}): {sched}"
    if unplaced:
        line += f" | Not enough weeks left for: {', '.join(unplaced)}"
    if plan.get("window_shorter_than_deadline"):
        line += " | (forecast doesn't reach the deadline yet; later weeks not yet valued)"
    return line
