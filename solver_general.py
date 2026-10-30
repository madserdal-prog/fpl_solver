"""
FPL Solver -- general version (fpl-auto-pipeline-guide.md, Part 3)
==================================================================
Maximizes expected points for your own squad. No rival model, no Monte
Carlo -- this is the "pure" version that answers "what is the best legal
squad given the forecasts", i.e. the same question your friend's system solves.

Using real data (once Parts 1/2 are built):
  - Replace make_synthetic_pool() with loading forecast_gw{N}.json
    (per player: a single xP point estimate from the forecast model, not samples)
  - Load my_team.json for budget/locked players/free transfers/chips

Dependencies: numpy, scipy (>=1.9 for scipy.optimize.milp). No PuLP/CBC
needed -- scipy's built-in HiGHS solver is used instead, since this
runtime environment has no network access to install PuLP. Mathematically
this is the same class of solver (MILP) the guide originally suggested
PuLP/CBC for.
"""

import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ---------------------------------------------------------------------------
# Constants (official FPL rules)
# ---------------------------------------------------------------------------

SQUAD_REQ = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN_MAX = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
BUDGET_DEFAULT = 100.0
HIT_COST = 4  # TRUE points per transfer beyond free transfers -- what FPL actually charges,
              # used for REPORTING (hit_cost, expected_points), never inflated.
HIT_SAFETY_MARGIN = 10  # extra decision-only buffer per hit (reported cost stays the true 4).
                        # Counters the "optimizer's curse": picking the highest FORECASTS
                        # systematically picks over-estimates, so marginal-looking gains
                        # don't materialise. Backtest on a simulated season (GW2-9): every
                        # margin from 8 to 12 beat the old code on both 1-week and 3-week
                        # scoring; 2-6 did not. 10 = middle of that region. Simulated, so
                        # re-check with backtest.py on real weeks (--set ...=8/12).
BENCH_WEIGHT = 0.25  # bench xP only counts via autosubs (a starter blanking), so a bench
                     # player is worth a fraction of a starter. Was 0.5, which over-valued the
                     # bench and pulled money away from starters. Heuristic, not fitted.
FREE_TRANSFER_VALUE = 1.0  # decision-only cost of USING a free transfer instead of banking it
                           # (a banked FT has option value). Stops churn for trivial gains.
MAX_BANKED_FREE_TRANSFERS = 5
FT_TERMINAL_VALUE = 1.0  # value of each free transfer still banked at the END of the planning
                         # horizon (option value beyond what the plan can see). Heuristic.


# ---------------------------------------------------------------------------
# Data loading / synthetic test data
# ---------------------------------------------------------------------------

def make_synthetic_pool(seed: int = 42, n_teams: int = 10):
    """
    Synthetic player pool for testing. Replaced with real loading from
    forecast_gw{N}.json once the forecast model (Part 2) is built.
    Here 'xp' is a single point estimate per player (5-round horizon, the
    round-1 value is used in this simple demo).
    """
    rng = np.random.default_rng(seed)
    counts = {"GK": 6, "DEF": 15, "MID": 15, "FWD": 14}
    base_points = {"GK": 3.0, "DEF": 3.3, "MID": 4.0, "FWD": 4.3}
    pool = []
    pid = 1
    for pos, n in counts.items():
        for _ in range(n):
            quality = float(rng.beta(2.0, 4.0))  # skewed toward cheap/weaker players
            price = round(max(3.9, 3.9 + quality * 10.0 + rng.normal(0, 0.2)), 1)
            play_prob = float(np.clip(0.55 + quality * 0.4 + rng.normal(0, 0.05), 0.05, 0.99))
            xp = round(base_points[pos] * (0.5 + quality) * play_prob, 2)
            pool.append({
                "id": pid,
                "name": f"{pos}{pid}",
                "team": f"Team{(pid % n_teams) + 1}",
                "position": pos,
                "price": price,
                "xp": xp,
                "rotation_value": xp,  # no real multi-week horizon in this synthetic demo --
                                        # defaults to the same as xp so old tests still behave
                                        # identically; see the dedicated rotation-value test
                                        # for a scenario where the two genuinely diverge.
                "xp_horizon_sum": xp * 5,  # same simplification -- treats each of the 5 weeks
                                            # as identical to this one, since this demo has no
                                            # real per-week horizon data to sum.
            })
            pid += 1
    return pool


HORIZON_DISCOUNT = 0.85  # per gameweek further out -- reflects that forecasts further
                          # into the future are less certain, so they should count for
                          # less when deciding whether a player is worth having in the
                          # squad at all (not for deciding who starts THIS week).


def load_pool_from_forecast(forecast_path: str, master_data_path: str, horizon_gw: int = 0):
    """
    Real data loading. forecast_gw{N}.json gives xP per player per round in
    the 5-round horizon (`horizon_gw`=0 -> this round, 1 -> next, etc.).
    master_data (from the bootstrap-static dump) gives position/price/team.

    Three separate values are produced per player:
      "xp"             -- THIS round's estimate only. Used to decide who
                           starts and who's captain, since that's a
                           decision about one specific set of fixtures.
      "rotation_value" -- a discounted average across the WHOLE horizon.
                           Used to decide whether a player is worth having
                           in the 15-man squad at all -- e.g. a player who
                           looks mediocre this week but strong from GW+2
                           onward (returning from injury, a fixture swing,
                           a double gameweek coming up) should still be
                           attractive to hold on the bench now, not just
                           judged on this week's number alone.
      "xp_horizon_sum" -- a plain, UNDISCOUNTED sum of xP across the whole
                           horizon (not an average, not decision-weighted).
                           This is purely for human-readable reporting --
                           e.g. "how many points do I expect this player to
                           produce over the next 5 fixtures" -- and isn't
                           used anywhere in the optimization itself.
    """
    with open(forecast_path, encoding="utf-8") as f:
        forecast = json.load(f)
    with open(master_data_path, encoding="utf-8") as f:
        master = json.load(f)
    master_by_id = {p["id"]: p for p in master["elements"]}
    pool = []
    for entry in forecast["players"]:
        master_player = master_by_id[entry["id"]]
        xp_series = entry["xP"]
        weights = [HORIZON_DISCOUNT ** i for i in range(len(xp_series))]
        rotation_value = sum(w * xp for w, xp in zip(weights, xp_series)) / sum(weights)
        pool.append({
            "id": entry["id"],
            "name": master_player["web_name"],
            "team": master_player["team"],
            "position": master_player["element_type"],  # map to GK/DEF/MID/FWD on your side
            "price": master_player["now_cost"] / 10.0,
            "xp": float(xp_series[horizon_gw]),
            "rotation_value": round(rotation_value, 3),
            "xp_horizon_sum": round(sum(xp_series[horizon_gw:]), 3),
            "xp_series": [float(v) for v in xp_series[horizon_gw:]],  # per-week xP, used by solve()
        })
    return pool


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _xp_series(player, horizon_len):
    """Per-week xP for the horizon. Pool entries from load_pool_from_forecast()
    carry 'xp_series'; older/synthetic entries only have 'xp' (1-week horizon)."""
    series = list(player.get("xp_series") or [player["xp"]])
    if len(series) < horizon_len:
        series += [series[-1]] * (horizon_len - len(series))
    return series[:horizon_len]


def prune_pool(pool, keep_ids, per_position_top=45, per_position_cheapest=10, per_position_value=15):
    """Candidate pruning for speed: keeps owned/locked players plus, per position,
    the best by horizon xP, the cheapest (budget enablers) and the best xP-per-£.
    A heuristic -- a player outside all three lists is almost never optimal."""
    keep = set(keep_ids)
    for pos in SQUAD_REQ:
        group = [p for p in pool if p["position"] == pos]
        horizon = lambda p: p.get("xp_horizon_sum", p["xp"])
        for ranked, k in ((sorted(group, key=lambda p: -horizon(p)), per_position_top),
                          (sorted(group, key=lambda p: p["price"]), per_position_cheapest),
                          (sorted(group, key=lambda p: -horizon(p) / max(p["price"], 3.5)), per_position_value)):
            keep.update(p["id"] for p in ranked[:k])
    return [p for p in pool if p["id"] in keep]


def solve(pool, current_squad_ids=None, budget=BUDGET_DEFAULT, free_transfers=1,
          locked_ids=None, unlimited_transfers=False, risk_penalty=None,
          blocked_captain_ids=None, bench_weight=None, sell_values=None,
          hit_safety_margin=None, free_transfer_value=None, plan_future_transfers=True,
          horizon_weeks=None, forbid_week0_squads=None, prune=True):
    """
    Multi-week MILP. For every week of the horizon it chooses the squad (so
    transfers can be PLANNED across weeks), the starting XI and the captain.
    Objective = discounted sum over weeks of (starters + captain bonus +
    BENCH_WEIGHT * bench) xP, minus discounted hit costs, plus a small value
    for free transfers still banked at the end of the horizon.

    Free-transfer rules modelled exactly: +1 per week, max 5 banked, each
    transfer beyond what's banked costs a hit, and taking any hit resets the
    following week to 1 free transfer.

    Only THIS week's decisions are actionable; later weeks are a plan that is
    re-optimised every run as forecasts update.

    plan_future_transfers=False -> squad held fixed after this week (the
      previous behaviour); FREE_TRANSFER_VALUE then stands in for banking.
    unlimited_transfers=True -> this week is a Wildcard/initial build: any
      number of changes, no hit, and the banked count is preserved (+1).
    risk_penalty: {player_id: share 0..1} of THIS week's xP removed.
    blocked_captain_ids: cannot captain THIS week.
    bench_weight: overrides BENCH_WEIGHT for THIS week only (Bench Boost).
    forbid_week0_squads: list of squads (sets of ids) this week's squad must
      differ from -- used to find the runner-up plan.
    """
    import scipy.sparse as sp

    current_squad_ids = set(current_squad_ids or [])
    locked_ids = set(locked_ids or [])
    risk_penalty = risk_penalty or {}
    blocked_captain_ids = set(blocked_captain_ids or [])
    sell_values = sell_values or {}
    hit_safety_margin = HIT_SAFETY_MARGIN if hit_safety_margin is None else hit_safety_margin
    free_transfer_value = FREE_TRANSFER_VALUE if free_transfer_value is None else free_transfer_value
    effective_hit_cost = HIT_COST + hit_safety_margin

    if prune and len(pool) > 150:
        pool = prune_pool(pool, current_squad_ids | locked_ids | set(risk_penalty))
    N = len(pool)
    T = max(len(p.get("xp_series") or [p["xp"]]) for p in pool)
    if horizon_weeks:
        T = min(T, horizon_weeks)
    xp = np.array([_xp_series(p, T) for p in pool], dtype=float).T   # (T, N) nominal
    xp_adj = xp.copy()
    risk_share = np.clip(np.array([risk_penalty.get(p["id"], 0.0) for p in pool]), 0.0, 1.0)
    xp_adj[0] *= (1.0 - risk_share)
    discount = np.array([HORIZON_DISCOUNT ** t for t in range(T)])
    week_bw = np.full(T, BENCH_WEIGHT)
    if bench_weight is not None:
        week_bw[0] = bench_weight

    # ---- variable layout -------------------------------------------------
    blocks = {}
    nxt = 0
    for name, size in (("x", T * N), ("s", T * N), ("c", T * N), ("tin", T * N), ("tout", T * N),
                       ("n", T), ("h", T), ("ft", T + 1), ("z", T)):
        blocks[name] = nxt
        nxt += size
    num_vars = nxt
    X = lambda t, i: blocks["x"] + t * N + i
    S = lambda t, i: blocks["s"] + t * N + i
    C = lambda t, i: blocks["c"] + t * N + i
    TIN = lambda t, i: blocks["tin"] + t * N + i
    TOUT = lambda t, i: blocks["tout"] + t * N + i
    NN = lambda t: blocks["n"] + t
    H = lambda t: blocks["h"] + t
    FT = lambda t: blocks["ft"] + t
    Z = lambda t: blocks["z"] + t

    rows, cols, vals, lbs, ubs = [], [], [], [], []
    def add_row(col_idx, coeffs, lb, ub):
        r = len(lbs)
        rows.extend([r] * len(col_idx)); cols.extend(col_idx); vals.extend(coeffs)
        lbs.append(lb); ubs.append(ub)

    by_pos = {pos: [i for i, p in enumerate(pool) if p["position"] == pos] for pos in SQUAD_REQ}
    teams = sorted({p["team"] for p in pool})
    cost = [sell_values.get(p["id"], p["price"]) for p in pool]
    owned = [1 if p["id"] in current_squad_ids else 0 for p in pool]
    fresh_build = not current_squad_ids

    for t in range(T):
        for pos, n in SQUAD_REQ.items():
            add_row([X(t, i) for i in by_pos[pos]], [1] * len(by_pos[pos]), n, n)
        add_row([X(t, i) for i in range(N)], cost, -np.inf, budget)
        for team in teams:
            members = [i for i, p in enumerate(pool) if p["team"] == team]
            add_row([X(t, i) for i in members], [1] * len(members), -np.inf, MAX_PER_CLUB)
        add_row([S(t, i) for i in range(N)], [1] * N, 11, 11)
        for pos, (lo, hi) in XI_MIN_MAX.items():
            add_row([S(t, i) for i in by_pos[pos]], [1] * len(by_pos[pos]), lo, hi)
        add_row([C(t, i) for i in range(N)], [1] * N, 1, 1)
        for i in range(N):
            add_row([S(t, i), X(t, i)], [1, -1], -np.inf, 0)
            add_row([C(t, i), S(t, i)], [1, -1], -np.inf, 0)
            # squad flow: x_t - x_{t-1} = in - out   (x_{-1} = current squad)
            if t == 0:
                add_row([X(0, i), TIN(0, i), TOUT(0, i)], [1, -1, 1], owned[i], owned[i])
            else:
                add_row([X(t, i), X(t - 1, i), TIN(t, i), TOUT(t, i)], [1, -1, -1, 1], 0, 0)
        add_row([NN(t)] + [TIN(t, i) for i in range(N)], [1] + [-1] * N, 0, 0)  # n_t = sum(in)
        # free transfers & hits
        week_unlimited = (t == 0 and (unlimited_transfers or fresh_build))
        if week_unlimited:
            add_row([H(t)], [1], 0, 0)
            add_row([FT(t + 1), FT(t)], [1, -1], -np.inf, 1)       # banked count kept (+1)
        else:
            add_row([H(t), NN(t), FT(t)], [1, -1, 1], 0, np.inf)    # h >= n - ft
            add_row([H(t), Z(t)], [1, -15], -np.inf, 0)             # any hit -> z = 1
            add_row([FT(t + 1), FT(t), NN(t), H(t)], [1, -1, 1, -1], -np.inf, 1)  # ft' <= ft - n + h + 1
            add_row([FT(t + 1), Z(t)], [1, 4], -np.inf, 5)          # hit taken -> ft' <= 1
        if t >= 1 and not plan_future_transfers:
            add_row([NN(t)], [1], 0, 0)

    for forbidden in (forbid_week0_squads or []):
        idx = [i for i, p in enumerate(pool) if p["id"] in set(forbidden)]
        add_row([X(0, i) for i in idx], [1] * len(idx), -np.inf, len(idx) - 1)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(lbs), num_vars))
    lower = np.zeros(num_vars); upper = np.ones(num_vars)
    for t in range(T):
        lower[NN(t)], upper[NN(t)] = 0, 15
        lower[H(t)], upper[H(t)] = 0, 15
        for i, p in enumerate(pool):
            if p["id"] in locked_ids:
                lower[X(t, i)] = 1
    for t in range(T + 1):
        lower[FT(t)], upper[FT(t)] = 0, MAX_BANKED_FREE_TRANSFERS
    lower[FT(0)] = upper[FT(0)] = min(free_transfers, MAX_BANKED_FREE_TRANSFERS)
    for i, p in enumerate(pool):
        if p["id"] in blocked_captain_ids:
            upper[C(0, i)] = 0
    integrality = np.ones(num_vars)

    # ---- objective (milp minimises) -------------------------------------
    c = np.zeros(num_vars)
    for t in range(T):
        for i in range(N):
            c[X(t, i)] = -discount[t] * week_bw[t] * xp_adj[t][i]
            c[S(t, i)] = -discount[t] * (1 - week_bw[t]) * xp_adj[t][i]
            c[C(t, i)] = -discount[t] * xp_adj[t][i]
        c[H(t)] = discount[t] * effective_hit_cost
    if plan_future_transfers:
        c[FT(T)] = -FT_TERMINAL_VALUE * discount[-1]   # value of FTs still banked at the end
    elif not unlimited_transfers and not fresh_build and free_transfers < MAX_BANKED_FREE_TRANSFERS:
        c[NN(0)] = free_transfer_value                  # old single-decision banking heuristic

    result = milp(c=c, constraints=[LinearConstraint(A, np.array(lbs), np.array(ubs))],
                  integrality=integrality, bounds=Bounds(lower, upper),
                  options={"mip_rel_gap": 1e-4, "time_limit": 120})
    if result.x is None:
        raise RuntimeError(f"MILP failed to solve the problem: {result.message}")
    x = result.x
    on = lambda v: v > 0.5

    pool_by_id = {p["id"]: p for p in pool}
    ids = [p["id"] for p in pool]
    def week(t):
        squad = [ids[i] for i in range(N) if on(x[X(t, i)])]
        xi = [ids[i] for i in range(N) if on(x[S(t, i)])]
        cap = next(ids[i] for i in range(N) if on(x[C(t, i)]))
        return squad, xi, cap
    squad_ids, xi_ids, captain_id = week(0)
    adj0 = {p["id"]: xp_adj[0][i] for i, p in enumerate(pool)}

    vice_candidates = [i for i in xi_ids if i != captain_id and i not in blocked_captain_ids]
    vice_id = max(vice_candidates, key=lambda i: adj0[i]) if vice_candidates else None
    bench_ids = sorted((i for i in squad_ids if i not in xi_ids),
                       key=lambda i: (pool_by_id[i]["position"] != "GK", -adj0[i]))

    def paired(ins, outs):
        pairs = []
        hz = lambda i: pool_by_id[i].get("xp_horizon_sum", pool_by_id[i]["xp"])
        for pos in SQUAD_REQ:
            a = sorted([i for i in ins if pool_by_id[i]["position"] == pos], key=lambda i: -hz(i))
            b = sorted([i for i in outs if pool_by_id[i]["position"] == pos], key=lambda i: -hz(i))
            pairs.extend(zip(a, b))
        return pairs

    transfers_in = sorted(set(squad_ids) - current_squad_ids) if current_squad_ids else []
    transfers_out = sorted(current_squad_ids - set(squad_ids))
    pairs = paired(transfers_in, transfers_out)
    transfers_made = len(transfers_out)
    hit_taken = 0 if (unlimited_transfers or fresh_build) else max(0, transfers_made - free_transfers)
    hit_cost = HIT_COST * hit_taken

    # True free-transfer counts, replayed with FPL's rules from the chosen
    # transfer counts. (The model only uses FTs as an upper bound, so its own
    # FT variables can sit below the real value without affecting decisions.)
    true_ft = [min(free_transfers, MAX_BANKED_FREE_TRANSFERS)]
    for t in range(T):
        n_t = int(round(x[NN(t)]))
        if t == 0 and (unlimited_transfers or fresh_build):
            true_ft.append(min(MAX_BANKED_FREE_TRANSFERS, true_ft[0] + 1))
        elif n_t > true_ft[t]:
            true_ft.append(1)
        else:
            true_ft.append(min(MAX_BANKED_FREE_TRANSFERS, true_ft[t] - n_t + 1))

    # Future-week plan (informational; re-optimised every run)
    plan, prev = [], set(squad_ids)
    for t in range(1, T):
        sq, xi_t, cap_t = week(t)
        ins, outs = sorted(set(sq) - prev), sorted(prev - set(sq))
        plan.append({
            "week_offset": t,
            "free_transfers_available": true_ft[t],
            "transfers": [(pool_by_id[a]["name"], pool_by_id[b]["name"]) for a, b in paired(ins, outs)],
            "hit_cost": HIT_COST * int(round(x[H(t)])),
            "captain": pool_by_id[cap_t]["name"],
            "starting_xi_ids": xi_t, "squad_ids": sq, "captain_id": cap_t,
        })
        prev = set(sq)

    expected_points = sum(pool_by_id[i]["xp"] for i in xi_ids) + pool_by_id[captain_id]["xp"] - hit_cost
    horizon_points = sum(xp[t][i] * (on(x[S(t, i)]) + on(x[C(t, i)])) for t in range(T) for i in range(N)) \
        - sum(HIT_COST * round(x[H(t)]) for t in range(T))
    squad_value = round(sum(sell_values.get(i, pool_by_id[i]["price"]) for i in squad_ids), 1)

    return {
        "squad": [pool_by_id[i]["name"] for i in squad_ids],
        "squad_ids": squad_ids,
        "starting_xi": [pool_by_id[i]["name"] for i in xi_ids],
        "starting_xi_ids": xi_ids,
        "bench": [pool_by_id[i]["name"] for i in bench_ids],
        "bench_ids": bench_ids,
        "captain": pool_by_id[captain_id]["name"],
        "captain_id": captain_id,
        "vice_captain": pool_by_id[vice_id]["name"] if vice_id else None,
        "transfers_in": [pool_by_id[a]["name"] for a, _ in pairs],
        "transfers_out": [pool_by_id[b]["name"] for _, b in pairs],
        "transfer_point_gains": [round(pool_by_id[a].get("xp_horizon_sum", pool_by_id[a]["xp"])
                                       - pool_by_id[b].get("xp_horizon_sum", pool_by_id[b]["xp"]), 2)
                                 for a, b in pairs],
        "transfers_made": transfers_made,
        "free_transfers_used": min(transfers_made, free_transfers),
        "hit_taken": hit_taken,
        "hit_cost": hit_cost,
        "expected_points": round(float(expected_points), 2),
        "horizon_expected_points": round(float(horizon_points), 2),
        "horizon_weeks": T,
        "future_plan": plan,
        "objective_value": round(float(-result.fun), 3),
        "bench_points": round(sum(pool_by_id[i]["xp"] for i in bench_ids), 2),
        "squad_value": squad_value,
        "bank_remaining": round(budget - squad_value, 1),
        "risk_adjusted_players": [pool_by_id[pid]["name"] for pid in risk_penalty
                                   if pid in squad_ids and risk_penalty[pid] > 0],
        "blocked_from_captaincy_in_squad": [pool_by_id[pid]["name"] for pid in blocked_captain_ids
                                             if pid in squad_ids],
    }


# ---------------------------------------------------------------------------
# Demo / self-test with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pool = make_synthetic_pool()

    # Synthetic "current_squad" -- in practice loaded from my_team.json
    rng = np.random.default_rng(1)
    all_ids = [p["id"] for p in pool]
    current_squad_ids = list(rng.choice(all_ids, size=15, replace=False))

    print("--- Normal round (1 free transfer) ---")
    solution = solve(pool, current_squad_ids=current_squad_ids, budget=100.0, free_transfers=1)
    print(json.dumps(solution, indent=2, ensure_ascii=False))

    print("\n--- Wildcard round (unlimited transfers, no penalty) ---")
    wc_solution = solve(pool, current_squad_ids=current_squad_ids, budget=100.0,
                         free_transfers=1, unlimited_transfers=True)
    print(json.dumps(wc_solution, indent=2, ensure_ascii=False))
