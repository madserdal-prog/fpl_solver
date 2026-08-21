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
HIT_COST = 4  # points per transfer beyond free transfers
BENCH_WEIGHT = 0.5  # value of bench players relative to starters (per the guide)


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
            })
            pid += 1
    return pool


def load_pool_from_forecast(forecast_path: str, master_data_path: str, horizon_gw: int = 0):
    """
    Real data loading. forecast_gw{N}.json gives xP per player per round in
    the 5-round horizon (`horizon_gw`=0 -> this round, 1 -> next, etc.).
    master_data (from the bootstrap-static dump) gives position/price/team.
    """
    with open(forecast_path, encoding="utf-8") as f:
        forecast = json.load(f)
    with open(master_data_path, encoding="utf-8") as f:
        master = json.load(f)
    master_by_id = {p["id"]: p for p in master["elements"]}
    pool = []
    for entry in forecast["players"]:
        m = master_by_id[entry["id"]]
        pool.append({
            "id": entry["id"],
            "name": m["web_name"],
            "team": m["team"],
            "position": m["element_type"],  # map to GK/DEF/MID/FWD on your side
            "price": m["now_cost"] / 10.0,
            "xp": float(entry["xP"][horizon_gw]),
        })
    return pool


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(pool, current_squad_ids=None, budget=BUDGET_DEFAULT, free_transfers=1,
          locked_ids=None, unlimited_transfers=False, risk_penalty=None,
          blocked_captain_ids=None):
    """
    Solves the MILP for the best legal squad given the forecasts.

    unlimited_transfers=True is used for wildcard/free hit rounds (no
    transfer penalty, build the squad from scratch).

    risk_penalty: dict {player_id: points deduction}, used for ROBUST
      optimization (cf. Ramezani 2025, arXiv:2505.02170): the objective
      uses (xP - uncertainty_margin) instead of raw xP, so players with
      substantial uncertainty around playing time/injury/rotation
      (typically set by critic.py after a news search) become less
      attractive to the solver -- NOT just excluded from a single captaincy
      check, but actually worth less throughout the entire objective
      function (squad selection, starting XI, captaincy).

    blocked_captain_ids: set of player IDs that can NEVER be selected as
      captain, regardless of xP -- used for players with known high
      variance in playing time (e.g. recently back from a long-term
      injury) where even a small risk_penalty isn't enough to prevent
      captaincy selection due to the doubling bonus.

    Variables:
      x[p]   -- in the 15-man squad
      s[p]   -- in the starting XI (s <= x)
      cap[p] -- captain (cap <= s, exactly one)
      extra_transfers -- continuous, >= max(0, transfers - free transfers)

    Objective (maximized):
      BENCH_WEIGHT * sum(xp_adj*x) + (1-BENCH_WEIGHT) * sum(xp_adj*s) + sum(xp_adj*cap)
      - HIT_COST * extra_transfers

    where xp_adj = xP - risk_penalty. This gives starting players full
    value (BENCH_WEIGHT + (1-BENCH_WEIGHT) = 1 * xp_adj), while bench
    players only count BENCH_WEIGHT * xp_adj -- exactly as described in
    the guide (bench weighting is optional, but reduces the risk of the
    solver "hiding" good value unused on the bench).
    """
    current_squad_ids = set(current_squad_ids or [])
    locked_ids = set(locked_ids or [])
    risk_penalty = risk_penalty or {}
    blocked_captain_ids = set(blocked_captain_ids or [])
    P = len(pool)
    xp = np.array([p["xp"] - risk_penalty.get(p["id"], 0.0) for p in pool])

    # Variable indexing: [x_0..x_{P-1}, s_0..s_{P-1}, cap_0..cap_{P-1}, extra_transfers]
    n_vars = 3 * P + 1
    X, S, CAP, ET = 0, P, 2 * P, 3 * P

    constraints = []

    # Position requirements in the 15-man squad
    for pos, req in SQUAD_REQ.items():
        row = np.zeros(n_vars)
        for i, p in enumerate(pool):
            if p["position"] == pos:
                row[X + i] = 1
        constraints.append(LinearConstraint(row, req, req))

    # Budget
    row = np.zeros(n_vars)
    for i, p in enumerate(pool):
        row[X + i] = p["price"]
    constraints.append(LinearConstraint(row, -np.inf, budget))

    # Max 3 per club
    for t in sorted(set(p["team"] for p in pool)):
        row = np.zeros(n_vars)
        for i, p in enumerate(pool):
            if p["team"] == t:
                row[X + i] = 1
        constraints.append(LinearConstraint(row, -np.inf, MAX_PER_CLUB))

    # Starting XI size
    row = np.zeros(n_vars)
    row[S:S + P] = 1
    constraints.append(LinearConstraint(row, 11, 11))

    # Formation
    for pos, (lo, hi) in XI_MIN_MAX.items():
        row = np.zeros(n_vars)
        for i, p in enumerate(pool):
            if p["position"] == pos:
                row[S + i] = 1
        constraints.append(LinearConstraint(row, lo, hi))

    # s <= x
    for i in range(P):
        row = np.zeros(n_vars)
        row[S + i] = 1
        row[X + i] = -1
        constraints.append(LinearConstraint(row, -np.inf, 0))

    # cap <= s
    for i in range(P):
        row = np.zeros(n_vars)
        row[CAP + i] = 1
        row[S + i] = -1
        constraints.append(LinearConstraint(row, -np.inf, 0))

    # exactly one captain
    row = np.zeros(n_vars)
    row[CAP:CAP + P] = 1
    constraints.append(LinearConstraint(row, 1, 1))

    # Transfer penalty: extra_transfers >= (number sold from current_squad) - free_transfers
    # number sold = |current_squad| - sum_{p in current_squad} x[p]
    if current_squad_ids and not unlimited_transfers:
        row = np.zeros(n_vars)
        row[ET] = 1
        for i, p in enumerate(pool):
            if p["id"] in current_squad_ids:
                row[X + i] = 1
        lo = len(current_squad_ids) - free_transfers
        constraints.append(LinearConstraint(row, lo, np.inf))

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    for i, p in enumerate(pool):
        if p["id"] in locked_ids:
            lb[X + i] = 1
    ub[ET] = 15.0  # can swap at most all 15
    for i, p in enumerate(pool):
        if p["id"] in blocked_captain_ids:
            ub[CAP + i] = 0  # can never be selected as captain, regardless of xP
    bounds = Bounds(lb, ub)

    integrality = np.ones(n_vars)
    integrality[ET] = 0  # extra_transfers can be continuous, the solver sets it to an integer anyway

    # Objective (minimize negative value)
    c = np.zeros(n_vars)
    c[X:X + P] = -BENCH_WEIGHT * xp
    c[S:S + P] = -(1 - BENCH_WEIGHT) * xp
    c[CAP:CAP + P] = -xp
    c[ET] = HIT_COST

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"MILP failed to solve the problem: {res.message}")

    x = res.x[X:X + P]
    s = res.x[S:S + P]
    cap = res.x[CAP:CAP + P]
    extra_transfers = res.x[ET]

    squad_ids = [pool[i]["id"] for i in range(P) if x[i] > 0.5]
    xi_ids = [pool[i]["id"] for i in range(P) if s[i] > 0.5]
    captain_id = next(pool[i]["id"] for i in range(P) if cap[i] > 0.5)

    transfers_in = sorted(set(squad_ids) - current_squad_ids)
    transfers_out = sorted(current_squad_ids - set(squad_ids))
    transfers_made = len(transfers_out)
    hit_taken = 0 if unlimited_transfers else max(0, transfers_made - free_transfers)
    hit_cost = HIT_COST * hit_taken

    pool_by_id = {p["id"]: p for p in pool}
    expected_points = float(
        sum(pool_by_id[i]["xp"] for i in xi_ids)
        + pool_by_id[captain_id]["xp"]  # captain bonus
        - hit_cost
    )

    return {
        "squad": [pool_by_id[i]["name"] for i in squad_ids],
        "squad_ids": squad_ids,  # actual FPL player IDs -- copy these straight into my_team.json
        "starting_xi": [pool_by_id[i]["name"] for i in xi_ids],
        "starting_xi_ids": xi_ids,
        "captain": pool_by_id[captain_id]["name"],
        "captain_id": captain_id,
        "transfers_in": [pool_by_id[i]["name"] for i in transfers_in],
        "transfers_out": [pool_by_id[i]["name"] for i in transfers_out],
        "transfers_made": transfers_made,
        "free_transfers_used": min(transfers_made, free_transfers),
        "hit_taken": hit_taken,
        "hit_cost": hit_cost,
        "expected_points": round(expected_points, 2),  # nominal xP, NOT risk-adjusted
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
