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
HIT_SAFETY_MARGIN = 4  # extra buffer used ONLY inside the optimization's decision-making, not
                        # in what gets reported/charged. A plain linear optimizer has no concept
                        # of caution -- it takes a hit for a net gain of +0.01 exactly as readily
                        # as +20, both being "optimal" given the exact numbers it's fed. Real FPL
                        # practice is explicit that a hit should only be taken for a CLEAR margin,
                        # not a marginal one, precisely because projections carry real uncertainty.
                        # This makes the solver require the combined gain from a hit-taking swap
                        # to exceed HIT_COST + HIT_SAFETY_MARGIN (default: 8, not 4) before it's
                        # recommended, while still reporting/charging the true 4-per-hit cost.
BENCH_WEIGHT = 0.25  # bench xP only counts via autosubs (a starter blanking), so a bench
                     # player is worth a fraction of a starter. Was 0.5, which over-valued the
                     # bench and pulled money away from starters. Heuristic, not fitted.
FREE_TRANSFER_VALUE = 1.0  # decision-only cost of USING a free transfer instead of banking it
                           # (a banked FT has option value). Stops churn for trivial gains.
MAX_BANKED_FREE_TRANSFERS = 5


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
    carry 'xp_series'; older/synthetic entries only have 'xp' (treated as a
    1-week horizon, i.e. the old single-week behaviour)."""
    series = list(player.get("xp_series") or [player["xp"]])
    if len(series) < horizon_len:
        series += [series[-1]] * (horizon_len - len(series))
    return series[:horizon_len]


def solve(pool, current_squad_ids=None, budget=BUDGET_DEFAULT, free_transfers=1,
          locked_ids=None, unlimited_transfers=False, risk_penalty=None,
          blocked_captain_ids=None, bench_weight=None, sell_values=None,
          hit_safety_margin=None, free_transfer_value=None):
    """
    Multi-week MILP: picks ONE 15-man squad (this week's transfers), then the
    best starting XI + captain for EACH week of the forecast horizon, holding
    that squad fixed. Objective = discounted sum over the horizon of
    (starters' xP + captain's xP + BENCH_WEIGHT * bench xP), minus hit cost.

    WHY (bug fix): the previous single-week version valued every starter by
    THIS WEEK's xP only -- the horizon only influenced bench players. Verified
    directly: given two otherwise identical candidates, it bought one worth
    ~10 pts over 5 GWs instead of one worth ~42, because of a 0.1-pt edge this
    week. That made fixture runs nearly irrelevant for starters, and made
    spending the bank on a premium with a good run rarely look worthwhile.

    Assumption: the squad is held for the whole horizon (future transfers are
    not modelled). That's the standard simplification -- it can only
    UNDER-state the value of flexibility, never invent value.

    risk_penalty: {player_id: share 0..1} of THIS WEEK's xP removed (proportional). The
      critic assesses near-term risk (news, rotation, a knock) and re-runs
      every week, so it shouldn't depress all five weeks.
    blocked_captain_ids: cannot captain THIS week (later weeks unaffected).
    bench_weight: overrides BENCH_WEIGHT for THIS WEEK only -- used for the
      Bench Boost alternative (1.0), since the chip only applies to one week.
    hit_safety_margin: extra decision-only buffer on top of HIT_COST (see
      HIT_SAFETY_MARGIN). Reported hit_cost always uses the true HIT_COST.
    free_transfer_value: decision-only cost of using a free transfer instead
      of banking it (see FREE_TRANSFER_VALUE). Zero when 5 are already banked.
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

    N = len(pool)
    T = max(len(p.get("xp_series") or [p["xp"]]) for p in pool)
    xp = np.array([_xp_series(p, T) for p in pool]).T          # shape (T, N), nominal
    xp_adj = xp.copy()
    # risk_penalty values are the SHARE of this week's xP to remove (0..1), not
    # flat points. BUG FIX: a flat -8 for a "high" flag exceeded most players'
    # whole weekly xP, so an AVAILABLE player (Isak, managed minutes) was
    # valued below a player who was OUT. Proportional scaling can't go below 0.
    risk_share = np.clip(np.array([risk_penalty.get(p["id"], 0.0) for p in pool]), 0.0, 1.0)
    xp_adj[0] *= (1.0 - risk_share)  # this week only
    discount = np.array([HORIZON_DISCOUNT ** t for t in range(T)])
    week_bench_weight = np.full(T, BENCH_WEIGHT)
    if bench_weight is not None:
        week_bench_weight[0] = bench_weight

    # Variable layout: [squad (N)] [start week0..T-1 (T*N)] [captain week0..T-1 (T*N)] [extra transfers (1)]
    SQ = 0
    def ST(t): return N + t * N
    def CP(t): return N + T * N + t * N
    EX = N + 2 * T * N
    num_vars = EX + 1

    rows, cols, vals, lbs, ubs = [], [], [], [], []
    def add_row(col_idx, coeffs, lb, ub):
        r = len(lbs)
        rows.extend([r] * len(col_idx)); cols.extend(col_idx); vals.extend(coeffs)
        lbs.append(lb); ubs.append(ub)

    idx = np.arange(N)
    by_pos = {pos: [i for i, p in enumerate(pool) if p["position"] == pos] for pos in SQUAD_REQ}

    for pos, n in SQUAD_REQ.items():                      # squad shape
        add_row([SQ + i for i in by_pos[pos]], [1] * len(by_pos[pos]), n, n)
    add_row(list(SQ + idx), [sell_values.get(p["id"], p["price"]) for p in pool], -np.inf, budget)
    for team in sorted({p["team"] for p in pool}):        # max 3 per club
        members = [i for i, p in enumerate(pool) if p["team"] == team]
        add_row([SQ + i for i in members], [1] * len(members), -np.inf, MAX_PER_CLUB)

    for t in range(T):                                     # a lineup per week
        add_row(list(ST(t) + idx), [1] * N, 11, 11)
        for pos, (lo, hi) in XI_MIN_MAX.items():
            add_row([ST(t) + i for i in by_pos[pos]], [1] * len(by_pos[pos]), lo, hi)
        add_row(list(CP(t) + idx), [1] * N, 1, 1)
        for i in range(N):
            add_row([ST(t) + i, SQ + i], [1, -1], -np.inf, 0)   # start only if owned
            add_row([CP(t) + i, ST(t) + i], [1, -1], -np.inf, 0)  # captain only if starting

    owned_idx = [i for i, p in enumerate(pool) if p["id"] in current_squad_ids]
    if current_squad_ids and not unlimited_transfers:      # extra >= sold - free
        add_row([EX] + [SQ + i for i in owned_idx], [1] + [1] * len(owned_idx),
                len(current_squad_ids) - free_transfers, np.inf)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(lbs), num_vars))
    constraints = [LinearConstraint(A, np.array(lbs), np.array(ubs))]

    lower = np.zeros(num_vars); upper = np.ones(num_vars)
    for i, p in enumerate(pool):
        if p["id"] in locked_ids:
            lower[SQ + i] = 1
        if p["id"] in blocked_captain_ids:
            upper[CP(0) + i] = 0
    upper[EX] = 15.0
    integrality = np.ones(num_vars); integrality[EX] = 0

    # Objective (milp minimises, so negate). Per player per week t:
    #   bench:   d_t * bw_t * xp      starter: d_t * xp      captain: d_t * 2 * xp
    c = np.zeros(num_vars)
    c[SQ:SQ + N] = -(discount[:, None] * week_bench_weight[:, None] * xp_adj).sum(axis=0)
    for t in range(T):
        c[ST(t):ST(t) + N] = -discount[t] * (1 - week_bench_weight[t]) * xp_adj[t]
        c[CP(t):CP(t) + N] = -discount[t] * xp_adj[t]
    c[EX] = effective_hit_cost
    if (current_squad_ids and not unlimited_transfers
            and free_transfers < MAX_BANKED_FREE_TRANSFERS and free_transfer_value):
        # Each transfer = (owned count) - (owned kept); rewarding "kept" by
        # free_transfer_value is the same as charging it per transfer made.
        for i in owned_idx:
            c[SQ + i] -= free_transfer_value

    result = milp(c=c, constraints=constraints, integrality=integrality,
                  bounds=Bounds(lower, upper), options={"mip_rel_gap": 1e-4})
    if not result.success:
        raise RuntimeError(f"MILP failed to solve the problem: {result.message}")
    x = result.x

    pool_by_id = {p["id"]: p for p in pool}
    squad_ids = [pool[i]["id"] for i in range(N) if x[SQ + i] > 0.5]
    xi_ids = [pool[i]["id"] for i in range(N) if x[ST(0) + i] > 0.5]
    captain_id = next(pool[i]["id"] for i in range(N) if x[CP(0) + i] > 0.5)
    adj0 = {p["id"]: xp_adj[0][i] for i, p in enumerate(pool)}

    # Vice-captain: best remaining starter this week who isn't blocked.
    vice_candidates = [i for i in xi_ids if i != captain_id and i not in blocked_captain_ids]
    vice_id = max(vice_candidates, key=lambda i: adj0[i]) if vice_candidates else None

    # Bench ORDER matters for FPL autosubs: GK slot first (fixed by FPL),
    # then outfielders by this week's xP, best first.
    bench_ids = [i for i in squad_ids if i not in xi_ids]
    bench_ids.sort(key=lambda i: (pool_by_id[i]["position"] != "GK", -adj0[i]))

    transfers_in = sorted(set(squad_ids) - current_squad_ids)
    transfers_out = sorted(current_squad_ids - set(squad_ids))
    transfers_made = len(transfers_out)
    hit_taken = 0 if unlimited_transfers else max(0, transfers_made - free_transfers)
    hit_cost = HIT_COST * hit_taken

    # Pair IN/OUT by POSITION (FPL requires like-for-like), highest-xP first
    # within each position -- the old version paired by sorted id, which could
    # pair e.g. a DEF in against a MID out and show a misleading gain.
    pairs = []
    for pos in SQUAD_REQ:
        ins = sorted([i for i in transfers_in if pool_by_id[i]["position"] == pos],
                     key=lambda i: -pool_by_id[i].get("xp_horizon_sum", pool_by_id[i]["xp"]))
        outs = sorted([i for i in transfers_out if pool_by_id[i]["position"] == pos],
                      key=lambda i: -pool_by_id[i].get("xp_horizon_sum", pool_by_id[i]["xp"]))
        pairs.extend(zip(ins, outs))
    transfer_point_gains = [
        round(pool_by_id[a].get("xp_horizon_sum", pool_by_id[a]["xp"])
              - pool_by_id[b].get("xp_horizon_sum", pool_by_id[b]["xp"]), 2)
        for a, b in pairs
    ]

    expected_points = sum(pool_by_id[i]["xp"] for i in xi_ids) + pool_by_id[captain_id]["xp"] - hit_cost
    horizon_points = sum(
        xp[t][i] * (x[ST(t) + i] > 0.5) + xp[t][i] * (x[CP(t) + i] > 0.5)
        for t in range(T) for i in range(N)) - hit_cost
    squad_value = round(sum(sell_values.get(i, pool_by_id[i]["price"]) for i in squad_ids), 1)

    return {
        "squad": [pool_by_id[i]["name"] for i in squad_ids],
        "squad_ids": squad_ids,
        "starting_xi": [pool_by_id[i]["name"] for i in xi_ids],
        "starting_xi_ids": xi_ids,
        "bench": [pool_by_id[i]["name"] for i in bench_ids],        # in recommended autosub order
        "bench_ids": bench_ids,
        "captain": pool_by_id[captain_id]["name"],
        "captain_id": captain_id,
        "vice_captain": pool_by_id[vice_id]["name"] if vice_id else None,
        "transfers_in": [pool_by_id[a]["name"] for a, _ in pairs],
        "transfers_out": [pool_by_id[b]["name"] for _, b in pairs],
        "transfer_point_gains": transfer_point_gains,  # per IN/OUT pair, summed over the horizon
        "transfers_made": transfers_made,
        "free_transfers_used": min(transfers_made, free_transfers),
        "hit_taken": hit_taken,
        "hit_cost": hit_cost,
        "expected_points": round(float(expected_points), 2),       # THIS week, nominal, excl. bench
        "horizon_expected_points": round(float(horizon_points), 2),  # undiscounted, over T weeks
        "horizon_weeks": T,
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
