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
                "rotation_value": xp,  # no real multi-week horizon in this synthetic demo --
                                        # defaults to the same as xp so old tests still behave
                                        # identically; see the dedicated rotation-value test
                                        # for a scenario where the two genuinely diverge.
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

    Two separate values are produced per player:
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
        })
    return pool


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(pool, current_squad_ids=None, budget=BUDGET_DEFAULT, free_transfers=1,
          locked_ids=None, unlimited_transfers=False, risk_penalty=None,
          blocked_captain_ids=None, bench_weight=None):
    """
    Solves the MILP for the best legal squad given the forecasts.

    bench_weight: overrides the module-level BENCH_WEIGHT constant when
      given. Used to answer "what's the best squad if bench points count
      fully this week" (bench_weight=1.0) -- i.e. evaluating a genuine
      Bench Boost alternative, WITHOUT touching transfer limits (Bench
      Boost does not grant extra free transfers, it only changes whether
      bench points count).

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

    Decision variables per player (each is a yes/no 0/1 switch, except the
    last one which is a plain number):
      in_squad[player]     -- is this player one of the 15?
      in_starting_xi[player] -- is this player in the starting XI? (only
                                 possible if in_squad[player] is also yes)
      is_captain[player]     -- is this player the captain? (only possible
                                 if in_starting_xi[player] is also yes;
                                 exactly one player gets "yes" overall)
      extra_transfers_count  -- how many transfers this squad needs beyond
                                 the free ones (a plain number, not yes/no)

    Objective (maximized):
      BENCH_WEIGHT * sum(adjusted_rotation_value * in_squad)
      + sum((adjusted_points - BENCH_WEIGHT * adjusted_rotation_value) * in_starting_xi)
      + sum(adjusted_points * is_captain)
      - HIT_COST * extra_transfers_count

    where adjusted_points = this round's xP - risk_penalty (used for
    starting/captaincy decisions -- a decision about ONE specific set of
    fixtures), and adjusted_rotation_value = a discounted multi-week
    average xP - risk_penalty (used for squad membership -- whether a
    player is worth HOLDING at all, including bench players who may not
    play this week but could next week). The starting-xi term is written
    to cancel out the squad term's rotation_value contribution, so a
    starter ends up valued at pure adjusted_points (not a blend), while a
    bench player ends up valued at BENCH_WEIGHT * adjusted_rotation_value.
    """
    current_squad_ids = set(current_squad_ids or [])
    locked_ids = set(locked_ids or [])
    risk_penalty = risk_penalty or {}
    blocked_captain_ids = set(blocked_captain_ids or [])
    bench_weight = BENCH_WEIGHT if bench_weight is None else bench_weight
    num_players = len(pool)
    # Squad membership is valued by rotation_value (horizon-aware -- "is this
    # player worth HOLDING at all, given where their fixtures/role are
    # trending"), while starting/captaincy is valued by xp (this week's
    # number only -- "who should actually play THIS specific gameweek").
    adjusted_points = np.array([player["xp"] - risk_penalty.get(player["id"], 0.0) for player in pool])
    adjusted_rotation_value = np.array(
        [player.get("rotation_value", player["xp"]) - risk_penalty.get(player["id"], 0.0) for player in pool])

    # Every player gets THREE yes/no variables (in_squad, in_starting_xi,
    # is_captain), laid out back-to-back in one long array, plus one final
    # plain number (extra_transfers_count) at the very end. These offsets
    # say where each player's block of three variables begins.
    num_variables = 3 * num_players + 1
    SQUAD_OFFSET = 0
    STARTING_XI_OFFSET = num_players
    CAPTAIN_OFFSET = 2 * num_players
    EXTRA_TRANSFERS_INDEX = 3 * num_players

    constraints = []

    # Position requirements in the 15-man squad (e.g. exactly 2 goalkeepers)
    for position, required_count in SQUAD_REQ.items():
        constraint_row = np.zeros(num_variables)
        for i, player in enumerate(pool):
            if player["position"] == position:
                constraint_row[SQUAD_OFFSET + i] = 1
        constraints.append(LinearConstraint(constraint_row, required_count, required_count))

    # Budget: total price of the 15-man squad must not exceed what's available
    constraint_row = np.zeros(num_variables)
    for i, player in enumerate(pool):
        constraint_row[SQUAD_OFFSET + i] = player["price"]
    constraints.append(LinearConstraint(constraint_row, -np.inf, budget))

    # Max 3 players from any one club
    for team_name in sorted(set(player["team"] for player in pool)):
        constraint_row = np.zeros(num_variables)
        for i, player in enumerate(pool):
            if player["team"] == team_name:
                constraint_row[SQUAD_OFFSET + i] = 1
        constraints.append(LinearConstraint(constraint_row, -np.inf, MAX_PER_CLUB))

    # Starting XI must be exactly 11 players
    constraint_row = np.zeros(num_variables)
    constraint_row[STARTING_XI_OFFSET:STARTING_XI_OFFSET + num_players] = 1
    constraints.append(LinearConstraint(constraint_row, 11, 11))

    # Formation: e.g. between 3 and 5 defenders in the starting XI
    for position, (min_starters, max_starters) in XI_MIN_MAX.items():
        constraint_row = np.zeros(num_variables)
        for i, player in enumerate(pool):
            if player["position"] == position:
                constraint_row[STARTING_XI_OFFSET + i] = 1
        constraints.append(LinearConstraint(constraint_row, min_starters, max_starters))

    # A player can only start if they're actually in the squad
    for i in range(num_players):
        constraint_row = np.zeros(num_variables)
        constraint_row[STARTING_XI_OFFSET + i] = 1
        constraint_row[SQUAD_OFFSET + i] = -1
        constraints.append(LinearConstraint(constraint_row, -np.inf, 0))

    # A player can only be captain if they're actually starting
    for i in range(num_players):
        constraint_row = np.zeros(num_variables)
        constraint_row[CAPTAIN_OFFSET + i] = 1
        constraint_row[STARTING_XI_OFFSET + i] = -1
        constraints.append(LinearConstraint(constraint_row, -np.inf, 0))

    # Exactly one captain, no more, no less
    constraint_row = np.zeros(num_variables)
    constraint_row[CAPTAIN_OFFSET:CAPTAIN_OFFSET + num_players] = 1
    constraints.append(LinearConstraint(constraint_row, 1, 1))

    # Transfer penalty: extra_transfers_count >= (players sold from current
    # squad) - free_transfers. Players sold = how many of your CURRENT
    # players are NOT in the new squad.
    if current_squad_ids and not unlimited_transfers:
        constraint_row = np.zeros(num_variables)
        constraint_row[EXTRA_TRANSFERS_INDEX] = 1
        for i, player in enumerate(pool):
            if player["id"] in current_squad_ids:
                constraint_row[SQUAD_OFFSET + i] = 1
        min_transfers_required = len(current_squad_ids) - free_transfers
        constraints.append(LinearConstraint(constraint_row, min_transfers_required, np.inf))

    # Bounds: every yes/no variable is 0 or 1; extra_transfers_count can be
    # any number from 0 up to 15 (can't transfer more than a full squad)
    lower_bounds = np.zeros(num_variables)
    upper_bounds = np.ones(num_variables)
    for i, player in enumerate(pool):
        if player["id"] in locked_ids:
            lower_bounds[SQUAD_OFFSET + i] = 1  # force this player into the squad
    upper_bounds[EXTRA_TRANSFERS_INDEX] = 15.0
    for i, player in enumerate(pool):
        if player["id"] in blocked_captain_ids:
            upper_bounds[CAPTAIN_OFFSET + i] = 0  # can never be selected as captain, regardless of xP
    bounds = Bounds(lower_bounds, upper_bounds)

    is_integer_variable = np.ones(num_variables)
    is_integer_variable[EXTRA_TRANSFERS_INDEX] = 0  # can be continuous; the solver sets it to a whole number anyway

    # Objective: scipy.optimize.milp always MINIMIZES, so every coefficient
    # here is negated -- minimizing negative points is the same as
    # maximizing points.
    #
    # The three terms per player must ADD UP correctly depending on which
    # combination of (in_squad, in_starting_xi, is_captain) ends up 1:
    #   bench (in_squad=1 only):            BENCH_WEIGHT * rotation_value
    #   starter, not captain:                xp                     <- pure xp, no rotation_value mixed in
    #   captain (also a starter):            2 * xp
    # Since in_squad=1 is ALWAYS true whenever in_starting_xi=1 (a player
    # can't start without being in the squad), the squad-term's contribution
    # would otherwise leak into the starter's total. The starting-xi term is
    # therefore set to (xp - BENCH_WEIGHT * rotation_value) specifically so
    # it exactly cancels that leakage out, leaving starters valued by pure
    # xp and bench players valued by discounted rotation_value.
    objective_coefficients = np.zeros(num_variables)
    objective_coefficients[SQUAD_OFFSET:SQUAD_OFFSET + num_players] = -bench_weight * adjusted_rotation_value
    objective_coefficients[STARTING_XI_OFFSET:STARTING_XI_OFFSET + num_players] = \
        -(adjusted_points - bench_weight * adjusted_rotation_value)
    objective_coefficients[CAPTAIN_OFFSET:CAPTAIN_OFFSET + num_players] = -adjusted_points
    objective_coefficients[EXTRA_TRANSFERS_INDEX] = HIT_COST

    solver_result = milp(c=objective_coefficients, constraints=constraints,
                          integrality=is_integer_variable, bounds=bounds)
    if not solver_result.success:
        raise RuntimeError(f"MILP failed to solve the problem: {solver_result.message}")

    squad_flags = solver_result.x[SQUAD_OFFSET:SQUAD_OFFSET + num_players]
    starting_xi_flags = solver_result.x[STARTING_XI_OFFSET:STARTING_XI_OFFSET + num_players]
    captain_flags = solver_result.x[CAPTAIN_OFFSET:CAPTAIN_OFFSET + num_players]
    extra_transfers_count = solver_result.x[EXTRA_TRANSFERS_INDEX]

    squad_ids = [pool[i]["id"] for i in range(num_players) if squad_flags[i] > 0.5]
    xi_ids = [pool[i]["id"] for i in range(num_players) if starting_xi_flags[i] > 0.5]
    captain_id = next(pool[i]["id"] for i in range(num_players) if captain_flags[i] > 0.5)

    transfers_in = sorted(set(squad_ids) - current_squad_ids)
    transfers_out = sorted(current_squad_ids - set(squad_ids))
    transfers_made = len(transfers_out)
    hit_taken = 0 if unlimited_transfers else max(0, transfers_made - free_transfers)
    hit_cost = HIT_COST * hit_taken

    pool_by_id = {player["id"]: player for player in pool}
    bench_ids = [i for i in squad_ids if i not in xi_ids]
    bench_points = round(sum(pool_by_id[i]["xp"] for i in bench_ids), 2)
    expected_points = float(
        sum(pool_by_id[i]["xp"] for i in xi_ids)
        + pool_by_id[captain_id]["xp"]  # captain bonus
        - hit_cost
    )
    squad_value = round(sum(pool_by_id[i]["price"] for i in squad_ids), 1)
    bank_remaining = round(budget - squad_value, 1)  # what's left of the budget after this squad

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
        "expected_points": round(expected_points, 2),  # nominal xP, NOT risk-adjusted, EXCLUDES bench
        "bench_points": bench_points,  # sum of xp for the 4 bench players (what Bench Boost would add)
        "squad_value": squad_value,  # total price of the 15-man squad
        "bank_remaining": bank_remaining,  # budget left over after buying this squad
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
