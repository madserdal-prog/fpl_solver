"""
"My Team" -- my_team.json (fpl-auto-pipeline-guide.md, Part 8)
================================================================
Handles the only manual input data in the pipeline beyond the transfers
themselves: budget, free transfers, available chips, and locked players.

This file:
  - defines and validates the my_team.json schema
  - converts a loaded my_team.json to exactly the parameters
    solver_general.solve() expects (current_squad_ids, budget,
    free_transfers, locked_ids, unlimited_transfers)
  - can generate a valid example my_team.json from a player pool, for
    testing (in practice you write this file yourself, or fetch it
    automatically from the entry/{team_id}/ endpoint -- see note at the bottom)
"""

import json

VALID_CHIPS = {"wildcard", "free_hit", "bench_boost", "triple_captain"}
REQUIRED_SQUAD_SIZE = 15
MAX_FREE_TRANSFERS = 5  # current rollover rule -- check against the season's actual rules


class MyTeamValidationError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_my_team(my_team: dict):
    """Raises MyTeamValidationError with a clear explanation on invalid data."""
    required_fields = ["current_squad", "bank", "free_transfers", "chips_available", "locked_players"]
    for field in required_fields:
        if field not in my_team:
            raise MyTeamValidationError(f"my_team.json is missing the field '{field}'")

    squad = my_team["current_squad"]
    if len(squad) != REQUIRED_SQUAD_SIZE:
        raise MyTeamValidationError(
            f"current_squad has {len(squad)} players, expected {REQUIRED_SQUAD_SIZE}")
    if len(set(squad)) != len(squad):
        raise MyTeamValidationError("current_squad contains duplicate player IDs")

    if my_team["bank"] < 0:
        raise MyTeamValidationError("bank cannot be negative")

    ft = my_team["free_transfers"]
    if not (0 <= ft <= MAX_FREE_TRANSFERS):
        raise MyTeamValidationError(
            f"free_transfers={ft} is outside the valid range 0-{MAX_FREE_TRANSFERS} "
            f"(check whether the season's rollover rules have changed)")

    invalid_chips = set(my_team["chips_available"]) - VALID_CHIPS
    if invalid_chips:
        raise MyTeamValidationError(f"Unknown chips: {invalid_chips} (valid: {VALID_CHIPS})")

    locked_not_in_squad = set(my_team["locked_players"]) - set(squad)
    if locked_not_in_squad:
        raise MyTeamValidationError(
            f"locked_players contains IDs not present in current_squad: {locked_not_in_squad}")

    # purchase_prices is optional (older files won't have it) -- but if present,
    # every key must be a string of a current_squad id (JSON object keys are
    # always strings, even though these represent integer player IDs).
    if "purchase_prices" in my_team:
        squad_str = {str(i) for i in squad}
        bad_keys = set(my_team["purchase_prices"].keys()) - squad_str
        if bad_keys:
            raise MyTeamValidationError(
                f"purchase_prices contains IDs not present in current_squad: {bad_keys}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_my_team(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        my_team = json.load(f)
    validate_my_team(my_team)
    return my_team


# ---------------------------------------------------------------------------
# Sell-price rule
# ---------------------------------------------------------------------------

def compute_sell_value(purchase_price: float, current_price: float) -> float:
    """
    FPL's real rule: you get your purchase price back, plus 50% of any
    profit, ROUNDED DOWN to the nearest 0.1m. If the price has fallen
    instead, you take the full loss (sell at the current, lower price).

    Prices are worked in tenths internally to avoid floating-point rounding
    surprises (e.g. 0.1 + 0.2 != 0.3 in raw floats).
    """
    purchase_tenths = round(purchase_price * 10)
    current_tenths = round(current_price * 10)

    if current_tenths <= purchase_tenths:
        return current_tenths / 10.0  # full loss passed through, no rounding needed

    profit_tenths = current_tenths - purchase_tenths
    half_profit_tenths = profit_tenths // 2  # integer division = rounds down, matches FPL's rule
    return (purchase_tenths + half_profit_tenths) / 10.0


def compute_squad_sell_values(my_team: dict, pool: list) -> dict:
    """
    Returns {player_id: sell_value} for every player in current_squad,
    using purchase_prices if tracked, falling back to the pool's current
    price (i.e. assume no profit/loss) for anyone not yet tracked -- this
    keeps older my_team.json files (without purchase_prices) working
    exactly as before, just without the sell-tax correction for them.
    """
    pool_by_id = {p["id"]: p for p in pool}
    purchase_prices = my_team.get("purchase_prices", {})

    sell_values = {}
    for player_id in my_team["current_squad"]:
        current_price = pool_by_id[player_id]["price"] if player_id in pool_by_id else None
        if current_price is None:
            continue
        purchase_price = purchase_prices.get(str(player_id))
        if purchase_price is None:
            sell_values[player_id] = current_price  # not tracked yet -- assume no change
        else:
            sell_values[player_id] = compute_sell_value(purchase_price, current_price)
    return sell_values


def update_purchase_prices(my_team: dict, pool: list) -> dict:
    """
    Call this once per refresh (e.g. right after fetch_my_team.py updates
    current_squad) to keep purchase_prices in sync: carries forward the
    tracked price for anyone still owned, and records today's current
    price as the starting estimate for anyone new (a real transaction
    price isn't available without logging into your FPL account, which
    this pipeline deliberately never does -- see my_team.py's module
    docstring). Returns the updated purchase_prices dict; does NOT mutate
    my_team in place.
    """
    pool_by_id = {p["id"]: p for p in pool}
    old_prices = my_team.get("purchase_prices", {})
    new_prices = {}
    for player_id in my_team["current_squad"]:
        key = str(player_id)
        if key in old_prices:
            new_prices[key] = old_prices[key]  # still owned -- keep the original tracked price
        elif player_id in pool_by_id:
            new_prices[key] = pool_by_id[player_id]["price"]  # newly owned -- best available estimate
    return new_prices


# ---------------------------------------------------------------------------
# Conversion to solver parameters
# ---------------------------------------------------------------------------

def to_solver_kwargs(my_team: dict, active_chip: str = None, base_budget: float = 100.0,
                      pool: list = None) -> dict:
    """
    Translates my_team.json + an optional chip-for-this-round into exactly
    the arguments solver_general.solve() expects.

    active_chip: "wildcard" or "free_hit" -> unlimited transfers this round.
                 "bench_boost"/"triple_captain" don't affect squad selection
                 itself (only scoring afterward), so the solver doesn't
                 handle them -- they're just passed through in the output
                 for traceability.

    pool: if given, current_squad players' contribution to the budget
    constraint uses their real FPL sell value (purchase price + 50% of
    profit, rounded down) instead of their current market price -- see
    compute_squad_sell_values(). Without it, falls back to the old
    behavior (current price for everyone), same as before this existed.
    """
    if active_chip is not None and active_chip not in VALID_CHIPS:
        raise MyTeamValidationError(f"Unknown chip: {active_chip}")
    if active_chip is not None and active_chip not in my_team["chips_available"]:
        raise MyTeamValidationError(
            f"Chip '{active_chip}' is not available (already used, or misspelled)")

    unlimited = active_chip in ("wildcard", "free_hit")

    kwargs = {
        "current_squad_ids": my_team["current_squad"],
        "budget": round(base_budget + my_team["bank"], 1),
        "free_transfers": my_team["free_transfers"],
        "locked_ids": my_team["locked_players"],
        "unlimited_transfers": unlimited,
    }
    if pool is not None:
        kwargs["sell_values"] = compute_squad_sell_values(my_team, pool)
    return kwargs


# ---------------------------------------------------------------------------
# Example data / self-test
# ---------------------------------------------------------------------------

def make_example_my_team(master_path: str, out_path: str, seed: int = 1, bank: float = 0.3,
                          free_transfers: int = 1, chips_available=None, locked_players=None):
    """
    Generates a VALID example my_team.json from a master_data.json (from
    forecast.py), for use in testing. In practice you write this file
    yourself manually (the only manual input in the pipeline), or connect
    it to your entry/{team_id}/ endpoint if you want to fetch your team
    automatically -- that requires login and should be treated as a
    separate step, since it involves your FPL account.
    """
    import numpy as np
    with open(master_path, encoding="utf-8") as f:
        master = json.load(f)

    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for e in master["elements"]:
        by_pos[e["element_type"]].append(e["id"])

    rng = np.random.default_rng(seed)
    req = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    squad = []
    for pos, n in req.items():
        squad += list(rng.choice(by_pos[pos], size=n, replace=False))
    squad = [int(x) for x in squad]

    my_team = {
        "current_squad": squad,
        "bank": bank,
        "free_transfers": free_transfers,
        "chips_available": chips_available or ["wildcard", "bench_boost", "free_hit", "triple_captain"],
        "locked_players": locked_players or [],
        "notes": "Synthetic example generated by make_example_my_team() -- replace with your actual team.",
    }
    validate_my_team(my_team)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(my_team, f, ensure_ascii=False, indent=2)
    return my_team


if __name__ == "__main__":
    # Requires forecast.py to have been run first (produces master_data.json)
    my_team = make_example_my_team("master_data.json", "my_team.json")
    print("Example my_team.json generated and validated:")
    print(json.dumps(my_team, indent=2, ensure_ascii=False))

    print("\nSolver parameters without a chip:")
    print(json.dumps(to_solver_kwargs(my_team), indent=2, ensure_ascii=False))

    print("\nSolver parameters with wildcard active:")
    print(json.dumps(to_solver_kwargs(my_team, active_chip="wildcard"), indent=2, ensure_ascii=False))
