"""
Informational-only checks -- price momentum and Bench Boost alternative
==========================================================================
Both functions here are DELIBERATELY informational, not decision-driving:
they never change what the solver picks, only add context to the report.
This matches FPL community consensus (see the design discussion): team
value should never be chased at the cost of points, and price-timing
advice is only useful for WHEN to make a transfer you've already decided
on -- never for WHICH player to pick.
"""

# Rough thresholds for flagging "likely rising/falling soon" from net
# transfer momentum. FPL's real price-change algorithm isn't fully public
# even among fan sites -- this is a coarse approximation using the one
# genuinely free, real-time signal available (transfers_in_event /
# transfers_out_event), NOT a precise reproduction of FPL's hidden formula.
RISE_THRESHOLD_PCT = 0.06   # net transfers in, as % of all managers
FALL_THRESHOLD_PCT = 0.06   # net transfers out, as % of all managers


def price_momentum_note(player_name: str, element: dict, total_managers: int) -> str:
    """
    Returns a short informational string if a player looks likely to rise
    or fall in price soon, based on this gameweek's net transfer momentum.
    Returns None if there's nothing notable or total_managers is unknown.
    """
    if not total_managers:
        return None

    net = element.get("transfers_in_event", 0) - element.get("transfers_out_event", 0)
    pct = 100.0 * net / total_managers

    if pct >= RISE_THRESHOLD_PCT:
        return f"{player_name} looks likely to RISE in price soon (net +{pct:.2f}% of managers this GW)"
    if pct <= -FALL_THRESHOLD_PCT:
        return f"{player_name} looks likely to FALL in price soon (net {pct:.2f}% of managers this GW)"
    return None


def collect_price_momentum_notes(transfers_in: list, transfers_out: list, master_data: dict) -> list:
    """
    Checks ONLY the players the solver already decided to buy/sell --
    never used to pick different players, only to note timing on the
    solver's own choices. transfers_in: buy soon before a rise.
    transfers_out: sell soon before a fall.
    """
    elements_by_name = {e["web_name"]: e for e in master_data["elements"]}
    total_managers = master_data.get("total_managers", 0)
    notes = []

    for name in transfers_in:
        element = elements_by_name.get(name)
        if element:
            note = price_momentum_note(name, element, total_managers)
            if note and "RISE" in note:
                notes.append(note)

    for name in transfers_out:
        element = elements_by_name.get(name)
        if element:
            note = price_momentum_note(name, element, total_managers)
            if note and "FALL" in note:
                notes.append(note)

    return notes


# ---------------------------------------------------------------------------
# Bench Boost alternative
# ---------------------------------------------------------------------------

# Feature flag: set to False to disable the full re-optimized Bench Boost
# alternative (the expensive check) while keeping everything else running.
# Easy single-line toggle, per your request.
RUN_BENCH_BOOST_ALTERNATIVE = True


def bench_boost_summary(normal_solution: dict, bb_solution: dict) -> str:
    """
    Compares the NORMAL solution's bench (as picked, unweighted) against a
    full re-optimization done with bench_weight=1.0 (bb_solution). Returns
    a short comparison string for the report.
    """
    normal_total_if_boosted = normal_solution["expected_points"] + normal_solution["bench_points"]
    bb_total = bb_solution["expected_points"] + bb_solution["bench_points"]
    gain = round(bb_total - normal_total_if_boosted, 2)

    return (
        f"Boosting THIS week's normal bench: +{normal_solution['bench_points']} pts "
        f"(total {round(normal_total_if_boosted, 2)}). "
        f"Restructuring specifically for Bench Boost instead: {round(bb_total, 2)} pts "
        f"({'+' if gain >= 0 else ''}{gain} vs. just boosting the normal squad)."
    )
