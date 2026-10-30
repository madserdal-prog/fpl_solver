from helpers import P, base_squad
import solver_general as solver


def test_starters_valued_over_whole_horizon():
    # Bug: starters used to be valued on THIS week only -> bought a 10-pt player over a 42-pt one.
    pool, squad = base_squad(weeks=5)
    pool += [P(97, "MID", "T17", 6.0, [6, 9, 9, 9, 9]), P(98, "MID", "T16", 6.0, [6.1, 1, 1, 1, 1])]
    s = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1,
                     locked_ids=[i for i in squad if i != 24])
    assert s["transfers_in"] == ["MID97"]


def test_hit_needs_clear_margin_and_reports_true_cost():
    pool, squad = base_squad(fwd_xp=10.0)
    locked = [i for i in squad if i != 24]
    marginal = solver.solve(pool + [P(99, "MID", "T16", 6.0, 9.0)], current_squad_ids=squad,
                            budget=100, free_transfers=0, locked_ids=locked)
    clear = solver.solve(pool + [P(99, "MID", "T16", 6.0, 20.0)], current_squad_ids=squad,
                         budget=100, free_transfers=0, locked_ids=locked)
    assert "MID99" not in marginal["transfers_in"]         # +5 < 4 + safety margin
    assert "MID99" in clear["transfers_in"] and clear["hit_cost"] == 4


def test_blocked_captain_respected():
    pool, squad = base_squad(fwd_xp=10.0)
    s = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1, blocked_captain_ids={30, 31, 32})
    assert s["captain_id"] not in {30, 31, 32}


def test_risk_penalty_is_proportional_never_below_injured_player():
    # Bug: flat -8 put an AVAILABLE striker (Isak) below an injured player on the bench.
    pool, squad = base_squad()
    pool[-1]["xp"] = pool[-1]["xp_series"][0] = 6.5          # "Isak", FWD32
    pool[2]["xp"] = pool[2]["xp_series"][0] = 0.0            # injured DEF10
    s_med = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1,
                         locked_ids=squad, risk_penalty={32: 0.3})
    s_high = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1,
                          locked_ids=squad, risk_penalty={32: 0.7})
    assert 32 in s_med["starting_xi_ids"]
    assert s_high["bench_ids"].index(32) < s_high["bench_ids"].index(10)


def test_free_transfers_bank_and_cap_at_five():
    pool, squad = base_squad(weeks=5)
    roll = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1)
    capped = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=5)
    assert [w["free_transfers_available"] for w in roll["future_plan"]] == [2, 3, 4, 5]
    assert [w["free_transfers_available"] for w in capped["future_plan"]] == [5, 5, 5, 5]


def test_planner_waits_for_returning_player():
    pool, squad = base_squad(weeks=5)
    pool.append(P(99, "MID", "T16", 6.0, [0, 0, 9, 9, 9]))
    s = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1)
    assert s["transfers_in"] == []
    assert any(w["week_offset"] == 2 and w["transfers"] and w["transfers"][0][0] == "MID99" for w in s["future_plan"])


def test_hit_resets_next_week_to_one_free_transfer():
    pool, squad = base_squad(weeks=3, fwd_xp=10.0)
    pool += [P(98, "MID", "T16", 6.0, [20, 20, 20]), P(99, "MID", "T17", 6.0, [20, 20, 20])]
    s = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1)
    assert s["hit_taken"] == 1
    assert s["future_plan"][0]["free_transfers_available"] == 1


def test_runner_up_differs_from_best():
    pool, squad = base_squad(weeks=3)
    pool += [P(98, "MID", "T16", 6.0, [7, 7, 7]), P(99, "MID", "T17", 6.0, [6.9, 6.9, 6.9])]
    best = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1)
    runner = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1,
                          forbid_week0_squads=[best["squad_ids"]])
    assert set(runner["squad_ids"]) != set(best["squad_ids"])
    assert runner["objective_value"] <= best["objective_value"]


def test_transfers_paired_like_for_like():
    pool, squad = base_squad(weeks=1)
    pool += [P(98, "DEF", "T16", 5.0, 9.0), P(99, "FWD", "T17", 7.0, 15.0)]
    s = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=2)
    pos = {p["name"]: p["position"] for p in pool}
    assert all(pos[i] == pos[o] for i, o in zip(s["transfers_in"], s["transfers_out"]))


def test_bench_ordered_gk_first():
    pool, squad = base_squad()
    s = solver.solve(pool, current_squad_ids=squad, budget=100, free_transfers=1, locked_ids=squad)
    assert s["bench"][0].startswith("GK")
