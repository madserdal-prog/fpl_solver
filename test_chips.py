from helpers import P
import chips, solver_general as solver


def _scenario():
    pool = [P(1, "GK", "T1", 4.5, [4, 4, 4, 4, 0]), P(2, "GK", "T2", 4.0, [1, 1, 6, 1, 1])]
    pool += [P(10 + i, "DEF", f"T{3+i}", 5.0, [4, 4, 4, 4, 0]) for i in range(3)]
    pool += [P(13, "DEF", "T6", 4.0, [1, 1, 6, 1, 1]), P(14, "DEF", "T7", 4.0, [1, 1, 6, 1, 1])]
    pool += [P(20 + i, "MID", f"T{8+i}", 6.0, [5, 5, 5, 5, 0]) for i in range(4)] + [P(24, "MID", "T12", 4.5, [1, 1, 6, 1, 1])]
    pool += [P(30, "FWD", "T13", 9.0, [8, 8, 8, 16, 0]), P(31, "FWD", "T14", 7.0, [5, 5, 5, 5, 0]),
             P(32, "FWD", "T15", 7.0, [5, 5, 5, 5, 0])]
    squad = [p["id"] for p in pool]
    pool += [P(100 + i, "MID", f"T{16+i%4}", 6.0, [4, 4, 4, 4, 6]) for i in range(8)]
    pool += [P(120 + i, "DEF", f"T{16+i%4}", 5.0, [3, 3, 3, 3, 5]) for i in range(6)]
    pool += [P(140 + i, "FWD", f"T{16+i%4}", 7.0, [4, 4, 4, 4, 6]) for i in range(4)] + [P(160, "GK", "T19", 4.5, [3, 3, 3, 3, 5])]
    return pool, squad


def test_chips_go_to_their_best_weeks():
    pool, squad = _scenario()
    kw = {"current_squad_ids": squad, "budget": 100.0, "free_transfers": 1, "sell_values": {}}
    main = solver.solve(chips.pool_for_horizon(pool, 5), **kw)
    plan = chips.plan_chips(pool, main, list(chips.CHIP_NAMES), 14, kw)
    assert plan["assignment"]["free_hit"] == 18          # blank week
    assert plan["assignment"]["triple_captain"] == 17    # captain double gameweek
    assert plan["assignment"]["bench_boost"] == 16       # strong-bench week
    assert plan["play_now"] is None                      # wildcard gain below the bar


def test_deadline_forces_chips_when_weeks_run_out():
    pool, squad = _scenario()
    kw = {"current_squad_ids": squad, "budget": 100.0, "free_transfers": 1, "sell_values": {}}
    main = solver.solve(chips.pool_for_horizon(pool, 5), **kw)
    plan = chips.plan_chips(pool, main, list(chips.CHIP_NAMES), 15, kw)   # 4 weeks left, 4 chips
    assert len(plan["assignment"]) == 4 and plan["play_now"] is not None


def test_best_lineup_respects_formation():
    pos = {i: p for i, p in zip(range(15), ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)}
    xp = {i: (10 if pos[i] == "FWD" else 1) for i in range(15)}
    xi, cap, bench, *_ = chips.best_lineup(list(range(15)), xp, pos)
    assert len(xi) == 11 and sum(pos[i] == "GK" for i in xi) == 1 and sum(pos[i] == "DEF" for i in xi) >= 3
