"""Shared builders for small, hand-checkable test squads."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def P(pid, pos, team, price, series):
    series = series if isinstance(series, list) else [series]
    return {"id": pid, "name": f"{pos}{pid}", "team": team, "position": pos, "price": price,
            "xp": series[0], "xp_series": series, "xp_horizon_sum": sum(series)}


def base_squad(weeks=1, fwd_xp=5.0):
    w = lambda v: [v] * weeks
    pool = [P(1, "GK", "T1", 4.5, w(3.0)), P(2, "GK", "T2", 4.0, w(2.0))]
    pool += [P(10 + i, "DEF", f"T{3+i}", 5.0, w(3.0)) for i in range(5)]
    pool += [P(20 + i, "MID", f"T{8+i}", 6.0, w(4.0)) for i in range(5)]
    pool += [P(30 + i, "FWD", f"T{13+i}", 7.0, w(fwd_xp)) for i in range(3)]
    return pool, [p["id"] for p in pool]
