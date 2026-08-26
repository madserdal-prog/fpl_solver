"""
run_real.py -- run the pipeline on REAL data (not synthetic demo data)
========================================================================
Assumes you have already run:
  1. python3 collector.py            -> master_data.json, fixtures.json
  2. python3 fetch_my_team.py --team-id X --gw Y   -> my_team.json

Then run:
  python3 run_real.py --gw 4

The run happens in up to three steps:
  1. Raw mathematical solution (same as before)
  2. A critic checks the proposal against injury/news signals
  3. If the critic found something -- re-solve with risk adjustment
     (robust optimization, cf. Ramezani 2025 arXiv:2505.02170)

Step 2 defaults to the FREE critic (critic_free.py) -- zero API cost,
uses FPL's own status/news fields only (see critic_free.py for exactly
what that does and doesn't catch). Pass --paid-critic to use critic.py
instead (Anthropic API + live web search, costs money per run, but catches
more -- e.g. transfer sagas, rotation risk for a fit-but-recently-returned
player). Pass --skip-critic to skip both.
"""

import argparse
import json
import os

import my_team as my_team_module
import solver_general as solver
import forecast
import critic_free
import collector
import checks


def resolve_gw(gw_arg, events_path="events.json"):
    """
    --gw auto (or omitted) -> read events.json (written by collector.py)
    and pick the current/next gameweek automatically. Falls back to
    requiring an explicit --gw if events.json is missing or stale.
    """
    if gw_arg is not None and gw_arg != "auto":
        return int(gw_arg)
    try:
        with open(events_path, encoding="utf-8") as f:
            events = json.load(f)
        current = collector.get_current_gameweek(events)
        print(f"Auto-detected gameweek: GW{current['id']} (deadline {current['deadline_time']})")
        return current["id"]
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(
            f"Could not auto-detect gameweek ({e}). Run collector.py first, or pass --gw N explicitly.")


def main():
    parser = argparse.ArgumentParser(description="Run the FPL solver on real data")
    parser.add_argument("--gw", default="auto",
                         help="Gameweek to optimize for, or 'auto' (default) to detect it "
                              "from events.json (written by collector.py)")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--master", default="master_data.json")
    parser.add_argument("--fixtures", default="fixtures.json")
    parser.add_argument("--my-team", default="my_team.json")
    parser.add_argument("--chip", default=None, choices=[None, "wildcard", "free_hit",
                                                           "bench_boost", "triple_captain"])
    parser.add_argument("--skip-critic", action="store_true",
                         help="Skip the news check entirely (raw mathematical solver only).")
    parser.add_argument("--paid-critic", action="store_true",
                         help="Use critic.py (Anthropic API + live web search) instead of the "
                              "free critic. Costs money per run, requires ANTHROPIC_API_KEY. "
                              "Catches more than the free critic (transfer sagas, rotation risk).")
    parser.add_argument("--initial-team", action="store_true",
                         help="Build the best possible squad from scratch, ignoring my_team.json's "
                              "current_squad and applying NO transfer penalty. Use this for your "
                              "very first GW1 squad, before FPL's first deadline has passed -- real "
                              "FPL doesn't limit changes at that point, so the solver shouldn't "
                              "either. This does NOT consume a wildcard chip; it's a separate, "
                              "correct case (see my_team.json's chips_available, which stays untouched).")
    parser.add_argument("--no-bench-boost-check", action="store_true",
                         help="Skip the full Bench Boost re-optimization check (feature flag -- "
                              "see checks.RUN_BENCH_BOOST_ALTERNATIVE for the same toggle in code).")
    args = parser.parse_args()
    args.gw = resolve_gw(args.gw)

    with open(args.master, encoding="utf-8") as f:
        master = json.load(f)
    with open(args.fixtures, encoding="utf-8") as f:
        fixtures = json.load(f)

    fc = forecast.build_forecast(master, fixtures, start_gw=args.gw, horizon=args.horizon)
    forecast_path = f"forecast_gw{args.gw}.json"
    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"Wrote {forecast_path}")

    pool = solver.load_pool_from_forecast(forecast_path, args.master, horizon_gw=0)

    my_team = my_team_module.load_my_team(args.my_team)
    kwargs = my_team_module.to_solver_kwargs(my_team, active_chip=args.chip)

    if args.initial_team:
        kwargs["unlimited_transfers"] = True
        print("(--initial-team set: building from scratch, current_squad ignored for transfer-cost "
              "purposes, no chip consumed)")

    # --- Step 1: raw mathematical solution (no news awareness yet) ---
    draft_solution = solver.solve(pool, **kwargs)
    out_path = f"solver_solution_gw{args.gw}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft_solution, f, ensure_ascii=False, indent=2)

    print(f"\n=== Draft solver proposal (before news check) ===")
    print(f"Captain: {draft_solution['captain']}  |  Transfers in: {draft_solution['transfers_in']}")

    final_solution = draft_solution
    critic_summary = None

    if args.skip_critic:
        print("\n(--skip-critic set: skipping news check, using the raw proposal directly)")
    else:
        critic_flags_path = f"critic_flags_gw{args.gw}.json"

        if args.paid_critic:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("\nWARNING: --paid-critic requires ANTHROPIC_API_KEY, which is not set.")
                print("Falling back to the free critic instead.")
                critic_free.run_free_critic(out_path, args.master, critic_flags_path)
            else:
                import critic
                critic.run_critic(out_path, args.master, critic_flags_path)
        else:
            critic_free.run_free_critic(out_path, args.master, critic_flags_path)

        with open(critic_flags_path, encoding="utf-8") as f:
            critic_result = json.load(f)

        risk_penalty = {int(k): v for k, v in critic_result["risk_penalty"].items()}
        blocked_captain_ids = set(critic_result["blocked_captain_ids"])
        critic_summary = critic_result["critic_output"].get("summary")

        if risk_penalty or blocked_captain_ids:
            # --- Step 3: re-solve with the critic's risk adjustments ---
            print("\n=== Re-solving with the critic's risk adjustments ===")
            final_solution = solver.solve(pool, risk_penalty=risk_penalty,
                                           blocked_captain_ids=blocked_captain_ids, **kwargs)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_solution, f, ensure_ascii=False, indent=2)
        else:
            print("\nThe critic found no risk flags requiring a re-run.")

    print(f"\n=== Final solution for GW{args.gw}{' (chip: ' + args.chip + ')' if args.chip else ''} ===")
    print(json.dumps(final_solution, indent=2, ensure_ascii=False))
    if critic_summary:
        print(f"\nCritic summary: {critic_summary}")
    if args.initial_team:
        print(f"\n=== Copy these into my_team.json's \"current_squad\" once you've set this team on FPL ===")
        print(json.dumps(final_solution["squad_ids"], ensure_ascii=False))

    # --- Price momentum (informational only -- never changes the pick) ---
    price_notes = checks.collect_price_momentum_notes(
        final_solution["transfers_in"], final_solution["transfers_out"], master)
    if price_notes:
        print("\n=== Price momentum notes (timing only, doesn't affect the recommendation) ===")
        for note in price_notes:
            print(f"  - {note}")
    final_solution["price_momentum_notes"] = price_notes

    # --- Bench Boost alternative (feature-flagged, see checks.py) ---
    bench_boost_note = None
    if checks.RUN_BENCH_BOOST_ALTERNATIVE and not args.no_bench_boost_check \
            and "bench_boost" in my_team.get("chips_available", []):
        print("\n=== Bench Boost alternative check ===")
        bb_solution = solver.solve(pool, bench_weight=1.0, **kwargs)
        bench_boost_note = checks.bench_boost_summary(final_solution, bb_solution)
        print(f"  {bench_boost_note}")
    final_solution["bench_boost_note"] = bench_boost_note

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
