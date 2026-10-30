"""
should_run.py -- deadline-aware gate for the hourly scheduled workflow
=======================================================================
The workflow ticks every hour; this decides whether a tick does real work:
  - the daily run (DAILY_RUN_HOUR_UTC), and
  - a pre-deadline run when the next deadline is PRE_DEADLINE_HOURS away
    (+/- half an hour), to catch press-conference team news.
Manual runs (workflow_dispatch) always run.

Prints "true"/"false", and writes run=true/false to $GITHUB_OUTPUT when set.
"""

import json
import os
import sys
from datetime import datetime, timezone

DAILY_RUN_HOUR_UTC = 9
PRE_DEADLINE_HOURS = 3.0


def next_deadline(events, now):
    upcoming = [datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
                for e in events if e.get("deadline_time")]
    upcoming = [d for d in upcoming if d > now]
    return min(upcoming) if upcoming else None


def should_run(events, now, event_name):
    if event_name != "schedule":
        return True, "manual run"
    if now.hour == DAILY_RUN_HOUR_UTC:
        return True, "daily run"
    deadline = next_deadline(events, now)
    if deadline is not None:
        hours_left = (deadline - now).total_seconds() / 3600
        if PRE_DEADLINE_HOURS - 0.5 <= hours_left < PRE_DEADLINE_HOURS + 0.5:
            return True, f"pre-deadline run ({hours_left:.1f}h before the deadline)"
    return False, "not a scheduled slot"


def main():
    event_name = os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch")
    try:
        with open("events.json", encoding="utf-8") as f:
            events = json.load(f)
    except (OSError, ValueError):
        events = []
    run, reason = should_run(events, datetime.now(timezone.utc), event_name)
    print(f"{'true' if run else 'false'} ({reason})", file=sys.stderr)
    print("true" if run else "false")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"run={'true' if run else 'false'}\n")


if __name__ == "__main__":
    main()
