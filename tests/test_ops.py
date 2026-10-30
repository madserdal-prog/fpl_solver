from datetime import datetime, timezone
import should_run

EVENTS = [{"id": 9, "deadline_time": "2026-10-31T11:00:00Z"}]
at = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_gate_daily_predeadline_and_manual():
    assert should_run.should_run(EVENTS, at("2026-10-30T09:05:00"), "schedule")[0]
    assert should_run.should_run(EVENTS, at("2026-10-31T08:05:00"), "schedule")[0]
    assert not should_run.should_run(EVENTS, at("2026-10-30T15:05:00"), "schedule")[0]
    assert should_run.should_run(EVENTS, at("2026-10-30T15:05:00"), "workflow_dispatch")[0]
