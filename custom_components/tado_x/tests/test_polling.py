"""Tests for the poll planner. No Home Assistant needed."""
import importlib.util
import pathlib
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "tado_x_polling", pathlib.Path(__file__).parent.parent / "polling.py"
)
polling = importlib.util.module_from_spec(_spec)
sys.modules["tado_x_polling"] = polling
_spec.loader.exec_module(polling)

from tado_x_polling import (  # noqa: E402
    BACKOFF_STEPS,
    DAY,
    MIN_SPACING,
    PollPlanner,
    PollReason,
    QuotaLevel,
    base_interval,
)

MIN = 60.0
T0 = 1_000_000.0


def planner(temp=20.0, setpoint=21.0, off=False, rooms=(1,), remaining=1000, reset_in=DAY):
    p = PollPlanner()
    p.update_quota(remaining, 1000, T0 + reset_in)
    p.set_rooms(set(rooms), T0)
    for room in rooms:
        p.update_reading(room, temp, setpoint, off, T0)
    p.poll_succeeded(T0, True)
    return p


@pytest.mark.parametrize(
    ("temp", "setpoint", "off", "expected", "reason"),
    [
        (21.6, 21.0, False, 30, PollReason.ABOVE_SETPOINT),
        (21.5, 21.0, False, 3, PollReason.NEAR_SETPOINT),
        (21.0, 21.0, False, 3, PollReason.NEAR_SETPOINT),
        (20.9, 21.0, False, 5, PollReason.BELOW_SETPOINT),
        (15.0, 21.0, True, 30, PollReason.OFF),
        (None, 21.0, False, 30, PollReason.OFF),
        (20.0, None, False, 30, PollReason.OFF),
    ],
)
def test_base_table(temp, setpoint, off, expected, reason):
    assert base_interval(temp, setpoint, off) == (expected * MIN, reason)


def test_float_noise_at_band_edge():
    assert base_interval(21.5000000001, 21.0, False)[0] == 3 * MIN


def run_due(p, now):
    """Poll at the planned moment and return it."""
    at = p.plan_next(now).at
    p.poll_succeeded(at, False)
    return at


def test_temperature_change_polls_then_backs_off_then_base():
    p = planner(temp=20.0)
    now = T0 + 10 * MIN
    p.last_poll = now - 5 * MIN
    assert p.update_reading(1, 20.1, 21.0, False, now) is PollReason.TEMPERATURE_CHANGE
    assert p.plan_next(now).at == now

    times = [run_due(p, now)]
    for _ in BACKOFF_STEPS:
        assert p.plan_next(times[-1]).reason is PollReason.BACKOFF
        times.append(run_due(p, times[-1]))
    assert [t2 - t1 for t1, t2 in zip(times, times[1:])] == list(BACKOFF_STEPS)
    plan = p.plan_next(times[-1])
    assert plan.reason is PollReason.BELOW_SETPOINT
    assert plan.at - times[-1] == 5 * MIN


def test_another_change_resets_backoff():
    p = planner(temp=20.0)
    now = T0 + 10 * MIN
    p.update_reading(1, 20.1, 21.0, False, now)
    now = run_due(p, now)
    now = run_due(p, now)
    now = run_due(p, now)
    assert p.rooms[1].backoff_step == 3
    now += 30
    p.update_reading(1, 20.2, 21.0, False, now)
    assert p.rooms[1].backoff_step == 0
    at = run_due(p, now)
    assert p.plan_next(at).at - at == BACKOFF_STEPS[0]


def test_change_under_a_tenth_and_a_bounce_are_ignored():
    p = planner(temp=20.0)
    now = T0 + 20 * MIN
    assert p.update_reading(1, 20.04, 21.0, False, now) is None  # rounds to 20.0
    assert p.update_reading(1, 20.1, 21.0, False, now) is PollReason.TEMPERATURE_CHANGE
    assert p.update_reading(1, 20.0, 21.0, False, now + 60) is None  # flicks back
    assert p.update_reading(1, 20.1, 21.0, False, now + 120) is None  # and forth
    # A real fall past the old value counts, and so does a late return.
    assert p.update_reading(1, 19.9, 21.0, False, now + 180) is PollReason.TEMPERATURE_CHANGE
    assert p.update_reading(1, 20.1, 21.0, False, now + 3000) is PollReason.TEMPERATURE_CHANGE


def test_setpoint_and_mode_changes_are_urgent_but_debounced():
    p = planner(temp=20.0)
    p.last_poll = now = T0 + 20 * MIN
    assert p.update_reading(1, 20.0, 22.0, False, now + 1) is PollReason.SETPOINT_CHANGE
    plan = p.plan_next(now + 1)
    assert plan.reason is PollReason.SETPOINT_CHANGE
    assert plan.at == now + 15  # the urgent spacing beats the 5 s debounce here
    p.poll_succeeded(plan.at, False)
    assert p.update_reading(1, 20.0, 22.0, True, plan.at + 100) is PollReason.SETPOINT_CHANGE


def test_minimum_spacing_holds_back_a_temperature_trigger():
    p = planner(temp=20.0)
    p.last_poll = T0 + 100
    p.update_reading(1, 20.1, 21.0, False, T0 + 110)
    assert p.plan_next(T0 + 110).at == T0 + 100 + MIN_SPACING


def test_earliest_room_wins_and_others_count_the_poll():
    p = planner(temp=20.0, setpoint=20.0, rooms=(1, 2))  # both near: 3 min
    p.update_reading(2, 25.0, 20.0, False, T0 + 1)
    # Room 2 is now above setpoint and backing off; room 1 is on its base.
    now = T0 + MIN
    p.poll_succeeded(now, False)
    assert p.rooms[2].next_due == now + BACKOFF_STEPS[0]
    assert p.rooms[1].next_due == now + 3 * MIN
    assert p.plan_next(now).room_id == 2


def test_setpoint_change_beats_a_due_room_held_by_spacing():
    p = planner(temp=20.0, rooms=(1, 2))
    p.last_poll = now = T0 + 5 * MIN
    p.rooms[1].next_due = now  # due, but must wait out the 60 s spacing
    p.update_reading(2, 20.0, 23.0, False, now)
    assert p.plan_next(now).at == now + 15


@pytest.mark.parametrize(
    ("remaining", "hours_left", "level"),
    [
        (1000, 24, QuotaLevel.NORMAL),
        (400, 12, QuotaLevel.NORMAL),
        (200, 12, QuotaLevel.LOW),
        (100, 12, QuotaLevel.CRITICAL),
        (5, 12, QuotaLevel.EXHAUSTED),
    ],
)
def test_quota_levels(remaining, hours_left, level):
    p = planner(remaining=remaining, reset_in=hours_left * 3600)
    assert p.quota_level(T0) is level


def test_low_quota_keeps_triggers_for_rooms_that_could_be_heating():
    below = planner(temp=20.0, remaining=200, reset_in=12 * 3600)
    below.last_poll = T0
    assert below.update_reading(1, 20.1, 21.0, False, T0 + 700) is PollReason.TEMPERATURE_CHANGE

    above = planner(temp=25.0, remaining=200, reset_in=12 * 3600)
    above.last_poll = T0
    assert above.update_reading(1, 25.1, 21.0, False, T0 + 700) is None


def test_critical_quota_drops_temperature_triggers_but_not_setpoints():
    p = planner(temp=20.0, remaining=100, reset_in=12 * 3600)
    assert p.update_reading(1, 20.1, 21.0, False, T0 + 700) is None
    assert p.update_reading(1, 20.1, 22.0, False, T0 + 800) is PollReason.SETPOINT_CHANGE


def test_pressure_stretches_idle_rooms_more_than_heating_rooms():
    heating = planner(temp=20.0, remaining=100, reset_in=12 * 3600)
    idle = planner(temp=25.0, remaining=100, reset_in=12 * 3600)
    assert idle.plan_next(T0).at - T0 == 120 * MIN  # 30 min x4
    heat_gap = heating.plan_next(T0).at - T0
    assert 7 * MIN < heat_gap < 10 * MIN  # 12 h over 95 calls, not the 80 for idle use


def test_no_polling_when_exhausted_except_a_probe():
    p = planner(remaining=3, reset_in=6 * 3600)
    plan = p.plan_next(T0)
    assert plan.level is QuotaLevel.EXHAUSTED
    assert plan.at == T0 + 3600


def test_rate_limit_waits_then_clears():
    p = planner(reset_in=20 * MIN)
    p.poll_rate_limited(T0 + 1)
    plan = p.plan_next(T0 + 1)
    assert plan.reason is PollReason.RATE_LIMITED
    assert plan.at == T0 + 20 * MIN
    p.poll_succeeded(plan.at, False)
    assert p.rate_limited_until is None


def test_failures_retry_with_growing_gaps():
    p = planner()
    now = T0
    gaps = []
    for _ in range(6):
        p.poll_failed(now)
        at = p.plan_next(now).at
        gaps.append(at - now)
        now = at
    assert gaps == [1 * MIN, 2 * MIN, 4 * MIN, 8 * MIN, 8 * MIN, 8 * MIN]
    p.poll_succeeded(now, False)
    assert p.failures == 0


def test_full_refresh_cadence():
    p = planner()
    assert not p.needs_full_refresh(T0 + 59 * MIN)
    assert p.needs_full_refresh(T0 + 60 * MIN)
    low = planner(remaining=200, reset_in=12 * 3600)
    assert not low.needs_full_refresh(T0 + 90 * MIN)
    assert low.needs_full_refresh(T0 + 180 * MIN)
    critical = planner(remaining=100, reset_in=12 * 3600)
    assert not critical.needs_full_refresh(T0 + 200 * MIN)  # still critical then
    critical.request_command_poll(T0)
    assert critical.needs_full_refresh(T0)


def test_stale_reset_date_rolls_forward():
    p = planner(reset_in=-3600)
    assert p.sustainable_spacing(T0) == pytest.approx((DAY - 3600) / 980)


@pytest.mark.parametrize("step", [30, 60, 120, 400])
def test_worst_case_never_runs_out(step):
    """Readings changing non-stop, for a day, can't use up the quota."""
    p = planner(temp=20.0, remaining=1000)
    remaining, now, temp, next_change = 1000, T0, 20.0, T0
    polls = 0
    while now < T0 + DAY - 1:
        plan = p.plan_next(now)
        if next_change < plan.at:
            now = next_change
            temp = round(temp + 0.1, 1) if polls % 2 else round(temp + 0.2, 1)
            if temp > 21.4:
                temp = 19.0
            p.update_reading(1, temp, 21.0, False, now)
            next_change = now + step
            continue
        now = plan.at
        if now >= T0 + DAY:
            break
        remaining -= 1
        polls += 1
        p.update_quota(remaining, 1000, T0 + DAY)
        p.poll_succeeded(now, False)
    assert remaining >= 1
