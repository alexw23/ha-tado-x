"""Quota-aware poll planning for Tado X.

This module decides *when* to call the Tado API. It holds no Home Assistant
imports and does no I/O: the coordinator feeds it readings and quota figures and
asks when the next poll is due. All times are POSIX seconds.

One ``GET /rooms`` returns every room, so there is one poll timer per home however
many rooms there are. Each room states when it next needs fresh data, and the
earliest room wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

MINUTE = 60.0

# Base intervals, by where the temperature sits relative to the setpoint.
BASE_ABOVE = 30 * MINUTE  # more than BAND above the setpoint
BASE_NEAR = 3 * MINUTE  # from the setpoint to BAND above it
BASE_BELOW = 5 * MINUTE  # below the setpoint
BASE_OFF = 30 * MINUTE  # off, or no setpoint or temperature to compare
BAND = 0.5

# Steps after a temperature or setpoint change, before returning to the base.
BACKOFF_STEPS = (1 * MINUTE, 2 * MINUTE, 4 * MINUTE, 8 * MINUTE, 12 * MINUTE)

# Readings arrive in 0.1 degree steps and often flick between two neighbours. A
# return to the previous value this soon is treated as noise, not a change.
TEMP_STEP = 0.1
BOUNCE_WINDOW = 10 * MINUTE

# Spacing between polls.
MIN_SPACING = 1 * MINUTE
MIN_SPACING_URGENT = 15.0  # someone just changed a setpoint or a mode
SETPOINT_DEBOUNCE = 5.0  # lets a slider settle and the change reach Tado

# Calls held back for commands, and the smaller part that polls for heating
# state may still use once everything else has stopped.
RESERVE_MIN = 20
RESERVE_FRACTION = 0.02
RESERVE_COMMANDS = 5

# Quota pressure, as the poll spacing that would use up what's left by the reset.
LEVEL_LOW_AT = 150.0
LEVEL_CRITICAL_AT = 360.0

# How much longer a room far from needing heat waits under pressure.
IDLE_STRETCH_LOW = 2
IDLE_STRETCH_CRITICAL = 4

# Full refreshes also read devices, presence and the optional features.
FULL_REFRESH_NORMAL = 60 * MINUTE
FULL_REFRESH_LOW = 180 * MINUTE

# Retry spacing after a failed poll, and the longest wait while rate limited
# (a probe, in case the reset time is only an estimate).
RETRY_STEPS = (1 * MINUTE, 2 * MINUTE, 4 * MINUTE, 8 * MINUTE)
RATE_LIMIT_PROBE = 60 * MINUTE

DUE_TOLERANCE = 5.0
DAY = 86400.0


class QuotaLevel(StrEnum):
    """How much the remaining quota limits polling."""

    NORMAL = "normal"
    LOW = "low"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


class PollReason(StrEnum):
    """Why a poll is scheduled."""

    STARTUP = "startup"
    ABOVE_SETPOINT = "above_setpoint"
    NEAR_SETPOINT = "near_setpoint"
    BELOW_SETPOINT = "below_setpoint"
    OFF = "off"
    TEMPERATURE_CHANGE = "temperature_change"
    SETPOINT_CHANGE = "setpoint_change"
    BACKOFF = "backoff"
    COMMAND = "command"
    RETRY = "retry"
    RATE_LIMITED = "rate_limited"


URGENT_REASONS = frozenset({PollReason.SETPOINT_CHANGE, PollReason.COMMAND})


def base_interval(
    temperature: float | None, setpoint: float | None, is_off: bool
) -> tuple[float, PollReason]:
    """Return the base interval for a room, from the requested table."""
    if is_off or temperature is None or setpoint is None:
        return BASE_OFF, PollReason.OFF
    # Rounded so float noise can't move a reading across a band edge.
    delta = round(temperature - setpoint, 2)
    if delta > BAND:
        return BASE_ABOVE, PollReason.ABOVE_SETPOINT
    if delta >= 0:
        return BASE_NEAR, PollReason.NEAR_SETPOINT
    return BASE_BELOW, PollReason.BELOW_SETPOINT


@dataclass
class RoomPollState:
    """What the planner knows about one room."""

    room_id: int
    temperature: float | None = None
    setpoint: float | None = None
    is_off: bool = False
    last_temperature: float | None = None  # the value before ``temperature``
    temperature_changed_at: float = 0.0
    seeded: bool = False
    backoff_step: int | None = None  # next index into BACKOFF_STEPS
    next_due: float = 0.0
    reason: PollReason = PollReason.STARTUP

    @property
    def heating_relevant(self) -> bool:
        """Return whether this room could plausibly be heating or about to."""
        if self.is_off or self.temperature is None or self.setpoint is None:
            return False
        return round(self.temperature - self.setpoint, 2) <= BAND


@dataclass(frozen=True)
class PollPlan:
    """When the next poll should run, and why."""

    at: float
    interval: float  # seconds from the previous poll, for diagnostics
    reason: PollReason
    room_id: int | None
    level: QuotaLevel


class PollPlanner:
    """Plans polls for one home."""

    def __init__(self) -> None:
        """Start with no rooms and no quota information."""
        self.rooms: dict[int, RoomPollState] = {}
        self.last_poll: float = 0.0
        self.last_full: float = 0.0
        self.failures: int = 0
        self.rate_limited_until: float | None = None
        self.remaining: int | None = None
        self.limit: int | None = None
        self.reset_at: float | None = None
        self._command_pending = False

    # ── Quota ────────────────────────────────────────────────────────────────

    def update_quota(
        self, remaining: int | None, limit: int | None, reset_at: float | None
    ) -> None:
        """Record the latest quota figures."""
        self.remaining = remaining
        self.limit = limit
        self.reset_at = reset_at

    def _seconds_to_reset(self, now: float) -> float:
        """Return seconds to the next reset; a date in the past rolls forward."""
        if self.reset_at is None:
            return DAY
        seconds = self.reset_at - now
        while seconds <= 0:
            seconds += DAY
        return seconds

    def _reserve(self) -> int:
        """Return the calls held back from non-heating polls."""
        return max(RESERVE_MIN, int((self.limit or 0) * RESERVE_FRACTION))

    def sustainable_spacing(self, now: float, reserve: int | None = None) -> float:
        """Return the poll spacing that would use up the quota exactly at reset."""
        if self.remaining is None:
            return 0.0
        held_back = self._reserve() if reserve is None else reserve
        return self._seconds_to_reset(now) / max(1, self.remaining - held_back)

    def quota_level(self, now: float) -> QuotaLevel:
        """Return the current quota pressure."""
        if self.rate_limited_until is not None and now < self.rate_limited_until:
            return QuotaLevel.EXHAUSTED
        if self.remaining is None:
            return QuotaLevel.NORMAL
        if self.remaining <= RESERVE_COMMANDS:
            return QuotaLevel.EXHAUSTED
        spacing = self.sustainable_spacing(now)
        if spacing >= LEVEL_CRITICAL_AT:
            return QuotaLevel.CRITICAL
        if spacing >= LEVEL_LOW_AT:
            return QuotaLevel.LOW
        return QuotaLevel.NORMAL

    # ── Readings and triggers ────────────────────────────────────────────────

    def set_rooms(self, room_ids: set[int], now: float) -> None:
        """Add new rooms, due straight away, and forget removed ones."""
        for room_id in room_ids - self.rooms.keys():
            self.rooms[room_id] = RoomPollState(room_id=room_id, next_due=now)
        for room_id in self.rooms.keys() - room_ids:
            del self.rooms[room_id]

    def update_reading(
        self,
        room_id: int,
        temperature: float | None,
        setpoint: float | None,
        is_off: bool,
        now: float,
        just_polled: bool = False,
    ) -> PollReason | None:
        """Take a room's reading and return the trigger it caused, if any.

        A reading from the API (``just_polled``) starts the backoff without asking
        for another poll, since the data in hand is already current.
        """
        room = self.rooms.get(room_id)
        if room is None:
            return None
        if temperature is not None:
            temperature = round(temperature, 1)

        if not room.seeded:
            room.temperature, room.setpoint, room.is_off = temperature, setpoint, is_off
            room.temperature_changed_at = now
            room.seeded = True
            return None

        level = self.quota_level(now)
        setpoint_changed = (
            setpoint is not None and setpoint != room.setpoint
        ) or is_off != room.is_off
        temperature_changed = self._note_temperature(room, temperature, now)
        if setpoint is not None:
            room.setpoint = setpoint
        room.is_off = is_off

        # Heating demand only moves when a setpoint or mode does, or the room
        # crosses the setpoint, so a setpoint change goes ahead at any quota level
        # that has calls left.
        if setpoint_changed and level is not QuotaLevel.EXHAUSTED:
            room.backoff_step = 0
            if just_polled:
                self._advance_backoff(room, now)
            else:
                room.reason = PollReason.SETPOINT_CHANGE
                room.next_due = now + SETPOINT_DEBOUNCE
            return PollReason.SETPOINT_CHANGE

        if temperature_changed and self._temperature_triggers(room, level):
            room.backoff_step = 0
            if just_polled:
                self._advance_backoff(room, now)
            else:
                room.reason = PollReason.TEMPERATURE_CHANGE
                room.next_due = now
            return PollReason.TEMPERATURE_CHANGE

        # No trigger, but a reading can still move the room to a shorter base.
        if room.backoff_step is None and not just_polled:
            interval, reason = self._room_base(room, level)
            if self.last_poll + interval < room.next_due:
                room.next_due = self.last_poll + interval
                room.reason = reason
        return None

    @staticmethod
    def _note_temperature(
        room: RoomPollState, temperature: float | None, now: float
    ) -> bool:
        """Store a temperature and return whether it's a meaningful change."""
        if temperature is None or room.temperature is None:
            if temperature is not None:
                room.temperature = temperature
                room.temperature_changed_at = now
            return False
        if abs(temperature - room.temperature) < TEMP_STEP - 0.001:
            return False
        if (
            room.last_temperature is not None
            and abs(temperature - room.last_temperature) < 0.001
            and now - room.temperature_changed_at < BOUNCE_WINDOW
        ):
            return False  # flicked back: keep the held value and its timestamp
        room.last_temperature = room.temperature
        room.temperature = temperature
        room.temperature_changed_at = now
        return True

    @staticmethod
    def _temperature_triggers(room: RoomPollState, level: QuotaLevel) -> bool:
        """Return whether a temperature change is worth a call at this level."""
        if level is QuotaLevel.NORMAL:
            return True
        return level is QuotaLevel.LOW and room.heating_relevant

    def request_command_poll(self, now: float) -> None:
        """Poll soon, and fully, because a command was just sent to Tado."""
        self._command_pending = True
        for room in self.rooms.values():
            room.next_due = now
            room.reason = PollReason.COMMAND

    # ── Results ──────────────────────────────────────────────────────────────

    def _room_base(
        self, room: RoomPollState, level: QuotaLevel
    ) -> tuple[float, PollReason]:
        """Return a room's base interval, stretched under quota pressure."""
        interval, reason = base_interval(room.temperature, room.setpoint, room.is_off)
        if not room.heating_relevant:
            if level is QuotaLevel.LOW:
                interval *= IDLE_STRETCH_LOW
            elif level in (QuotaLevel.CRITICAL, QuotaLevel.EXHAUSTED):
                interval *= IDLE_STRETCH_CRITICAL
        return interval, reason

    def _advance_backoff(self, room: RoomPollState, now: float) -> None:
        """Move a room to its next backoff step, or back to its base."""
        if room.backoff_step is not None and room.backoff_step < len(BACKOFF_STEPS):
            room.next_due = now + BACKOFF_STEPS[room.backoff_step]
            room.backoff_step += 1
            room.reason = PollReason.BACKOFF
            return
        room.backoff_step = None
        interval, room.reason = self._room_base(room, self.quota_level(now))
        room.next_due = now + interval

    def poll_succeeded(self, now: float, full: bool) -> None:
        """Advance every room after a poll that returned data."""
        self.last_poll = now
        self.failures = 0
        self.rate_limited_until = None
        self._command_pending = False
        if full:
            self.last_full = now
        level = self.quota_level(now)
        for room in self.rooms.values():
            # Backoff is the first thing to go under pressure, except in a room
            # that could be heating while calls last.
            if room.backoff_step is not None and (
                level in (QuotaLevel.CRITICAL, QuotaLevel.EXHAUSTED)
                or (level is QuotaLevel.LOW and not room.heating_relevant)
            ):
                room.backoff_step = None
                room.next_due = now  # falls through to the base below
            if room.next_due <= now + DUE_TOLERANCE:
                self._advance_backoff(room, now)
            elif room.backoff_step is None:
                # Polled early for another room: this room's data is fresh too.
                interval, room.reason = self._room_base(room, level)
                room.next_due = now + interval

    def poll_failed(self, now: float) -> None:
        """Record a failed poll; plan_next() then schedules a retry."""
        self.last_poll = now
        self.failures += 1

    def poll_rate_limited(self, now: float) -> None:
        """Stop polling until the reset, probing now and then in case it's wrong."""
        self.last_poll = now
        self.rate_limited_until = now + min(
            self._seconds_to_reset(now), RATE_LIMIT_PROBE
        )

    def needs_full_refresh(self, now: float) -> bool:
        """Return whether the next poll should read devices and presence too."""
        if self.last_full == 0.0 or self._command_pending:
            return True
        level = self.quota_level(now)
        if level in (QuotaLevel.CRITICAL, QuotaLevel.EXHAUSTED):
            return False
        wait = FULL_REFRESH_NORMAL if level is QuotaLevel.NORMAL else FULL_REFRESH_LOW
        return now - self.last_full >= wait

    # ── Planning ─────────────────────────────────────────────────────────────

    def plan_next(self, now: float) -> PollPlan:
        """Return when the next poll should run."""
        level = self.quota_level(now)

        if self.rate_limited_until is not None and now < self.rate_limited_until:
            return self._plan(self.rate_limited_until, PollReason.RATE_LIMITED, None, level)

        if self.failures:
            step = min(self.failures - 1, len(RETRY_STEPS) - 1)
            return self._plan(
                self.last_poll + RETRY_STEPS[step], PollReason.RETRY, None, level
            )

        if level is QuotaLevel.EXHAUSTED:
            # Only the calls kept for commands are left: wait for the reset, with a
            # probe now and then in case the reset time is only an estimate.
            at = now + min(self._seconds_to_reset(now), RATE_LIMIT_PROBE)
            return self._plan(at, PollReason.RATE_LIMITED, None, level)

        if not self.rooms:
            return self._plan(
                self.last_poll + BASE_OFF, PollReason.OFF, None, level
            )

        at, room = min(
            (
                (max(r.next_due, self.last_poll + self._spacing(r, level, now)), r)
                for r in self.rooms.values()
            ),
            key=lambda item: item[0],
        )
        return self._plan(at, room.reason, room.room_id, level)

    def _spacing(self, room: RoomPollState, level: QuotaLevel, now: float) -> float:
        """Return the least time to leave after the previous poll."""
        if room.reason in URGENT_REASONS:
            return MIN_SPACING_URGENT
        if level is QuotaLevel.NORMAL:
            return MIN_SPACING
        if room.heating_relevant:
            # Heating detection may eat into the reserve, down to the last few.
            return max(MIN_SPACING, self.sustainable_spacing(now, RESERVE_COMMANDS))
        return max(MIN_SPACING, self.sustainable_spacing(now))

    def _plan(
        self, at: float, reason: PollReason, room_id: int | None, level: QuotaLevel
    ) -> PollPlan:
        """Build a plan."""
        return PollPlan(
            at=at,
            interval=max(0.0, at - self.last_poll) if self.last_poll else 0.0,
            reason=reason,
            room_id=room_id,
            level=level,
        )
