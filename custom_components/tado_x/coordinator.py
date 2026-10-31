"""DataUpdateCoordinator for Tado X."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from homeassistant.components.climate import (
    ATTR_CURRENT_TEMPERATURE,
    DOMAIN as CLIMATE_DOMAIN,
    HVACMode,
)
from homeassistant.const import ATTR_TEMPERATURE, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HassJob,
    HomeAssistant,
    callback,
)
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .api import TadoXApi, TadoXApiError, TadoXAuthError, TadoXRateLimitError
from .const import (
    DEVICE_TYPE_SENSOR,
    DEVICE_TYPE_THERMOSTAT,
    DEVICE_TYPE_VALVE,
    DOMAIN,
    SCAN_INTERVAL_AUTO_ASSIST,
    SCAN_INTERVAL_FREE_TIER,
)
from .polling import PollPlan, PollPlanner, PollReason, QuotaLevel

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry

_LOGGER = logging.getLogger(__name__)

# Which device speaks for a room when several of them are on Matter
LOCAL_DEVICE_PRIORITY = {
    DEVICE_TYPE_THERMOSTAT: 0,
    DEVICE_TYPE_SENSOR: 1,
    DEVICE_TYPE_VALVE: 2,
}

# How long a failing API may go unnoticed, showing the last data, before entities
# turn unavailable. Without this, each DNS blip blanked them until the next poll.
STALE_GRACE_SECONDS = 1800


@dataclass
class PollingStatus:
    """Adaptive polling diagnostics."""

    adaptive: bool = True
    interval: float = 0.0  # seconds between the last poll and the next
    last_update: datetime | None = None
    next_update: datetime | None = None
    reason: PollReason | None = None
    level: QuotaLevel | None = None
    last_error: str | None = None
    rooms: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TadoXDevice:
    """Representation of a Tado X device."""

    serial_number: str
    device_type: str
    firmware_version: str
    connection_state: str
    battery_state: str | None = None
    temperature_measured: float | None = None
    temperature_offset: float = 0.0
    mounting_state: str | None = None
    child_lock_enabled: bool = False
    room_id: int | None = None
    room_name: str | None = None


@dataclass
class TadoXRoom:
    """Representation of a Tado X room."""

    room_id: int
    name: str
    current_temperature: float | None = None
    target_temperature: float | None = None
    humidity: float | None = None
    heating_power: int = 0
    power: str = "OFF"
    connection_state: str = "DISCONNECTED"
    manual_control_active: bool = False
    manual_control_remaining_seconds: int | None = None
    manual_control_type: str | None = None
    boost_mode: bool = False
    open_window_detected: bool = False
    next_schedule_change: str | None = None
    next_schedule_temperature: float | None = None
    devices: list[TadoXDevice] = field(default_factory=list)
    # Running times data (heating duration today)
    running_time_today_seconds: int = 0


@dataclass
class TadoXWeather:
    """Representation of Tado weather data."""

    outdoor_temperature: float | None = None
    solar_intensity: float | None = None
    weather_state: str | None = None


@dataclass
class TadoXMobileDevice:
    """Representation of a Tado mobile device for geofencing."""

    device_id: int
    name: str
    device_metadata: dict = field(default_factory=dict)
    location: str | None = None  # "HOME", "AWAY", or None if location not available
    at_home: bool = False
    geofencing_enabled: bool = False


@dataclass
class TadoXRoomAirComfort:
    """Air comfort data for a room."""

    room_id: int
    freshness: str | None = None  # FRESH, FAIR, STALE
    comfort_level: str | None = None  # Based on temperature/humidity


@dataclass
class TadoXData:
    """Data from Tado X API."""

    home_id: int
    home_name: str
    rooms: dict[int, TadoXRoom] = field(default_factory=dict)
    devices: dict[str, TadoXDevice] = field(default_factory=dict)
    other_devices: list[TadoXDevice] = field(default_factory=list)
    presence: str | None = None  # HOME, AWAY, or None if not locked
    presence_locked: bool = False  # Whether presence is manually set
    api_calls_today: int = 0
    api_reset_time: datetime | None = None
    has_auto_assist: bool = False
    # Real values from Tado API response headers
    api_quota_limit: int | None = None  # From ratelimit-policy header (q=)
    api_quota_remaining: int | None = None  # From ratelimit header (r=)
    # Weather data
    weather: TadoXWeather | None = None
    # Mobile devices for geofencing
    mobile_devices: dict[int, TadoXMobileDevice] = field(default_factory=dict)
    # Running times data (raw response for additional processing if needed)
    running_times: dict[str, Any] = field(default_factory=dict)
    # Air comfort data per room
    air_comfort: dict[int, TadoXRoomAirComfort] = field(default_factory=dict)
    # Rate limit status
    rate_limited: bool = False
    rate_limit_reset: datetime | None = None
    # Flow temperature optimization
    max_flow_temperature: int | None = None
    flow_temp_min: int | None = None
    flow_temp_max: int | None = None
    flow_temp_auto_adaptation: bool = False
    flow_temp_auto_value: int | None = None
    has_flow_temp_control: bool = False
    # Adaptive polling diagnostics, shared with the coordinator
    polling: PollingStatus = field(default_factory=PollingStatus)


class TadoXDataUpdateCoordinator(DataUpdateCoordinator[TadoXData]):
    """Class to manage fetching Tado X data."""

    def __init__(
        self,
        hass: HomeAssistant,
        api: TadoXApi,
        home_id: int,
        home_name: str,
        save_api_stats_callback: Callable[[], None] | None = None,
        scan_interval: int | None = None,
        enable_weather: bool = True,
        enable_mobile_devices: bool = True,
        enable_air_comfort: bool = True,
        enable_running_times: bool = True,
        enable_flow_temp: bool = True,
        adaptive_polling: bool = True,
        config_entry: ConfigEntry | None = None,
    ) -> None:
        """Initialize the coordinator."""
        # Determine scan interval based on subscription tier if not explicitly set
        if scan_interval is None:
            scan_interval = (
                SCAN_INTERVAL_AUTO_ASSIST if api.has_auto_assist
                else SCAN_INTERVAL_FREE_TIER
            )

        # Adaptive polling runs its own timer, so the base class must not run one.
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name=DOMAIN,
            update_interval=(
                None if adaptive_polling else timedelta(seconds=scan_interval)
            ),
        )
        self.api = api
        self.home_id = home_id
        self.home_name = home_name
        self.api.home_id = home_id
        self._save_api_stats_callback = save_api_stats_callback
        self._scan_interval = scan_interval

        # Feature toggles for optional API calls
        self.enable_weather = enable_weather
        self.enable_mobile_devices = enable_mobile_devices
        self.enable_air_comfort = enable_air_comfort
        self.enable_running_times = enable_running_times
        self.enable_flow_temp = enable_flow_temp

        # Adaptive polling
        self.adaptive_polling = adaptive_polling
        self.planner = PollPlanner()
        self.polling_status = PollingStatus(adaptive=adaptive_polling)
        self._poll_running = False
        self._stopped = False
        self._auth_failed = False
        self._unsub_timer: CALLBACK_TYPE | None = None
        self._unsub_local: CALLBACK_TYPE | None = None
        self._local_by_room: dict[int, str] = {}
        self._room_by_local: dict[str, int] = {}
        self._last_success: float = 0.0
        self._known_ids: set[int | str] | None = None
        # The parts of a full refresh that light polls reuse
        self._cached_rooms_devices: dict[str, Any] = {}
        self._cached_home_state: dict[str, Any] = {}
        self._cached_mobile_devices: list[dict[str, Any]] = []

        _LOGGER.info(
            "Tado X coordinator initialized with %s (%s tier)",
            "adaptive polling" if adaptive_polling else f"a {scan_interval} second interval",
            "Auto-Assist" if api.has_auto_assist else "Free"
        )
        _LOGGER.info(
            "Optional features - Weather: %s, Mobile devices: %s, Air comfort: %s, Running times: %s, Flow temp: %s",
            enable_weather, enable_mobile_devices, enable_air_comfort, enable_running_times, enable_flow_temp
        )

    def update_scan_interval(self, new_interval: int) -> None:
        """Update the scan interval dynamically (fixed-interval mode only)."""
        self._scan_interval = new_interval
        if not self.adaptive_polling:
            self.update_interval = timedelta(seconds=new_interval)
        _LOGGER.info("Scan interval updated to %d seconds", new_interval)

    # ── Adaptive polling ─────────────────────────────────────────────────────

    @callback
    def async_start_adaptive_polling(self) -> None:
        """Start listening to Matter and schedule the next poll."""
        if not self.adaptive_polling:
            return
        self._async_resolve_local_entities()
        self._async_reschedule()
        self.async_update_listeners()  # the sensors exist only now

    @callback
    def async_stop_adaptive_polling(self) -> None:
        """Stop the timer and the Matter listener."""
        self._stopped = True
        self._async_cancel_timer()
        if self._unsub_local:
            self._unsub_local()
            self._unsub_local = None

    async def async_shutdown(self) -> None:
        """Stop adaptive polling along with the coordinator."""
        self.async_stop_adaptive_polling()
        await super().async_shutdown()

    async def async_request_refresh(self) -> None:
        """Refresh after a command, through the planner so it can't double up."""
        if not self.adaptive_polling:
            await super().async_request_refresh()
            return
        self.planner.request_command_poll(dt_util.utcnow().timestamp())
        self._async_reschedule()

    def local_reading(
        self, room_id: int
    ) -> tuple[float | None, float | None, bool] | None:
        """Return (temperature, setpoint, is_off) from Matter, if it has them."""
        entity_id = self._local_by_room.get(room_id)
        state = self.hass.states.get(entity_id) if entity_id else None
        if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return None
        return (
            state.attributes.get(ATTR_CURRENT_TEMPERATURE),
            state.attributes.get(ATTR_TEMPERATURE),
            state.state == HVACMode.OFF,
        )

    def local_entity(self, room_id: int) -> str | None:
        """Return the Matter climate entity behind a room, if any."""
        return self._local_by_room.get(room_id)

    @callback
    def _async_resolve_local_entities(self) -> None:
        """Find each room's Matter climate entity from its Tado serial numbers.

        Matter devices identify themselves as ("matter", "serial_<serial>"), with
        the same serial number that Tado reports, so no setup is needed and a room
        with no Matter device simply stays on API readings.
        """
        if not self.data:
            return
        device_registry = dr.async_get(self.hass)
        entity_registry = er.async_get(self.hass)
        found: dict[int, str] = {}
        for room_id, room in self.data.rooms.items():
            for device in sorted(
                room.devices,
                key=lambda d: LOCAL_DEVICE_PRIORITY.get(d.device_type, 99),
            ):
                matter_device = device_registry.async_get_device(
                    identifiers={("matter", f"serial_{device.serial_number}")}
                )
                if matter_device is None:
                    continue
                entity_id = next(
                    (
                        entry.entity_id
                        for entry in er.async_entries_for_device(
                            entity_registry, matter_device.id
                        )
                        if entry.domain == CLIMATE_DOMAIN
                        and entry.platform == "matter"
                        and not entry.disabled
                    ),
                    None,
                )
                if entity_id:
                    found[room_id] = entity_id
                    break

        if found == self._local_by_room and (self._unsub_local or not found):
            return
        self._local_by_room = found
        self._room_by_local = {entity: room for room, entity in found.items()}
        if self._unsub_local:
            self._unsub_local()
            self._unsub_local = None
        if found and not self._stopped:
            self._unsub_local = async_track_state_change_event(
                self.hass, list(found.values()), self._async_local_state_changed
            )
        _LOGGER.debug("Local Matter entities by room: %s", found)

    @callback
    def _async_local_state_changed(self, event: Event[EventStateChangedData]) -> None:
        """React to a Matter thermostat changing."""
        room_id = self._room_by_local.get(event.data["entity_id"])
        reading = self.local_reading(room_id) if room_id is not None else None
        if reading is not None:
            trigger = self.planner.update_reading(
                room_id, *reading, dt_util.utcnow().timestamp()
            )
            if trigger:
                _LOGGER.debug("Room %s: %s", room_id, trigger)
            self._async_reschedule()
        # Entities show Matter's temperature and setpoint straight away.
        self.async_update_listeners()

    @callback
    def _async_cancel_timer(self) -> None:
        """Cancel the pending poll."""
        if self._unsub_timer:
            self._unsub_timer()
            self._unsub_timer = None

    @callback
    def _async_reschedule(self) -> None:
        """Set the timer for the planner's next poll."""
        self._async_cancel_timer()
        if not self.adaptive_polling or self._stopped or self._auth_failed:
            return
        now = dt_util.utcnow().timestamp()
        plan = self.planner.plan_next(now)
        self._async_update_status(plan, now)
        if self.config_entry and self.config_entry.pref_disable_polling:
            return
        self._unsub_timer = async_call_later(
            self.hass,
            max(0.0, plan.at - now),
            HassJob(self._async_timer_fired, "tado_x poll", cancel_on_shutdown=True),
        )

    @callback
    def _async_timer_fired(self, _now: datetime) -> None:
        """Run the poll that the timer was set for."""
        self._unsub_timer = None
        self.hass.async_create_background_task(
            self._async_run_poll(), "tado_x poll"
        )

    async def _async_run_poll(self) -> None:
        """Poll once; a poll already running covers this one."""
        if self._poll_running:
            return
        self._poll_running = True
        try:
            await self.async_refresh()
        finally:
            self._poll_running = False
            self._async_reschedule()
            self.async_update_listeners()

    @callback
    def _async_update_status(self, plan: PollPlan, now: float) -> None:
        """Copy the plan into the diagnostics that sensors read."""
        status = self.polling_status
        status.interval = plan.interval
        status.next_update = dt_util.utc_from_timestamp(plan.at)
        status.reason = plan.reason
        status.level = plan.level
        names = self.data.rooms if self.data else {}
        status.rooms = {
            (names[room_id].name if room_id in names else str(room_id)): {
                "source": "matter" if room_id in self._local_by_room else "api",
                "local_entity": self._local_by_room.get(room_id),
                "temperature": room.temperature,
                "setpoint": room.setpoint,
                "reason": room.reason.value,
                "backoff_step": room.backoff_step,
                "next_due": dt_util.utc_from_timestamp(room.next_due).isoformat(),
            }
            for room_id, room in self.planner.rooms.items()
        }

    @callback
    def _async_after_poll(self, data: TadoXData, full: bool, now: float) -> None:
        """Feed a successful poll to the planner and look for new hardware."""
        data.polling = self.polling_status
        self._last_success = now
        self.polling_status.last_update = dt_util.utc_from_timestamp(now)
        self.polling_status.last_error = None

        self._async_check_for_new_hardware(data)
        if not self.adaptive_polling:
            return

        self.planner.update_quota(
            data.api_quota_remaining,
            data.api_quota_limit,
            data.api_reset_time.timestamp() if data.api_reset_time else None,
        )
        self.planner.set_rooms(set(data.rooms), now)
        previous_data, self.data = self.data, data
        if full or previous_data is None:
            self._async_resolve_local_entities()
        for room_id, room in data.rooms.items():
            # Matter is current; the API is the fallback for rooms without it.
            reading = self.local_reading(room_id) or (
                room.current_temperature,
                room.target_temperature,
                room.power == "OFF",
            )
            self.planner.update_reading(room_id, *reading, now, just_polled=True)
        self.planner.poll_succeeded(now, full)

    @callback
    def _async_check_for_new_hardware(self, data: TadoXData) -> None:
        """Reload when a new room or device appears, so it gets its entities."""
        current: set[int | str] = {*data.rooms, *data.devices}
        if self._known_ids is None:
            self._known_ids = current
            return
        if new := current - self._known_ids:
            _LOGGER.info("New Tado rooms or devices found, reloading: %s", new)
            self._known_ids |= new
            if self.config_entry:
                self.hass.config_entries.async_schedule_reload(
                    self.config_entry.entry_id
                )

    def get_api_calls_per_update(self) -> int:
        """Calculate the number of API calls per update based on enabled features."""
        # Base calls: get_rooms, get_rooms_and_devices, get_home_state
        calls = 3
        if self.enable_weather:
            calls += 1
        if self.enable_mobile_devices:
            calls += 1
        if self.enable_air_comfort:
            calls += 1
        if self.enable_running_times:
            calls += 1
        return calls

    async def _async_update_data(self) -> TadoXData:
        """Fetch data from Tado X API.

        Rooms carry everything heating detection needs for one call, so most polls
        read only those and reuse the rest from the last full refresh.
        """
        now = dt_util.utcnow().timestamp()
        previous = self.data
        full = (
            not self.adaptive_polling
            or previous is None
            or self.planner.needs_full_refresh(now)
        )
        try:
            # Get rooms with current state (required)
            rooms_data = await self.api.get_rooms()

            if full:
                # Get rooms with devices (required)
                self._cached_rooms_devices = await self.api.get_rooms_and_devices()

                # Get home presence state (required)
                self._cached_home_state = await self.api.get_home_state()
            rooms_devices_data = self._cached_rooms_devices
            home_state = self._cached_home_state
            presence = home_state.get("presence")
            presence_locked = home_state.get("presenceLocked", False)

            # Get weather data (optional)
            weather = previous.weather if previous and self.enable_weather else None
            if full and self.enable_weather:
                weather_data = await self.api.get_weather()
                outdoor_temp_data = weather_data.get("outsideTemperature") or {}
                solar_data = weather_data.get("solarIntensity") or {}
                weather_state_data = weather_data.get("weatherState") or {}

                weather = TadoXWeather(
                    outdoor_temperature=outdoor_temp_data.get("celsius"),
                    solar_intensity=solar_data.get("percentage"),
                    weather_state=weather_state_data.get("value"),
                )

            # Get mobile devices for geofencing (optional)
            if not self.enable_mobile_devices:
                self._cached_mobile_devices = []
            elif full:
                self._cached_mobile_devices = await self.api.get_mobile_devices()
            mobile_devices_data = self._cached_mobile_devices

            # Process the data
            data = TadoXData(
                home_id=self.home_id,
                home_name=self.home_name,
                presence=presence,
                presence_locked=presence_locked,
                weather=weather,
            )

            # Process rooms and devices
            rooms_info = rooms_devices_data.get("rooms", [])
            room_devices_map: dict[int, list[dict]] = {}

            for room_info in rooms_info:
                room_id = room_info.get("roomId")
                if room_id:
                    room_devices_map[room_id] = room_info.get("devices", [])

            # Process room states
            for room_data in rooms_data:
                room_id = room_data.get("id")
                if not room_id:
                    continue

                # Debug: log raw room data for power/setting analysis
                setting = room_data.get("setting") or {}
                _LOGGER.debug(
                    "Room %s (%s) - setting: %s, manualControl: %s",
                    room_id,
                    room_data.get("name"),
                    setting,
                    room_data.get("manualControlTermination"),
                )

                # Get sensor data (use 'or {}' to handle None values)
                sensor_data = room_data.get("sensorDataPoints") or {}
                inside_temp = sensor_data.get("insideTemperature") or {}
                humidity_data = sensor_data.get("humidity") or {}

                # Get setting (use 'or {}' to handle None values)
                setting = room_data.get("setting") or {}
                target_temp = setting.get("temperature") or {}

                # Get manual control info
                manual_control = room_data.get("manualControlTermination")
                manual_active = manual_control is not None
                manual_remaining = None
                manual_type = None
                if manual_control:
                    manual_remaining = manual_control.get("remainingTimeInSeconds")
                    manual_type = manual_control.get("type")

                # Get next schedule change (use 'or {}' to handle None values)
                next_change = room_data.get("nextScheduleChange") or {}
                next_change_time = next_change.get("start")
                next_change_setting = next_change.get("setting") or {}
                next_change_temp_obj = next_change_setting.get("temperature") or {}
                next_change_temp = next_change_temp_obj.get("value")

                # Get heating power and connection (use 'or {}' to handle None values)
                heating_power_data = room_data.get("heatingPower") or {}
                connection_data = room_data.get("connection") or {}

                room = TadoXRoom(
                    room_id=room_id,
                    name=room_data.get("name", f"Room {room_id}"),
                    current_temperature=inside_temp.get("value"),
                    target_temperature=target_temp.get("value"),
                    humidity=humidity_data.get("percentage"),
                    heating_power=heating_power_data.get("percentage", 0),
                    power=setting.get("power", "OFF"),
                    connection_state=connection_data.get("state", "DISCONNECTED"),
                    manual_control_active=manual_active,
                    manual_control_remaining_seconds=manual_remaining,
                    manual_control_type=manual_type,
                    boost_mode=room_data.get("boostMode") is not None,
                    open_window_detected=room_data.get("openWindow") is not None,
                    next_schedule_change=next_change_time,
                    next_schedule_temperature=next_change_temp,
                )

                # Add devices for this room
                for device_data in room_devices_map.get(room_id, []):
                    device_connection = device_data.get("connection") or {}
                    device = TadoXDevice(
                        serial_number=device_data.get("serialNumber", ""),
                        device_type=device_data.get("type", ""),
                        firmware_version=device_data.get("firmwareVersion", ""),
                        connection_state=device_connection.get("state", "DISCONNECTED"),
                        battery_state=device_data.get("batteryState"),
                        temperature_measured=device_data.get("temperatureAsMeasured"),
                        temperature_offset=device_data.get("temperatureOffset", 0.0),
                        mounting_state=device_data.get("mountingState"),
                        child_lock_enabled=device_data.get("childLockEnabled", False),
                        room_id=room_id,
                        room_name=room.name,
                    )
                    room.devices.append(device)
                    data.devices[device.serial_number] = device

                data.rooms[room_id] = room

            # Process other devices (bridge, thermostat controller)
            # First, find the room with the most devices (for thermostat association)
            room_with_most_devices: int | None = None
            max_device_count = 0
            for room_id, room in data.rooms.items():
                device_count = len(room.devices)
                if device_count > max_device_count:
                    max_device_count = device_count
                    room_with_most_devices = room_id

            for device_data in rooms_devices_data.get("otherDevices") or []:
                other_device_connection = device_data.get("connection") or {}
                other_room_id = device_data.get("roomId")
                other_room_name = None
                device_type = device_data.get("type", "")

                # If device has a room association from API, use it
                if other_room_id and other_room_id in data.rooms:
                    other_room_name = data.rooms[other_room_id].name
                # For Wireless Receiver X (TR04) without room, associate with the room
                # that has the most devices (typically the main room it controls)
                elif device_type == "TR04" and room_with_most_devices:
                    other_room_id = room_with_most_devices
                    other_room_name = data.rooms[room_with_most_devices].name
                    _LOGGER.debug(
                        "Associating Wireless Receiver X %s with room %s (%s) - room has %d devices",
                        device_data.get("serialNumber"),
                        other_room_id,
                        other_room_name,
                        max_device_count,
                    )

                device = TadoXDevice(
                    serial_number=device_data.get("serialNumber", ""),
                    device_type=device_type,
                    firmware_version=device_data.get("firmwareVersion", ""),
                    connection_state=other_device_connection.get("state", "DISCONNECTED"),
                    room_id=other_room_id,
                    room_name=other_room_name,
                )

                # If device has a room, add it to the room's device list
                if other_room_id and other_room_id in data.rooms:
                    data.rooms[other_room_id].devices.append(device)

                data.other_devices.append(device)
                data.devices[device.serial_number] = device

            # Process mobile devices
            for mobile_data in mobile_devices_data:
                device_id = mobile_data.get("id")
                if not device_id:
                    continue

                # Get location info
                location_data = mobile_data.get("location") or {}
                settings_data = mobile_data.get("settings") or {}
                geo_tracking = settings_data.get("geoTrackingEnabled", False)

                # Determine if at home based on location
                at_home = location_data.get("atHome", False)
                location_state = "HOME" if at_home else "AWAY" if location_data else None

                mobile_device = TadoXMobileDevice(
                    device_id=device_id,
                    name=mobile_data.get("name", f"Mobile {device_id}"),
                    device_metadata=mobile_data.get("deviceMetadata", {}),
                    location=location_state,
                    at_home=at_home,
                    geofencing_enabled=geo_tracking,
                )
                data.mobile_devices[device_id] = mobile_device

            # Between full refreshes, the optional data stays as last fetched.
            if not full and previous:
                data.running_times = previous.running_times
                data.air_comfort = previous.air_comfort
                for room_id, room in data.rooms.items():
                    if room_id in previous.rooms:
                        room.running_time_today_seconds = previous.rooms[
                            room_id
                        ].running_time_today_seconds
                data.has_flow_temp_control = previous.has_flow_temp_control
                data.max_flow_temperature = previous.max_flow_temperature
                data.flow_temp_min = previous.flow_temp_min
                data.flow_temp_max = previous.flow_temp_max
                data.flow_temp_auto_adaptation = previous.flow_temp_auto_adaptation
                data.flow_temp_auto_value = previous.flow_temp_auto_value

            # Fetch running times data for today (optional)
            if full and self.enable_running_times:
                try:
                    today = date.today().isoformat()
                    running_times_data = await self.api.get_running_times(today, today)
                    data.running_times = running_times_data

                    # Process running times per zone/room
                    # The API returns running times with zone IDs that correspond to room IDs
                    running_times_list = running_times_data.get("runningTimes", [])
                    for rt_entry in running_times_list:
                        zones = rt_entry.get("zones", [])
                        for zone_data in zones:
                            zone_id = zone_data.get("id")
                            zone_running_seconds = zone_data.get("runningTimeInSeconds", 0)
                            if zone_id and zone_id in data.rooms:
                                data.rooms[zone_id].running_time_today_seconds = zone_running_seconds

                    _LOGGER.debug("Running times fetched: %s zones", len(running_times_list))
                except Exception as err:
                    # Running times endpoint might not be available for all accounts
                    # Log warning but don't fail the entire update
                    _LOGGER.warning("Failed to fetch running times data: %s", err)
                    data.running_times = {}

            # Fetch air comfort data (optional)
            if full and self.enable_air_comfort:
                try:
                    air_comfort_data = await self.api.get_air_comfort()
                    comfort_list = air_comfort_data.get("comfort", [])
                    for comfort_entry in comfort_list:
                        room_id = comfort_entry.get("roomId")
                        if room_id:
                            # Get humidity level (HUMID, COMFY, DRY)
                            # The API returns humidityLevel directly in comfort_entry
                            freshness = comfort_entry.get("humidityLevel")

                            # Get temperature comfort level (COLD, COMFY, WARM)
                            # The API returns temperatureLevel directly in comfort_entry
                            comfort_level = comfort_entry.get("temperatureLevel")

                            room_air_comfort = TadoXRoomAirComfort(
                                room_id=room_id,
                                freshness=freshness,
                                comfort_level=comfort_level,
                            )
                            data.air_comfort[room_id] = room_air_comfort

                    _LOGGER.debug("Air comfort fetched for %d rooms", len(data.air_comfort))
                except Exception as err:
                    # Air comfort endpoint might not be available for all accounts
                    _LOGGER.warning("Failed to fetch air comfort data: %s", err)

            # Fetch flow temperature optimization settings (if available and enabled)
            if full and self.enable_flow_temp:
                try:
                    flow_data = await self.api.get_flow_temperature_optimization()
                    if flow_data:
                        data.has_flow_temp_control = True
                        data.max_flow_temperature = flow_data.get("maxFlowTemperature")
                        constraints = flow_data.get("maxFlowTemperatureConstraints", {})
                        data.flow_temp_min = constraints.get("min")
                        data.flow_temp_max = constraints.get("max")
                        auto_adapt = flow_data.get("autoAdaptation", {})
                        data.flow_temp_auto_adaptation = auto_adapt.get("enabled", False)
                        data.flow_temp_auto_value = auto_adapt.get("maxFlowTemperature")
                        _LOGGER.debug(
                            "Flow temperature: %s°C (range %s-%s), auto=%s",
                            data.max_flow_temperature,
                            data.flow_temp_min,
                            data.flow_temp_max,
                            data.flow_temp_auto_adaptation,
                        )
                except Exception as err:
                    # Flow temperature endpoint might not be available for all setups
                    # (requires OpenTherm-compatible boiler control device)
                    _LOGGER.debug("Flow temperature optimization not available: %s", err)
                    data.has_flow_temp_control = False

            # Populate API stats (prefer real values from headers when available)
            data.api_calls_today = self.api.api_calls_today
            data.api_reset_time = self.api.api_reset_time
            data.has_auto_assist = self.api.has_auto_assist
            data.api_quota_limit = self.api.api_quota_limit
            data.api_quota_remaining = self.api.api_quota_remaining

            # Save API stats for persistence
            if self._save_api_stats_callback:
                self._save_api_stats_callback()

            self._async_after_poll(data, full, now)
            return data

        except TadoXRateLimitError as err:
            # Handle rate limit gracefully - return previous data with rate_limited flag
            _LOGGER.warning(
                "Rate limit hit. Suspending API calls until %s. Using cached data.",
                err.reset_time,
            )
            self.planner.update_quota(
                self.api.api_quota_remaining,
                self.api.api_quota_limit,
                err.reset_time.timestamp() if err.reset_time else None,
            )
            self.planner.poll_rate_limited(now)
            self.polling_status.last_error = str(err)
            if self.data:
                # Return previous data with rate_limited flag set
                self.data.rate_limited = True
                self.data.rate_limit_reset = err.reset_time
                return self.data
            # No previous data - create minimal data with rate limited status
            return TadoXData(
                home_id=self.home_id,
                home_name=self.home_name,
                rate_limited=True,
                rate_limit_reset=err.reset_time,
                polling=self.polling_status,
            )
        except TadoXAuthError as err:
            # Trigger reauthentication flow instead of just failing
            # This will prompt the user to re-authenticate via the UI
            self._auth_failed = True
            raise ConfigEntryAuthFailed(
                f"Authentication failed: {err}. Please re-authenticate."
            ) from err
        except TadoXApiError as err:
            self.planner.poll_failed(now)
            self.polling_status.last_error = str(err)
            if (
                self.adaptive_polling
                and previous
                and now - self._last_success < STALE_GRACE_SECONDS
            ):
                _LOGGER.warning("Tado API call failed, keeping the last data: %s", err)
                return previous
            raise UpdateFailed(f"API error: {err}") from err
        except Exception as err:
            self.planner.poll_failed(now)
            self.polling_status.last_error = str(err)
            _LOGGER.exception("Unexpected error fetching Tado X data")
            raise UpdateFailed(f"Unexpected error: {err}") from err
