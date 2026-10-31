"""Tests for adaptive polling inside Home Assistant, with the API mocked."""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.tado_x.api import TadoXApiError, TadoXAuthError, TadoXRateLimitError
from custom_components.tado_x.const import DOMAIN

SERIAL_A = "RU0000000001"
SERIAL_B = "VA0000000002"
MATTER = "climate.matter_thermostat"


def room(room_id, name, temp=20.0, target=21.0, power_pct=0):
    return {
        "id": room_id,
        "name": name,
        "sensorDataPoints": {
            "insideTemperature": {"value": temp},
            "humidity": {"percentage": 50},
        },
        "setting": {"power": "ON", "temperature": {"value": target}},
        "heatingPower": {"percentage": power_pct},
        "connection": {"state": "CONNECTED"},
    }


def devices(*rooms_):
    return {
        "rooms": [
            {
                "roomId": rid,
                "devices": [
                    {
                        "serialNumber": serial,
                        "type": kind,
                        "firmwareVersion": "1",
                        "connection": {"state": "CONNECTED"},
                        "batteryState": "NORMAL",
                        "temperatureOffset": 0.0,
                    }
                ],
            }
            for rid, serial, kind in rooms_
        ],
        "otherDevices": [],
    }


@pytest.fixture
def api():
    """Replace the API client's calls."""
    with (
        patch("custom_components.tado_x.api.TadoXApi.refresh_access_token", AsyncMock(return_value=True)),
        patch(
            "custom_components.tado_x.api.TadoXApi.get_rooms",
            AsyncMock(return_value=[room(1, "Bedroom"), room(2, "Lounge", temp=25.0)]),
        ) as get_rooms,
        patch(
            "custom_components.tado_x.api.TadoXApi.get_rooms_and_devices",
            AsyncMock(return_value=devices((1, SERIAL_A, "RU04"), (2, SERIAL_B, "VA04"))),
        ) as get_devices,
        patch(
            "custom_components.tado_x.api.TadoXApi.get_home_state",
            AsyncMock(return_value={"presence": "HOME", "presenceLocked": False}),
        ) as get_home,
    ):
        yield type("Api", (), {"rooms": get_rooms, "devices": get_devices, "home": get_home})


@pytest.fixture
async def entry(hass, api, freezer):
    """Set up Tado X beside a Matter thermostat with a matching serial number."""
    freezer.move_to(datetime(2026, 11, 2, 10, 0, tzinfo=UTC))
    matter_entry = MockConfigEntry(domain="matter")
    matter_entry.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=matter_entry.entry_id,
        identifiers={("matter", f"serial_{SERIAL_A}")},
    )
    er.async_get(hass).async_get_or_create(
        "climate", "matter", "unique", suggested_object_id="matter_thermostat",
        device_id=device.id, config_entry=matter_entry,
    )
    set_matter(hass, 20.0, 21.0)

    config = MockConfigEntry(
        domain=DOMAIN,
        data={
            "home_id": 99, "home_name": "Home", "access_token": "a",
            "refresh_token": "b", "token_expiry": None, "has_auto_assist": False,
            "enable_weather": False, "enable_mobile_devices": False,
            "enable_air_comfort": False, "enable_running_times": False,
            "enable_flow_temp": False,
        },
    )
    config.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config.entry_id)
    await hass.async_block_till_done()
    yield config
    await hass.config_entries.async_unload(config.entry_id)
    await hass.async_block_till_done()


def set_matter(hass, temp, setpoint, mode="heat"):
    hass.states.async_set(
        MATTER, mode, {"current_temperature": temp, "temperature": setpoint}
    )


async def advance(hass, freezer, seconds):
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


def coordinator(hass, entry):
    return hass.data[DOMAIN][entry.entry_id]


async def test_matter_found_by_serial_and_one_climate_per_room(hass, entry):
    c = coordinator(hass, entry)
    assert c.local_entity(1) == MATTER
    assert c.local_entity(2) is None
    ids = hass.states.async_entity_ids("climate")
    assert {"climate.bedroom", "climate.lounge"} <= set(ids)


async def test_startup_does_one_full_refresh_then_light_polls(hass, entry, api, freezer):
    assert (api.rooms.await_count, api.devices.await_count, api.home.await_count) == (1, 1, 1)
    await advance(hass, freezer, 5 * 60 + 1)  # below setpoint in the bedroom
    assert (api.rooms.await_count, api.devices.await_count, api.home.await_count) == (2, 1, 1)


async def test_full_refresh_returns_hourly(hass, entry, api, freezer):
    for _ in range(13):
        await advance(hass, freezer, 5 * 60 + 1)
    assert api.devices.await_count == 2
    assert api.home.await_count == 2


async def test_temperature_change_polls_once_and_backs_off(hass, entry, api, freezer):
    await advance(hass, freezer, 5 * 60 + 1)
    assert api.rooms.await_count == 2
    await advance(hass, freezer, 70)
    set_matter(hass, 20.1, 21.0)
    await advance(hass, freezer, 1)
    assert api.rooms.await_count == 3
    set_matter(hass, 20.1, 21.0)  # same reading, nothing new
    await advance(hass, freezer, 30)
    assert api.rooms.await_count == 3
    await advance(hass, freezer, 31)  # first backoff step
    assert api.rooms.await_count == 4
    await advance(hass, freezer, 121)  # second
    assert api.rooms.await_count == 5
    assert coordinator(hass, entry).planner.rooms[1].backoff_step == 3


async def test_setpoint_change_polls_quickly(hass, entry, api, freezer):
    await advance(hass, freezer, 60)
    set_matter(hass, 20.0, 23.0)
    await hass.async_block_till_done()
    assert api.rooms.await_count == 1  # waits out the 5 s debounce
    await advance(hass, freezer, 6)
    assert api.rooms.await_count == 2


async def test_climate_prefers_matter_and_reports_heating(hass, entry, api, freezer):
    set_matter(hass, 20.4, 22.5)
    await hass.async_block_till_done()
    state = hass.states.get("climate.bedroom")
    assert state.attributes["current_temperature"] == 20.4
    assert state.attributes["temperature"] == 22.5
    assert state.attributes["hvac_action"] == "idle"
    assert state.state == "auto"

    api.rooms.return_value = [room(1, "Bedroom", power_pct=40), room(2, "Lounge", temp=25.0)]
    await advance(hass, freezer, 20)
    assert hass.states.get("climate.bedroom").attributes["hvac_action"] == "heating"

    set_matter(hass, 20.4, 22.5, mode="off")
    await hass.async_block_till_done()
    state = hass.states.get("climate.bedroom")
    assert state.state == "off"
    assert state.attributes["hvac_action"] == "off"

    # The room without Matter uses the API.
    assert hass.states.get("climate.lounge").attributes["current_temperature"] == 25.0


async def test_commands_cannot_double_up(hass, entry, api, freezer):
    await advance(hass, freezer, 60)
    c = coordinator(hass, entry)
    await c.async_request_refresh()
    await c.async_request_refresh()
    set_matter(hass, 20.0, 24.0)
    await advance(hass, freezer, 20)
    assert api.rooms.await_count == 2
    assert api.devices.await_count == 2  # a command refreshes everything


async def test_a_poll_in_flight_blocks_a_second(hass, entry, api, freezer):
    c = coordinator(hass, entry)
    c._poll_running = True
    await c._async_run_poll()
    assert api.rooms.await_count == 1
    c._poll_running = False


async def test_failure_keeps_entities_and_retries(hass, entry, api, freezer):
    api.rooms.side_effect = TadoXApiError("Network error: DNS")
    await advance(hass, freezer, 5 * 60 + 1)
    assert api.rooms.await_count == 2
    assert hass.states.get("climate.bedroom").state != "unavailable"
    assert hass.states.get("sensor.home_home_last_api_update").state != "unavailable"
    await advance(hass, freezer, 61)
    assert api.rooms.await_count == 3
    api.rooms.side_effect = None
    await advance(hass, freezer, 121)
    assert api.rooms.await_count == 4
    assert coordinator(hass, entry).polling_status.last_error is None


async def test_long_outage_goes_unavailable_then_recovers(hass, entry, api, freezer):
    api.rooms.side_effect = TadoXApiError("down")
    for _ in range(8):
        await advance(hass, freezer, 8 * 60 + 1)
    assert hass.states.get("binary_sensor.bedroom_heating").state == "unavailable"
    api.rooms.side_effect = None
    await advance(hass, freezer, 8 * 60 + 1)
    assert hass.states.get("binary_sensor.bedroom_heating").state == "off"


async def test_only_a_real_auth_failure_stops_polling(hass, entry, api, freezer):
    api.rooms.side_effect = TadoXAuthError("refused")
    await advance(hass, freezer, 5 * 60 + 1)
    count = api.rooms.await_count
    await advance(hass, freezer, 3600)
    assert api.rooms.await_count == count


async def test_token_refresh_network_error_is_not_an_auth_failure(hass):
    from homeassistant.helpers.aiohttp_client import async_get_clientsession
    from custom_components.tado_x.api import TadoXApi
    from aiohttp import ClientError

    api = TadoXApi(async_get_clientsession(hass), refresh_token="x")
    with patch.object(api._session, "post", side_effect=ClientError("dns")):
        with pytest.raises(TadoXApiError):
            await api.refresh_access_token()


async def test_rate_limit_stops_until_reset(hass, entry, api, freezer):
    api.rooms.side_effect = TadoXRateLimitError(
        "429", reset_time=dt_util.utcnow() + timedelta(hours=5)
    )
    await advance(hass, freezer, 5 * 60 + 1)
    count = api.rooms.await_count
    c = coordinator(hass, entry)
    assert c.polling_status.reason == "rate_limited"
    await advance(hass, freezer, 30 * 60)
    assert api.rooms.await_count == count
    assert hass.states.get("sensor.home_home_quota_level").state == "exhausted"


async def test_new_room_reloads_the_entry(hass, entry, api, freezer):
    api.rooms.return_value = [
        room(1, "Bedroom"), room(2, "Lounge", temp=25.0), room(3, "Office"),
    ]
    api.devices.return_value = devices(
        (1, SERIAL_A, "RU04"), (2, SERIAL_B, "VA04"), (3, "RU0000000003", "RU04")
    )
    await advance(hass, freezer, 5 * 60 + 1)
    await hass.async_block_till_done()
    assert hass.states.get("climate.office") is not None


async def test_unload_stops_everything(hass, entry, api, freezer):
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    count = api.rooms.await_count
    set_matter(hass, 25.0, 30.0)
    await advance(hass, freezer, 3600)
    assert api.rooms.await_count == count


async def test_diagnostic_sensors(hass, entry, api, freezer):
    interval = hass.states.get("sensor.home_home_polling_interval")
    assert float(interval.state) == 5.0
    assert interval.attributes["rooms"]["Bedroom"]["source"] == "matter"
    assert interval.attributes["rooms"]["Lounge"]["source"] == "api"
    assert hass.states.get("sensor.home_home_polling_reason").state == "below_setpoint"
    assert hass.states.get("sensor.home_home_quota_level").state == "normal"
    assert hass.states.get("sensor.home_home_last_api_update").state.startswith("2026-11-02T10:00")
    assert hass.states.get("sensor.home_home_next_api_update").state.startswith("2026-11-02T10:05")


async def test_fixed_interval_option_still_works(hass, api, freezer):
    freezer.move_to(datetime(2026, 11, 2, 10, 0, tzinfo=UTC))
    config = MockConfigEntry(
        domain=DOMAIN,
        data={
            "home_id": 99, "home_name": "Home", "access_token": "a",
            "refresh_token": "b", "token_expiry": None, "scan_interval": 360,
            "adaptive_polling": False,
        },
    )
    config.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config.entry_id)
    await hass.async_block_till_done()
    await advance(hass, freezer, 361)
    assert api.rooms.await_count == 2
    assert api.devices.await_count == 2
    await hass.config_entries.async_unload(config.entry_id)
