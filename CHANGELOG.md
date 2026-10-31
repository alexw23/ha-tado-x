# Changelog

Changes in this fork, on top of [exabird/ha-tado-x](https://github.com/exabird/ha-tado-x) v1.8.11.

## 1.9.0

### Added

- Quota-aware adaptive polling, on by default, with an option to return to a fixed interval. See the README for the rules.
- Rooms are linked to their Matter thermostats automatically by serial number, so local changes can trigger polls.
- Climate entities show Matter's temperature and setpoint as soon as they change, when the room has a Matter device.
- Diagnostic sensors: Polling interval, Last API update, Next API update, Polling reason and Quota level.
- New rooms and devices get entities without a manual reload.
- Tests under `custom_components/tado_x/tests`.

### Changed

- Most polls now read rooms only (1 call). Devices, presence and optional features refresh hourly.
- API calls run one at a time, at least 1 second apart, with a 30-second timeout.
- The quota reset estimate is now 06:25 UTC, based on one account's history, and moves when a reset is observed or the rate limit header gives the time.
- Failed polls keep the last values for 30 minutes before entities become unavailable.

### Fixed

- A network or server error during token refresh no longer counts as an authentication failure, which stopped polling until a restart.
- `hvac_action` no longer reports `off` for a room idling on its schedule.
- Services no longer use a stale coordinator after the entry reloads.
