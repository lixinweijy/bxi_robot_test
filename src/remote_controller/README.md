# Remote controller

Deployment and robot button operations are documented in the
[repository README](../../README.md). This package reads input devices and publishes
motion commands; its `system` actions start and stop the robot launcher.

## Input configuration

The default [xbox_default.yaml](config/xbox_default.yaml) selects the Battle Dragon
joystick at `/dev/input/jsBattleDragon`. Button and axis indices depend on the
controller and connection mode. The historical [PS4 index diagram](ps4_key_map.png)
is a reference for that layout, not a substitute for checking the active device.
In the current mapping, pressing the right stick (R3, `js.button.14`) starts the
robot; the physical START button (`js.button.11`) stops it and turns motor power off.

Joystick axes are signed values near -32767 to 32767. Buttons report pressed/released
states, D-pad axes are discrete, and trigger axes may rest at the negative endpoint.
The robot configuration maps these raw signals to controls and output fields.

## Device selection

Each `sources.<name>` entry is a candidate; exactly one device is active. A higher
priority source must remain available for `promote_stable_ms` before taking over.
Switching stops the old driver, clears its signals, publishes a zero-motion command,
starts the new driver, and waits for readiness. Edge actions are suppressed during
switching: release buttons after reconnecting, then press R3 to start or START to stop.

`loss_timeout_ms` controls disconnection and `cooldown_ms` controls retry timing.
Per-signal `timeout_ms` is intended for continuously refreshed network fields;
joystick/CRSF connection state is handled by their drivers instead.

Joystick, keyboard, and CRSF drivers are built in. The bundled robot configuration
only enables the joystick. CRSF uses `crsf.channel.1` through `crsf.channel.16`,
normalizes the receiver range 174..1811 to -1..1, and requires recent valid CRC frames.
To use it, add a source with the correct serial device, baud rate, signal mappings,
and control bindings for that receiver. Unsupported driver types are logged and
skipped; if none are ready, the input manager remains in its safe stopped state.
`--driver <type>` and `--keyboard` are diagnostic filters, not normal deployment options.

## Diagnostics

Only run one receiver for the robot. These manual commands are alternatives to the
systemd receiver, not additional nodes to start alongside it:

```bash
ros2 launch remote_controller remote_controller.launch.py DEBUG:=true
# Or run the executable directly with an explicit configuration:
ros2 run remote_controller remote_controller --DEBUG --config /path/to/xbox_default.yaml
```

`--DEBUG` (also `--debug`) enables periodic input-driver diagnostics. CRSF diagnostics
include the latest valid channel snapshot. `ros2 launch --debug` is a launch-system
option and does not enable driver diagnostics.

## Adding a driver

Implement `InputDriver` or derive from `InputDriverBase`, then register the type with
`register_input_driver_factory`. `InputDriverBase::set_signal()` synchronizes updates
with the mapper. Availability probes must not block; readiness must require an actual
safe input snapshot rather than an open device alone. Keep application button mappings
in YAML instead of hard-coding them in the transport driver.
