# Script directory

Deployment and operation instructions are in the [repository README](../README.md).

| File | Purpose |
| --- | --- |
| [ros_elf_launch.service](ros_elf_launch.service) | NVIDIA/Jazzy receiver service; runs as root with Domain 0, Fast DDS, and localhost discovery. |
| [bxi-motor-power](bxi-motor-power) | Sends the serial power board an `o` or `c` byte at 921600 baud; supports device and timing overrides. |
| [bxi-motor-ros](bxi-motor-ros) | Owns the motor-power lock and ROS process group; powers on before launch, cleans up and powers off on exit, and provides `--stop`. |
| [bxi-rl-ros](bxi-rl-ros) | Starts the unified ELF3 controller through `bxi-motor-ros` at 200 Hz without a second receiver. |
| [bxi-dev.rules](bxi-dev.rules) | Creates stable IMU, serial-power, and gamepad device names and sets device permissions. |
| [bxi-battle-dragon-link](bxi-battle-dragon-link) | Maintains `/dev/input/jsBattleDragon` when matching controllers connect or disconnect. |
| [bxi-hid.conf](bxi-hid.conf) | Loads `uhid` at boot for Bluetooth HID input registration. |
| [80-bxi-can.network](80-bxi-can.network) | Configures can0-can3 for CAN-FD with 1 Mbit/s arbitration and 5 Mbit/s data rates. |
| [test_nvidia_scripts.py](test_nvidia_scripts.py) | Offline checks for serial bytes, startup failures, locking, process-group shutdown, and deployment using a simulated system. |
