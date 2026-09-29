#!/bin/bash
# Deploy the NVIDIA receiver without launching the robot or switching motor power.
set -Ee -o pipefail
trap 'printf "ERROR at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

die() { echo "ERROR: $*" >&2; exit 1; }
case "${1:-}" in
  ""|--dry-run) ;;
  -h|--help)
    echo "Usage: ./deploy.sh [--dry-run]"
    echo "Run as nvidia with sudo access, motors off and the paired controller connected."
    exit 0 ;;
  *) die "Unknown option: $1" ;;
esac
[[ $# -le 1 ]] || die "Use ./deploy.sh [--dry-run]"

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$repo"
service=ros_elf_launch.service
prefix=/opt/bxi/bxi_rl_controller_ros2_example
hardware=/opt/bxi/bxi_ros2_pkg

echo "[1/5] Checking the target and existing deployment"
(( EUID != 0 )) || die "Run ./deploy.sh as nvidia, without sudo; privileged steps request sudo themselves."
[[ $(uname -m) == aarch64 ]] || die "An ARM64 NVIDIA Jetson is required."
source /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 24.04 ]] || die "Ubuntu 24.04 is required."
[[ -d /run/systemd/system ]] || die "systemd must be running."
for command in sudo systemctl pgrep flock ip dpkg-query readlink; do
  command -v "$command" >/dev/null || die "Missing system tool: $command"
done
[[ -r /opt/ros/jazzy/setup.bash ]] || die "Install ROS 2 Jazzy first."
[[ -r $hardware/local_setup.bash && -x $hardware/lib/hardware_elf3/hardware_elf3 ]] ||
  die "Install the ARM64/Jazzy hardware packages at $hardware first."
if [[ -e $prefix || -L $prefix ]]; then
  [[ $(readlink -f "$prefix") == "$repo" ]] || die "$prefix points elsewhere; resolve it before deployment."
fi
if pgrep -af '(^|/)(hardware_elf3|bxi_example_py_elf3_(unified|demo|run|mjlab|vibration|suspended_tests|test_wire))([[:space:]]|$)|^/bin/bash /usr/local/bin/bxi-motor-ros( launch|$)'; then
  die "A robot control process is running. Support the robot and press START to stop before deploying."
else
  [[ $? == 1 ]] || die "Unable to check running robot processes."
fi
for can in can0 can1 can2 can3; do
  [[ -d /sys/class/net/$can ]] || die "Missing $can; check the CAN driver."
  if command -v nmcli >/dev/null && systemctl is-active --quiet NetworkManager.service; then
    state=$(LC_ALL=C nmcli -g GENERAL.STATE device show "$can")
    [[ $state == '10 '* ]] || die "NetworkManager manages $can; assign CAN to systemd-networkd first."
  fi
done
if [[ ${1:-} == --dry-run ]]; then
  echo "Prerequisites passed. No changes made. Deployment would:"
  echo "  Stop the receiver; install missing dependencies; build this checkout."
  echo "  Install power scripts, udev/uhid/CAN rules and the receiver service."
  echo "  Check devices, model inference, CAN settings and live gamepad readiness."
  exit 0
fi

exec 8<"$repo/deploy.sh"
flock -n 8 || die "Another deployment of this checkout is running."
sudo -v
# Keep a failed/partial deployment from starting automatically at the next boot.
if systemctl cat "$service" >/dev/null 2>&1; then
  sudo systemctl disable --now "$service"
fi
receiver_started=0
cleanup() {
  status=$?
  if (( status != 0 && receiver_started )); then
    sudo systemctl disable --now "$service" || true
    echo "Receiver stopped because deployment verification failed." >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[2/5] Installing missing dependencies and building"
missing=()
for package in build-essential cmake libyaml-cpp-dev python3-colcon-common-extensions \
  python3-yaml python3-numpy python3-pip ros-jazzy-rmw-fastrtps-cpp bluez; do
  [[ $(dpkg-query -W -f='${Status}' "$package" 2>/dev/null) == 'install ok installed' ]] || missing+=("$package")
done
if (( ${#missing[@]} )); then
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi
if ! sudo /usr/bin/python3 -c 'import numpy, onnxruntime'; then
  sudo /usr/bin/python3 -m pip install --break-system-packages 'onnxruntime==1.23.2'
fi
bash build.sh

# Use the same root Python and ROS overlays as the installed service. No ROS node
# is created here; importing the controller and running one inference is offline.
sudo bash -c '
set -e
source "$1/setup_env.sh"
libraries=$(ldd -r /opt/bxi/bxi_ros2_pkg/lib/hardware_elf3/hardware_elf3 2>&1)
if [[ $libraries == *"not found"* || $libraries == *"undefined symbol"* ]]; then
  echo "$libraries" >&2
  exit 1
fi
exec /usr/bin/python3 -
' bash "$repo" <<'PY'
from pathlib import Path
import numpy as np
from ament_index_python.packages import get_package_prefix, get_package_share_path
from bxi_example_py_elf3.bxi_example_unified import WalkingPolicy, DOF_NUM

expected = Path.cwd() / 'install'
for package in ('remote_controller', 'bxi_example_py_elf3'):
    assert Path(get_package_prefix(package)).resolve() == expected.resolve(), package
assert Path(get_package_prefix('hardware_elf3')).resolve() == Path('/opt/bxi/bxi_ros2_pkg').resolve()
policy = WalkingPolicy()
policy.initialize_onnx(str(get_package_share_path('bxi_example_py_elf3') / 'data/amp_terrain.onnx'))
policy.action = np.zeros(DOF_NUM, dtype=np.float32)
policy.inference_step(np.zeros((1, policy.num_obs), dtype=np.float32))
assert np.isfinite(policy.action).all(), 'Model returned non-finite actions'
print('Root ROS imports and walking-model inference passed.')
PY

echo "[3/5] Installing scripts, device rules and startup configuration"
sudo install -d /opt/bxi /usr/local/bin /usr/local/sbin /etc/modules-load.d /etc/systemd/network
if [[ ! -e $prefix && ! -L $prefix ]]; then
  sudo ln -s "$repo" "$prefix"
fi
sudo install -o root -g root -m 0755 script/bxi-motor-power /usr/local/sbin/bxi-motor-power
sudo install -o root -g root -m 0755 \
  script/bxi-motor-ros script/bxi-rl-ros script/bxi-battle-dragon-link /usr/local/bin/
getent group imu >/dev/null || sudo groupadd --system imu
sudo install -D -o root -g root -m 0644 script/bxi-dev.rules /etc/udev/rules.d/bxi-dev.rules
sudo install -D -o root -g root -m 0644 script/bxi-hid.conf /etc/modules-load.d/bxi-hid.conf
sudo install -D -o root -g root -m 0644 script/80-bxi-can.network /etc/systemd/network/80-bxi-can.network
sudo install -D -o root -g root -m 0644 script/ros_elf_launch.service /etc/systemd/system/ros_elf_launch.service
sudo systemctl daemon-reload
sudo modprobe uhid
sudo systemctl enable --now bluetooth.service systemd-networkd.service
sudo udevadm control --reload-rules
for subsystem in input tty iio; do
  sudo udevadm trigger --action=change --subsystem-match="$subsystem"
done
sudo udevadm settle
sudo /usr/local/bin/bxi-battle-dragon-link
sudo networkctl reload
sudo networkctl reconfigure can0 can1 can2 can3

echo "[4/5] Checking devices and CAN (keep controller buttons released)"
for device in /dev/uhid /dev/bxi_imu /dev/ttyMotorPower /dev/input/jsBattleDragon; do
  [[ -c $device ]] || die "Missing $device. Check the hardware; reconnect the paired gamepad after loading uhid, then rerun."
done
# networkd applies configuration asynchronously; allow it to settle.
sleep 2
ip -details -json link show type can | /usr/bin/python3 -c '
import json, sys
links = {item["ifname"]: item for item in json.load(sys.stdin)}
for name in ("can0", "can1", "can2", "can3"):
    link = links.get(name, {})
    data = link.get("linkinfo", {}).get("info_data", {})
    assert "UP" in link.get("flags", []), name + " is down"
    assert "FD" in data.get("ctrlmode", []), name + " has CAN-FD disabled"
    assert data.get("bittiming", {}).get("bitrate") == 1000000, name + " has wrong arbitration rate"
    assert data.get("data_bittiming", {}).get("bitrate") == 5000000, name + " has wrong data rate"
print("can0-can3: UP, CAN-FD, 1M/5M")
'

echo "[5/5] Starting and verifying the receiver"
receiver_started=1
sudo systemctl start "$service"
invocation=$(systemctl show "$service" -p InvocationID --value)
[[ -n $invocation ]] || die "The receiver has no systemd invocation ID."
ready=0
for (( attempt=0; attempt<20; attempt++ )); do
  [[ $(systemctl show "$service" -p InvocationID --value) == "$invocation" ]] || die "The receiver restarted during verification."
  output=$(sudo journalctl "_SYSTEMD_INVOCATION_ID=$invocation" --no-pager -o cat)
  state=$(printf '%s\n' "$output" | grep -E 'input candidate (ready; accepting commands: gamepad|stopped: gamepad)' || true)
  if [[ ${state##*$'\n'} == *'input candidate ready; accepting commands: gamepad'* ]]; then
    ready=$((ready + 1))
    (( ready >= 3 )) && break
  else
    ready=0
  fi
  sleep 1
done
(( ready >= 3 )) || die "Gamepad not ready. Check its connection; see: sudo journalctl -u $service -n 50"
systemctl is-active --quiet "$service"
sudo systemctl enable "$service"
echo "Deployment complete: receiver ready and enabled at boot."
echo "Press the right stick (R3) to initialize; press START to stop and turn power off."
echo "Motor power and motion still require the supported-robot start/stop test."
