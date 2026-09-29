#!/usr/bin/env python3
"""Offline check: temp paths, a PTY and fake ROS; no sudo or real hardware."""

import os
import json
from pathlib import Path
import pty
import select
import signal
import shutil
import subprocess
import tempfile
import time


def check_deploy(source):
    """Exercise deployment ordering/failures with fake system commands, never sudo."""
    with tempfile.TemporaryDirectory(prefix="bxi-deploy-check-") as directory:
        root = Path(directory)
        repo = root / "repo"
        repo.mkdir()
        shutil.copytree(source, repo / "script")
        script = (source.parent / "deploy.sh").read_text()
        for prefix in ("/opt/", "/etc/", "/sys/", "/dev/", "/run/systemd/", "/usr/local/"):
            script = script.replace(prefix, f"{root}{prefix}")
        # The target guard is checked separately; allow this sandbox test as root too.
        script = script.replace("(( EUID != 0 ))", "true")
        (repo / "deploy.sh").write_text(script)
        for relative in ("opt/ros/jazzy/setup.bash", "opt/bxi/bxi_ros2_pkg/local_setup.bash",
                         "opt/bxi/bxi_ros2_pkg/lib/hardware_elf3/hardware_elf3"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(":\n")
            path.chmod(0o755)
        (root / "etc").mkdir()
        (root / "etc/os-release").write_text('ID=ubuntu\nVERSION_ID="24.04"\n')
        (root / "run/systemd/system").mkdir(parents=True)
        for name in ("can0", "can1", "can2", "can3"):
            (root / "sys/class/net" / name).mkdir(parents=True)
        for name in ("uhid", "bxi_imu", "ttyMotorPower", "input/jsBattleDragon"):
            path = root / "dev" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to("/dev/null")
        bin_dir = root / "bin"
        bin_dir.mkdir()
        mock = bin_dir / "mock"
        mock.write_text(r'''#!/usr/bin/python3
import json, os, pathlib, subprocess, sys
root = pathlib.Path(os.environ['DEPLOY_TEST_ROOT'])
mode = os.environ['DEPLOY_TEST_MODE']
name, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
with (root / 'calls').open('a') as stream:
    stream.write(json.dumps([name, *args]) + '\n')
if name == 'sudo':
    if args == ['-v']: sys.exit(0)
    name, args = pathlib.Path(args[0]).name, args[1:]
if name == 'uname': print('x86_64' if mode == 'wrong_arch' else 'aarch64')
elif name == 'pgrep':
    if mode == 'busy': print('123 hardware_elf3')
    sys.exit(0 if mode == 'busy' else (2 if mode == 'process_check_fail' else 1))
elif name == 'dpkg-query':
    if mode == 'missing_packages': sys.exit(1)
    print('install ok installed', end='')
elif name == 'systemctl':
    operation = args[0]
    if operation == 'is-active':
        if 'NetworkManager.service' in args: sys.exit(0 if mode == 'nm_conflict' else 3)
        sys.exit(0 if (root / 'active').exists() else 3)
    elif operation == 'disable' and 'ros_elf_launch.service' in args:
        (root / 'active').unlink(missing_ok=True)
        (root / 'enabled').unlink(missing_ok=True)
    elif operation == 'start':
        if mode == 'start_fail': sys.exit(1)
        (root / 'active').touch()
        (root / 'show_count').write_text('0')
    elif operation == 'enable' and 'ros_elf_launch.service' in args:
        (root / 'enabled').touch()
    elif operation == 'show':
        count = int((root / 'show_count').read_text()) + 1
        (root / 'show_count').write_text(str(count))
        print('restarted' if mode == 'restart' and count > 1 else 'current')
elif name == 'nmcli': print('100 (connected)')
elif name == 'python3':
    sys.exit(1 if mode == 'onnx_missing' and args[0] == '-c' else 0)
elif name == 'bash':
    if args == ['build.sh']: sys.exit(1 if mode == 'build_fail' else 0)
    sys.stdin.read()
    sys.exit(1 if mode == 'runtime_fail' else 0)
elif name == 'install':
    clean = []
    while args:
        arg = args.pop(0)
        if arg in ('-o', '-g'): args.pop(0)
        else: clean.append(arg)
    assert str(root) in clean[-1], clean
    subprocess.run(['/usr/bin/install', *clean], check=True)
elif name == 'ln':
    assert args[0] == '-s' and args[2].startswith(str(root))
    pathlib.Path(args[2]).symlink_to(args[1])
elif name == 'ip':
    print(json.dumps([{'ifname': 'can'+str(i), 'flags': ['UP'], 'linkinfo': {'info_data': {
        'ctrlmode': ['FD'], 'bittiming': {'bitrate': 500000 if mode == 'bad_can' else 1000000},
        'data_bittiming': {'bitrate': 5000000}}}} for i in range(4)]))
elif name == 'journalctl':
    assert '_SYSTEMD_INVOCATION_ID=current' in args, args
    if mode != 'not_ready': print('input candidate ready; accepting commands: gamepad')
    if mode == 'disconnected': print('input candidate stopped: gamepad (availability lost)')
elif name not in ('getent', 'groupadd', 'modprobe', 'udevadm', 'networkctl',
                  'bxi-battle-dragon-link', 'apt-get', 'sleep'):
    raise AssertionError('Unexpected system command: ' + name)
''')
        mock.chmod(0o755)
        for name in ("sudo", "uname", "pgrep", "dpkg-query", "systemctl", "nmcli",
                     "bash", "getent", "sleep", "ip"):
            (bin_dir / name).symlink_to(mock)
        env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DEPLOY_TEST_ROOT=directory)

        def run(mode="ok", *args, code=0):
            (root / "calls").write_text("")
            result = subprocess.run(["/bin/bash", str(repo / "deploy.sh"), *args],
                                    env=dict(env, DEPLOY_TEST_MODE=mode), text=True,
                                    capture_output=True, timeout=15)
            assert result.returncode == code, (mode, result.stdout, result.stderr)
            calls = [json.loads(line) for line in (root / "calls").read_text().splitlines()]
            assert not any(Path(call[1]).name in ("bxi-motor-power", "bxi-motor-ros", "bxi-rl-ros")
                           for call in calls if call[0] == "sudo"), calls
            return calls

        for mode, option in (("ok", "--dry-run"), ("ok", "--help")):
            assert not any(call[0] == "sudo" for call in run(mode, option))
        for mode in ("wrong_arch", "busy", "process_check_fail", "nm_conflict"):
            assert not any(call[0] == "sudo" for call in run(mode, code=1))
        prefix = root / "opt/bxi/bxi_rl_controller_ros2_example"
        prefix.symlink_to(root)
        assert not any(call[0] == "sudo" for call in run(code=1))
        prefix.unlink()
        for mode in ("ok", "ok", "missing_packages", "onnx_missing"):
            calls = run(mode)
            assert (root / "active").exists() and (root / "enabled").exists()
            assert prefix.resolve() == repo
            assert (root / "etc/systemd/system/ros_elf_launch.service").read_bytes() == (source / "ros_elf_launch.service").read_bytes()
            assert (root / "usr/local/sbin/bxi-motor-power").stat().st_mode & 0o111
            assert ["sudo", "systemctl", "enable", "ros_elf_launch.service"] == calls[-1]
            if mode == "missing_packages": assert ["sudo", "apt-get", "update"] in calls
            if mode == "onnx_missing": assert any("onnxruntime==1.23.2" in call for call in calls)
        for mode in ("build_fail", "runtime_fail", "bad_can", "start_fail", "restart", "not_ready", "disconnected"):
            calls = run(mode, code=1)
            assert not (root / "active").exists() and not (root / "enabled").exists(), mode
            if mode in ("build_fail", "runtime_fail", "bad_can"):
                assert ["sudo", "systemctl", "start", "ros_elf_launch.service"] not in calls
        (root / "dev/input/jsBattleDragon").unlink()
        calls = run(code=1)
        assert ["sudo", "systemctl", "start", "ros_elf_launch.service"] not in calls
    print("PASS: deployment dry-run, guards, repeat install, dependencies and failure cleanup (mocked system)")


def main():
    source = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="bxi-nvidia-check-") as directory:
        root = Path(directory)
        env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", CHECK_DIR=directory,
                   BXI_MOTOR_POWER_OPEN_DELAY="0", BXI_MOTOR_POWER_ON_DELAY="0")
        replacements = {
            "/usr/local/bin/bxi-motor-ros": str(root / "bxi-motor-ros"),
            "/usr/local/sbin/bxi-motor-power": str(root / "fake-power"),
            "/run/lock/": f"{root}/",
            "/opt/ros/jazzy/setup.bash": str(root / "setup.bash"),
            "/opt/bxi/bxi_ros2_pkg/local_setup.bash": str(root / "setup.bash"),
            "/opt/bxi/bxi_rl_controller_ros2_example/install/local_setup.bash": str(root / "setup.bash"),
        }
        for name in ("bxi-motor-power", "bxi-motor-ros", "bxi-rl-ros"):
            subprocess.run(["bash", "-n", str(source / name)], check=True)
            text = (source / name).read_text()
            for old, new in replacements.items():
                text = text.replace(old, new)
            path = root / name
            path.write_text(text)
            path.chmod(0o755)

        def run(name, *args, code=0, **variables):
            result = subprocess.run([str(root / name), *args], env=dict(env, **variables),
                                    capture_output=True, text=True, timeout=12)
            assert result.returncode == code, (name, result.returncode, result.stderr)
            return result

        # Real stty/write on a pseudo-terminal; never resolve a hardware device.
        master, slave = pty.openpty()
        try:
            for action, expected in (("on", b"o"), ("off", b"c")):
                run("bxi-motor-power", action, BXI_MOTOR_POWER_DEVICE=os.ttyname(slave))
                assert select.select([master], [], [], 1)[0], "missing serial byte"
                assert os.read(master, 16) == expected
            run("bxi-motor-power", "invalid", code=2)
            run("bxi-motor-power", "on", code=1, BXI_MOTOR_POWER_DEVICE=str(root / "missing"))
            assert not select.select([master], [], [], 0)[0], "unexpected serial data"
        finally:
            os.close(master)
            os.close(slave)

        (root / "setup.bash").write_text(":\n")
        (root / "fake-power").write_text(
            '#!/bin/bash\necho "$1" >> "$CHECK_DIR/events"\n'
            '[[ "$1" != on || "${CHECK_MODE:-}" != power-fail ]] || exit 7\n'
        )
        (root / "ros2").write_text(
            '#!/bin/bash\necho "ros2 $*" >> "$CHECK_DIR/events"\n'
            'case "${CHECK_MODE:-}" in\n'
            '  launch-fail) exit 7 ;;\n'
            '  hold) echo $$ > "$CHECK_DIR/group"; trap "" TERM; '
            'sleep 60 & echo $! > "$CHECK_DIR/child"; wait ;;\n'
            'esac\n'
        )
        for name in ("fake-power", "ros2"):
            (root / name).chmod(0o755)

        events = root / "events"
        for mode, expected_code in (("", 0), ("launch-fail", 7), ("power-fail", 7)):
            events.write_text("")
            run("bxi-rl-ros", code=expected_code, CHECK_MODE=mode)
            lines = events.read_text().splitlines()
            assert lines[0] == "on" and lines[-1] == "off", lines
            if mode == "power-fail":
                assert lines == ["on", "off"], lines
            else:
                assert lines[1] == (
                    "ros2 launch bxi_example_py_elf3 example_launch_unified_hw.launch.py "
                    "start_remote_controller:=false control_rate_hz:=200.0 motion_button_mode:=momentary"
                ), lines

        events.write_text("")
        run("bxi-motor-ros")
        assert events.read_text().splitlines() == [
            "on", "ros2 launch hardware_elf3 hardware_elf3_launch.py", "off"]
        events.write_text("")
        (root / "setup.bash").write_text("false\n")
        run("bxi-rl-ros", code=1)
        assert events.read_text() == "", "powered on despite failed ROS setup"
        (root / "setup.bash").write_text(":\n")

        proc = subprocess.Popen([str(root / "bxi-rl-ros")], env=dict(env, CHECK_MODE="hold"),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not (root / "child").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (root / "child").exists(), "mock ROS did not start"
            before = events.read_text()
            run("bxi-rl-ros", code=1)
            assert events.read_text() == before, "second launcher touched power"
            run("bxi-motor-ros", "--stop")
            assert proc.wait(timeout=5) == 143
            child_pid = int((root / "child").read_text())
            stat = Path(f"/proc/{child_pid}/stat")
            assert not stat.exists() or stat.read_text().split()[2] == "Z", "orphan survived"
            assert events.read_text().splitlines()[-1] == "off"
        finally:
            if (root / "group").exists():
                try:
                    os.killpg(int((root / "group").read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=5)
    print("PASS: serial o/c, setup/power/launch failures, shared lock, stop/group cleanup")
    check_deploy(source)


if __name__ == "__main__":
    main()
