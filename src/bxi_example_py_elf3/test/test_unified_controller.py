"""Exercise the real unified node in an isolated ROS domain, without hardware."""

import os
from pathlib import Path
import subprocess
import sys


def test_unified_controller_modes_and_safety():
    env = dict(os.environ, ROS_DOMAIN_ID='198', ROS_LOCALHOST_ONLY='0',
               ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST', ROS_STATIC_PEERS='',
               PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(Path(__file__).parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-B', __file__, 'probe'], env=env,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS unified' in result.stdout


def probe():
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    import rclpy
    from communication.msg import MotionCommands
    from sensor_msgs.msg import JointState, Imu
    from std_srvs.srv import SetBool
    from bxi_example_py_elf3.bxi_example_unified import UnifiedControlNode
    from bxi_example_py_elf3.control.elf3 import JOINT_NAMES, JOINT_NOMINAL_POS

    rclpy.init(args=['--ros-args', '-p', 'topic_prefix:=unified_probe/',
                    '-r', 'motion_commands:=unified_probe/motion_commands',
                    '-p', 'joint_test_required:=false', '-p', 'control_rate_hz:=200.0',
                    '-p', 'initialization_sec:=0.1', '-p', 'amplitude_rad:=0.23',
                    '-p', 'start_frequency_hz:=10.0', '-p', 'end_frequency_hz:=20.0',
                    '-p', 'limb_test_range_speed_deg_s:=180.0',
                    '-p', 'require_joint_state:=true', '-p', 'log_csv_path:=/tmp/unified_probe.csv'])
    n = UnifiedControlNode()
    for timer in n.timers:
        timer.cancel()
    # No hardware reset service is invoked. Exercise both acknowledgements.
    resets = []
    def reset(req):
        resets.append((req.reset_step, req.release))
        return SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(is_success=True))
    n.reset_client = SimpleNamespace(service_is_ready=lambda: True, call_async=reset)
    commands = []
    actual_publish = n.actuator_pub.publish
    def publish(msg):
        commands.append(msg)
        actual_publish(msg)
    n.actuator_pub.publish = publish
    now = 100.0
    def tick(button='', feedback=True, imu=True, remote=True, velocity=0., position_offset=0.):
        nonlocal now
        now += .01
        with patch('time.monotonic', return_value=now):
            if feedback:
                joint = JointState()
                # Deliberately reordered hardware feedback with two extra head joints.
                names = ['head_z_joint'] + list(reversed(JOINT_NAMES)) + ['head_y_joint']
                values = dict(zip(JOINT_NAMES, n.last_command_positions))
                values['l_elbow_y_joint'] += position_offset
                joint.name = names
                joint.position = [float(values.get(name, 0.)) for name in names]
                joint.velocity = [velocity] * 31
                n._joint_callback(joint)
            if imu:
                msg = Imu(); msg.orientation.w = 1.
                n._imu_callback(msg)
            if remote:
                msg = MotionCommands()
                if button:
                    setattr(msg, button, 1)
                msg.vel_des.x = 1.; msg.vel_des.y = .1
                n._motion_callback(msg)
            n._timer_callback()
        return n.mode, n.phase
    def until(predicate, limit=2000):
        for _ in range(limit):
            tick()
            assert not n.shutdown_requested.is_set() and not n.safety_fault
            if predicate():
                return
        raise AssertionError((n.mode, n.phase, n.queued_mode, n.pending_mode))
    def active(mode):
        return n.mode == mode and n.phase == 'active' and not n.returning_to_center
    def press(button):
        tick(button); tick()
    try:
        assert n.mode == 'pd' and not n.test_enabled and not n.run_enabled
        tick('btn_10')  # Held on the first received sample must not start Y.
        until(lambda: n.reset_stage == 2)
        assert resets == [(1, False), (2, False)]
        for command in commands:
            np.testing.assert_allclose(command.pos, n.pd_positions)
        # DDS graph: this is the only actuator publisher in the isolated domain.
        for _ in range(30):
            rclpy.spin_once(n, timeout_sec=.01)
            if n.count_publishers('unified_probe/actuators_cmds') == 1:
                break
        assert n.count_publishers('unified_probe/actuators_cmds') == 1
        modes = n.BUTTON_MODES
        for button, mode in modes.items():
            press(button)
            until(lambda: active(mode))
            before = len(commands)
            for _ in range(5):
                tick(button)  # A new press stops, holding does not restart.
            tick()
            assert n.mode == 'pd' and not n.queued_mode
            until(lambda: active('pd'))
            np.testing.assert_allclose(n.last_command_positions, n.pd_positions)
            assert len(commands) > before
        # Every distinct pair must visit PD and settle before starting the next.
        for source_button, source in modes.items():
            press(source_button); until(lambda: active(source))
            for target_button, target in modes.items():
                if source == target:
                    continue
                press(target_button)
                assert n.mode == 'pd' and n.phase == 'return_pd' and n.queued_mode == target
                tick()
                assert n.mode == 'pd' and not n.test_enabled and not n.limb_test_running
                until(lambda: active(target))
                press(source_button); until(lambda: active(source))
            press(source_button); until(lambda: active('pd'))
        # A repeated pending button cancels the request while completing PD return.
        press('btn_8'); until(lambda: active('arm_rom'))
        press('btn_10'); assert n.queued_mode == 'vibration'
        press('btn_10'); assert not n.queued_mode
        until(lambda: active('pd'))
        # Moving feedback must never count as settled even at the target position.
        with patch.object(n, '_queue_diagnostic_log') as diagnostic:
            press('btn_5')
            for _ in range(150):
                tick(velocity=1.)
            messages = [call.args[1] for call in diagnostic.call_args_list]
            assert any('Mode request accepted: auto_walk -> auto_walk' in m for m in messages)
            waiting = [m for m in messages if 'Waiting for pose to settle:' in m]
            assert 1 <= len(waiting) <= 2  # Throttle logging outside the 200 Hz control path.
            assert any('1.000 rad/s (limit 0.200)' in m for m in waiting)
        assert n.mode == 'pd'
        with patch.object(n, '_queue_diagnostic_log') as diagnostic:
            for _ in range(150):
                tick(position_offset=n.limb_test_start_tolerance_rad + .1)
            assert n.mode == 'pd' and n.queued_mode == 'auto_walk'
            assert any('position=l_elbow_y_joint' in call.args[1]
                       for call in diagnostic.call_args_list)
            press('btn_5')
            assert not n.queued_mode
            assert any('auto_walk -> pd' in call.args[1] for call in diagnostic.call_args_list)
        press('btn_5')
        until(lambda: active('auto_walk'))
        # Preserve the existing forward/reverse schedule in the actual model input.
        for elapsed, expected in ((.2, .5), (1.6, -.5), (3.0, .5)):
            n.mode_started_at = now + .01 - elapsed
            n.next_policy_at = 0.
            tick()
            assert np.isclose(n.policy.input_buffer.reshape(10,96)[-1,6], expected)
        # Services share arbitration and repeated enable is idempotent.
        request = SetBool.Request(); request.data = True
        with patch('time.monotonic', return_value=now):
            n._enable_service_callback(request, SetBool.Response())
        assert n.phase == 'return_pd' and n.queued_mode == 'vibration'
        with patch('time.monotonic', return_value=now):
            n._enable_service_callback(request, SetBool.Response())
        assert n.queued_mode == 'vibration'
        request.data = False
        with patch('time.monotonic', return_value=now):
            n._enable_service_callback(request, SetBool.Response())
        assert not n.queued_mode
        until(lambda: active('pd'))
        request.data = True
        with patch('time.monotonic', return_value=now):
            n._limb_test_enable_service_callback(request, SetBool.Response())
        until(lambda: active('body_rom'))
        with patch('time.monotonic', return_value=now):
            n._run_enable_service_callback(request, SetBool.Response())
        assert n.phase == 'return_pd' and n.queued_mode == 'auto_walk'
        until(lambda: active('auto_walk'))
        # Vibration uses its nominal center and naturally completes into PD.
        press('btn_10'); until(lambda: active('vibration'))
        np.testing.assert_allclose(n.center_positions, JOINT_NOMINAL_POS)
        n.test_started_at = now - n.duration_sec
        tick(); assert n.mode == 'pd' and n.phase == 'return_pd'
        until(lambda: active('pd'))
        assert all(len(c.pos) == len(c.kp) == len(c.kd) == 29 for c in commands)
        assert all(np.all(np.isfinite(c.pos)) for c in commands)
        # Loss of IMU while walking must latch a fault and stop publishing.
        press('btn_6'); until(lambda: active('remote_walk'))
        np.testing.assert_allclose(n.policy.input_buffer.reshape(10,96)[-1,6:9], [2.5,.2,0.])
        for _ in range(25):
            tick(imu=False)
        assert n.safety_fault
        count = len(commands)
        tick(); press('btn_5')
        assert len(commands) == count and not n.queued_mode
    finally:
        n.destroy_node()
        rclpy.shutdown()
    print('PASS unified: five toggles, 20 directed switches, PD settling, services, 29/31 mapping, AMP, IMU fault')


if __name__ == '__main__':
    probe()


def test_launch_shutdown_runs_power_off_hook(tmp_path):
    import importlib.util
    from unittest.mock import patch
    from launch import LaunchDescription, LaunchService
    from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
    from launch.event_handlers import OnProcessExit, OnShutdown
    from launch.events import Shutdown

    path = Path(__file__).parents[1] / 'launch/example_launch_vibration_hw.launch.py'
    spec = importlib.util.spec_from_file_location('power_off_launch_probe', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    description = module.generate_launch_description()
    cleanup = next(action for action in description.entities
                   if isinstance(action, RegisterEventHandler)
                   and isinstance(action.event_handler, OnShutdown))
    real_run = subprocess.run
    marker = tmp_path / 'motor-off'
    def fake_motor_power(cmd, **kwargs):
        assert cmd == ['/usr/local/sbin/bxi-motor-power', 'off']
        return real_run([sys.executable, '-c',
                         'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("off")',
                         str(marker)], **kwargs)
    # Both normal exit and an uncatchable controller kill must run actual cleanup.
    for body in ('pass', 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)'):
        process = ExecuteProcess(cmd=[sys.executable, '-c', body])
        service = LaunchService()
        service.include_launch_description(LaunchDescription([
            cleanup,
            RegisterEventHandler(OnProcessExit(target_action=process,
                on_exit=[EmitEvent(event=Shutdown(reason='mock controller exited'))])),
            process,
        ]))
        with patch.object(module.subprocess, 'run', side_effect=fake_motor_power):
            service.run()
        assert marker.read_text() == 'off'
        marker.unlink()
