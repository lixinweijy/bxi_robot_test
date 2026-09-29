"""Exit regressions using isolated ROS nodes, never hardware or motor topics."""

import os
from pathlib import Path
import subprocess
import sys


def test_controller_exit_keeps_context_until_callbacks_finish():
    program = r"""
import os
import signal
import sys
import time
from threading import Event, current_thread, main_thread
import rclpy
from rclpy.node import Node
from bxi_example_py_elf3.control.ros_runtime import run_controller

mode = sys.argv[1]
state = {"callback_finished": False, "destroyed": False}
class Probe(Node):
    def __init__(self):
        super().__init__("controller_exit_probe", enable_rosout=False,
                         start_parameter_services=False)
        self.shutdown_requested = Event()
        self.timer = self.create_timer(0.01, self.callback)

    def callback(self):
        assert current_thread() is main_thread(), "controller must use a single-threaded executor"
        self.timer.cancel()
        try:
            if mode == "error":
                raise RuntimeError("normal callback failure")
            if mode == "safety":
                self.shutdown_requested.set()
            else:
                os.kill(os.getpid(), getattr(signal, mode))
            time.sleep(0.15)
            assert rclpy.ok(), "context closed while callback was active"
        finally:
            state["callback_finished"] = True

    def destroy_node(self):
        assert state["callback_finished"], "node destroyed before callback finished"
        assert rclpy.ok(), "context closed before node cleanup"
        state["destroyed"] = True
        return super().destroy_node()

try:
    run_controller(Probe)
except RuntimeError as exc:
    assert mode == "error" and str(exc) == "normal callback failure"
else:
    assert mode != "error", "normal callback exception was swallowed"
assert state["destroyed"] and not rclpy.ok()
print("PASS", mode)
"""
    env = dict(os.environ, ROS_DOMAIN_ID="199", ROS_LOCALHOST_ONLY="0",
               ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", ROS_STATIC_PEERS="",
               PYTHONDONTWRITEBYTECODE="1")
    package_root = str(Path(__file__).parents[1])
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    for mode in ("SIGINT", "SIGTERM", "safety", "error"):
        result = subprocess.run([sys.executable, "-B", "-c", program, mode],
                                env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASS " + mode in result.stdout
        assert "context is invalid" not in result.stderr
        assert "exception was never retrieved" not in result.stderr


def test_shutdown_blocks_control_publish_and_reset():
    import ast
    from threading import Event, RLock
    from types import SimpleNamespace

    path = Path(__file__).parents[1] / "bxi_example_py_elf3" / "bxi_example_vibration.py"
    tree = ast.parse(path.read_text())
    controller = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "VibrationTestNode")
    names = {"_timer_callback", "_publish_command", "_call_robot_reset"}
    methods = [n for n in controller.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"time": SimpleNamespace(monotonic=lambda: 0.0)}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    stopped = Event()
    stopped.set()
    node = SimpleNamespace(shutdown_requested=stopped, state_lock=RLock(), safety_fault=False)
    assert namespace["_timer_callback"](node) is None
    assert namespace["_publish_command"](node, None, None, None) is False
    assert namespace["_call_robot_reset"](node, 1, False, 0.0) is False


def test_remote_start_selects_unified_controller():
    import ast
    import yaml

    source_root = Path(__file__).parents[2]
    config = yaml.safe_load((source_root / "remote_controller/config/xbox_default.yaml").read_text())
    signals = config["sources"]["gamepad"]["signals"]
    assert signals["gamepad.lb"]["from"] == "js.button.6"
    assert signals["gamepad.rb"]["from"] == "js.button.7"
    assert config["outputs"]["publish_on_change"] is False
    commands = config["system"]
    assert any("ros2 launch bxi_example_py_elf3 example_launch_unified_hw.launch.py " in command
               for command in commands["start"])
    assert any("bxi_example_py_elf3_mjlab" in command for command in commands["stop"])
    assert {"output": "btn_5=1", "when": ["button.lb_event"]} in config["outputs"]["level"]
    assert {"output": "btn_6=1", "when": ["button.rb_event"]} in config["outputs"]["level"]
    path = source_root / "bxi_example_py_elf3/bxi_example_py_elf3/bxi_example_mjlab.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert any(n.func.id == "SingleThreadedExecutor" and not n.args and not n.keywords for n in calls)
    assert not any(n.func.id == "MultiThreadedExecutor" for n in calls)
    launch = ast.parse((source_root / "bxi_example_py_elf3/launch/example_walk_hw.launch.py").read_text())
    nodes = [n for n in ast.walk(launch) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "Node"]
    assert len(nodes) == 2
    assert all(any(k.arg == "on_exit" and isinstance(k.value, ast.Call)
                   and isinstance(k.value.func, ast.Name) and k.value.func.id == "Shutdown"
                   for k in n.keywords) for n in nodes)


def test_walk_model_loads_without_constructing_robot_node():
    import ast
    from types import SimpleNamespace
    import numpy as np
    from ament_index_python.packages import get_package_share_path
    from bxi_example_py_elf3.bxi_example_mjlab import BxiExample

    probe = SimpleNamespace()
    path = get_package_share_path("bxi_example_py_elf3") / "data/model_normal.onnx"
    BxiExample.initialize_onnx(probe, str(path))
    metadata = probe.session.get_modelmeta().custom_metadata_map
    assert "joint_names" in metadata
    for key in ("joint_stiffness", "joint_damping", "action_scale", "default_joint_pos"):
        assert len(ast.literal_eval(metadata[key])) == 29
    actions = BxiExample.inference_step(probe, np.zeros((1, 96), dtype=np.float32))
    assert actions.shape == (29,) and np.all(np.isfinite(actions))


def test_walk_gravity_projection_preserves_rotation_and_validation():
    import numpy as np
    from bxi_example_py_elf3.bxi_example_mjlab import projected_gravity_from_quat

    gravity = np.array([0.0, 0.0, -1.0])
    for quaternion, expected in (
        ([0, 0, 0, 2], [0, 0, -1]),
        ([1, 0, 0, 1], [0, -1, 0]),
        ([0, 1, 0, 1], [1, 0, 0]),
        ([0, 0, 1, 1], [0, 0, -1]),
    ):
        np.testing.assert_allclose(projected_gravity_from_quat(quaternion, gravity), expected, atol=1e-12)
    for invalid in ([0, 0, 0, 0], [0, 0, 1], [0, 0, 0, float("nan")]):
        try:
            projected_gravity_from_quat(invalid, gravity)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid quaternion accepted")


def test_walk_accepts_31_joint_feedback_without_changing_policy_dimensions():
    from threading import Lock
    from types import SimpleNamespace
    import numpy as np
    from sensor_msgs.msg import JointState
    from bxi_example_py_elf3.bxi_example_mjlab import BxiExample, joint_name

    probe = SimpleNamespace(lock_in=Lock(), qpos=np.zeros(29), qvel=np.zeros(29))
    names = ["head_z_joint"] + list(reversed(joint_name)) + ["head_y_joint"]
    positions = {name: float(i) for i, name in enumerate(joint_name)}
    positions.update(head_z_joint=1000.0, head_y_joint=2000.0)
    message = JointState()
    message.name = names
    message.position = [positions[name] for name in names]
    message.velocity = [-positions[name] for name in names]
    BxiExample.joint_callback(probe, message)
    np.testing.assert_array_equal(probe.qpos, np.arange(29))
    np.testing.assert_array_equal(probe.qvel, -np.arange(29))
    assert probe.qpos.shape == probe.qvel.shape == (29,)
    legacy = JointState()
    legacy.position = list(map(float, range(29)))
    legacy.velocity = list(map(float, range(29)))
    BxiExample.joint_callback(probe, legacy)
    np.testing.assert_array_equal(probe.qvel, np.arange(29))

    for bad in (
        SimpleNamespace(name=[], position=list(message.position), velocity=list(message.velocity)),
        SimpleNamespace(name=names[:-1], position=list(message.position), velocity=list(message.velocity)),
        SimpleNamespace(name=[names[0]] + names[:-1], position=list(message.position), velocity=list(message.velocity)),
        SimpleNamespace(name=["unknown"] + list(joint_name[1:]), position=list(legacy.position), velocity=list(legacy.velocity)),
        SimpleNamespace(name=names, position=list(message.position), velocity=list(message.velocity)[:-1]),
        SimpleNamespace(name=[], position=[float("nan")] + list(legacy.position)[1:], velocity=list(legacy.velocity)),
    ):
        before = (probe.qpos.copy(), probe.qvel.copy())
        try:
            BxiExample.joint_callback(probe, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid feedback accepted")
        np.testing.assert_array_equal(probe.qpos, before[0])
        np.testing.assert_array_equal(probe.qvel, before[1])


def test_walk_buttons_toggle_once_per_press_with_idle_heartbeat():
    import ast
    from threading import Lock
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from bxi_example_py_elf3 import bxi_example_mjlab as walk

    logs = []
    probe = SimpleNamespace(lock_in=Lock(), walk_test_mode=0, sprint_remote_mode=False,
                            shuttle_started_at=0.0, vx=0.0, vy=0.0, dyaw=0.0,
                            get_logger=lambda: SimpleNamespace(info=logs.append))
    # Use the real constructor's button settings without constructing a ROS node.
    tree = ast.parse(Path(walk.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BxiExample")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignments = [n for n in init.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr in
                           ("shuttle_button", "sprint_button") for t in n.targets)]
    assert len(assignments) == 2
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(walk.__file__), "exec"),
         {"self": probe, "RemoteButtonEdge": walk.RemoteButtonEdge})

    def feed(now, lb=0, rb=0):
        message = SimpleNamespace(btn_5=lb, btn_6=rb,
                                  vel_des=SimpleNamespace(x=0.0, y=0.0), yawdot_des=0.0)
        with patch.object(walk.time, "monotonic", return_value=now):
            walk.BxiExample.joy_callback(probe, message)

    feed(1.0, lb=1, rb=1)  # An initially held button must not start motion.
    feed(1.1)
    for now in np.arange(1.2, 11.2, 0.01):
        feed(float(now))
    assert not logs and probe.walk_test_mode == 0 and not probe.sprint_remote_mode
    feed(11.21, lb=1)
    assert probe.walk_test_mode == 1
    feed(11.22, lb=1)
    feed(11.23)
    assert probe.walk_test_mode == 1 and len(logs) == 1
    feed(11.24, lb=1)
    feed(11.25)
    assert probe.walk_test_mode == 0 and len(logs) == 2
    feed(11.26, rb=1)
    feed(11.27, rb=1)
    feed(11.28)
    assert probe.sprint_remote_mode and len(logs) == 3
    feed(11.29, rb=1)
    feed(11.30)
    assert not probe.sprint_remote_mode and len(logs) == 4
    feed(13.0, lb=1, rb=1)  # A real publisher gap still resynchronizes safely.
    feed(13.1)
    assert probe.walk_test_mode == 0 and not probe.sprint_remote_mode and len(logs) == 4


def test_yamaxun_amp_contract_and_half_speed_shuttle():
    import hashlib
    from threading import Lock
    from types import SimpleNamespace, MethodType
    from unittest.mock import patch
    import numpy as np
    from builtin_interfaces.msg import Time
    from bxi_example_py_elf3 import bxi_example_mjlab as walk
    from bxi_example_py_elf3.control.yamaxun_joints import ELF3_ISAAC_PARAMETERS

    path = Path(__file__).parents[1] / "data/amp_terrain.onnx"
    blob = path.read_bytes()
    assert hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest() == "f5240049e219780dce176fd62c63351fe1dea230"
    probe = SimpleNamespace()
    walk.BxiExample.initialize_onnx(probe, str(path))
    assert probe.num_obs == 960 and probe.output_info.shape == [1, 32]
    probe.action = np.zeros(29, dtype=np.float32)
    native = ELF3_ISAAC_PARAMETERS
    indices = [walk.joint_name.index(name) for name in native.layout.names]
    history = []
    for step in range(12):
        q = probe.default_joint_pos + np.arange(29, dtype=np.float32) * (step / 10000.0)
        dq = np.arange(29, dtype=np.float32) * 0.01
        command = [0.5 if step % 2 == 0 else -0.5, 0.0, 0.0]
        expected = np.concatenate(([0.1, 0.2, 0.3], [0, 0, -1], command,
                                   q[indices] - native.default_position, dq[indices],
                                   probe.action[indices])).astype(np.float32)
        history = [expected.copy()] * 10 if step == 0 else history[1:] + [expected.copy()]
        actual = walk.BxiExample.build_policy_input(probe, q, dq, [0, 0, 0, 1],
                                                    [0.1, 0.2, 0.3], command, reset_history=step == 0)
        np.testing.assert_allclose(actual, np.asarray(history).reshape(1, 960), atol=1e-7)
        raw = probe.session.run([probe.output_info.name], {probe.input_info.name: actual})[0][0]
        probe.action[:] = walk.BxiExample.inference_step(probe, actual)
        np.testing.assert_allclose(probe.action[indices], raw[:29], atol=1e-7)
        target = probe.default_joint_pos + probe.action_scale * probe.action
        np.testing.assert_allclose(target[indices], native.default_position + native.action_scale * raw[:29], atol=1e-7)
        np.testing.assert_allclose(probe.joint_stiffness[indices], native.kp)
        np.testing.assert_allclose(probe.joint_damping[indices], native.kd)

    messages = []
    probe.step = 2
    probe.loop_count = 0
    probe.lock_in = Lock()
    probe.qpos, probe.qvel = probe.default_joint_pos.copy(), np.zeros(29)
    probe.quat, probe.omega = np.array([0, 0, 0, 1]), np.zeros(3)
    probe.walk_test_mode, probe.shuttle_started_at = 1, 100.0
    probe.build_policy_input = MethodType(walk.BxiExample.build_policy_input, probe)
    probe.inference_step = MethodType(walk.BxiExample.inference_step, probe)
    probe.act_pub = SimpleNamespace(publish=messages.append)
    probe.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time()))
    # Decimal subtraction can put 102.8 - 100 just below 2.8.
    # Check both sides of each boundary instead of assuming decimal arithmetic.
    for now, speed in (
        (100.0, 0.5), (101.49, 0.5), (101.5, -0.5),
        (np.nextafter(102.8, -np.inf), -0.5),
        (np.nextafter(102.8, np.inf), 0.5),
        (104.29, 0.5), (104.31, -0.5), (105.59, -0.5), (105.61, 0.5),
    ):
        with patch.object(walk.time, "monotonic", return_value=now):
            walk.BxiExample.timer_callback(probe)
        np.testing.assert_array_equal(probe.input_buffer.reshape(10, 96)[-1, 6:9], [speed, 0, 0])
        assert tuple(messages[-1].actuators_name) == walk.joint_name
        assert len(messages[-1].pos) == 29 and np.all(np.isfinite(messages[-1].pos))


if __name__ == "__main__":
    test_controller_exit_keeps_context_until_callbacks_finish()
    test_shutdown_blocks_control_publish_and_reset()
    test_remote_start_selects_unified_controller()
    test_walk_model_loads_without_constructing_robot_node()
    test_walk_gravity_projection_preserves_rotation_and_validation()
    test_walk_accepts_31_joint_feedback_without_changing_policy_dimensions()
    test_walk_buttons_toggle_once_per_press_with_idle_heartbeat()
    test_yamaxun_amp_contract_and_half_speed_shuttle()
    print("PASS: SIGINT, SIGTERM, safety exit and callback error propagation")


def test_b_two_motion_loop_stop_and_return_to_a():
    from types import SimpleNamespace, MethodType
    from threading import RLock
    import inspect
    import numpy as np
    from bxi_example_py_elf3.bxi_example_suspended_tests import SuspendedTestNode, ARM_TEST_GROUPS
    from bxi_example_py_elf3.control.elf3 import DOF_NUM, JOINT_NAMES
    from bxi_example_py_elf3.control.limb_sequence import LIMB_TEST_GROUPS, build_safe_ranges
    from bxi_example_py_elf3.control.remote import RemoteButtonEdge

    assert 'self.arm_load_test_button = RemoteButtonEdge(\n            "momentary"' in inspect.getsource(SuspendedTestNode)
    button = RemoteButtonEdge("momentary", .5)
    assert [button.update(v, t) for v,t in ((0,1.),(1,1.1),(1,1.2),(0,1.3),(1,1.4))] == [False, True, False, False, True]
    errors, logs, returns = [], [], []
    state = SimpleNamespace(
        feedback_lock=RLock(), measured_positions=np.zeros(DOF_NUM),
        last_command_positions=np.zeros(DOF_NUM), limb_test_center_positions=np.zeros(DOF_NUM),
        center_positions=np.zeros(DOF_NUM), limb_test_start_tolerance_rad=np.deg2rad(10),
        limb_test_move_sec=2.0, whole_body_test_move_sec=.9, limb_test_range_speed_deg_s=180.,
        limb_test_target_ranges=build_safe_ranges(), limb_test_group_index=0, limb_test_failures=0,
        limb_test_segment_start=np.zeros(DOF_NUM), limb_test_segment_target=np.zeros(DOF_NUM),
        _joint_feedback_ready=lambda now: True,
        get_logger=lambda: SimpleNamespace(error=errors.append),
        _queue_diagnostic_log=lambda level,msg: logs.append(msg),
        _begin_smooth_return_locked=lambda owner,duration_sec: returns.append((owner,duration_sec)),
    )
    for name in ("_limb_segment_duration", "_load_limb_group_locked"):
        setattr(state, name, MethodType(getattr(SuspendedTestNode, name), state))
    assert SuspendedTestNode._prepare_limb_test_locked(state, "remote B", ARM_TEST_GROUPS)
    assert state.active_limb_test_groups == ARM_TEST_GROUPS
    for side in ("l", "r"):
        assert state.limb_test_center_positions[JOINT_NAMES.index(side+"_elbow_y_joint")] == np.pi/2
    state.last_command_positions[:] = state.limb_test_center_positions
    state._load_limb_group_locked(1.)
    assert state.limb_test_segment_duration_sec == 2.0
    for tick in range(18):  # Three complete cycles, same state machine as A.
        assert len(state.limb_test_waypoints) == (2 if state.limb_test_group_index == 0 else 4)
        state.last_command_positions[:] = state.limb_test_segment_target
        SuspendedTestNode._start_next_limb_segment_locked(state, 2.+tick)
    assert state.limb_test_group_index == 0 and state.limb_test_segment_index == 0
    assert sum("CYCLE COMPLETE" in msg for msg in logs) == 3
    SuspendedTestNode._stop_limb_test_locked(state, "remote B")
    assert not state.limb_test_running and state.pending_mode == ""
    assert returns[-1][0] == "limb_test_stop"
    np.testing.assert_array_equal(state.center_positions, state.limb_test_center_positions)
    state.measured_positions[:] = state.last_command_positions
    assert SuspendedTestNode._prepare_limb_test_locked(state, "remote A", LIMB_TEST_GROUPS)
    np.testing.assert_array_equal(state.limb_test_center_positions, np.zeros(DOF_NUM))
    assert state.active_limb_test_groups == LIMB_TEST_GROUPS
    state.measured_positions[JOINT_NAMES.index("l_elbow_y_joint")] += np.deg2rad(11)
    assert not SuspendedTestNode._prepare_limb_test_locked(state, "remote B", ARM_TEST_GROUPS)
    assert errors[-1].startswith("remote B rejected")
