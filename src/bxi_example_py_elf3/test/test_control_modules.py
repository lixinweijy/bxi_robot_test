import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bxi_example_py_elf3.control.elf3 import (
    DOF_NUM,
    JOINT_KD,
    JOINT_KP,
    JOINT_NAMES,
    JOINT_NOMINAL_POS,
    JOINT_POSITION_MAX,
    JOINT_POSITION_MIN,
    SUSPENDED_RUN_NOMINAL_POS,
    position_limit_violations,
)
from bxi_example_py_elf3.control.remote import RemoteButtonEdge
from bxi_example_py_elf3.control.limb_sequence import (
    WHOLE_BODY_TEST_GROUPS,
    LIMB_TEST_GROUPS,
    build_safe_ranges,
    compact_posture,
    full_range_waypoints,
    velocity_limited_duration,
)
from bxi_example_py_elf3.control.trajectory import (
    JointTrajectory,
    minimum_jerk_progress,
)
from bxi_example_py_elf3.suspended_states import (
    SuspendedLimbTestState,
    SuspendedRunningState,
    SuspendedVibrationState,
    create_button_states,
)


def test_shared_joint_configuration_is_consistent():
    assert DOF_NUM == 29
    assert len(JOINT_NAMES) == DOF_NUM
    for vector in (
        JOINT_KP,
        JOINT_KD,
        JOINT_NOMINAL_POS,
        JOINT_POSITION_MIN,
        JOINT_POSITION_MAX,
        SUSPENDED_RUN_NOMINAL_POS,
    ):
        assert vector.shape == (DOF_NUM,)
        assert np.all(np.isfinite(vector))
    assert not position_limit_violations(JOINT_NOMINAL_POS)
    assert SUSPENDED_RUN_NOMINAL_POS[16] == 0.2
    assert SUSPENDED_RUN_NOMINAL_POS[23] == -0.2


def test_toggle_button_ignores_initial_state_and_activates_on_each_change():
    button = RemoteButtonEdge("toggle", resync_sec=0.5)
    assert not button.update(1, 1.0)
    assert not button.update(1, 1.1)
    assert button.update(0, 1.2)
    assert not button.update(0, 1.3)
    assert button.update(1, 1.4)


def test_momentary_button_only_activates_on_rising_edge():
    button = RemoteButtonEdge("momentary", resync_sec=0.5)
    assert not button.update(0, 1.0)
    assert button.update(1, 1.1)
    assert not button.update(1, 1.2)
    assert not button.update(0, 1.3)


def test_button_resynchronizes_after_publisher_gap():
    button = RemoteButtonEdge("toggle", resync_sec=0.5)
    assert not button.update(0, 1.0)
    assert button.update(1, 1.1)
    assert not button.update(0, 2.0)
    assert button.update(1, 2.1)


def test_x_and_y_use_independent_toggle_edges():
    run_button = RemoteButtonEdge("toggle", resync_sec=0.5)
    vibration_button = RemoteButtonEdge("toggle", resync_sec=0.5)
    assert not run_button.update(0, 1.0)
    assert not vibration_button.update(0, 1.0)
    assert run_button.update(1, 1.1)
    assert not vibration_button.update(0, 1.1)
    assert not run_button.update(1, 1.2)
    assert vibration_button.update(1, 1.2)


def test_suspended_button_states_have_independent_remote_fields():
    states = create_button_states()
    assert isinstance(states[0], SuspendedRunningState)
    assert isinstance(states[1], SuspendedVibrationState)
    assert isinstance(states[2], SuspendedLimbTestState)
    assert [state.button for state in states] == ["X", "Y", "A"]
    assert [state.message_field for state in states] == [
        "btn_9",
        "btn_10",
        "btn_7",
    ]
    assert len({state.name for state in states}) == len(states)
    assert len({state.message_field for state in states}) == len(states)
    state_file = Path(inspect.getsourcefile(SuspendedRunningState))
    assert "mods" in state_file.parts
    assert "com.bxi.suspended_tests" in state_file.parts


def test_minimum_jerk_blend_has_clamped_endpoints_and_is_monotonic():
    samples = [minimum_jerk_progress(index / 20.0) for index in range(21)]
    assert minimum_jerk_progress(-1.0) == 0.0
    assert minimum_jerk_progress(2.0) == 1.0
    assert samples == sorted(samples)


def test_whole_body_full_range_order_and_mirrored_motion():
    groups = WHOLE_BODY_TEST_GROUPS
    assert [group.category for group in groups].count("arms") == 7
    assert [group.category for group in groups].count("torso") == 3
    assert [group.category for group in groups].count("legs") == 6
    assert groups[0].joint_names == (
        "l_wrist_z_joint",
        "r_wrist_z_joint",
    )
    assert groups[-1].joint_names == (
        "l_hip_y_joint",
        "r_hip_y_joint",
    )
    safe_ranges = build_safe_ranges()
    motion_names, wrist_waypoints = full_range_waypoints(
        JOINT_NOMINAL_POS,
        groups[0],
        safe_ranges,
    )
    left = JOINT_NAMES.index("l_wrist_z_joint")
    right = JOINT_NAMES.index("r_wrist_z_joint")
    assert motion_names == groups[0].joint_names
    assert np.rad2deg(wrist_waypoints[0][left]) == -40.0
    assert np.rad2deg(wrist_waypoints[0][right]) == 40.0
    assert np.rad2deg(wrist_waypoints[1][left]) == 40.0
    assert np.rad2deg(wrist_waypoints[1][right]) == -40.0


def test_full_range_shoulder_motion_uses_compact_elbows_and_unfolds():
    safe_ranges = build_safe_ranges()
    shoulder_group = WHOLE_BODY_TEST_GROUPS[6]
    motion_names, waypoints = full_range_waypoints(
        np.zeros(DOF_NUM), shoulder_group, safe_ranges
    )
    left_elbow = JOINT_NAMES.index("l_elbow_y_joint")
    right_elbow = JOINT_NAMES.index("r_elbow_y_joint")
    assert compact_posture(shoulder_group.joint_names) == {
        "l_elbow_y_joint": np.deg2rad(-45.0),
        "r_elbow_y_joint": np.deg2rad(-45.0),
    }
    assert "l_elbow_y_joint" in motion_names
    assert np.rad2deg(waypoints[0][left_elbow]) == -45.0
    assert np.rad2deg(waypoints[0][right_elbow]) == -45.0
    np.testing.assert_allclose(waypoints[-1], np.zeros(DOF_NUM))


def test_full_range_duration_limits_minimum_jerk_peak_speed():
    start = np.zeros(DOF_NUM)
    target = start.copy()
    target[JOINT_NAMES.index("waist_z_joint")] = np.deg2rad(160.0)
    duration = velocity_limited_duration(
        start,
        target,
        ("waist_z_joint",),
        minimum_move_sec=1.5,
        range_speed_deg_s=20.0,
    )
    assert duration == 15.0


def test_trajectory_rejects_position_limit_violation():
    positions = JOINT_NOMINAL_POS.reshape(1, -1).copy()
    positions[0, 0] = JOINT_POSITION_MAX[0] + 0.01
    try:
        JointTrajectory(positions, "test", 50.0)
    except ValueError as exc:
        assert JOINT_NAMES[0] in str(exc)
    else:
        raise AssertionError("position-limit violation was accepted")


def test_trajectory_playback_wraps_without_aliasing_source_data():
    positions = np.vstack(
        [JOINT_NOMINAL_POS, JOINT_NOMINAL_POS + 0.001]
    )
    trajectory = JointTrajectory(positions, "test", 50.0)
    assert trajectory.next_is_first_frame
    first = trajectory.next()
    assert not trajectory.next_is_first_frame
    first[0] = 99.0
    np.testing.assert_allclose(trajectory.next(), positions[1])
    assert trajectory.next_is_first_frame
    np.testing.assert_allclose(trajectory.next(), positions[0])


def test_a_rom_is_faster_without_changing_b_or_peak_speed():
    # Load only the pure duration method; never construct a ROS/hardware node.
    path = Path(__file__).parents[1] / "bxi_example_py_elf3" / "bxi_example_suspended_tests.py"
    tree = ast.parse(path.read_text())
    controller = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "SuspendedTestNode")
    method = next(n for n in controller.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_limb_segment_duration")
    namespace = dict(velocity_limited_duration=velocity_limited_duration,
                     LIMB_TEST_GROUPS=LIMB_TEST_GROUPS)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    duration = namespace["_limb_segment_duration"]
    arms = tuple(g for g in LIMB_TEST_GROUPS if g.category == "arms")
    state = SimpleNamespace(
        limb_test_segment_start=np.zeros(DOF_NUM),
        limb_test_segment_target=np.zeros(DOF_NUM),
        limb_test_motion_names=("l_wrist_x_joint",),
        whole_body_test_move_sec=0.9, limb_test_move_sec=1.5,
        limb_test_range_speed_deg_s=180.0,
        active_limb_test_groups=LIMB_TEST_GROUPS,
    )
    index = JOINT_NAMES.index("l_wrist_x_joint")
    state.limb_test_segment_target[index] = np.deg2rad(30.0)
    assert duration(state) == 0.9
    state.active_limb_test_groups = arms
    assert duration(state) == 1.5
    state.limb_test_segment_target[index] = np.deg2rad(310.0)
    for groups in (LIMB_TEST_GROUPS, arms):
        state.active_limb_test_groups = groups
        assert np.isclose(1.875 * 310.0 / duration(state), 180.0)


def test_a_rom_only_moves_arms_and_legs():
    assert len(LIMB_TEST_GROUPS) == 13
    assert sum(g.category == "arms" for g in LIMB_TEST_GROUPS) == 7
    assert sum(g.category == "legs" for g in LIMB_TEST_GROUPS) == 6
    waist_indices = [i for i, name in enumerate(JOINT_NAMES) if name.startswith("waist_")]
    center = np.zeros(DOF_NUM)
    ranges = build_safe_ranges(collision_margin_deg=10.0, mechanical_margin_deg=2.0)
    total = 0.0
    segments = 0
    current = center.copy()
    for group in LIMB_TEST_GROUPS:
        names, waypoints = full_range_waypoints(center, group, ranges)
        assert not any(name.startswith("waist_") for name in names)
        for target in waypoints:
            np.testing.assert_array_equal(target[waist_indices], center[waist_indices])
            total += velocity_limited_duration(current, target, names, 0.9, 180.0) + 0.2
            segments += 1
            current = target
    assert segments == 51
    assert np.isclose(total, 75.00104166666667)


def test_only_suspended_hardware_controller_uses_local_discovery():
    launch_dir = Path(__file__).parents[1] / "launch"
    base = ast.parse((launch_dir / "example_launch_vibration_hw.launch.py").read_text())
    function = next(n for n in base.body if isinstance(n, ast.FunctionDef)
                    and n.name == "generate_launch_description")
    defaults = dict(zip([a.arg for a in function.args.args][-len(function.args.defaults):],
                        function.args.defaults))
    assert ast.literal_eval(defaults["controller_localhost_only"]) is False
    assignment = next(n for n in ast.walk(function) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "vibration_node"
                              for t in n.targets))
    env = next(k.value for k in assignment.value.keywords if k.arg == "additional_env")
    expression = compile(ast.Expression(env), "controller_env", "eval")
    assert eval(expression, {"controller_localhost_only": True}) == {
        "ROS_LOCALHOST_ONLY": "0", "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
    }
    assert eval(expression, {"controller_localhost_only": False}) is None
    wrapper = ast.parse((launch_dir / "example_launch_suspended_tests_hw.launch.py").read_text())
    call = next(n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "generate_launch_description")
    assert any(k.arg == "controller_localhost_only" and ast.literal_eval(k.value) is True
               for k in call.keywords)


def test_b_load_waypoints_match_xml_geometry_and_leave_a_unchanged():
    import xml.etree.ElementTree as ET
    from bxi_example_py_elf3.control.limb_sequence import (
        ARM_LOAD_TEST_GROUPS, arm_load_rest_pose, arm_load_waypoints,
    )
    xml = ET.parse(Path(__file__).parents[1] / "data/elf3.xml")

    def directions(pose, side):
        # Read the actual model's axes and link offsets; no ROS or motors.
        body = xml.find(".//body[@name='%s_shoulder_y_link']" % side)
        rotation, position, origins = np.eye(3), np.zeros(3), []
        for suffix in ("shoulder_y", "shoulder_x", "shoulder_z", "elbow_y", "wrist_x"):
            assert body.attrib["name"] == side + "_" + suffix + "_link"
            assert not any(k in body.attrib for k in ("quat", "euler", "axisangle"))
            position = position + rotation @ np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            origins.append(position.copy())
            joint = body.find("joint")
            axis = np.fromstring(joint.attrib["axis"], sep=" ")
            axis /= np.linalg.norm(axis)
            x, y, z = axis
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            angle = pose[JOINT_NAMES.index(joint.attrib["name"])]
            rotation = rotation @ (np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew))
            body = body.find("body")
        upper, fore = origins[3]-origins[0], origins[4]-origins[3]
        return upper/np.linalg.norm(upper), fore/np.linalg.norm(fore), rotation[:, 2]

    base = JOINT_NOMINAL_POS.copy()
    # Prove helpers preserve non-moving joint commands, not just zero values.
    base[JOINT_NAMES.index("l_wrist_z_joint")] = 0.1
    unchanged = base.copy()
    rest = arm_load_rest_pose(base)
    assert len(ARM_LOAD_TEST_GROUPS) == 2
    all_points = []
    seconds = 0.0
    current = rest
    for group in ARM_LOAD_TEST_GROUPS:
        names, points = arm_load_waypoints(rest, group)
        indices = [JOINT_NAMES.index(n) for n in names]
        other = [i for i in range(DOF_NUM) if i not in indices]
        assert all("wrist" not in n for n in names)
        for target in points:
            np.testing.assert_array_equal(target[other], base[other])
            assert not position_limit_violations(target, np.deg2rad(2.0))
            seconds += velocity_limited_duration(current, target, names, 2.0, 180.0) + 0.2
            for fraction in np.linspace(0, 1, 31):
                pose = current + minimum_jerk_progress(fraction) * (target-current)
                assert not position_limit_violations(pose, np.deg2rad(2.0))
            current = target
        np.testing.assert_array_equal(points[-1], rest)
        all_points.extend(points)
    assert len(all_points) == 6
    assert abs(seconds - 13.2) < 1e-9
    for side, sign in (("l", 1), ("r", -1)):
        upper, fore, normal = directions(rest, side)
        np.testing.assert_allclose(upper, (0, sign*np.sin(np.deg2rad(10)), -np.cos(np.deg2rad(10))), atol=1e-12)
        np.testing.assert_allclose(fore, upper, atol=1e-12)
        upper, fore, normal = directions(all_points[0], side)
        np.testing.assert_allclose(normal, (0, -sign*np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10))), atol=1e-12)
        # Command adds 10 degrees above model-horizontal; physical level needs calibration.
        np.testing.assert_allclose(upper, (0, sign*np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10))), atol=1e-12)
        np.testing.assert_allclose(fore, upper, atol=1e-12)
        diagonal = (.5, sign*.5, -np.sqrt(.5))
        upper, fore, _ = directions(all_points[2], side)
        np.testing.assert_allclose(upper, diagonal, atol=1e-12)
        np.testing.assert_allclose(fore, upper, atol=1e-12)
        upper_bent, fore_bent, _ = directions(all_points[3], side)
        np.testing.assert_allclose(upper_bent, diagonal, atol=1e-12)
        np.testing.assert_allclose(fore_bent, (0, 0, 1), atol=1e-12)
        internal_angle = np.rad2deg(np.arccos(np.clip(np.dot(-upper_bent, fore_bent), -1, 1)))
        assert abs(internal_angle - 45.0) < 1e-10
    np.testing.assert_array_equal(base, unchanged)
    assert len(LIMB_TEST_GROUPS) == 13
    assert sum(len(full_range_waypoints(np.zeros(DOF_NUM), g, build_safe_ranges())[1])
               for g in LIMB_TEST_GROUPS) == 51
    for bad_center in (np.full(DOF_NUM, np.nan), np.full(DOF_NUM, 100.0)):
        try:
            arm_load_waypoints(bad_center, ARM_LOAD_TEST_GROUPS[0])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid B pose accepted")
