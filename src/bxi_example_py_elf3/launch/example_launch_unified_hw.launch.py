"""Unified PD, LB/RB policy walking and A/B/Y hardware tests."""

import importlib.util

from ament_index_python.packages import get_package_share_path


def generate_launch_description():
    launch_path = (
        get_package_share_path("bxi_example_py_elf3")
        / "launch/example_launch_vibration_hw.launch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "unified_hardware_launch_base",
        launch_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description(
        controller_executable_default=(
            "bxi_example_py_elf3_unified"
        ),
        controller_name_default="bxi_example_py_elf3_unified",
        joint_test_required_default="false",
        allow_hardware_without_joint_test_default="true",
        start_remote_controller_default="false",
        # The onboard receiver is local; ignore other robots' same-domain topics.
        controller_localhost_only=True,
        control_rate_hz_default="200.0",
    )
