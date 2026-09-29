"""One control loop for PD, policy walking and the existing A/B/Y tests."""

import math
import time

import numpy as np
from ament_index_python.packages import get_package_share_path
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_srvs.srv import SetBool

from .bxi_example_mjlab import BxiExample, quaternion_to_euler_array
from .bxi_example_suspended_tests import SuspendedTestNode, ARM_TEST_GROUPS
from .bxi_example_vibration import VibrationTestNode
from .control.elf3 import DOF_NUM, JOINT_NAMES, JOINT_KP, JOINT_KD, JOINT_NOMINAL_POS
from .control.limb_sequence import LIMB_TEST_GROUPS, velocity_limited_duration
from .control.remote import RemoteButtonEdge
from .control.ros_runtime import run_controller
from .control.trajectory import minimum_jerk_progress
from .control.walking_policy import WalkingPolicy


class UnifiedControlNode(SuspendedTestNode):
    BUTTON_MODES = {
        'btn_5': 'auto_walk', 'btn_6': 'remote_walk',
        'btn_8': 'arm_rom', 'btn_7': 'body_rom', 'btn_10': 'vibration',
    }
    WALK_MODES = ('auto_walk', 'remote_walk')

    def __init__(self):
        super().__init__()
        model = str(self.declare_parameter(
            'onnx_file', str(get_package_share_path('bxi_example_py_elf3') / 'data/amp_terrain.onnx')
        ).value)
        self.policy = WalkingPolicy()
        self.policy.initialize_onnx(model)
        self.policy.action = np.zeros(DOF_NUM, dtype=np.float32)
        # Warm the session before initialization can enable motor commands.
        self.policy.inference_step(np.zeros((1, self.policy.num_obs), dtype=np.float32))
        self.pd_positions = self.policy.default_joint_pos.copy()
        self.center_positions[:] = self.pd_positions
        self.last_command_positions[:] = self.pd_positions
        self.mode = 'pd'
        self.phase = 'active'
        self.queued_mode = ''
        self.stable_since = None
        self.last_settle_log_at = -math.inf
        self.mode_started_at = 0.0
        self.next_policy_at = 0.0
        self.reset_policy_history = True
        self.walk_command = self.pd_positions.copy()
        self.last_kp = self.policy.joint_stiffness.copy()
        self.last_kd = self.policy.joint_damping.copy()
        self.return_start_kp = self.last_kp.copy()
        self.return_start_kd = self.last_kd.copy()
        self.return_target_kp = self.last_kp.copy()
        self.return_target_kd = self.last_kd.copy()
        self.buttons = {name: RemoteButtonEdge('momentary', self.motion_command_resync_sec)
                        for name in self.BUTTON_MODES}
        self.remote_command = np.zeros(3)
        self.last_remote_at = 0.0
        self.qpos = self.measured_positions  # Existing validated 29/31-joint mapping.
        self.qvel = np.zeros(DOF_NUM)
        self.lock_in = self.feedback_lock
        self.quat = np.array([0., 0., 0., 1.])
        self.omega = np.zeros(3)
        self.last_imu_at = 0.0
        self.pd_settle_sec = float(self.declare_parameter('pd_settle_sec', 0.3).value)
        self.pd_velocity_tolerance = float(self.declare_parameter('pd_velocity_tolerance_rad_s', 0.2).value)
        self.imu_timeout_sec = float(self.declare_parameter('imu_timeout_sec', 0.2).value)
        self.remote_timeout_sec = float(self.declare_parameter('remote_timeout_sec', 0.5).value)
        for value in (self.pd_settle_sec, self.pd_velocity_tolerance,
                      self.imu_timeout_sec, self.remote_timeout_sec):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('unified settle/feedback timeout parameters must be finite and positive')
        if self.control_rate_hz < 50 or self.auto_start or self.joint_test_required:
            raise ValueError('unified control requires >=50 Hz, auto_start=false, joint_test_required=false')
        self.imu_sub = self.create_subscription(
            Imu, self.topic_prefix + 'imu_data', self._imu_callback, qos_profile_sensor_data)
        self.mode_services = [self.create_service(
            SetBool, mode + '_enable',
            lambda req, res, mode=mode: self._mode_service(mode, req, res))
            for mode in ('auto_walk', 'remote_walk', 'arm_rom')]
        self.get_logger().info(self._remote_help_message())

    def _remote_help_message(self):
        return ('Unified controller: startup=PD; LB=auto walk, RB=remote walk (2.5 m/s), '
                'B=arm ROM, A=body ROM, Y=vibration. Same button returns to PD; '
                'another button queues its mode after PD settles. X is unused.')

    def _joint_callback(self, msg):
        try:
            BxiExample.joint_callback(self, msg)
        except (ValueError, TypeError) as exc:
            self._latch_safety_fault(str(exc))
            return
        super()._joint_callback(msg)

    def _imu_callback(self, msg):
        q, w = msg.orientation, msg.angular_velocity
        quat = np.array([q.x, q.y, q.z, q.w])
        omega = np.array([w.x, w.y, w.z])
        norm = np.linalg.norm(quat)
        if not np.all(np.isfinite(omega)) or not math.isfinite(norm) or norm < 1e-6:
            if self.reset_stage == 2:
                self._latch_safety_fault('invalid IMU quaternion/angular velocity')
            return
        self.quat[:] = quat / norm
        self.omega[:] = omega
        self.last_imu_at = time.monotonic()

    def _motion_callback(self, msg):
        now = time.monotonic()
        if self.count_publishers(self.motion_topic) > 1:
            self._latch_safety_fault('multiple motion_commands publishers')
            return
        values = np.array([msg.vel_des.x, msg.vel_des.y, msg.yawdot_des])
        if not np.all(np.isfinite(values)):
            self._latch_safety_fault('non-finite remote velocity command')
            return
        self.remote_command[:] = np.clip(values, -1., 1.) * [2.5, 2., 2.]
        self.last_remote_at = now
        pressed = [mode for name, mode in self.BUTTON_MODES.items()
                   if self.buttons[name].update(getattr(msg, name) != 0, now)]
        if len(pressed) == 1:
            self._select_mode(pressed[0], toggle=True)
        elif pressed:
            self._select_mode('pd')

    def _select_mode(self, requested, toggle=False):
        with self.state_lock:
            if self.safety_fault or self.shutdown_requested.is_set() or self.reset_stage != 2:
                self._queue_diagnostic_log(
                    'warning', 'Mode request rejected: %s; reset_stage=%s safety_fault=%s shutdown=%s'
                    % (requested, self.reset_stage, self.safety_fault,
                       self.shutdown_requested.is_set()))
                return False
            if requested not in ('pd', *self.BUTTON_MODES.values()):
                return False
            original_request = requested
            if toggle and requested in (self.mode, self.queued_mode):
                requested = 'pd'
            elif not toggle and requested != 'pd' and requested in (self.mode, self.queued_mode):
                return True
            self.queued_mode = '' if requested == 'pd' else requested
            self.last_settle_log_at = -math.inf
            self._queue_diagnostic_log(
                'info', 'Mode request accepted: %s -> %s; current=%s phase=%s queued=%s'
                % (original_request, requested, self.mode, self.phase,
                   self.queued_mode or 'none'))
            if self.phase == 'return_pd':
                return True
            if self.mode == 'pd' and self.phase == 'active':
                self.stable_since = None
                return True
            self._return_to_pd()
            return True

    def _return_to_pd(self):
        self.test_enabled = False
        self._close_csv()
        self.limb_test_running = False
        self.limb_test_phase = 'idle'
        self.run_enabled = False
        self.pending_mode = ''
        self.mode = 'pd'
        self.phase = 'return_pd'
        self.stable_since = None
        self.center_positions[:] = self.pd_positions
        duration = velocity_limited_duration(
            self.last_command_positions, self.pd_positions, JOINT_NAMES,
            self.limb_test_move_sec, self.limb_test_range_speed_deg_s)
        self._begin_smooth_return_locked('unified_pd', duration_sec=duration)
        self.get_logger().info('Returning to PD; queued mode: ' + (self.queued_mode or 'none'))

    def _begin_smooth_return_locked(self, owner, duration_sec=None):
        super()._begin_smooth_return_locked(owner, duration_sec)
        self.return_start_kp = self.last_kp.copy()
        self.return_start_kd = self.last_kd.copy()
        self.return_target_kp, self.return_target_kd = self._mode_gains()

    def _mode_gains(self):
        if self.mode in ('pd', *self.WALK_MODES):
            return self.policy.joint_stiffness, self.policy.joint_damping
        return JOINT_KP, JOINT_KD

    def _initialization_command(self, ramp):
        return self.pd_positions, self.policy.joint_stiffness * ramp, self.policy.joint_damping

    def _settled(self, now):
        feedback_ready = self._joint_feedback_ready(now)
        with self.feedback_lock:
            position_error = np.abs(self.measured_positions - self.center_positions)
            velocity = np.abs(self.qvel)
        ready = (feedback_ready
                 and np.max(position_error) <= self.limb_test_start_tolerance_rad
                 and np.max(velocity) <= self.pd_velocity_tolerance)
        if not ready:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = now
        stable_sec = 0.0 if self.stable_since is None else now - self.stable_since
        settled = ready and stable_sec >= self.pd_settle_sec
        if not settled and now - self.last_settle_log_at >= 1.0:
            self.last_settle_log_at = now
            pos_index, vel_index = int(np.argmax(position_error)), int(np.argmax(velocity))
            self._queue_diagnostic_log(
                'warning',
                'Waiting for pose to settle: mode=%s phase=%s queued=%s; feedback_ready=%s; '
                'position=%s %.3f deg (limit %.3f); velocity=%s %.3f rad/s (limit %.3f); '
                'stable=%.3f/%.3f s'
                % (self.mode, self.phase, self.queued_mode or 'none', feedback_ready,
                   JOINT_NAMES[pos_index], math.degrees(position_error[pos_index]),
                   math.degrees(self.limb_test_start_tolerance_rad),
                   JOINT_NAMES[vel_index], velocity[vel_index], self.pd_velocity_tolerance,
                   stable_sec, self.pd_settle_sec))
        return settled

    def _begin_mode(self, mode, now):
        self.queued_mode = ''
        self.mode = mode
        self.stable_since = None
        if mode in self.WALK_MODES:
            if not self._walking_feedback_safe(now):
                return
            self.phase = 'active'
            self.mode_started_at = now
            self.next_policy_at = 0.0
            self.policy.action[:] = 0.0
            self.reset_policy_history = True
            self.walk_command[:] = self.pd_positions
        else:
            self.phase = 'prepare'
            if mode in ('arm_rom', 'body_rom'):
                groups = ARM_TEST_GROUPS if mode == 'arm_rom' else LIMB_TEST_GROUPS
                if not self._prepare_limb_test_locked('unified ' + mode, groups):
                    self._return_to_pd()
                    return
            else:
                self._prepare_vibration_locked('unified Y')
        self.get_logger().info('Selected mode: ' + mode)

    def _walking_feedback_safe(self, now):
        if not self._joint_feedback_ready(now):
            self._latch_safety_fault('walking joint feedback became incomplete or stale')
        elif self.last_imu_at <= 0 or now - self.last_imu_at > self.imu_timeout_sec:
            self._latch_safety_fault('walking IMU feedback became stale')
        elif self.mode == 'remote_walk' and now - self.last_remote_at > self.remote_timeout_sec:
            self._latch_safety_fault('remote walking command heartbeat lost')
        elif np.any(np.abs(quaternion_to_euler_array(self.quat)[:2]) > math.pi / 3.):
            self._latch_safety_fault('walking tilt safety limit exceeded')
        return not self.safety_fault

    def _idle_command(self, now):
        if self.returning_to_center:
            return self.center_positions.copy(), *self._mode_gains()
        if self.phase == 'return_pd':
            if self._settled(now):
                self.phase = 'active'
                self.get_logger().info('PD settled')
            else:
                return self.center_positions.copy(), *self._mode_gains()
        if self.mode == 'pd' and self.queued_mode and self._settled(now):
            self._begin_mode(self.queued_mode, now)
        if self.phase == 'prepare' and not self.returning_to_center and self._settled(now):
            self.pending_mode = ''
            if self.mode in ('arm_rom', 'body_rom'):
                started = self._start_limb_test_locked()
            else:
                # Preserve the commanded nominal center, rather than moving it to noisy feedback.
                started = VibrationTestNode._start_test(
                    self, request_received_at=now, center_override=JOINT_NOMINAL_POS,
                    envelope_slack_rad=self.vibration_start_envelope_slack_rad)
            if started:
                self.phase = 'active'
                self.get_logger().info('Active mode: ' + self.mode)
            else:
                self.get_logger().error('Mode start rejected; returning to PD: ' + self.last_start_message)
                self._return_to_pd()
        if self.mode in self.WALK_MODES and self.phase == 'active':
            if self._walking_feedback_safe(now) and now + 1e-9 >= self.next_policy_at:
                command = self.remote_command
                if self.mode == 'auto_walk':
                    phase = (now - self.mode_started_at) % 2.8
                    command = [0.5 if phase < 1.5 else -0.5, 0., 0.]
                obs = self.policy.build_policy_input(
                    self.qpos, self.qvel, self.quat, self.omega, command,
                    reset_history=self.reset_policy_history)
                self.policy.action[:] = self.policy.inference_step(obs)
                self.walk_command[:] = self.pd_positions + self.policy.action_scale * self.policy.action
                self.reset_policy_history = False
                # Preserve the trained 50 Hz policy rate under the 100 Hz publisher.
                self.next_policy_at += 0.02
                if self.next_policy_at <= now:
                    self.next_policy_at = now + 0.02
            return self.walk_command.copy(), *self._mode_gains()
        if self.limb_test_running:
            return self._limb_test_command_locked(now), JOINT_KP, JOINT_KD
        return self.center_positions.copy(), *self._mode_gains()

    def _publish_command(self, positions, kp, kd):
        if self.reset_stage == 2:
            kp, kd = self._mode_gains()
            if self.phase in ('return_pd', 'prepare'):
                progress = minimum_jerk_progress(
                    (time.monotonic() - self.return_started_at) / self.return_duration_sec)
                kp = self.return_start_kp + progress * (self.return_target_kp - self.return_start_kp)
                kd = self.return_start_kd + progress * (self.return_target_kd - self.return_start_kd)
        published = super()._publish_command(positions, kp, kd)
        if published:
            self.last_kp, self.last_kd = np.array(kp), np.array(kd)
        return published

    def _return_to_center_command(self, now):
        # Keep the existing collision check on every transition, including PD return.
        owner = self.return_owner
        self.return_owner = 'prepare_limb_test'
        command = super()._return_to_center_command(now)
        if self.returning_to_center:
            self.return_owner = owner
        return command

    def _stop_test(self, reason):
        if not self.test_enabled:
            return False
        self.queued_mode = ''
        self._return_to_pd()
        return True

    def _latch_safety_fault(self, reason):
        self.queued_mode = ''
        super()._latch_safety_fault(reason)

    def _mode_service(self, mode, request, response):
        with self.state_lock:
            target = mode if request.data else 'pd'
            relevant = mode in (self.mode, self.queued_mode)
            response.success = (self._select_mode(target) if request.data or relevant else True)
            response.message = ('accepted; mode=%s phase=%s queued=%s' %
                                (self.mode, self.phase, self.queued_mode)) if response.success else 'initializing or safety fault'
        return response

    def _enable_service_callback(self, request, response):
        return self._mode_service('vibration', request, response)

    def _limb_test_enable_service_callback(self, request, response):
        return self._mode_service('body_rom', request, response)

    def _run_enable_service_callback(self, request, response):
        return self._mode_service('auto_walk', request, response)

    def _joint_test_service_callback(self, request, response):
        response.success = False
        response.message = 'Legacy precheck unavailable in unified mode; use A/body ROM'
        return response


def main(args=None):
    run_controller(UnifiedControlNode, args=args)


if __name__ == '__main__':
    main()
