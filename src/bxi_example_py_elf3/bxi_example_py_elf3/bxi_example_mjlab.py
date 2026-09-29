import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg 
import sensor_msgs.msg
from threading import Lock
import numpy as np
# import torch
import time
import sys
import os
import math
from collections import deque
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

import onnxruntime as ort
import ast
from .control.walking_policy import WalkingPolicy, projected_gravity_from_quat

from .control.remote import RemoteButtonEdge

robot_name = "elf3"

dof_num = 29

joint_name = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    
    "l_hip_y_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_z_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴

    "r_hip_y_joint",   # 右腿_髋关节_z轴    
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_z_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴

    "l_shoulder_y_joint",   # 左臂_肩关节_y轴
    "l_shoulder_x_joint",   # 左臂_肩关节_x轴
    "l_shoulder_z_joint",   # 左臂_肩关节_z轴
    "l_elbow_y_joint",   # 左臂_肘关节_y轴
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    
    "r_shoulder_y_joint",   # 右臂_肩关节_y轴   
    "r_shoulder_x_joint",   # 右臂_肩关节_x轴
    "r_shoulder_z_joint",   # 右臂_肩关节_z轴
    "r_elbow_y_joint",    # 右臂_肘关节_y轴
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
    )   

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])



class BxiExample(Node, WalkingPolicy):

    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        print('topic_prefix:', self.topic_prefix)

        
        self.declare_parameter('/onnx_file', 'default_value')
        self.onnx_file = self.get_parameter('/onnx_file').get_parameter_value().string_value        
        print("onnx_file:", self.onnx_file)

        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability, reliability=qos_profile_sensor_data.reliability)
        
        self.act_pub = self.create_publisher(bxiMsg.ActuatorCmds, self.topic_prefix+'actuators_cmds', qos)  # CHANGE
        
        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, self.topic_prefix+'odom', self.odom_callback, qos)
        self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', self.joint_callback, qos)
        self.imu_sub = self.create_subscription(sensor_msgs.msg.Imu, self.topic_prefix+'imu_data', self.imu_callback, qos)
        self.touch_sub = self.create_subscription(bxiMsg.TouchSensor, self.topic_prefix+'touch_sensor', self.touch_callback, qos)
        self.joy_sub = self.create_subscription(bxiMsg.MotionCommands, 'motion_commands', self.joy_callback, qos)

        self.rest_srv = self.create_client(bxiSrv.RobotReset, self.topic_prefix+'robot_reset')
        self.sim_rest_srv = self.create_client(bxiSrv.SimulationReset, self.topic_prefix+'sim_reset')
        
        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()
        
        self.initialize_onnx(self.onnx_file)
        self.lock_in = Lock()
        self.lock_ou = self.lock_in #Lock()
        self.qpos = np.zeros(self.num_action,dtype=np.double)
        self.qvel = np.zeros(self.num_action,dtype=np.double)
        self.omega = np.zeros(3,dtype=np.double)
        self.quat = np.zeros(4,dtype=np.double)
        
        self.target_q = np.zeros(self.num_action, dtype=np.double)
        self.action = np.zeros(self.num_action, dtype=np.double)

        policy_input = np.zeros([1, self.num_obs], dtype=np.float32)
        print("policy test")

        warm_action = self.inference_step(policy_input)
        if self.num_obs == 96:
            self.action[:] = warm_action

        self.vx = 0.0
        self.vy = 0
        self.dyaw = 0
        self.shuttle_button = RemoteButtonEdge("momentary", 0.5)
        self.sprint_button = RemoteButtonEdge("momentary", 0.5)
        self.walk_test_mode = 0
        self.sprint_remote_mode = False
        self.shuttle_started_at = 0.0

        self.step = 0
        self.loop_count = 0
        self.dt = 0.02  # loop @100Hz
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)

    # 初始化部分（完整版）


    # 循环推理部分（极速版）

    def timer_callback(self):
        
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_reset(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (10./self.dt): # 延迟10s
            self.robot_reset(2, True) # first reset
            print('robot reset 2!')
            self.loop_count = 0
            self.step = 2
            return
        
        if self.step == 1:
            soft_start = self.loop_count/(1./self.dt) # 1秒关节缓启动
            if soft_start > 1:
                soft_start = 1
                
            soft_joint_kp = self.joint_stiffness * soft_start #* 0.2
            soft_joint_kd = self.joint_damping #* 0.2
                
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            msg.pos = self.default_joint_pos.tolist()
            msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.kp = soft_joint_kp.tolist()
            msg.kd = soft_joint_kd.tolist()
            self.act_pub.publish(msg)
            
        elif self.step == 2:
            with self.lock_in:
                q = self.qpos
                dq = self.qvel
                quat = self.quat
                omega = self.omega
                
                if self.walk_test_mode:
                    phase = (time.monotonic() - self.shuttle_started_at) % 2.8
                    speed = 0.5
                    x_vel_cmd = speed if phase < 1.5 else -speed
                    y_vel_cmd = 0.0
                    yaw_vel_cmd = 0.0
                else:
                    x_vel_cmd = self.vx
                    y_vel_cmd = self.vy
                    yaw_vel_cmd = self.dyaw
            
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            if (np.abs(eu_ang[0]) > (math.pi/3.0)) or (np.abs(eu_ang[1]) > (math.pi/3.0)):
                raise RuntimeError("walking tilt safety limit exceeded")
            policy_input = self.build_policy_input(
                q, dq, quat, omega, (x_vel_cmd, y_vel_cmd, yaw_vel_cmd),
                reset_history=self.loop_count == 0,
            )

            self.action[:] = self.inference_step(policy_input)
            # self.action = np.clip(self.action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)
            self.target_q = self.action * self.action_scale
            qpos = self.default_joint_pos.copy()
            qpos[:] += self.target_q[:]
            
            kp = self.joint_stiffness #* 0.9
            kd = self.joint_damping #* 0.2
            
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            msg.pos = qpos.tolist()
            msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.kp = kp.tolist()
            msg.kd = kd.tolist()
            self.act_pub.publish(msg)
            self.last_action=self.action.copy()

        self.loop_count += 1
    
    def robot_reset(self, reset_step, release):
        req = bxiSrv.RobotReset.Request()
        req.reset_step = reset_step
        req.release = release
        req.header.frame_id = robot_name
    
        while not self.rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.rest_srv.call_async(req)
        
    def sim_robot_reset(self):        
        req = bxiSrv.SimulationReset.Request()
        req.header.frame_id = robot_name

        base_pose = Pose()
        base_pose.position.x = 0.0
        base_pose.position.y = 0.0
        base_pose.position.z = 1.0
        base_pose.orientation.x = 0.0
        base_pose.orientation.y = 0.0
        base_pose.orientation.z = 0.0
        base_pose.orientation.w = 1.0        

        joint_state = JointState()
        joint_state.name = joint_name
        joint_state.position = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.velocity = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.effort = np.zeros(dof_num, dtype=np.float32).tolist()
        
        req.base_pose = base_pose
        req.joint_state = joint_state
    
        while not self.sim_rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.sim_rest_srv.call_async(req)
    
    def joint_callback(self, msg):
        positions = np.asarray(msg.position, dtype=np.float64)
        velocities = np.asarray(msg.velocity, dtype=np.float64)
        if positions.shape not in ((dof_num,), (31,)) or velocities.shape != positions.shape:
            raise ValueError("expected matching position/velocity feedback for 29 or 31 joints")
        if msg.name:
            if len(msg.name) != len(positions) or len(set(msg.name)) != len(msg.name):
                raise ValueError("joint feedback names must be unique and match array lengths")
            indices_by_name = {name: index for index, name in enumerate(msg.name)}
            missing = [name for name in joint_name if name not in indices_by_name]
            if missing:
                raise ValueError("joint feedback missing model joints: " + ", ".join(missing))
            # The 31-joint hardware includes two head joints; the policy still controls 29.
            indices = [indices_by_name[name] for name in joint_name]
            positions = positions[indices]
            velocities = velocities[indices]
        elif len(positions) != dof_num:
            raise ValueError("31-joint feedback requires names to map the 29-joint policy safely")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("non-finite model joint feedback")
        with self.lock_in:
            self.qpos[:] = positions
            self.qvel[:] = velocities

    def joy_callback(self, msg):
        now = time.monotonic()
        shuttle_activated = self.shuttle_button.update(msg.btn_5 != 0, now)
        sprint_activated = self.sprint_button.update(msg.btn_6 != 0, now)
        with self.lock_in:
            if shuttle_activated:
                self.walk_test_mode = 0 if self.walk_test_mode == 1 else 1
                self.shuttle_started_at = now
                self.get_logger().info(
                    "0.5 m/s shuttle (forward 1.5 s, reverse 1.3 s) %s" % ("started" if self.walk_test_mode else "stopped")
                )
            if sprint_activated:
                self.sprint_remote_mode = not self.sprint_remote_mode
                self.walk_test_mode = 0
                self.vx = self.vy = self.dyaw = 0.0
                self.get_logger().info(
                    "RB remote walking mode %s: joystick speed limit 2.5 m/s"
                    % ("enabled" if self.sprint_remote_mode else "disabled")
                )
            if not self.walk_test_mode:
                speed_limit = 2.5 if self.sprint_remote_mode else 2.0
                self.vx = np.clip(msg.vel_des.x * speed_limit, -speed_limit, speed_limit)
                self.vy = msg.vel_des.y * 2
                self.dyaw = msg.yawdot_des * 2
        
    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity
        acc = msg.linear_acceleration

        quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)

        with self.lock_in:
            self.quat = quat_tmp1
            self.omega = np.array([avel.x, avel.y, avel.z])

    def touch_callback(self, msg):
        foot_force = msg.value
        
    def odom_callback(self, msg): # 全局里程计（上帝视角，仅限仿真使用）
        base_pose = msg.pose
        base_twist = msg.twist

def main(args=None):
   
    time.sleep(5)
    
    rclpy.init(args=args)
    node = BxiExample()
    
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()
    
