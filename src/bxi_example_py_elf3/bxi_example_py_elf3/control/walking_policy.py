"""Shared walking inference; no ROS node or command publisher."""

import ast
import numpy as np
import onnxruntime as ort
from ..utils.tfs import quat_rotate_inverse
from .elf3 import JOINT_NAMES as joint_name, DOF_NUM as dof_num

def projected_gravity_from_quat(quaternion, gravity=np.array([0, 0, -9.81])):
    """Project world gravity into the body using an [x, y, z, w] quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("expected a finite, nonzero quaternion with shape (4,)")
    # Match Rotation.from_quat normalization; the existing helper uses [w, x, y, z].
    return quat_rotate_inverse((q / norm)[[3, 0, 1, 2]], gravity)



class WalkingPolicy:
    def initialize_onnx(self, model_path):
        # 配置执行提供者（根据硬件选择最优后端）
        providers = [
            'CUDAExecutionProvider',  # 优先使用GPU
            'CPUExecutionProvider'    # 回退到CPU
        ] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 创建推理会话
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=options
        )
        
        # 预存输入输出信息
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        
        # 预分配输入内存（可选，适合固定输入尺寸）
        self.input_buffer = np.zeros(
            self.input_info.shape,
            dtype=np.float32
        )

        self.num_action = dof_num
        self.num_obs = self.input_info.shape[-1]
        if self.input_info.shape == [1, 960] and self.output_info.shape == [1, 32]:
            from .yamaxun_joints import ELF3_ISAAC_PARAMETERS, ELF3_POLICY_JOINTS
            native = ELF3_ISAAC_PARAMETERS
            canonical = native.select(ELF3_POLICY_JOINTS)
            self._isaac_indices = [joint_name.index(name) for name in native.layout.names]
            self._hardware_indices = [native.layout.names.index(name) for name in joint_name]
            self.default_joint_pos = canonical.default_position.copy()
            self.joint_stiffness = canonical.kp.copy()
            self.joint_damping = canonical.kd.copy()
            self.action_scale = canonical.action_scale.copy()
            self._history = np.zeros((10, 96), dtype=np.float32)
        elif self.input_info.shape == [1, 96] and self.output_info.shape == [1, 29]:
            metadata = self.session.get_modelmeta().custom_metadata_map
            if tuple(metadata["joint_names"].split(",")) != joint_name:
                raise ValueError("model joint order does not match the controller")
            self.default_joint_pos = np.asarray(ast.literal_eval(metadata["default_joint_pos"]), dtype=np.float32)
            self.joint_stiffness = np.asarray(ast.literal_eval(metadata["joint_stiffness"]), dtype=np.float32)
            self.joint_damping = np.asarray(ast.literal_eval(metadata["joint_damping"]), dtype=np.float32)
            self.action_scale = np.asarray(ast.literal_eval(metadata["action_scale"]), dtype=np.float32)
        else:
            raise ValueError("unsupported walking model input/output dimensions")


    def build_policy_input(self, q, dq, quat, omega, command, reset_history=False):
        gravity = projected_gravity_from_quat(quat, np.array([0, 0, -1]))
        single = np.zeros(96, dtype=np.float32)
        single[:3] = omega
        single[3:6] = gravity
        if self.num_obs == 960:
            # yamaxun AMP: Isaac joint order, command first, oldest-to-newest history.
            indices = self._isaac_indices
            single[6:9] = command
            single[9:38] = np.asarray(q, dtype=np.float32)[indices] - self.default_joint_pos[indices]
            single[38:67] = np.asarray(dq)[indices]
            single[67:96] = self.action[indices]
            if reset_history:
                self._history[:] = single
            else:
                self._history[:-1] = self._history[1:]
                self._history[-1] = single
            return self._history.reshape(1, 960)
        single[6:35] = q - self.default_joint_pos
        single[35:64] = dq
        single[64:93] = self.action
        single[93:96] = command
        return single.reshape(1, 96)


    def inference_step(self, obs_data):
        # 使用预分配内存（如果适用）
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        
        # 极简推理（比原版快5-15%）
        output = self.session.run(
            [self.output_info.name],
            {self.input_info.name: self.input_buffer}
        )[0][0]
        if self.num_obs == 960:
            # The final three AMP outputs estimate velocity, not additional joints.
            output = output[:29][self._hardware_indices]
        if output.shape != (29,) or not np.all(np.isfinite(output)):
            raise ValueError("invalid walking policy actions")
        return output


