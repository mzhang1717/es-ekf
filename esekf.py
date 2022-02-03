import numpy as np
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


class ESEKF:
    def __init__(self, p_0=None, v_0=None, q_0=None):
        self.p = np.zeros(shape=(1, 3)) if p_0 is None else p_0
        self.v = np.zeros(shape=(1, 3)) if v_0 is None else v_0
        self.q = Quaternion(1.0, 0.0, 0.0, 0.0) if q_0 is None else q_0
        self.g = np.array([0, 0, -9.81])
        self.a_b = 0
        self.w_b = 0
        self.F = np.eye(9)
        self.Q = np.zeros((6, 6))
        self.p_cov = np.zeros(9)
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)  # measurement model jacobian
        self.var_imu_f = 0.10
        self.var_imu_w = 0.25

    def kinematics(self, a_m, w_m, dt):
        assert (dt > 0)
        R = self.q.to_mat()
        acce = R.dot(a_m - self.a_b)
        a = acce + self.g

        self.p += self.v*dt + 0.5*a*dt*dt
        self.v += a*dt
        self.q = self.q.quat_mult_left(Quaternion(axis_angle=dt * (w_m-self.w_b)), 'Quaternion')

        self.F[0:3, 3:6] = np.eye(3) * dt
        self.F[3:6, 6:9] = -skew_symmetric(acce) * dt
        self.Q[0:3, 0:3] = np.eye(3) * self.var_imu_f * (dt ** 2)
        self.Q[3:6, 3:6] = np.eye(3) * self.var_imu_w * (dt ** 2)
        # p_cov[k] = F @ p_cov[k - 1] @ F.T + l_jac @ Q @ l_jac.T
        self.p_cov = self.F @ self.p_cov @ self.F.T + self.l_jac @ self.Q @ self.l_jac.T

    def measurement_update(self, sensor_var, y_k):
        # 3.1 Compute Kalman Gain
        K = self.p_cov @ self.h_jac.T @ np.linalg.inv(self.h_jac @ self.p_cov @ self.h_jac.T + sensor_var)
        # 3.2 Compute error state
        delta_x = K.dot(y_k - self.p)
        # 3.3 Correct predicted state
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        delta_quat = Quaternion(axis_angle=delta_x[6:9])
        # print("q_axis = {}".format( Quaternion(axis_angle = delta_x[6:9]) ))
        # print("q_euler = {}".format( Quaternion(euler = delta_x[6:9]) ))
        self.q = delta_quat.quat_mult_left(self.q, 'Quaternion')
        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(9) - (K @ self.h_jac)) @ self.p_cov

    def get_states(self):
        return self.p, self.v, self.q, self.p_cov

    def set_imu_var(self, var_f, var_m):
        self.var_imu_f = var_f  # 0.10
        self.var_imu_w = var_m  # 0.25



