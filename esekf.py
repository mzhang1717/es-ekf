import numpy as np
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


class ESEKF:
    def __init__(self, p_0=None, v_0=None, q_0=None):
        """
        Initialization method to set initial states and other variables
        :param p_0: initial position
        :param v_0: initial velocity
        :param q_0: initial orientation in the format of quaternion
        """
        self.p = np.zeros(shape=(1, 3)) if p_0 is None else p_0  # estimated position
        self.v = np.zeros(shape=(1, 3)) if v_0 is None else v_0  # estimated velocity
        self.q = Quaternion(1.0, 0.0, 0.0, 0.0) if q_0 is None else q_0  # estimated orientation in quaternion
        self.g = np.array([0, 0, -9.81])  # vectorized acceleration of gravity
        self.a_b = 0  # accelerometer bias
        self.w_b = 0  # gyrometer bias
        self.F = np.eye(9)  # jacobian matrix of error-state kinematics w.r.t. the error state
        self.Q = np.zeros((6, 6))  # Covariance matrix of the perturbation impulses
        self.p_cov = np.zeros((9, 9))  # Covariance matrix of the error-state

        self.l_jac = np.zeros((9, 6)) # jacobian matrix of error-state kinematics w.r.t. the perturbation impulses vector
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian

        self.h_jac = np.zeros((3, 9)) # jacobian matrix of position observation w.r.t to nominal state
        self.h_jac[:, :3] = np.eye(3)  # measurement model jacobian
        self.var_imu_f = 0.10  # variance of force/acceleration measurements (i.e., accelerometer)
        self.var_imu_w = 0.25  # variance of rotational velocity measurement (i.e., gyrometer)

    def prediction(self, a_m, w_m, dt):
        """
        Prediction step for nominal states and error states
        :param a_m: acceleration measurement
        :param w_m: rotational velocity measurement
        :param dt: time interval to the next time step
        :return:
        """
        assert (dt > 0)
        R = self.q.to_mat()  # obtain current orientation in the format of a rotational matrix
        acce = R.dot(a_m - self.a_b)  # transform acceleration measurement from vehicle frame to world frame
        a = acce + self.g  # add gravity vector

        # Predict the next position, velocity & orientation by integrating the high-rate IMU measurements
        self.p += self.v*dt + 0.5*a*dt*dt
        self.v += a*dt
        self.q = self.q.quat_mult_left(Quaternion(axis_angle=dt * (w_m-self.w_b)), 'Quaternion')

        # Predict the covariance matrix of the next error-state
        self.F[0:3, 3:6] = np.eye(3) * dt
        self.F[3:6, 6:9] = -skew_symmetric(acce) * dt
        self.Q[0:3, 0:3] = np.eye(3) * self.var_imu_f * (dt ** 2)
        self.Q[3:6, 3:6] = np.eye(3) * self.var_imu_w * (dt ** 2)
        # p_cov[k] = F @ p_cov[k - 1] @ F.T + l_jac @ Q @ l_jac.T
        self.p_cov = self.F @ self.p_cov @ self.F.T + self.l_jac @ self.Q @ self.l_jac.T

    def correction(self, sensor_var, y_k):
        """
        Correct the predicted states & covariance by incorporating external measurments
        :param sensor_var: variance of the external sensor (Lidar or GNSS) measurements
        :param y_k: position measurement by the external sensor (Lidar or GNSS)
        :return:
        """
        # Compute Kalman Gain
        K = self.p_cov @ self.h_jac.T @ np.linalg.inv(self.h_jac @ self.p_cov @ self.h_jac.T + sensor_var)

        # Compute error state
        delta_x = K.dot(y_k - self.p)

        # Correct predicted state
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        delta_quat = Quaternion(axis_angle=delta_x[6:9])
        self.q = delta_quat.quat_mult_left(self.q, 'Quaternion')

        # Compute corrected covariance
        self.p_cov = (np.eye(9) - (K @ self.h_jac)) @ self.p_cov

    def get_states(self):
        """
        Return the estimated states
        """
        return self.p, self.v, self.q, self.p_cov

    def set_imu_var(self, var_f, var_m):
        """
        Set the variance of accelerometer and gyrometer
        :param var_f:
        :param var_m:
        :return:
        """
        self.var_imu_f = var_f  # 0.10
        self.var_imu_w = var_m  # 0.25
