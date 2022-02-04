import pickle
import sys

import numpy as np
from esekf import ESEKF
import matplotlib.pyplot as plt

from rotations import angle_normalize, rpy_jacobian_axis_angle, Quaternion


################################################################################################
# Transform the LIDAR data to the IMU frame using a known extrinsic calibration
# rotation matrix C_li and translation vector t_i_li.
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
################################################################################################

C_li = np.array([
   [0.99376, -0.09722,  0.05466],
   [0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975]
])

t_i_li = np.array([0.5, 0.1, 0.5])


def main():
    # load data from file into a dictionary
    with open('data/pt3_data.pkl', 'rb') as file:
        data = pickle.load(file)  # data's type is dictionary

    ################################################################################################
    # Each element of the data dictionary is stored as an item from the data dictionary, which we
    # will store in local variables, described by the following:
    #   gt: Data object containing ground truth. with the following fields:
    #     a: Acceleration of the vehicle, in the inertial frame
    #     v: Velocity of the vehicle, in the inertial frame
    #     p: Position of the vehicle, in the inertial frame
    #     alpha: Rotational acceleration of the vehicle, in the inertial frame
    #     w: Rotational velocity of the vehicle, in the inertial frame
    #     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
    #     _t: Timestamp in ms.
    #   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
    #     data: The actual data
    #     t: Timestamps in ms.
    #   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
    #     data: The actual data
    #     t: Timestamps in ms.
    #   gnss: StampedData object with the GNSS data.
    #     data: The actual data
    #     t: Timestamps in ms.
    #   lidar: StampedData object with the LIDAR data (positions only).
    #     data: The actual data
    #     t: Timestamps in ms.
    ################################################################################################
    gt = data['gt']
    imu_f = data['imu_f']
    imu_w = data['imu_w']
    gnss = data['gnss']
    lidar = data['lidar']
    lidar.data = (C_li @ lidar.data.T).T + t_i_li

    ################################################################################################
    # Setting the estimated sensor variances
    ################################################################################################
    var_imu_f = 0.01  # 0.10
    var_imu_w = 0.01  # 0.25
    var_gnss = 0.1  # 0.01
    var_lidar = 35   # 1.00

    ################################################################################################
    # Create arrays to store estimated states and covariance.
    ################################################################################################
    p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
    v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
    q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
    p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each time step

    ################################################################################################
    # Set initial values
    ################################################################################################
    p_est[0] = gt.p[0]
    v_est[0] = gt.v[0]
    q0 = Quaternion(euler=gt.r[0])
    q_est[0] = q0.to_numpy()
    p_cov[0] = np.zeros(9)  # covariance of estimate
    # gnss_i = 0
    # lidar_i = 0

    ################################################################################################
    # Now that everything is set up, we can start taking in the sensor data and creating estimates
    # for our state in a loop.
    ################################################################################################
    k_l = 0
    k_g = 0
    while lidar.t[k_l] < imu_f.t[1]:
        k_l += 1
    while gnss.t[k_g] < imu_f.t[1]:
        k_g += 1

    ################################################################################################
    # Create an ESEKF object with initial states
    # Set the sensor variances
    ################################################################################################
    es_efk_solver = ESEKF(np.copy(gt.p[0]), np.copy(gt.v[0]), q0)
    es_efk_solver.set_imu_var(var_f=var_imu_f, var_m=var_imu_w)

    ################################################################################################
    # Start taking in the sensor data and creating estimates for the states in a loop
    ################################################################################################
    for k in range(1, imu_f.data.shape[0]):  # start at 1 because we have initial prediction from gt
        delta_t = imu_f.t[k] - imu_f.t[k - 1]

        # Prediction step of the nominal & error states. This happens at a high-rate
        es_efk_solver.prediction(a_m=imu_f.data[k-1], w_m=imu_w.data[k-1], dt=delta_t)

        # If there is no external measurement to correct the predicted state, the predicted one will be considered
        # as estimated state
        p_hat, v_hat, q_hat, p_cov_hat = es_efk_solver.get_states()
        p_est[k] = p_hat
        v_est[k] = v_hat
        q_est[k] = q_hat.to_numpy()
        p_cov[k] = p_cov_hat

        # Correction by Lidar measurements. This happens at a lower rate
        if k_l < lidar.data.shape[0] and k < imu_f.data.shape[0] - 1 and lidar.t[k_l] >= imu_f.t[k] and lidar.t[k_l] < \
                imu_f.t[k + 1]:
            delta_tl = 1.0  # lidar.t[k_l] - lidar.t[k_l -1]
            es_efk_solver.correction(np.eye(3) * var_lidar * (delta_tl ** 2), lidar.data[k_l])

            p_hat, v_hat, q_hat, p_cov_hat = es_efk_solver.get_states()
            p_est[k] = p_hat
            v_est[k] = v_hat
            q_est[k] = q_hat.to_numpy()
            p_cov[k] = p_cov_hat
            k_l += 1

        # Correction by GNSS measurements. This happens at a lower rate
        if k_g < gnss.data.shape[0] and k < imu_f.data.shape[0] - 1 and gnss.t[k_g] >= imu_f.t[k] and gnss.t[k_g] < \
                imu_f.t[k + 1]:
            delta_tg = 1.0  # gnss.t[k_g] - gnss.t[k_g -1]
            es_efk_solver.correction(np.eye(3) * var_gnss * (delta_tg ** 2), gnss.data[k_g])

            p_hat, v_hat, q_hat, p_cov_hat = es_efk_solver.get_states()
            p_est[k] = p_hat
            v_est[k] = v_hat
            q_est[k] = q_hat.to_numpy()
            p_cov[k] = p_cov_hat
            k_g += 1

    ################################################################################################
    # Plot the results to show the ground truth and the estimated trajectories on the same plot.
    # Notice that the estimated trajectory continues past the ground truth. This is because we will be evaluating
    # your estimated poses from the part of the trajectory where you don't have ground truth!
    ################################################################################################

    est_traj_fig = plt.figure()
    ax = est_traj_fig.add_subplot(111, projection='3d')
    ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2], label='Estimated')
    ax.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], label='Ground Truth')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('Ground Truth and Estimated Trajectory')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_zlim(-2, 2)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_zticks([-2, -1, 0, 1, 2])
    ax.legend(loc=(0.62, 0.77))
    ax.view_init(elev=45, azim=-50)

    ################################################################################################
    # Plot the error for each of the 6 DOF, with estimates for the uncertainty included.
    # The error estimates are in blue, and the uncertainty bounds are red and dashed.
    # The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
    ################################################################################################
    error_fig, ax = plt.subplots(2, 3)
    error_fig.suptitle('Error Plots')
    num_gt = gt.p.shape[0]
    p_est_euler = []
    p_cov_euler_std = []

    # Convert estimated quaternions to euler angles
    for i in range(len(q_est)):
        qc = Quaternion(*q_est[i, :])
        p_est_euler.append(qc.to_euler())

        # First-order approximation of RPY covariance
        J = rpy_jacobian_axis_angle(qc.to_axis_angle())
        p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

    p_est_euler = np.array(p_est_euler)
    p_cov_euler_std = np.array(p_cov_euler_std)

    # Get uncertainty estimates from P matrix
    p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

    titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
    for i in range(3):
        ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
        ax[0, i].plot(range(num_gt), 3 * p_cov_std[:num_gt, i], 'r--')
        ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
        ax[0, i].set_title(titles[i])
    ax[0, 0].set_ylabel('Meters')

    for i in range(3):
        ax[1, i].plot(range(num_gt),
                      angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
        ax[1, i].plot(range(num_gt), 3 * p_cov_euler_std[:num_gt, i], 'r--')
        ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
        ax[1, i].set_title(titles[i + 3])
    ax[1, 0].set_ylabel('Radians')
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except:
        print("Whew!", sys.exc_info()[0], "occurred")
