from numpy import ndarray, diag, array, dot, cos, sin, tanh
from numpy.linalg import inv

from dsorlib import State


def rot_matrix_from_B_to_U(yaw):
    """
    Auxiliary function to compute the rotation matrix in the 2D plane for velocities
    expressed in the body frame to velocities expressed in the inertial frame
    :param yaw:
    :return:
    """
    return array([[cos(yaw), -sin(yaw)],
                  [sin(yaw), cos(yaw)]])


class AUVOuterLoop:
    """
    Class responsible for implementing the outer loop that given a reference coordinate [x, y],
    generates a desired surge speed and heading rate for the the inner loops of the vehicle to follow

    The work in this section is heavily inspired by the work of F. Vanni and A. Pascoal in 'Cooperative
    Path-Following of Underactuated Autonomous Marine Vehicles with Logic-based Communications'
    """

    def __init__(self,
                 K: ndarray = array([1.0, 1.0]),
                 delta: float = -1.0):
        """
        Constructor for the AUV outer-loop
        :param K: A gain vector of size 2, with elements > 0
        :param delta: The size of the ball of the error allowed < 0
        """

        # Save the Tuning gains for the importance given to the tracking error in [X, Y]
        self.K: ndarray = diag(K).reshape((2, 2))

        # Save the gamma parameter (which defines the ball of the allowed error)
        self.delta: float = float(delta)

        # Save the delta vector
        self.delta_vec = array([self.delta, 0.0])

        # Matrix delta matrix
        self.delta_matrix = array([[1.0, 0.0],
                                   [0.0, -self.delta]])

        # The inverted delta_matrix
        self.inv_delta_matrix = inv(self.delta_matrix)

    def __call__(self,
                 state: State,
                 desired_pos: ndarray,
                 desired_vel: ndarray,
                 v_disturbances: ndarray = array([0.0, 0.0])):

        # Compute the rotation matrix from the inertial frame to the body frame
        bRu = rot_matrix_from_B_to_U(state.eta_2[2]).transpose()

        # Compute the error in the inertial frame [x, y] - [x_des, y_des]
        error_I: ndarray = state.eta_1[0:2] - desired_pos

        # Convert the error to the body frame
        error_B = dot(bRu, error_I)

        # Get the sway velocity from the state vector
        v_sway = state.v_1[1]

        # Compute the feedback law
        u_d = dot(self.inv_delta_matrix, -dot(self.K, tanh(error_B - self.delta_vec)) - array([0.0, v_sway]) - v_disturbances + dot(bRu, desired_vel))

        # Return the (desired surge speed, desired yaw rate)
        return u_d[0], u_d[1]
