from numpy import array, dot, cos, sin, arcsin, arctan2, pi, power, sqrt, clip
from numpy.linalg import norm, inv

from dsorlib.vehicles.state.state import State


class DesiredState:
    """
    DesiredState class is essentially a structure to save all the desired values
    for our quadrotor to follow
    """

    def __init__(self, pos=array([0.0, 0.0, -5.0]),
                 vel=array([0.0, 0.0, 0.0]),
                 acc=array([0.0, 0.0, 0.0]),
                 yaw: float = 0.0,
                 yaw_rate: float = 0.0):
        """
        :param pos: The desired position for the quadrotor [x, y, z]
        :param vel: The desired velocity for the quadrotor [x_dot, y_dot, z_dot]
        :param acc: The desired acceleration for the quadrotor [x_ddot, y_ddot, z_ddot]
        :param yaw: The desired orientation (yaw angle)
        :param yaw_rate: The desired heading rate (velocity in yaw)
        """
        self.pos = array(pos)
        self.vel = array(vel)
        self.acc = array(acc)
        self.yaw: float = yaw
        self.yaw_rate: float = yaw_rate


class QuadrotorInnerLoop:

    def __init__(self,
                 Kp: float,
                 Kd: float,
                 k1: float,
                 k2: float,
                 m: float,
                 I: array,
                 g: float=9.8):

        # Constants for position control
        self.Kp = float(Kp)
        self.Kd = float(Kd)

        # Constants for attitude control
        self.k1 = float(k1)
        self.k2 = float(k2)

        self.g: float = float(g)
        self.m: float = float(m)
        self.I = I

    def pos_controller(self, desired_state: DesiredState, state: State):
        """
        Position controller - given a set of desired position, velocity and acceleration
        for the quadrotor and its current state:
            1) Compute the thrust to apply to the quadrotor
            2) Compute the 3rd column of the desired rotation matrix to be followed

        :param desired_state: A DesiredState object with desired position, velocity and acceleration
        :param state: The current state of the quadrotor AUV
        :return: A tuple with (thrust, r3d), where thrust is the total force in Z to apply to the quadrotor
        and r3d is the desired 3rd column of the rotation matrix
        """

        # Compute the error between the current position and desired position
        pos_error = array(state.eta_1) - desired_state.pos
        vel_error = array(state.eta_1_dot) - desired_state.vel

        # Construct the input signal vector for the translational dynamics controller
        u_t = -(self.Kp * pos_error) - (self.Kd * vel_error) - array([0, 0, self.g]) + desired_state.acc

        # Compute the thrust to apply to the quadrotor
        output = -self.m * norm(u_t)

        # Compute the desired angles for the attitude controller to follow
        yaw_des = desired_state.yaw
        Rz = array([[sin(yaw_des), -cos(yaw_des)],
                    [cos(yaw_des), sin(yaw_des)]])
        roll_pitch_des = dot(Rz, -u_t[0:2] / norm(u_t))
        angle_des = array([roll_pitch_des[0], roll_pitch_des[1], yaw_des])

        return output, angle_des

    def att_controller(self, angle_desired: array, desired_state: DesiredState, state: State):

        # Check if the desired angles are to big and clip them between -10 and 10 deg
        angle_desired[0:2] = clip(angle_desired[0:2], -0.17, 0.17)

        # Compute the error between the desired angles and the real angles
        angle_error = array(state.eta_2) - angle_desired
        angle_vel_error = array(state.v_2) - array([0.0, 0.0, desired_state.yaw_rate])

        # Wrap the angle error between -pi and pi
        while angle_error[0] > pi:
            angle_error[0] -= 2*pi
        while angle_error[0] < -pi:
            angle_error[0] += 2*pi

        while angle_error[1] > pi:
            angle_error[1] -= 2*pi
        while angle_error[1] < -pi:
            angle_error[1] += 2*pi

        # Compute the output
        output = -(self.k1 * angle_error) -(self.k2 * angle_vel_error)

        return dot(self.I, output)