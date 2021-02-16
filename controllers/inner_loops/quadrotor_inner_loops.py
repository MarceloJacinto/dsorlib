#  MIT License
#
#  Copyright (c) 2021 Marcelo Jacinto
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
from numpy import ndarray, array, ones, zeros, cos, sin, dot, arcsin, arctan2, clip
from numpy.linalg import norm

from dsorlib.controllers import PID


class QuadrotorInnerLoop:

    def __init__(self, Kp: ndarray, Kd: ndarray, K1: ndarray, K2: ndarray,
                 m: float, I: ndarray, g: float = 9.8, dt=0.01):
        # Constants for position control
        self.Kp: ndarray = array(Kp)
        self.Kd: ndarray = array(Kd)

        # Constants for attitude control
        self.K1: ndarray = array(K1)
        self.K2: ndarray = array(K2)

        # The quadrotor model constants
        self.g: float = float(g)
        self.m: float = float(m)
        self.I: ndarray = array(I).reshape((3, 3))

        # Define the limits for the position PD error to be -4 and 4
        self.pos_lower_bounds = -4.0 * ones(3)
        self.pos_upper_bounds = 4.00 * ones(3)

        # Define the limits for the angle PD error
        self.des_ang_lower_bounds = -10  # [deg]
        self.des_ang_upper_bounds = 10  # [deg]

        # The PD to actually do the computations for the position error
        self.position_pd = PID(num_states=3,
                               Kp=self.Kp,
                               Kd=self.Kd,
                               Ki=zeros(3),
                               dt=dt,
                               output_bounds=(self.pos_lower_bounds, self.pos_upper_bounds),
                               is_angle=[False, False, False])

        # The PD to do the computations for the angle errors
        self.angle_pd = PID(num_states=3,
                            Kp=self.K1,
                            Kd=self.K2,
                            Ki=zeros(3),
                            dt=dt,
                            output_bounds=None,
                            is_angle=[True, True, True])

    def position_controller(self,
                            desired_pos: ndarray,
                            eta_1: ndarray,
                            eta_2: ndarray,
                            eta_1_dot: ndarray):
        """
        """

        # Set the correct reference for the PD controller to follow
        self.position_pd.reference = array(desired_pos)

        # Compute the actual feedback value
        feedback = self.position_pd(sys_output=eta_1,
                                    sys_output_derivative=eta_1_dot)

        # Compute the value of u_t
        u_t = feedback - array([0.0, 0.0, self.g])

        # Compute the desired thrust
        thrust = self.m * norm(u_t)

        # Normalize the feedback vector
        r3d = -(1.0 / norm(u_t)) * u_t

        # Get the current yaw angle
        yaw = eta_2[2]

        # Compute the rotation matrix about the z-axis
        Rot_yaw = array([[cos(yaw), - sin(yaw), 0],
                         [sin(yaw), cos(yaw), 0],
                         [0, 0, 1]])

        # Compute the desired roll and pitch
        Z = self.m * dot(Rot_yaw, r3d)
        des_roll = arcsin(-Z[1])
        des_pitch = arctan2(Z[0], Z[2])

        return thrust, des_roll, des_pitch

    def attitude_controller(self, des_roll: float, des_pitch: float, des_yaw: float, eta_2: ndarray, v_2: ndarray):
        # Clip the desired angles so that they vary only between acceptable bounds
        des_roll = clip(des_roll, self.des_ang_lower_bounds, self.des_ang_upper_bounds)
        des_pitch = clip(des_pitch, self.des_ang_lower_bounds, self.des_ang_upper_bounds)

        # Set the correct reference for the PD controller to follow
        self.angle_pd.reference = array([des_roll, des_pitch, des_yaw])

        # Get the actual feedback value
        feedback = self.angle_pd(sys_output=eta_2, sys_output_derivative=v_2)

        # Multiply by the inertia matrix and return
        torques = dot(self.I, feedback)

        return torques[0], torques[1], torques[2]
