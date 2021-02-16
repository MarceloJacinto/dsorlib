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
from numpy import abs
from dsorlib.controllers.inner_loops.pid import PID
from dsorlib.vehicles.state.state import State


class surge_PI:

    def __init__(self, m: float, Xu: float, Xu_dot: float, Xuu: float,
                 surge_gains=(1.0, 1.0),
                 output_bounds=(-50.0, 50.0),
                 dt: float = 0.01):
        """
        This class implements the surge controller for an AUV vehicle operating in a 2D plane
        - A PI (proportional integral) controller for following surge speeds [m/s]

        :param m: The mass of the vehicle in [Kg]
        :param Xu: The linear damping in surge
        :param Xu_dot: The added mass in surge
        :param Xuu: The quadratic damping in surge

        :param surge_gains: The (kp, Ti) PI gains
        :param output_bounds: The minimum and maximum force (in newtons) allowed for the output of the controller
        :param dt: The sampling time used (in seconds)
        """

        # Save the variables of the dynamics of the vehicle
        self._m = m  # The mass of the vehicle in Kg
        self._Xu = Xu  # The linear damping in surge
        self._Xu_dot = Xu_dot  # The added mass in surge
        self._Xuu = Xuu  # The quadratic damping in surge

        # The values for the Kp and Ki
        self._Kp = surge_gains[0]
        self._Ki = surge_gains[1]

        # Surge controller gains and corresponding PI controller
        self._mu = m - Xu_dot  # Total mass in surge = mass - "added mass"
        self._surge_pid = PID(num_states=1, Kp=self._Kp, Ki=self._Ki, Kd=0.0, dt=dt, output_bounds=output_bounds, is_angle=False)

    def follow_surge(self, desired_surge: float, state: State):
        """
        Compute the force in surge to apply using a PI controller
        :param desired_surge: The reference for the surge velocity [m/s]
        :param state: The current state of the vehicle we are trying to control
        :return: The force to apply in surge - Fx
        """

        # Get the current surge velocity in surge
        surge_speed = state.v_1[0]

        # Compute the forward gain to apply
        feed_forward = self._compute_surge_feed_forward(surge_speed)

        # Compute the feedback gain from the PI controller
        self._surge_pid.reference = desired_surge
        feed_back = float(self._surge_pid(sys_output=surge_speed))

        return feed_forward + (self._mu * feed_back)

    def _compute_surge_feed_forward(self, surge_speed: float):
        """
        :param surge_speed: The current vehicle speed in surge [m/s]
        :return: the feed forward control term in surge
        """

        # Compute the damping du
        du = -self._Xu - self._Xuu * abs(surge_speed)

        return du * surge_speed


class yaw_PD:

    def __init__(self, Iz: float, Nr: float, Nr_dot: float, Nrr: float,
                 yaw_gains=(1.0, 1.0),
                 dt: float = 0.01):
        """
        This class implements the heading controller for an AUV vehicle operating in a 2D plane
        - A PD (proportional derivative) controller for following yaw-reference angles [rad]

        :param Iz: The moment of inertia about the z-axis
        :param Nr: The linear damping in yaw_rate
        :param Nr_dot: The added inertia in yaw_rate
        :param Nrr: The quadratic damping in yaw_rate
        :param yaw_gains: The (kp, kd) PD gains

        :param dt: The sampling time used (in seconds)
        """

        # Save the variables of the dynamics of the vehicle
        self._Iz = Iz  # The inertia about z-axis
        self._Nr = Nr  # The linear damping about z-axis
        self._Nr_dot = Nr_dot  # The added mass in yaw-rate
        self._Nrr = Nrr  # The quadratic damping about z-axis

        # The values for the Kp and Ki
        self._Kp = yaw_gains[0]
        self._Kd = yaw_gains[1]

        # Yaw controller gains
        self._mr = Iz - Nr_dot  # The total inertia in yaw_rate = Inertia - "added inertia"
        self._yaw_pid = PID(num_states=1, Kp=self._Kp, Kd=self._Kd, Ki=0.0, dt=dt, is_angle=True)

    def follow_yaw(self, desired_yaw: float, state: State):
        """
        Compute the torque to apply about the z-axis using a PD controller
        :param desired_yaw: The reference for the yaw angle [rad]
        :param state: The current state of the vehicle
        :return: The torque to apply about the z-axis
        """

        # Get the current yaw and yaw_rate velocity
        yaw = state.eta_2[2]
        yaw_rate = state.v_2[2]

        # Compute the forward gain to apply
        feed_forward = self._compute_yaw_feed_forward(yaw_rate)

        # Compute the feedback gain from the PD controller
        self._yaw_pid.reference = float(desired_yaw)
        feed_back = float(self._yaw_pid(sys_output=yaw, sys_output_derivative=yaw_rate))

        return feed_forward + (self._mr * feed_back)

    def _compute_yaw_feed_forward(self, yaw_rate: float):
        """
        :param yaw_rate: The current vehicle yaw rate in [rad/s]
        :return: the feed forward control term in yaw
        """

        # Compute the damping dr
        dr = -self._Nr - self._Nrr * abs(yaw_rate)

        return dr * yaw_rate


class yaw_rate_PI:

    def __init__(self, Iz: float, Nr: float, Nr_dot: float, Nrr: float,
                 yaw_rate_gains=(1.0, 1.0),
                 dt: float = 0.01):
        """
        This class implements the heading controller for an AUV vehicle operating in a 2D plane
        - A PI (proportional integral) controller for following a yaw-rate reference [rad/s]

        :param Iz: The moment of inertia about the z-axis
        :param Nr: The linear damping in yaw_rate
        :param Nr_dot: The added inertia in yaw_rate
        :param Nrr: The quadratic damping in yaw_rate
        :param yaw_rate_gains: The (kp, ki) PI gains

        :param dt: The sampling time used (in seconds)
        """

        # Save the variables of the dynamics of the vehicle
        self._Iz = Iz  # The inertia about z-axis
        self._Nr = Nr  # The linear damping about z-axis
        self._Nr_dot = Nr_dot  # The added mass in yaw-rate
        self._Nrr = Nrr  # The quadratic damping about z-axis

        # The values for the Kp and Ki
        self._Kp = yaw_rate_gains[0]
        self._Ki = yaw_rate_gains[1]

        # Yaw controller gains
        self._mr = Iz - Nr_dot  # The total inertia in yaw_rate = Inertia - "added inertia"
        self._yaw_pid = PID(num_states=1, Kp=self._Kp, Ki=self._Ki, Kd=0.0, dt=dt, is_angle=False)

    def follow_yaw_rate(self, desired_yaw_rate: float, state: State):
        """
        Compute the torque to apply about the z-axis using a PD controller
        :param desired_yaw_rate: The reference for the yaw rate [rad/s]
        :param state: The current state of the vehicle
        :return: The torque to apply about the z-axis
        """

        # Get the current yaw and yaw_rate velocity
        yaw_rate = state.v_2[2]

        # Compute the forward gain to apply
        feed_forward = self._compute_yaw_feed_forward(yaw_rate)

        # Compute the feedback gain from the PD controller
        self._yaw_pid.reference = float(desired_yaw_rate)
        feed_back = float(self._yaw_pid(sys_output=yaw_rate))

        return feed_forward + (self._mr * feed_back)

    def _compute_yaw_feed_forward(self, yaw_rate: float):
        """
        :param yaw_rate: The current vehicle yaw rate in [rad/s]
        :return: the feed forward control term in yaw
        """

        # Compute the damping dr
        dr = -self._Nr - self._Nrr * abs(yaw_rate)

        return dr * yaw_rate
