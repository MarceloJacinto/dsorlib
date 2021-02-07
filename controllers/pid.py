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
from numpy import pi
from dsorlib.utils import integrate


class PID:
    """
    PID class implements a simple PID controller
    """

    def __init__(self, Kp: float=0.0, Td: float=0.0, Ti: float=0.0,
                 dt:float=0.01,
                 reference: float=0.0,
                 output_bounds=(None, None),
                 is_angle=False):
        """
        Instantiates an instance of a PID controller
        :param Kp: The proportional gain
        :param Td: The derivative gain
        :param Ti: The integral gain
        :param dt: The time interval (in seconds)
        :param output_bounds: The (lower, upper) limits for the output control signal
        :param is_angle: A boolean to check whether the reference is an angle (to wrap the error or not)
        """

        # The sampling time expressed in [s]
        self._dt: float = float(dt)

        # Save the gains for the PID
        self.Kp: float = float(Kp)
        self.Ti: float = float(Ti)
        self.Td: float = float(Td)

        # Save the regulated level to maintain (reference to follow)
        self.reference: float = float(reference)

        # The lower and upper bounds of the control signal
        self._output_min = output_bounds[0]
        self._output_max = output_bounds[1]

        # Save whether this is a PID to control an angle or not (in order to wrap the error between -pi and pi)
        self._is_angle: bool = is_angle

        # Auxiliary variable for the integral and anti-windup mechanism
        # TODO - check if the _prev_output = 0.0 does not raise problems for derivative in the initial second
        # if the system varies quickly
        self._prev_output: float = 0.0 # The value on the previous iteration for the derivative
        self._integral_error: float = 0.0   # The integral accumulated error on the previous iteration
        self._anti_windup: float = 0.0  # The anti-windup gain

    def __call__(self, sys_output: float, sys_output_derivative=None):
        """
        Update the control law for the PID

        |Ref|--->|"piece of code"|--->|PID|--->|Plant|----->
                        ^                               |
                        |_______________________________|

        :param sys_output: The output of the system that we want to regulate
        :param sys_output_derivative: The derivative of the output (if have a direct measurement - to avoid noise)
        :return: the update control law
        """

        # Compute the proportional error
        error = sys_output - self._reference

        # Check if the error is an angle (the error between 2 angles cannot ever be superior to pi)
        if self._is_angle:
            while error > pi:
                error -= (2 * pi)
            while error < -pi:
                error += (2 * pi)

        # Compute the integral error-anti_windup
        integral_error = integrate(x_dot=error - self._anti_windup, x=self._integral_error, dt=self._dt)

        # Update the integral error for the next iteration
        self._integral_error = integral_error

        # If the derivative is given, then just save it
        if sys_output_derivative is not None:
            derivative_error = float(sys_output_derivative)
        else:
            # If the derivative of the signal is not given, then compute it
            derivative_error = (sys_output - self._prev_output) / self._dt

        # Save the current output for the next derivative gain update
        self._prev_output = sys_output

        # Compute the control law
        output = -self.Kp * (error + (self.Td * derivative_error) + (1.0 / self.Ti * integral_error))

        # Saturate the output between a minimum and maximum value
        # and compute the value for the anti-windup system to feedback
        if output < self._output_min:
            self._anti_windup = output - self._output_min
            output = self._output_min
        elif output > self._output_max:
            self._anti_windup = output - self._output_min
            output = self._output_max

        return output

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'Kp={self._Kp!r}, Ti={self._Ti!r}, Td={self._Td!r}, '
            'sample_time={self._dt!r}'')').format(self=self)

    def reset(self):
        """
        Reset the PID controller, but keeping the gains
        """
        self.reference: float = 0.0
        self._prev_output: float = 0.0
        self._integral_error: float = 0.0
        self._anti_windup: float = 0.0

    @property
    def sampling_time(self):
        """
        :return: the sampling time used by the controller expressed in seconds
        """
        return self._dt

    @sampling_time.setter
    def sampling_time(self, dt: float):
        """
        :param dt: The sampling time to be used by the pid (in seconds)
        """
        if float(dt) < 0:
            raise ValueError('The sampling time must be greater than 0.')
        self._dt = float(dt)

    @property
    def gains(self):
        """
        Returns the controller gains used
        and the gain_pid which is the gain to which the controller output is multiplied by
        :return: (Kp, Ti, Td)
        """
        return self._Kp, self._Ti, self._Td

    @gains.setter
    def gains(self, gains):
        """
        Sets the controller gains used (Kp, Ti, Td)
        where Kp-proportional gain, Ki-integral gain and Kd-derivative gain
        """
        if None in gains or (gains[0]<0 and gains[1]<0 and gains[2]<0):
            raise ValueError('The gains must all be positive floats')

        self._Kp, self._Ti, self._Td = gains

    @property
    def is_angle(self):
        """
        :return: If the reference to the controller is an angle (to be wrapped between -pi and pi)
        """
        return self._is_angle

    @is_angle.setter
    def is_angle(self, is_angle: bool):
        """
        Updates whether the reference value is an angle or not
        in order to wrap the error between -pi and pi accordingly
        :param is_angle: a boolean to check if the reference value is an angle or not
        """
        self._is_angle: bool = bool(is_angle)
