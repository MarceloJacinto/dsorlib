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
from numpy import ndarray, array, zeros, pi, greater
from dsorlib.utils import integrate

# Auxiliary type to make sure we receive a list of booleans
BooleanList = list[bool]


class PID:
    """
    PIDVector implements a PID controller for a vector case (in an efficient manner)
    """

    def __init__(self, num_states: int, Kp, Kd, Ki, reference=None,
                 dt: float = 0.01,
                 output_bounds: (ndarray, ndarray) = (None, None),
                 is_angle: BooleanList = None):

        # Define the number of states that we are controlling with the PID
        self.num_states: int = int(num_states)

        # The sampling time expressed in [s]
        self._dt: float = float(dt)

        # Validate the sampling time used
        if dt < 0.0:
            raise ValueError("The sampling time must be greater or equal than 0.")

        # The gains for the controller
        self.Kp: ndarray = array(Kp).reshape(num_states)
        self.Kd: ndarray = array(Kd).reshape(num_states)
        self.Ki: ndarray = array(Ki).reshape(num_states)

        # Check if the gains received are valid
        for i in range(num_states):
            if self.Kp[i] < 0.0 or self.Kd[i] < 0.0 or self.Ki[i] < 0.0:
                raise ValueError("The gains for the PID must be greater or equal than 0!")

        # The references for the PIDs
        if reference is None:
            self._reference = zeros(num_states)
        else:
            self._reference: ndarray = array(reference, dtype=float).reshape(num_states)

        # Set the output bounds for the controller error
        if output_bounds is not None and output_bounds[0] is not None and output_bounds[1] is not None:

            # If we only have one state, we must wrap the float with a list
            if num_states == 1:
                self._output_min: ndarray = array([output_bounds[0]], dtype=float)
                self._output_max: ndarray = array([output_bounds[1]], dtype=float)
            else:
                self._output_min: ndarray = array(output_bounds[0], dtype=float)
                self._output_max: ndarray = array(output_bounds[1], dtype=float)

            # Check if the controller output bounds are valid
            for i in range(num_states):
                if num_states > 1 and self._output_min[i] > self._output_max[i]:
                    raise ValueError("The lower bound for the output must be lower than the upper bounds!")
                elif num_states == 1 and self._output_min > self._output_max:
                    raise ValueError("The lower bound for the output must be lower than the upper bounds!")
        else:
            self._output_min = None
            self._output_max = None

        # Check if the is_angle list is valid
        if is_angle is not None:

            if num_states > 1:
                self._is_angle: BooleanList = is_angle
            else:
                self._is_angle = [is_angle]

            if len(self._is_angle) != self.num_states:
                raise ValueError("We should have a true or false for each state")
        else:
            self._is_angle = None

        # Auxiliary variable for the integral and anti-windup mechanism
        # if the system varies quickly
        self._prev_output: ndarray = zeros(self.num_states)     # The value on the previous iteration for the derivative
        self._integral_error: ndarray = zeros(self.num_states)  # The integral accumulated error on the previous iteration
        self._anti_windup: ndarray = zeros(self.num_states)     # The anti-windup gain

    def __call__(self, sys_output: ndarray, sys_output_derivative: ndarray = None):

        # Compute the proportional error
        error = sys_output - self._reference

        # Check if the error is an angle (the error between 2 angles is bounded between -pi, pi)
        for i in range(self.num_states):
            if self._is_angle[i]:
                while error[i] < -pi:
                    error[i] += (2 * pi)
                while error[i] > pi:
                    error[i] -= (2 * pi)

        # Compute the integral error-anti_windup
        integral_error = integrate(x_dot=error - self._anti_windup, x=self._integral_error, dt=self._dt)

        # Update the integral error for the next iteration
        self._integral_error = integral_error

        # If the derivative is given, then just save it
        if sys_output_derivative is not None:
            derivative_error: ndarray = array(sys_output_derivative).reshape(self.num_states)
        else:
            # If the derivative of the signal is not given, then compute it
            derivative_error: ndarray = (sys_output - self._prev_output) / self._dt

        # Save the current output for the next derivative gain update
        self._prev_output = sys_output

        # Compute the control law
        output = -(self.Kp * error) - (self.Kd * derivative_error) - (self.Ki * integral_error)

        # Saturate the output between a minimum and maximum value
        # and compute the value for the anti-windup system to feedback
        if self._output_min is not None:
            self._anti_windup[greater(self._output_min, output)] = output[greater(self._output_min, output)] - self._output_min[greater(self._output_min, output)]
            output[greater(self._output_min, output)] = self._output_min[greater(self._output_min, output)]

        if self._output_max is not None:
            self._anti_windup[greater(output, self._output_max)] = output[greater(output, self._output_max)] - self._output_min[greater(output, self._output_max)]
            output[greater(output, self._output_max)] = self._output_max[greater(output, self._output_max)]

        # If the number of states is just one, return the control law as a float an not as a numpy array
        if self.num_states == 1:
            return float(output)

        return output

    def reset(self):
        """
        Reset the PID controller, but keeping the gains
        """
        self._reference: ndarray = zeros(self.num_states)
        self._prev_output: ndarray = zeros(self.num_states)
        self._integral_error: ndarray = zeros(self.num_states)
        self._anti_windup: ndarray = zeros(self.num_states)

    @property
    def reference(self):
        """
        :return: Returns the reference the PID is trying to follow
        """
        return self._reference

    @reference.setter
    def reference(self, ref_value):
        """
        Update the reference for the PID to follow
        :param ref_value: A float or a numpy array with values for reference
        """
        self._reference = array(ref_value).reshape(self.num_states)

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
        :return: (Kp, Ki, Kd)
        """
        return self.Kp, self.Ki, self.Kd

    @gains.setter
    def gains(self, Kp: ndarray, Kd: ndarray, Ki: ndarray):
        """
        Sets the controller gains used (Kp, Ki, Kd)
        where Kp-proportional gain, Ki-integral gain and Kd-derivative gain
        """

        new_Kp = array(Kp).reshape(self.num_states)
        new_Kd = array(Kd).reshape(self.num_states)
        new_Ki = array(Ki).reshape(self.num_states)

        for i in range(self.num_states):
            if new_Kp[i] < 0.0 or new_Kd[i] < 0.0 or new_Ki[i] < 0.0:
                raise ValueError('The gains must all be positive floats')

        self.Kp = new_Kp
        self.Kd = new_Kd
        self.Ki = new_Ki

    @property
    def is_angle(self):
        """
        :return: If the reference to the controller is an angle (to be wrapped between -pi and pi)
        """
        return self._is_angle

    @is_angle.setter
    def is_angle(self, is_angle: BooleanList):
        """
        Updates whether the reference value is an angle or not
        in order to wrap the error between -pi and pi accordingly
        :param is_angle: a boolean list to check if the reference value is an angle or not
        """

        if is_angle is not None:
            self._is_angle: BooleanList = list(is_angle)
            if len(self._is_angle) != self.num_states:
                raise ValueError("We should have a true or false for each state")
        else:
            self._is_angle = None

