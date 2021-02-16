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
from numpy import ndarray, array, zeros, power, sqrt, exp, clip, around

from dsorlib.vehicles.thrusters.thruster import Thruster


class SimplePoleThruster(Thruster):

    def __init__(self,
                 min_input: float,
                 max_input: float,
                 thruster_map_positive: ndarray,
                 thruster_map_negative: ndarray,
                 delay: float,
                 pole: float,
                 dt: float):
        """
        QuadraticThruster is an implementation for a Thruster model that:
        - Dynamics can be described by a pole and a delay (in continuous time)
        - The map between force (in Newtons) and the input unit (%RPM, RPM or rad/s) can be mapped by using half of
        quadratic equations for the negative and positive sides

        For more information regarding implementing your own Thruster model, check the abstract class Thruster

        :param min_input: The minimum value for the thruster (for example -4500 RPM)
        :param max_input: The maximum value for the thruster (for example 4500 RPM)
        :param thruster_map_positive: Values for the positive side of the quadratic mapping between force and input [a, b]
        :param thruster_map_negative: Values for the negative side of the quadratic mapping between force and input [a, b]
        :param delay: The delay in the continuous time model of the thrusters
        :param pole: The position of the pole (in continuous time) for the thruster
        :param dt: The sampling period for this discretized model
        """

        # Save the values for the quadratic that allow the mapping between the forces and input values
        self.thruster_map_positive: ndarray = array(thruster_map_positive).reshape(2)
        self.thruster_map_negative: ndarray = array(thruster_map_negative).reshape(2)

        # Save the delay
        self.delay: float = float(delay)
        if delay < 0:
            raise ValueError("The delay of the motor cannot be negative")

        # Save the pole location
        self.pole: float = float(pole)

        # Save the sampling period
        self.period: float = float(dt)
        if self.period <= 0:
            raise ValueError("The sampling time must be greater than 0")

        # Auxiliary variables for the dynamics calculations (save the outputs in previous time steps)
        # Compute the number of delays that we get from the conversion of the continuous time model to discrete difference equations model
        self.num_delays_input = int(round(self.delay / self.period))

        # Create a circular buffer to store the previous input of the thruster model for the corresponding delays
        # Save input in (k-1, k-2, ..., k-num_delays)
        self.circular_buffer: ndarray = zeros(self.num_delays_input)

        # Create a variable to save the previous output (in k-1 timestep)
        self.y_1: float = 0.0

        # Initialized the Super Class (Thruster) elements
        super().__init__(min_input, max_input)

    def force_to_input_unit(self, force_val: float):
        """
        Convert the force in Newton [N] to the unit of input of the thruster dynamics system (in %RPM, RPM or rad/s for example)

        :param force_val: The force applied in Newton [N]
        :return: The corresponding force in the input unit (in %RPM, RPM or rad/s for example)
        """

        # Check whether the force applied is negative or positive
        if force_val >= 0:
            a = self.thruster_map_positive[0]
            b = self.thruster_map_positive[1]
        else:
            a = self.thruster_map_negative[0]
            b = self.thruster_map_negative[1]

        # Compute the corresponding input in %RPM, RPM or rad/s (for example)
        return (-b + sqrt(power(b, 2) + 4 * a * force_val)) / (2 * a)

    def input_unit_to_force(self, input_val: float):
        """
        Convert the unit of input of the system to Newton

        :param input_val: The input value for the thruster (in %RPM, RPM or rad/s for example)
        :return: The corresponding force in Newton [N]
        """

        # Limit the input
        min_val, max_val = self.input_bounds
        input_val = clip(input_val, min_val, max_val)

        # Check whether the input value is positive or negative and choose the parameters of the curve accordingly
        if input_val >= 0:
            a = self.thruster_map_positive[0]
            b = self.thruster_map_positive[1]
        else:
            a = self.thruster_map_negative[0]
            b = self.thruster_map_negative[1]

        # Return the force float Force[N] = a * input^2 + b*input
        return (a * power(input_val, 2)) + (b * input_val)

    def dynamic_model(self, input_val: float) -> float:
        """
        Simulates the dynamic model of the thruster. Receives as a parameters the desired
        input to give to the thruster (in RPM, %RPM or rad/s for example) and returns
        what the actual thruster will output (in the same unit)

        :param input_val: The thrust desired to apply to the thruster
        :return: The actual applied in thrust
        """

        # load buffer input = u[k-delay]
        u_old = self.circular_buffer[0]

        # Calculate the y[k] in the difference equations
        y = (exp(-self.pole * self.period) * self.y_1) + ((-exp(-self.pole * self.period) + 1) * u_old)

        # Rotate the buffer (and discard the oldest sample)
        self.circular_buffer[0:-1] = self.circular_buffer[1:]

        # Saturate the input between the allowed bounds and insert the new input in the end of the circular buffer
        input_val = clip(input_val, self._min_input, self._max_input)
        self.circular_buffer[-1] = input_val

        # Save the new output to used in the next iteration
        self.y_1 = y

        # Saturate the output between the minimum and maximum values
        y = clip(y, self._min_input, self._max_input)

        return around(y)

