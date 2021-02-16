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
from abc import ABC, abstractmethod


class Thruster(ABC):

    def __init__(self, min_input: float, max_input: float):
        """
        The Constructor for an abstract class called
        :param min_input: The minimum value for the thruster (for example -4500 RPM)
        :param max_input: The maximum value for the thruster (for example 4500 RPM)
        """

        if type(self) == Thruster:
            raise Exception("<Thruster> cannot be instantiated. It is an abstract class")

        # Save the minimum and maximum input values for the thrusters (in %RPM, RPM or rad/s for example)
        self._min_input: float = float(min_input)
        self._max_input: float = float(max_input)

        # Compute the corresponding minimum and maximum force (in Newtons) that the thruster can apply
        # corresponding to the minimum and maximum inputs
        self._min_force: float = self.input_unit_to_force(self._min_input)
        self._max_force: float = self.input_unit_to_force(self._max_input)

    @property
    def input_bounds(self):
        """
        Getter for the input bounds of the thruster (%RPM, RPM or rad/s for example)
        :return: A tuple of floats with (min_value, max_value)
        """
        return self._min_input, self._max_input

    @input_bounds.setter
    def input_bounds(self, min_input: float, max_input: float):
        """
        Setter for the input bounds of the thruster (%RPM, RPM or rad/s for example)
        :param min_input: The minimum value for the thruster (for example -4500 RPM)
        :param max_input: The maximum value for the thruster (for example 4500 RPM)
        """

        # Check that the minimum input is lower then the max input
        if type(min_input) != float or type(max_input) != float or min_input > max_input:
            raise ValueError("The minimum value must be lower then the maximum value.")

        # Save the new bounds for the thruster
        self._min_input: float = float(min_input)
        self._max_input: float = float(max_input)

        # Update the minimum and maximum force bounds (in Newtons)
        self._min_force: float = self.input_unit_to_force(self._min_input)
        self._max_force: float = self.input_unit_to_force(self._max_input)

    @property
    def force_bounds(self):
        """
        Getter for the minimum and maximum force that the thruster can apply (in Newtons)
        :return: A tuple of floats with (min_force, max_force)
        """
        return self._min_force, self._max_force

    @abstractmethod
    def force_to_input_unit(self, force_val: float):
        """
        Abstract method that when implemented should convert the force each thruster applied (in Newton)
        to the input value that the dynamics are working with (for example %RPM, RPM or rad/s)
        :param force_val: The force applied, in Newton
        :return: the corresponding value as input (in %RPM, RPM or rad/s) for example

        Note: This implementation must be consistent with the implementation of the method 'input_unit_to_force'
        """
        pass

    @abstractmethod
    def input_unit_to_force(self, input_val: float):
        """
        Abstract method that when implemented should convert the input value that the dynamics are working with (for
        example %RPM, RPM or rad/s) to the actual force applied by the thruster (in Newton)
        :param input_val: The input value (in %RPM, RPM or rad/s) - for example
        :return: the corresponding force applied, in Newton

        Note: This implementation must be consistent with the implementation of the method 'force_to_input_unit'
        """
        pass

    @abstractmethod
    def dynamic_model(self, input_val: float) -> float:
        """
        Abstract method that when implemented should propagate the desired input_val (in %RPM, RPM or rad/s - for example)
        and propagate through the dynamic model of the thruster. Should return the actual value applied by the thruster
        (in %RPM, RPM or rad/s - for example)
        :param input_val: The input value (in %RPM, RPM or rad/s) - for example
        :return: the actual applied value (in %RPM, RPM or rad/s) for example
        """
        pass
