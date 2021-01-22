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

from numpy import array

from dsorlib.vehicles.thrusters.abstract_thruster_model import AbstractThrusterModel


class PerfectThrusterModel(AbstractThrusterModel):
    """
    PerfectThrusterModel is an implementation for the thruster models where an input of the thruster
    model is exactly the same as the output of the thrusters

    We can think of this model of a direct connection between the input or the output, as if the motors
    had no dynamics - instantaneous motors

    This class inherits the AbstractThrusterModel class. For more information refer to the documentation for AbstractThrusterModel
    """

    def __init__(self,
                 number_of_thrusters: int,  # The number of thrusters
                 B: array,                  # The thruster allocation matrix
                 min_thrust: float,         # The minimum thrust that can be applied in Newton [N]
                 max_thrust: float):        # The maximum thrust that can be applied in Newton [N]
        """
        The Constructor for the linear model of the thrusters

        params:
            - number_of_thrusters: int - The number of thrusters
            - B: array - The thruster allocation matrix with size (6 x num_thrusters)
            - min_thrust: float - The minimum thrust that can be applied in Newton [N]
            - max_thrust: float - The maximum thrust that can be applied in Newton [N]
        """

        # Initialized the Super Class elements
        super().__init__(number_of_thrusters, B, min_thrust, max_thrust)

    def __copy__(self):
        return PerfectThrusterModel(self.number_of_thrusters, self.B, self.min_thrust, self.max_thrust)

    def thrust_to_input(self, thrust=[0.0, 0.0, 0.0, 0.0]):
        """
        params:
            thrust - a vector of the thrust for each individual motor, expressed in Newton [N]
        returns:
            a vector with the corresponding input signals to the motors
        """

        # Linear model - does nothing (output=input)
        return array(thrust)

    def input_to_thrust(self, input=[0.0, 0.0, 0.0, 0.0]):
        """
        params:
            input - a vector of the input signals to the motors (RPM%)

        returns:
            a vector with the corresponding thrust expressed in Newton [N]
        """

        # Linear model - does nothing (output=input)
        return array(input)

    def thrusters_dynamic_model(self, input=[0.0, 0.0, 0.0, 0.0]):
        """
        params:
            input - a vector of the input signals to the motors (RPM%)

        returns:
            a vector with the actual applied thrust expressed in the same units as the original input signal (RPM%)
        """

        # Linear model - does nothing (output=input)
        return array(input)
