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
from numpy import ndarray, array, zeros, dot, clip

from dsorlib.vehicles.thrusters.thruster import Thruster

# Type to ensure that the list passed to the ThrusterAllocator class is actual thrusters
ThrustersList = list[Thruster]


class ThrusterAllocator(ABC):

    def __init__(self, thrusters: ThrustersList):
        """
        The constructor for the abstract class ThrusterAllocator. The goal of this class is to provide
        an interface for thruster allocation
        :param thrusters: A list of thrusters objects
        """
        if type(self) == ThrusterAllocator:
            raise Exception("<ThrusterAllocator> cannot be instantiated. It is an abstract class")

        # Save and check the inputs received by the constructor
        self.number_of_thrusters: int = len(thrusters)

        # Check if the list of thrusters contains actual thruster objects
        self.thrusters = thrusters

    def _thrust_to_force(self, B: ndarray, thrusts: ndarray):
        """
        Thrust to Force method converts a vector of thrust applied to each motor in [N] to the
        generalized vector of forces and torques, that following the SNAME convention are given by:
        thrusts = [motor1, motor2, ..., motor_n]
        forces = [Frb, Nrb] = [X, Y, Z, K, M, N]

        :param B: The allocation matrix
        :param thrusts: = [thruster1, thruster2, ...] - the thrust in [N] applied to each individual motor

        :returns:
            a vector of generalized forces, given by the equation
            [Fx Fy Fz Tx Ty Tx]' = B * [thruster1 thruster2 thruster3 thruster4]'
            expressed in Newton [N]
            where B is the motor allocation matrix
        """

        return dot(B, array(thrusts).reshape(self.number_of_thrusters))

    def apply_thrusters_dynamics(self, thrusts: ndarray):
        """
        Propagates the desired thrusts through the thrusters dynamical models and returns the real
        thrust applied in each thruster in [N]

        :param thrusts: The desired thrust to apply to each motor (in Newton) [N]
        :return: The actual thrust applied by each motor (in Newton) [N]
        """

        actual_applied_thrusts = zeros(self.number_of_thrusters)

        # Propagate the dynamical model through each individual thruster
        for thruster, i in zip(self.thrusters, range(self.number_of_thrusters)):

            # Convert the force in [N] to the input unit of the system (for example RPM)
            input_value = thruster.force_to_input_unit(thrusts[i])

            # Saturate the input of the system between the minimum and maximum values
            min_val, max_val = thruster.input_bounds
            input_value = clip(input_value, min_val, max_val)

            # Apply the dynamics of that thruster
            actual_applied_value = thruster.dynamic_model(input_value)

            # Convert the actual applied value back to forces in [N]
            actual_applied_thrusts[i] = thruster.input_unit_to_force(actual_applied_value)

        return actual_applied_thrusts

    @abstractmethod
    def convert_thrusts_to_general_forces(self, thrusts: ndarray):
        pass

    @abstractmethod
    def convert_general_forces_to_thrusts(self, forces: ndarray):
        pass



