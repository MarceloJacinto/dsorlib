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
from numpy import dot, array
from numpy.linalg import pinv


class AbstractThrusterModel(ABC):
    """
    ThrusterDynamics is an abstract class that serves as a template for implementing the
    dynamics of the thrusters is a given marine AUV

    It provides API to convert from generalized forces and torques applied in the rigid body to a mapping
    of forces in each individual motor and vice-versa.

    NOTE: It is assumed in this class that the mapping between the motors and the generalized vector
    of forces and torques is static (i.e. the robot does not rotate its motors). If you are developing for
    a robot that has rotating axis for the thrusters, overwrite the methods:
        - thrust_to_forceN(self, thrusts)
        - force_to_thrustN(self, forces)

    In addition it also provides the API for converting thrust in [N] to the input
    signal used by the motor and vice-versa

    It also provides de API for the dynamics of the thrusters themselves.
    """

    def __init__(self,
                 number_of_thrusters: int,
                 B: array,
                 min_thrust: float,
                 max_thrust: float):
        """
        Constructor - This class cannot be instantiated and must be inherited

        params:
            number_of_thrusters:int - The number of thrusters in the AUV
            B:array - The thruster allocation matrix, such that [Fx Fy Fz Tx Ty Tx]' = B * [Force_motor_1 Force_motor_2 ... Force_motor_n]'
            min_thrust:float - The minimum thrust applied to the motor in Newton [N]
            max_thrust:float - The maximum thrust applied to the motor in Newton [N]
        """
        if type(self) == AbstractThrusterModel:
            raise Exception("<AbstractThrusterModel> cannot be instantiated. It is an abstract class")

        # The number of thrusters in the vehicle model
        self.number_of_thrusters = int(number_of_thrusters)

        # The allocation matrix B
        # [Fx Fy Fz Tx Ty Tx]' = B * [Force_motor_1 Force_motor_2 ... Force_motor_n]'
        self.B = array(B)

        # The minimum and maximum force we can apply to each motor
        self.min_thrust = float(min_thrust)
        self.max_thrust = float(max_thrust)

        # Calculate the pseudo-inverse of B (the inverse of the allocation matrix)
        # [Force_motor_1 Force_motor_2 ... Force_motor_n]' = pseudo_inv(B) * [Fx Fy Fz Tx Ty Tx]'
        self.pseudo_invB = pinv(self.B)

    def thrust_to_forceN(self, thrusts: array = array([0.0, 0.0, 0.0, 0.0])):
        """
        Thrust to Force method converts a vector of thrust applied to each motor in [N] to the
        generalized vector of forces and torques, that following the SNAME convention are given by:
        thrusts = [motor1, motor2, ..., motor_n]
        forces = [Frb, Nrb] = [X, Y, Z, K, M, N]

        params:
            thrusts = [motor1, motor2, ...] - the thrust in [N] applied to each individual motor

        returns:
            a vector of generalized forces, given by the equation
            [Fx Fy Fz Tx Ty Tx]' = B * [hor_left_thr hor_right_thr ver_back_thr ver_front_thr]'
            expressed in Newton [N]
            where B is the motor allocation matrix

        NOTE: It is assumed in this class that the mapping between the motors and the generalized vector
        of forces and torques is static (i.e. the robot does not rotate its motors). If you are developing for
        a robot that has rotating axis for the thrusters, overwrite the methods:
        - thrust_to_forceN(self, thrusts)
        - force_to_thrustN(self, forces)
        """
        return dot(self.B, array(thrusts).reshape((self.number_of_thrusters,)))

    def force_to_thrustN(self, forces: array = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        """
        Force to Thrust method converts a generalized vector of forces
        following the SNAME convention forces = [Frb, Nrb] = [X, Y, Z, K, M, N]
        and converts it to a vector of forces allocated to each individual motor in Newton [N]

        This method normalizes the forces applied to each thruster so that they are bounded between
        the values specified in the dynamics [dynamics.min_thrust, dynamics.max_thrust] in Newton [N]

        params:
            forces - generalized vector of forces and torques
            following the SNAME convention forces = [Frb, Nrb] = [X, Y, Z, K, M, N]

        returns:
            a vector of thrusts applied to each motor based on the equation
            [hor_left_thr hor_right_thr ver_back_thr ver_front_thr]' = pseudo_inv(B) * [Fx Fy Fz Tx Ty Tx]'
            expressed in Newton [N]

        NOTE: It is assumed in this class that the mapping between the motors and the generalized vector
        of forces and torques is static (i.e. the robot does not rotate its motors). If you are developing for
        a robot that has rotating axis for the thrusters, overwrite the methods:
        - thrust_to_forceN(self, thrusts)
        - force_to_thrustN(self, forces)
        """

        # Calculate how much thrust to allocate to each motor
        thrust_vector = dot(self.pseudo_invB, array(forces).reshape((6,)))

        # Saturate the Thrust in Newton [N] in each motor, so it only attains values
        # between the minimum and maximum force allowed by the motor
        # the goal to preserve the "direction" of the force and torque
        # while at the same time saturating the controls
        max_requested = max(thrust_vector)
        min_requested = min(thrust_vector)

        normalize = 1.0
        normalize = max([abs(min_requested / self.min_thrust), normalize])
        normalize = max([abs(max_requested / self.max_thrust), normalize])

        return (1.0 / normalize) * thrust_vector

    @abstractmethod
    def __copy__(self):
        """
        Implements a deep copy of the object that represents the Thrusters/motors model
        :return: a deep copy of the object
        """
        pass

    @abstractmethod
    def thrust_to_input(self, thrust: array = array([0.0, 0.0, 0.0, 0.0])):
        """
        Thrust to Input is an abstract method that should be implemented by the class that inherits it

        The goal of this method is to convert a given vector of thrusts for each individual motor
        expressed in [N] and convert it to an input signal to the actual motors - the input signal can be
        angular velocity (rad/s), angular velocity (RPM), RPM%, voltage (V) or any other input

        params:
            thrust - a vector of the thrust for each individual motor, expressed in Newton [N]
        returns:
            a vector with the corresponding input signals to the motors
        """
        pass

    @abstractmethod
    def input_to_thrust(self, input: array = array([0.0, 0.0, 0.0, 0.0])):
        """
        Input to Thrust is an abstract method that should be implemented by the class that inherits it

        The goal of this method is to convert a given vector of inputs for each individual motor
        (the input can be in any unit such as rad/s, RPM, RPM%, V, etc, but it has to be consistent with
        the method thrust_to_input which is the inverse model) and convert it to thrust in Newton [N]

        params:
            input - a vector of the input signals to the motors

        returns:
            a vector with the corresponding thrust expressed in Newton [N]
        """
        pass

    @abstractmethod
    def thrusters_dynamic_model(self, input: array = array([0.0, 0.0, 0.0, 0.0])):
        """
        Thrusters dynamic model is an abstract method that should be implemented by the class that inherits it

        The goal of this method is given a vector of inputs for each individual motor compute the actual
        applied input to the model (propagate its dynamics) and return the actual inputs applied to the system

        (the input can be in any unit such as rad/s, RPM, RPM%, V, etc, but it has to be consistent with
        the method input_to_thrust and thrust_to_input)

        params:
            input - a vector of the input signals to the motors

        returns:
            a vector with the actual applied input expressed in the same units as the original input signal
        """
        pass
