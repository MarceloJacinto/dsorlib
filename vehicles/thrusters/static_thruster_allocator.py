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
from numpy import ndarray, array, zeros, dot, clip
from numpy.linalg import pinv
from dsorlib.vehicles.thrusters.thruster_allocater import ThrusterAllocator, ThrustersList


class StaticThrusterAllocator(ThrusterAllocator):

    def __init__(self,
                 B: ndarray,
                 thrusters: ThrustersList,
                 normalize: bool = True):
        """
        A class that implements ThrusterAllocator class. In here, the thruster allocation
        is done statically resorting to the allocation matrix and its pseudo-inverse.

        When the force asked to each thruster goes beyond its minimum and maximum values, a
        scaling of the force applied to each thruster is scaled in order to preserve the
        final orientation of the forces and torques applied on the vehicle

        :param B: The allocation numpy matrix of size[num_of_thrusters, 6]
        :param thrusters: A list of thrusters to use
        :param normalize: A boolean to choose whether to normalize the vector of thrusts
            of just clamp the value of the allocation. If normalize=True will normalize, otherwise
            it will clamp between min and max values for each thruster
        """

        # Initialized the Super Class (Thruster) elements
        super().__init__(thrusters)

        # The allocation matrix B
        # [Fx Fy Fz Tx Ty Tx]' = B * [Force_motor_1 Force_motor_2 ... Force_motor_n]'
        self.B: ndarray = array(B).reshape((6, self.number_of_thrusters))

        # Calculate the pseudo-inverse of B (the inverse of the allocation matrix)
        # [Force_motor_1 Force_motor_2 ... Force_motor_n]' = pseudo_inv(B) * [Fx Fy Fz Tx Ty Tx]'
        self.pseudo_invB = pinv(self.B)

        # Whether to normalize or clamp the values for each thruster
        self.normalize: bool = bool(normalize)

        # Get an array of minimum and maximum forces that each thruster can apply
        self.min_force: ndarray = zeros(self.number_of_thrusters)
        self.max_force: ndarray = zeros(self.number_of_thrusters)

        for thruster, i in zip(self.thrusters, range(self.number_of_thrusters)):

            # Get the input bounds for the thrusters inputs (for example -25N, 25N)
            self.min_force[i], self.max_force[i] = thruster.force_bounds

    def convert_thrusts_to_general_forces(self, thrusts: ndarray):
        """
        Computes the forces and torques applied in the rigid body [Fx, Fy, Fz, Mx, My, Mz]
        corresponding to applying the given vector of thrusts in the motors
        :param thrusts: The thrusts applied in each motor [thrust_motor1, thrust_motor2, ...]
        :return: A numpy array with 6 elements corresponding to the forces and torques applied on the
        rigid body
        """
        return self._thrust_to_force(B=self.B, thrusts=thrusts)

    def convert_general_forces_to_thrusts(self, forces: ndarray):
        """
        Force to Thrust method converts a generalized vector of forces
        following the SNAME convention forces = [Frb, Nrb] = [X, Y, Z, K, M, N]
        and converts it to a vector of forces allocated to each individual motor in Newton [N]

        This method normalizes the forces applied to each thruster so that they are bounded between
        the values specified in the dynamics [dynamics.min_thrust, dynamics.max_thrust] in Newton [N]

        :param forces - generalized vector of forces and torques following the SNAME
                        convention forces = [Frb, Nrb] = [X, Y, Z, K, M, N]
        :return:
            a vector of thrusts applied to each motor based on the equation
            [thrust_motor_1 thrust_motor_2 ...]' = pseudo_inv(B) * [Fx Fy Fz Tx Ty Tx]'
            expressed in Newton [N]
        """

        # Compute how much thrust to allocate to each motor
        thrust_vector = dot(self.pseudo_invB, array(forces).reshape(6))

        # Check if we just want to clamp the thrust values to apply
        if not self.normalize:
            return clip(thrust_vector, self.min_force, self.max_force)

        # Otherwise, normalize all the thrusts so that the direction of the forces and torques are preserved
        normalize = 1.0

        for i in range(self.number_of_thrusters):
            # If we are asking a negative thrust, check whether we are asking bellow the minimum allowed for that thruster
            if thrust_vector[i] < 0:
                normalize = max(abs(thrust_vector[i] / self.min_force[i]), normalize)
            # If we are asking a positive thrust, check whether we are asking above the maximum allowed for each thruster
            else:
                normalize = max(abs(thrust_vector[i] / self.max_force[i]), normalize)

        return (1.0 / normalize) * thrust_vector
