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
#  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
from abc import ABC, abstractmethod
from numpy import array, zeros, eye, block, diag, sin, cos, dot, abs, concatenate
from numpy.linalg import inv

from dsorlib.utils import Smtrx
from dsorlib.vehicles.state.state import State


class AbstractAUVDynamics(ABC):
    """
    AUV Dynamics is a data holder class for the AUV dynamics parameters
    """

    def __init__(self,
                 m: float,  # Mass
                 inertia_tensor: array,  # The inertia tensor (a vector of 9 elements)
                 damping: array,  # The damping vector
                 quadratic_damping: array,  # The quadratic damping vector
                 added_mass: array,  # The added mass terms
                 g: float = 9.8,  # Gravity acceleration
                 fluid_density: float = 1000,  # Water/fluid density in [kg/m3]
                 rg: array = zeros(3),  # Center of gravity
                 rb: array = zeros(3),  # Center of buoyancy
                 sea_surface_z: float = 0.0  # The z-coordinate of the inertial frame where the sea surface is
                 ):
        if type(self) == AbstractAUVDynamics:
            raise Exception("<AbstractAUVDynamics> cannot be instantiated. It is an abstract class")

        # Save the mass of the vehicle, gravity acceleration
        self.m = float(m)
        self.g = float(g)
        self.fluid_density = float(fluid_density)
        self.sea_surface_z = float(sea_surface_z)

        # Save the center of gravity and buoyancy
        self.rg = array(rg)
        self.rb = array(rb)

        # Save the tensor of inertia
        self.inertia_tensor = array(inertia_tensor)

        # Check if damping is given as a vector of 3 terms (diagonal matrix)
        # or a vector with 9 terms (the complete matrix which might be non-diagonal)
        if self.inertia_tensor.size == 3:
            self.inertia_matrix = diag(self.inertia_tensor).reshape((3, 3))
        else:
            self.inertia_matrix = self.inertia_tensor.reshape((3, 3))

        # Save the damping and quadratic damping terms
        self.damping = array(damping)
        self.quadratic_damping = array(quadratic_damping)

        # Check if damping is given as a vector of 6 terms (diagonal matrix)
        # or a vector with 36 terms (the complete matrix which might be non-diagonal)
        if self.damping.size == 6:
            self.Dl = diag(self.damping).reshape((6, 6))
        else:
            self.Dl = self.damping.reshape((6, 6))

        # Check if quadratic damping is given as a vector of 6 terms (diagonal matrix)
        # or a vector with 36 terms  (the complete matrix which might be non-diagonal)
        if self.quadratic_damping.size == 6:
            self.Dq = diag(self.quadratic_damping).reshape((6, 6))
        else:
            self.Dq = self.quadratic_damping.reshape((6, 6))

        # Compute Mrb from the data
        self.Mrb = block([[m * eye(3), -m * Smtrx(rg)],
                          [m * Smtrx(rg), self.inertia_matrix - (m * Smtrx(rg) * Smtrx(rg))]])

        # Check if added mass is given as a vector of 6 terms (diagonal matrix)
        # or a vector with 36 terms  (the complete matrix which might be non-diagonal)
        self.added_mass = array(added_mass)
        if self.added_mass.size == 6:
            self.Ma = diag(self.added_mass).reshape((6, 6))
        else:
            self.Ma = self.added_mass.reshape((6, 6))

        # Compute the full mass matrix
        self.M = self.Ma + self.Mrb
        # Compute the inverse of the mass matrix (useful to update the dynamics)
        self.M_inv = inv(self.M)

    @abstractmethod
    def compute_gravitational_forces(self, state: State):
        """
        Abstract method that should be implemented by a class that inherits it.
        This calculation can be more or less complicated depending on the model
        of the vehicle

        :return: A array with 6 elements with the forces and torques resulting from
                buoyancy and gravitational force
        """

    def compute_coriolis_matrix(self, state: State):
        """
        :param state: The State object to make the calculations from
        :return: A 6x6 matrix with the coriolis terms
        """

        # Get the state vector corresponding to linear and angular velocities in the Body frame
        v = concatenate((state.v_1, state.v_2))

        # Compute the coriolis terms from the mass matrix and the velocity vector in the body frame
        s1 = Smtrx(dot(self.M[0:3, 0:3], v[0:3]) + dot(self.M[0:3, 3:6], v[3:6]))
        s2 = Smtrx(dot(self.M[3:6, 0:3], v[0:3]) + dot(self.M[3:6, 3:6], v[3:6]))
        c = zeros((6, 6))
        c[0:3, 3:6] = -s1
        c[3:6, 0:3] = -s1
        c[3:6, 3:6] = -s2

        return c

    def compute_damping_terms(self, state: State):
        """
        :param state: The State object to make the calculations from
        :return: A 6x6 matrix with the damping terms
        """

        # Get the state vector corresponding to linear and angular velocities in the Body frame
        v_vector = concatenate((state.v_1, state.v_2))

        # Compute the damping matrix according to the formulas
        return self.Dl + diag(dot(self.Dq, abs(v_vector)))
