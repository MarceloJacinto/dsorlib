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
from numpy import array, zeros

from dsorlib.vehicles.auv_dynamics.abstract_auv_dynamics import AbstractAUVDynamics
from dsorlib.vehicles.state.state import State


class NeutralBuoyancyAUVDynamics(AbstractAUVDynamics):

    def __init__(self,
                 m: float,  # Mass
                 inertia_tensor: array,  # The inertia tensor (a vector of 9 elements)
                 damping: array,  # The damping vector
                 quadratic_damping: array,  # The quadratic damping vector
                 added_mass: array,  # The added mass terms
                 g: float = 9.8,  # Gravity acceleration
                 fluid_density: float = 1000,  # Water/fluid density in [kg/m3]
                 rg: array = zeros(3),  # Center of gravity
                 sea_surface_z: float = 0.0  # The z-coordinate of the inertial frame where the sea surface is
                 ):

        # For a vehicle with neutral Buoyancy we can assume that the center of buoyancy
        # is the same as the center of the body coordinate frame hence the ofset with this frame is zero
        rb = zeros(3)

        # Initialized the Super Class elements
        super().__init__(m, inertia_tensor, damping, quadratic_damping, added_mass, g, fluid_density, rg, rb, sea_surface_z)

    def compute_buoyancy(self, state: State):
        # For neutral Buoyancy vehicles, B=W=m*g which simplifies the gravitational forces
        # which will become (inherently, by consequence)
        # which is a consequence of B=W therefore the terms where B-W = 0 which are g[0], g[1] and g[2]
        # g = [0.0,
        #      0.0,
        #      0.0,
        #      (-BG_y * W * cos(theta) * cos(phi)) + (BG_z * W * cos(theta) * sin(phi)),
        #      (BG_z * W * sin(theta)) + (BG_x * W * cos(theta) * cos(phi)),
        #      (-BG_x * W * cos(theta) * sin(phi)) - (BG_y * W * sin(theta))]
        # For more information regarding this, read Fossen - Handbook of marine craft hydrodynamics and motion control
        # Return B=W=m*g
        return self.m * self.g
