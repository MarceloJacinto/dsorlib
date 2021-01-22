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
from numpy import array, zeros, abs

from dsorlib.vehicles.auv_rb_dynamics.abstract_auv_dynamics import AbstractAUVDynamics
from dsorlib.vehicles.state.state import State


class PositiveBuoyancyAUVDynamics(AbstractAUVDynamics):
    """
    This class implements the AUV dynamics for a vehicle with POSITIVE BUOYANCY

    This class implements the volume of fluid displaced in a simplified way.
    Since the vehicle is to have positive buoyancy, then when the vehicle is completely
    submerged the volume of fluid displaced corresponds to the volume of the vehicle.

    When the vehicle is at the surface (z=0.0), then is is assumed that the volume of
    fluid displaced is such that W=B, where W is the gravity force (W = m * g) and B
    is the buoyancy (B = fluid_density * g * volume_of_water).

    Note: here we assume that the body frame of the coordinate system of the vehicle
    is placed in the geometric center of the vehicle.

    If the enter of buoyancy of the vehicle in z-axis is not the same as the geometric
    center of the vehicle, that is taken into consideration.

    When the vehicle is not fully submerged neither at z=0.0 (is somewhere in the middle),
    the a linear relation is applied.

    This is important because this is the approximation used to calculate the volume of
    fluid displaced.
    """

    def __init__(self,
                 vehicle_volume: float,  # Vehicle volume in [m^3]
                 vehicle_height: float,  # Height of the vehicle in [m]
                 m: float,  # Mass
                 inertia_tensor: array,  # The inertia tensor (a vector of 9 elements)
                 damping: array,  # The damping vector
                 quadratic_damping: array,  # The quadratic damping vector
                 added_mass: array,  # The added mass terms
                 g: float = 9.8,  # Gravity acceleration
                 fluid_density: float = 1000,  # Water/fluid density in [kg/m^3]
                 rg: array = zeros(3),  # Center of gravity
                 sea_surface_z: float = 0.0  # The z-coordinate of the inertial frame where the sea surface is
                 ):

        # Calculate W (the gravity force)
        self.W = float(m * g)

        # Compute the center of buoyancy offset
        V_water_displaced = self.W / (fluid_density * g) # The volume of water displaced when W=B (gravity=buoyancy)
        percentage_submerged = V_water_displaced / vehicle_volume   # The % [0, 1.0] of volume submerged when W=B
        height_submerged = vehicle_height * percentage_submerged    # The height of the vehicle that is submerged in [m]
        z_b = -(vehicle_height - height_submerged)  # Offset of the center of buoyancy of the vehicle with respect to the body frame

        # The offset of the center of buyancy with respect to the body frame
        rb = array([0.0, 0.0, z_b])

        # Initialized the Super Class elements
        super().__init__(m, inertia_tensor, damping, quadratic_damping, added_mass, g, fluid_density, rg, rb,
                         sea_surface_z)

        self.vehicle_volume = float(vehicle_volume)  # Vehicle volume in [m^3]
        self.vehicle_height = float(vehicle_height)  # Height of the vehicle in [m]

        # Volume displaced at (z=0.0) Assuming the vehicle has [phi, theta, psi] = [0.0, 0.0, 0.0]
        # and assuming positive buoyancy and such that W=B -> volume_of_water = m / fluid_density
        self.volume_of_water_at_zo = float(m / fluid_density)

        # The z coordinate in the inertial frame where the sea surface is located (usually is at z=0) but can be different
        self.zo = float(sea_surface_z)

        # Get the center of buoyancy of the vehicle (in the z-coordinate)
        # with respect to the center of the coordinate frame of the body frame of the vehicle
        zb = float(self.rb[2])

        # Check if the center of buoyancy is in the body frame of the vehicle, otherwise throw an exception
        if abs(zb) > self.vehicle_height / 2:
            raise Exception("The center of buoyancy must be inside the vehicle frame!")

        # Parameters for the linear model used for the calculation of fluid displaced as a function of depth
        self.slope1 = ((self.m / self.fluid_density) - 0) / ((self.zo - self.rb[2]) - (-self.vehicle_height / 2))
        self.b1 = (self.vehicle_height / 2) * self.slope1

        self.slope2 = (self.vehicle_volume - (self.m / self.fluid_density)) / (
                (self.vehicle_height / 2) - (self.zo - self.rb[2]))
        self.b2 = self.vehicle_volume - ((self.vehicle_height / 2) * self.slope2)

    def compute_buoyancy(self, state: State):
        """
        Compute the buoyancy of the vehicle, given by the formula:

        B=fluid_density * g * fluid_displaced [m^3]

        :param state: The state of the vehicle
        :return: A float with the Buoyancy force applied on the vehicle, B
        """

        # Compute the fluid displaced
        fluid_displaced = self.compute_volume_fluid_displaced(state)

        # Compute the Buoyancy B=fluid_density * g * fluid_displaced [m^3]
        return self.fluid_density * self.g * fluid_displaced

    def compute_volume_fluid_displaced(self, state: State):

        #TODO
        pass

    def compute_volume_fluid_displaced2(self, state: State):
        """
        Compute the volume of fluid displaced by the vehicle

        When the vehicle is fully submerged, the volume of fluid displaced corresponds
        approximately to the volume of the vehicle in [m^3].

        When the vehicle is partially submerged the volume of water displaced is
        given by an approximately linear law, such that when the buoyancy center
        of the vehicle is at zb=0, B=W -> therefore volume_displaced = m / fluid_density

        :param state: The state of the vehicle
        :return: A float with the volume of fluid displaced in [m^3]
        """

        z_o = float(self.zo)  # The z position of the water surface in inertial frame (in meters)
        z_b = float(self.rb[2])  # The z position of the center of buoyancy relative to the body frame of the vehicle
        # negative z_b means that the buoyancy center of the vehicle is above the vehicle's center of mass
        z_M = state.eta_1[2]  # The z position of the center of mass of the vehicle

        # If the vehicle is completely above water, then the volume of water displaced is zero
        if z_M <= z_o - self.vehicle_height / 2:
            return 0.0

        # Volume if partially submerged
        # Note that at the point where z_o = z_M + z_b we want W=B (force gravity = force buoyancy)
        # therefore volume_displaced = mass / density_of_fluid when (z_o = z_M + z_b)
        # for the other values make linear relations between 0.0, that point and full volume
        elif (z_o - self.vehicle_height / 2) < z_M <= (z_o - z_b):
            return self.b1 + self.slope1 * z_M
        elif (z_o - z_b) < z_M <= (z_o + self.vehicle_height / 2):
            return self.b2 + self.slope2 * z_M
        # Volume if completely submerged
        else:
            return self.vehicle_volume
