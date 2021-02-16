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
from numpy import array, power, pi, sin, cos

from dsorlib.vehicles.auv_dynamics.abstract_auv_dynamics import AbstractAUVDynamics
from dsorlib.vehicles.state.state import State


class SphereAUVDynamics(AbstractAUVDynamics):
    """
    This class implements the AUV dynamics for a vehicle with POSITIVE BUOYANCY

    This class implements the volume of fluid displaced in a simplified way.
    Since the vehicle is to have positive buoyancy, then when the vehicle is completely
    submerged the volume of fluid displaced corresponds to the volume of the vehicle.

    This is important because this is the approximation used to calculate the volume of
    fluid displaced.
    """

    def __init__(self,
                 vehicle_volume: float,  # Vehicle volume in [m^3]
                 m: float,  # Mass
                 inertia_tensor: array,  # The inertia tensor (a vector of 9 elements)
                 damping: array,  # The damping vector
                 quadratic_damping: array,  # The quadratic damping vector
                 added_mass: array,  # The added mass terms
                 g: float = 9.8,  # Gravity acceleration
                 fluid_density: float = 1000,  # Water/fluid density in [kg/m^3]
                 sea_surface_z: float = 0.0  # The z-coordinate of the inertial frame where the sea surface is
                 ):

        # Initialized the Super Class elements
        super().__init__(m=m,
                         inertia_tensor=inertia_tensor,
                         damping=damping,
                         quadratic_damping=quadratic_damping,
                         added_mass=added_mass,
                         g=g,
                         fluid_density=fluid_density,
                         sea_surface_z=sea_surface_z)

        # Calculate W (the gravity force)
        self.W = float(m * g)

        # Save the vehicle volume and approximate the vehicle for a sphere with a given radius
        # that we are going to calculate according to the formula of the volume of a sphere
        self.vehicle_volume = float(vehicle_volume)  # Vehicle volume in [m^3]
        self.vehicle_radius = float(power(3 / (4 * pi) * self.vehicle_volume, 1 / 3))

        # Compute the density of the vehicle
        self.vehicle_density = float(self.m / self.vehicle_volume)

    def compute_gravitational_forces(self, state: State):
        """
        :return: A array with 6 elements with the forces and torques resulting from
                buoyancy and gravitational force
        """

        # Compute the Weight W=m*g
        W = self.m * self.g

        # Compute the Buoyancy B=fluid_density * g * fluid_displaced [m^3]
        B = self.compute_buoyancy(state)

        # Get phi and theta angles from the state vectors
        phi = float(state.eta_2[0])
        theta = float(state.eta_2[1])

        # Get the center of gravity of the AUV relative to the origin of {B}
        xg = self.rg[0]
        yg = self.rg[1]
        zg = self.rg[2]

        # Get the center of buoyancy of the AUV relative to the origin of {B}
        self.rb = self.compute_buoyancy_center(state)
        xb = self.rb[0]
        yb = self.rb[1]
        zb = self.rb[2]

        # Calculate the Gravitational Force and return
        g = array([(W - B) * sin(theta),
                   -(W - B) * cos(theta) * sin(phi),
                   -(W - B) * cos(theta) * cos(phi),
                   -((yg * W) - (yb * B)) * cos(theta) * cos(phi) + ((zg * W) - (zb * B)) * cos(theta) * sin(phi),
                   ((zg * W) - (zb * B)) * sin(theta) + ((xg * W) - (xb * B)) * cos(theta) * cos(phi),
                   -((xg * W) - (xb * B)) * cos(theta) * sin(phi) - ((yg * W) - (yb * B)) * sin(theta)])

        return g

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

    def compute_buoyancy_center(self, state: State):
        """
        Recall that the Buoyancy center is, by definition, center of mass of the volume of water
        displaced by the vehicle. When the vehicle is fully submerged, this center of buoyancy corresponds
        to the geometrical center of the vehicle (if the density of the vehicle is uniformely distributed)
        which is the case here, since we are approximating our vehicle for a sphere
        :param state: The state of the vehicle
        :return: the buoyancy center defined with respect to the body reference frame
        """

        cm = state.eta_1[2]  # the z-coordinate of the center of mass of the sphere/vehicle
        zo = self.sea_surface_z  # the z-coordinate of the sea-surface
        r = self.vehicle_radius  # the radius of the sphere/vehicle

        # Compute the height of the sphere submerged
        if cm < zo - r:
            # If the sphere/vehicle is completely above water the height of the sphere submerged is zero
            h = 0
        elif zo - r <= cm < zo:
            # If the center of mass of the sphere/vehicle is above seawater level but the sphere
            # is partially submerged, the height of the submerged region is given by:
            h = r - (zo - cm)
        elif cm >= zo < cm - r:
            # If the center of mass of the sphere/vehicle is bellow seawater level
            # with the sphere partially submerged, the height of the submerged region is given by:
            h = r + (cm - zo)
        else:
            # If the sphere/vehicle is completely submerged, the height o the sphere that is submerged
            # is the same as the radius of the vehicle/sphere * 2
            h = r * 2

        # Distance from the center of buoyancy to the center of mass in the z-axis
        z_dist = (3.0 / 4.0) * power((2 * r - h), 2) / (3 * r - h)

        # New buoyancy center (expressed in the body reference frame)
        rb = array([0.0, 0.0, z_dist])

        return rb

    def compute_volume_of_sphere_cap(self, h: float):
        """
        This method computes the volume cap of the sphere (aka the robot) that is bellow
        water, given h
        :param h: The height of the cap of the sphere we want to compute the volume
        :return: The volume of the cap of a sphere with a height h
        """

        return ((3 / 4 * power((h / self.vehicle_radius), 2)) - (
                    1 / 4 * power(h / self.vehicle_radius, 3))) * 4 / 3 * pi * power(self.vehicle_radius, 3)

    def compute_volume_fluid_displaced(self, state: State):
        """
        Method to compute the volume of fluid displaced given the state of vehicle,
        more concretely the z-coordinate (expressed in the inertial frame) of the vehicle body frame location
        :param state: The state of the vehicle
        :return: the volume of water displaced by the vehicle
        """

        cm = state.eta_1[2]  # the z-coordinate of the center of mass of the sphere/vehicle
        zo = self.sea_surface_z  # the z-coordinate of the sea-surface
        r = self.vehicle_radius  # the radius of the sphere/vehicle
        v_vehicle = self.vehicle_volume  # the volume of the vehicle

        # If the vehicle is totally submerged in water,
        # the volume of water displaced corresponds to the volume of
        # the vehicle
        if zo <= cm - r:
            return v_vehicle

        # If the vehicle is partially submerged
        if cm - r < zo <= cm + r:

            # Compute the height of the sphere cap that is above water
            h = r - (zo - cm)

            # Compute the volume of the sphere cap that is above water
            vol_cap = self.compute_volume_of_sphere_cap(h)

            # Compute the volume of the part that is underwater
            return vol_cap

        # If the vehicle is totally above water, the volume of water
        # displaced is zero
        else:
            return 0.0
