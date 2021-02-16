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
from dsorlib.vehicles.vehicle import Vehicle
from dsorlib.vehicles.thrusters.thruster_allocater import ThrusterAllocator
from dsorlib.vehicles.uav_dynamics.quadrotor_dynamics import QuadrotorDynamics
from dsorlib.vehicles.state.state import State
from dsorlib.vehicles.disturbances.abstract_disturbance import AbstractDisturbance
from dsorlib.utils import integrate, rot_matrix_B_to_U, ang_vel_rot_B_to_U, wrapAngle

from numpy import ndarray, array, cross, dot, zeros


# noinspection DuplicatedCode
class Quadrotor(Vehicle):

    def __init__(self,
                 rigid_body_dynamics: QuadrotorDynamics,
                 thruster_dynamics: ThrusterAllocator,
                 initial_state: State,
                 wind: AbstractDisturbance = None,
                 dt: float = 0.01,
                 input_is_thrusts: bool = False):
        """
        Constructor for the Quadrotor Vehicle class. This class inherits from Abstract Vehicle
        and implements all the necessary methods to simulate a quadrotor UAV (Unmanned Aerial Vehicle)

        :param rigid_body_dynamics: The rigid body dynamics of the AUV model
        :param thruster_dynamics: The thruster dynamics of the thrusters used in the vehicle
        :param initial_state: The initial state of the vehicle
        :param dt: The sampling time used in seconds [s]
        :param input_is_thrusts: A boolean to check whether the input given to the model
            is the thrust to apply to each individual motor (in Newton [N]) or the generalized
            vector of forces and torques (array of 6 elements) to apply to the rigid body
        """

        # Initialized the Super Class with the update period (Discretized sampling period)
        super().__init__(initial_state=initial_state, dt=dt)

        # Setup the rigid body dynamics for the vehicles
        # This object is not passed as copy so that if we want multiple AUV sharing the same rigid body dynamics
        # and want to change them on the fly, we have the ability to do so
        self.rb_dynamics = rigid_body_dynamics

        # Setup the thruster model to be used by the AUV model
        self.thruster_dynamics = thruster_dynamics

        # Boolean to check whether the input will be a vector of thrusts to each motor
        # or will be a vector os general forces and torques applied in the rigid body
        self.input_is_thrusts = input_is_thrusts

        # Save the type of wind to use
        # The ocean currents is not passed as copy so that if we want to have multiple drones sharing the same
        # wind currents, we have the ability to do so
        self.wind = wind

    def update(self, desired_input: array = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        """
        Method to update the state of the vehicle given a desired:
            generalized vector of force and torques OR thrust to apply to each individual motor (depending
            on the boolean specified in the constructor of the AUV object)
        :param desired_input: generalized vector of force and torques OR thrust to apply to each individual motor
            (depending on the boolean specified in the constructor of the quadrotor object)

        Updates the values inside state and state_dot
        """

        # --------------------------------------------------------
        # -- Check if the input is generalized vector of forces --
        # -- in the rigid body or the vector of thrusts applied --
        # -- to each motor                                      --
        # --------------------------------------------------------

        # Check whether the forces vector is a generalized vector of desired forces and torques
        # to apply to the rigid body or already a vector of thrusts to apply to the vehicle thrusters
        if not self.input_is_thrusts:

            # Convert the desired general forces applied in the rigid body to the desired thrusts
            # and check if we are getting the correct size
            thrusts = self.thruster_dynamics.convert_general_forces_to_thrusts(array(desired_input).reshape(6))
        else:
            # The "forces" vector received is not general forces but rather thrusts applied directly to the motors
            # and check if we are getting as many desired thrust as the number of thrusters in our thruster model
            thrusts = array(desired_input).reshape(self.thruster_dynamics.number_of_thrusters)

        # --------------------------------------------------------
        # -- Propagate the desired thrust by the thruster model --
        # --------------------------------------------------------
        applied_thrusts = self.thruster_dynamics.apply_thrusters_dynamics(thrusts)
        applied_forces = self.thruster_dynamics.convert_thrusts_to_general_forces(applied_thrusts)

        # --------------------------------------------------------
        # --       Compute the wind to use by the model         --
        # --------------------------------------------------------

        # Update the ocean currents dynamics
        if self.wind is None:
            # Assume zero currents in [x, y, z] components
            wind = zeros(3)
        else:
            wind = self.wind.get_currents()

        # --------------------------------------------
        # -- Compute the dynamics of the rigid body --
        # --------------------------------------------
        self.state_dot.v_1 = self.translational_dynamics(applied_forces[0:3])
        self.state_dot.v_2 = self.rotational_dynamics(applied_forces[3:6])

        # Integrate the dynamics
        self.state.v_1 = integrate(x_dot=self.state_dot.v_1, x=self.state.v_1, dt=self.dt)
        self.state.v_2 = integrate(x_dot=self.state_dot.v_2, x=self.state.v_2, dt=self.dt)

        # ----------------------------------------------
        # -- Compute the kinematics of the rigid body --
        # -----------------------------------------------
        self.state_dot.eta_1, self.state_dot.eta_2 = self.kinematics(wind=wind)

        # Integrate the kinematics
        self.state.eta_1 = integrate(x_dot=self.state_dot.eta_1, x=self.state.eta_1, dt=self.dt)
        self.state.eta_2 = integrate(x_dot=self.state_dot.eta_2, x=self.state.eta_2, dt=self.dt)

        # Wrap the angles (phi, theta, psi) between -pi and pi
        for i in range(self.state.eta_2.size):
            self.state.eta_2[i] = wrapAngle(self.state.eta_2[i])

    def kinematics(self, wind: ndarray) -> ():
        """
        Kinematics: Updates the kinematics of the vehicle based on the equation
        |-       -| = |-    -| . |- -|     |-      -|
        |eta_1_dot|   | R   0|   |v_1|   + |  wind  |
        |eta_2_dot|   | 0   Q|   |v_2|     |   0    |
        |-       -|   |-    -|   |- -|     |-      -|
        with R being the rotation matrix from {U} - Inertial Frame to {B} - Body Frame
        and Q being the transformation matrix of angular velocities from {U} to {B}

        :param wind: a vector with the currents in [x, y, z] in m/s in the inertial frame
        :return: (state_dot_eta_1, state_dot_eta_2) - a tuple with
                state_dot_eta_1 = [x_dot, y_dot, z_dot]'
                state_dot_eta_2 = [phi_dot, theta_dot, psi_dot]'
        """
        phi = self.state.eta_2[0]
        theta = self.state.eta_2[1]
        psi = self.state.eta_2[2]

        # Update x_dot, y_dot and z_dot
        state_dot_eta_1 = dot(rot_matrix_B_to_U(phi, theta, psi), self.state.v_1) + wind

        # Update phi_dot, theta_dot, psi_dot
        state_dot_eta_2 = dot(ang_vel_rot_B_to_U(phi, theta, psi), self.state.v_2)

        return state_dot_eta_1, state_dot_eta_2

    def translational_dynamics(self, forces: array):
        """
        Computes the translational dynamics of the quadrotor

        :param forces: An array with 3 elements with the generalized forces applied to the rigid
        body [Fx, Fy, Fz]
        :return: An array with [u_dot, v_dot, w_dot]
        """

        # Get the rotation matrix from inertial frame to body frame
        roll = self.state.eta_2[0]
        pitch = self.state.eta_2[1]
        yaw = self.state.eta_2[2]

        Rbu = rot_matrix_B_to_U(roll, pitch, yaw).transpose()

        # Compute the coriolis terms
        coriolis = cross(self.state.v_2, self.state.v_1)

        # Compute gravity force
        gravity = array([0.0, 0.0, self.rb_dynamics.m * self.rb_dynamics.g])

        # Compute the translational dynamics in the body frame of reference
        v1_dot = (1.0 / self.rb_dynamics.m) * (forces + dot(Rbu, gravity) - coriolis)

        return v1_dot

    def rotational_dynamics(self, torques: array):
        """
        Computes the rotational dynamics of the quadrotor

        :param torques: An array with 3 elements with the generalized torques applied to the rigid
        body [Mx, My, Mz]
        :return: An array with [p_dot, q_dot, r_dot]
        """

        # Get the inertia matrix and its inverse
        I = self.rb_dynamics.inertia_matrix
        I_inv = self.rb_dynamics.inertia_matrix_inv

        # Compute the coriolis terms
        coriolis = cross(self.state.v_2, dot(I, self.state.v_2))

        # Compute the rotational dynamics in the body frame of reference
        v2_dot = dot(I_inv, torques - coriolis)

        return v2_dot
