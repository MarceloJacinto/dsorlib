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
from numpy import array, zeros, dot, concatenate

from dsorlib.utils import integrate, rot_matrix_B_to_U, ang_vel_rot_B_to_U, wrapAngle
from dsorlib.vehicles.state.state import State
from dsorlib.vehicles.vehicle import Vehicle
from dsorlib.vehicles.auv_dynamics.abstract_auv_dynamics import AbstractAUVDynamics
from dsorlib.vehicles.thrusters.thruster_allocater import ThrusterAllocator
from dsorlib.vehicles.ocean_currents.abstract_ocean_currents import AbstractOceanCurrents


class AUV(Vehicle):
    """
    AUV class that should be used as base to implement different AUV (Autonomous Underwater Vehicles)
    """

    def __init__(self,
                 rigid_body_dynamics: AbstractAUVDynamics,
                 thruster_dynamics: ThrusterAllocator,
                 initial_state: State = State(),
                 ocean_currents: AbstractOceanCurrents = None,
                 dt: float = 0.01,
                 input_is_thrusts: bool = False):
        """
        Constructor for the AUV Vehicle class. This class inherits from Abstract Vehicle
        and implements all the necessary methods to simulate an AUV (Autonomous Underwater Vehicle)

        :param rigid_body_dynamics: The rigid body dynamics of the AUV model
        :param thruster_dynamics: The thruster dynamics of the thrusters used in the vehicle
        :param initial_state: The initial state of the vehicle
        :param ocean_currents: The type of ocean currents to be used (can be None)
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

        # Save the type of ocean currents to use
        # The ocean currents is not passed as copy so that if we want to have multiple AUV sharing the same
        # ocean currents, we have the ability to do so
        self.ocean_currents = ocean_currents

        # Boolean to check whether the input will be a vector of thrusts to each motor
        # or will be a vector os general forces and torques applied in the rigid body
        self.input_is_thrusts = input_is_thrusts

    def update(self, desired_input: array = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        """
        Method to update the state of the vehicle given a desired:
            generalized vector of force and torques OR thrust to apply to each individual motor (depending
            on the boolean specified in the constructor of the AUV object)
        :param desired_input: generalized vector of force and torques OR thrust to apply to each individual motor
            (depending on the boolean specified in the constructor of the AUV object)

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
        #applied_forces = desired_input
        #print(applied_forces)

        # --------------------------------------------------------
        # --   Compute the ocean currents to use by the model   --
        # --------------------------------------------------------

        # Update the ocean currents dynamics
        if self.ocean_currents is None:
            # Assume zero currents in [x, y, z] components
            currents = zeros(3)
        else:
            currents = self.ocean_currents.get_currents()

        # --------------------------------------------------------
        # -- Compute the dynamics of the rigid body of the AUV  --
        # --------------------------------------------------------
        self.state_dot.v_1, self.state_dot.v_2 = self.dynamics(forces=applied_forces)

        # Integrate the dynamics
        self.state.v_1 = integrate(x_dot=self.state_dot.v_1, x=self.state.v_1, dt=self.dt)
        self.state.v_2 = integrate(x_dot=self.state_dot.v_2, x=self.state.v_2, dt=self.dt)

        # --------------------------------------------------------
        # --Compute the kinematics of the rigid body of the AUV --
        # --------------------------------------------------------
        self.state_dot.eta_1, self.state_dot.eta_2 = self.kinematics(currents=currents)

        # Integrate the kinematics
        self.state.eta_1 = integrate(x_dot=self.state_dot.eta_1, x=self.state.eta_1, dt=self.dt)
        self.state.eta_2 = integrate(x_dot=self.state_dot.eta_2, x=self.state.eta_2, dt=self.dt)

        # Wrap the angles (phi, theta, psi) between -pi and pi
        for i in range(self.state.eta_2.size):
            self.state.eta_2[i] = wrapAngle(self.state.eta_2[i])

    def kinematics(self, currents: array) -> ():
        """
        Kinematics: Updates the kinematics of the vehicle based on the equation
        |-       -| = |-    -| . |- -|     |-      -|
        |eta_1_dot|   | R   0|   |v_1|   + |currents|
        |eta_2_dot|   | 0   Q|   |v_2|     |   0    |
        |-       -|   |-    -|   |- -|     |-      -|
        with R being the rotation matrix from {U} - Inertial Frame to {B} - Body Frame
        and Q being the transformation matrix of angular velocities from {U} to {B}

        :param currents: a vector with the currents in [x, y, z] in m/s
        :return: (state_dot_eta_1, state_dot_eta_2) - a tuple with
                state_dot_eta_1 = [x_dot, y_dot, z_dot]'
                state_dot_eta_2 = [phi_dot, theta_dot, psi_dot]'
        """

        phi = self.state.eta_2[0]
        theta = self.state.eta_2[1]
        psi = self.state.eta_2[2]

        # Update x_dot, y_dot and z_dot
        state_dot_eta_1 = dot(rot_matrix_B_to_U(phi, theta, psi), self.state.v_1) + currents

        # Update phi_dot, theta_dot, psi_dot
        state_dot_eta_2 = dot(ang_vel_rot_B_to_U(phi, theta, psi), self.state.v_2)

        return state_dot_eta_1, state_dot_eta_2

    def dynamics(self, forces: array) -> ():
        """
        :param forces: An array with size with the generalized forces and torques []
        applied by the thrusters to the rigid body in Newton [N]

            ((Mrb + Ma) * v_dot) + (Crb + Ca(v))*v + (Dl + Dq(v))*v + g = forces
        <=> (M * v_dot) + C*v + D*v + g = forces
        <=> v_dot = inv(M) * (forces - g - (C + D)*v)

        :return: (state_dot_v_1, state_dot_v_2) - a tuple with
                state_dot_v_1 = [u_dot, v_dot, w_dot]'
                state_dot_v_2 = [p_dot, q_dot, r_dot]'
        """

        # Calculate the Coriolis Matrix (6x6)
        coriolis = self.rb_dynamics.compute_coriolis_matrix(self.state)

        # Calculate the Damping Matrix
        damping = self.rb_dynamics.compute_damping_terms(self.state)

        # Calculate Gravitational Forces Vector array((6,))
        gravity = self.rb_dynamics.compute_gravitational_forces(self.state)

        # Calculate the updated v_dot
        M_inv = self.rb_dynamics.M_inv
        v = concatenate((self.state.v_1, self.state.v_2))

        # Note: Damping terms already contain the (-) sign, therefore we should sum here
        v_dot = dot(M_inv, forces - gravity - dot(coriolis, v) + dot(damping, v))
        return v_dot[0:3].reshape(3), v_dot[3:6].reshape(3)
