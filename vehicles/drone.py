from dsorlib.vehicles.abstract_vehicle import AbstractVehicle
from dsorlib.vehicles.state.quadrotorstate import QuadrotorState
from dsorlib.utils import rot_matrix_B_to_U, ang_vel_rot_B_to_U, integrate

from numpy import pi, array, dot, cross, sin, cos, clip
from numpy.linalg import inv


def wrap_angle(val):
    """
    Wraps an angle between [-pi, pi]
    :param val: angle to wrap
    :return: Float with the corresponding angle wrapped
    """
    return (val + pi) % (2 * pi) - pi


class Controller:

    def __init__(self):
        # The max angles in degress that the drones are allowed to turn
        self.ang_limits = [-10.0, 10.0]

        # PD constants for the translational dynamics
        self.P_pos = [5.0, 5.0, 5.0]
        self.D_pos = [5.0, 5.0, 5.0]

        # PD constants for the rotational dynamics
        self.P_rot = [15.0, 15.0, 15.0]
        self.D_rot = [10.0, 10.0, 10.0]

        # Compute the allocation matrix
        self.Kt = 1.46e-5
        self.Km = 3.8e-7
        self.L = 0.29  # The arm size [m]

        # Allocation matrix
        self.B = array([[self.Kt, self.Kt, self.Kt, self.Kt],
                        [0.0, self.L * self.Kt, 0.0, -self.L * self.Kt],
                        [self.L * self.Kt, 0.0, -self.L * self.Kt, 0.0],
                        [self.Km, -self.Km, self.Km, -self.Km]])

        self.invB = inv(self.B)

        # Motor velocity limits (rad/s)
        self.motor_vel_limits = [0.0, 850.0]

        # Convert the angle limits to tilt limits in rad
        self.tilt_limits = [(self.ang_limits[0] / 180.0) * 3.14, (self.ang_limits[1] / 180.0) * 3.14]

    def propeller_model(self, forces):
        """
        Converts from forces and torques in Rigid Body to angular velocities
        Then saturates the velocities and converts back to forces and torques
        """

        # Convert the forces and momentum [Fz, Mx, My, Mz] -> [omega_1, omega_2, omega_3, omega_4]
        velocities = dot(self.invB, forces)

        # Clip the velocities between the minimum and maximum
        velocities = clip(velocities, self.motor_vel_limits[0], self.motor_vel_limits[1])

        print(velocities)

        # Convert back to forces and momentum
        forces_and_torques = dot(self.B, velocities)

        return forces_and_torques

    def __call__(self, des_pos, state: QuadrotorState):
        """
        :param des_pos: The desired position [x_des, y_des, z_des, yaw_des]
        :return: The forces and torques to apply to the quadrotor [Fz, Mx, My, Mz]
        """

        # Compute the error between the current position and the desired position
        x_error = des_pos[0] - state.eta_1[0]
        y_error = des_pos[1] - state.eta_1[1]
        z_error = des_pos[2] - state.eta_1[2]

        # Compute the PD law for the position error
        des_x_ddot = x_error * self.P_pos[0] + (0.0 - state.eta_1_dot[0]) * self.D_pos[0]
        des_y_ddot = y_error * self.P_pos[1] + (0.0 - state.eta_1_dot[1]) * self.D_pos[1]
        des_z_ddot = z_error * self.P_pos[2] + (0.0 - state.eta_1_dot[2]) * self.D_pos[2]

        u = array([des_x_ddot, des_y_ddot, des_z_ddot])

        # Current yaw=psi
        psi = state.eta_2[2]

        # Compute the desired angles based on the X and Y error commands
        des_phi = des_x_ddot * sin(psi) - des_y_ddot * cos(psi)
        des_theta = des_x_ddot * cos(psi) + des_y_ddot * sin(psi)
        des_psi = des_pos[3]

        # Limit the amount the desired angle between acceptable bounds (saturate)
        des_phi = clip(des_phi, self.tilt_limits[0], self.tilt_limits[1])
        des_theta = clip(des_theta, self.tilt_limits[0], self.tilt_limits[1])

        # Compute the error between the current angles and the desired angles
        phi_error = des_phi - state.eta_2[0]
        theta_error = des_theta - state.eta_2[1]
        psi_error = wrap_angle(des_psi - state.eta_2[2])

        # Compute the PD law for the rotation error
        des_phi_ddot = phi_error * self.P_rot[0] + (0.0 - state.v_2[0]) * self.D_rot[0]
        des_theta_ddot = theta_error * self.P_rot[1] + (0.0 - state.v_2[1]) * self.D_rot[1]
        des_psi_ddot = psi_error * self.P_rot[2] + (0.0 - state.v_2[2]) * self.D_rot[2]

        # Compute the Force in Z and Torque [Fz, Mx, My, Mz] and propagate through the propeller model
        forces_and_torques = [des_z_ddot, des_phi_ddot, des_theta_ddot, des_psi_ddot]
        #forces_and_torques = self.propeller_model(forces_and_torques)

        return forces_and_torques


class Drone(AbstractVehicle):

    def __init__(self,
                 initial_state: QuadrotorState = QuadrotorState(),
                 dt: float = 0.01):
        """
        :param initial_state: The initial position and velocity of the drone
        :param dt: The sampling time
        """

        # Initialized the Super Class with the update period (Discretized sampling period)
        super().__init__(initial_state, dt)

        self.g: float = 9.8  # gravity
        self.m = 0.976  # mass - Kg
        self.L = 0.29  # The arm size [m]

        # Compute the inertia matrix
        self.I = array([[0.0081, 0.0000, 0.0000],
                        [0.0000, 0.0081, 0.0000],
                        [0.0000, 0.0000, 0.0162]])

        self.invI = inv(self.I)

        # History of the states
        self.history = {'x': list(), 'y': list(), 'z': list(), 'x_dot': list(), 'y_dot': list(), 'z_dot': list(),
                        'roll': list(), 'pitch': list(), 'yaw': list(), 'p': list(), 'q': list(), 'r': list()}


    def update(self, desired_input=array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        # Desired thrust
        forces = array(desired_input[0:3])
        torques = array(desired_input[3:6])

        # Get the current angles
        phi = self.state.eta_2[0]
        theta = self.state.eta_2[1]
        psi = self.state.eta_2[2]

        # Compute the angular rotation matrix from body frame to inertial fram
        uQb = ang_vel_rot_B_to_U(phi, theta, psi)

        # Compute the rotation matrix from body frame to inertial frame
        uRb = rot_matrix_B_to_U(phi, theta, psi)

        # Compute the rotation matrix from inertial frame to body frame
        bRu = uRb.transpose()

        # Compute the linear accelerations [u_dot, v_dot, w_dot] and integrate them
        v_1_dot = dot(bRu, array([0.0, 0.0, self.g])) + (1.0 / self.m * forces) - (
                1.0 / self.m * cross(self.state.v_2, self.state.v_1))
        self.state.v_1 = integrate(x_dot=v_1_dot, x=self.state.v_1, dt=self.dt)

        # Compute the angular accelerations [p_dot, q_dot, r_dot] and integrate them
        v_2_dot = dot(self.invI, torques - cross(self.state.v_2, dot(self.I, self.state.v_2)))
        self.state.v_2 = integrate(x_dot=v_2_dot, x=self.state.v_2, dt=self.dt)

        # Compute [x_dot, y_dot, z_dot] from [u, v, w]
        self.state.eta_1_dot = dot(uRb, self.state.v_1)

        # Integrate to get the position
        self.state.eta_1 = integrate(x_dot=self.state.eta_1_dot, x=self.state.eta_1, dt=self.dt)

        # Compute [phi_dot, theta_dot, psi_dot] from [p, q, r]
        eta_2_dot = dot(uQb, self.state.v_2)

        # Integrate to get the angles
        self.state.eta_2 = integrate(x_dot=eta_2_dot, x=self.state.eta_2, dt=self.dt)

        # Wrap angles between -pi and pi
        self.state.eta_2 = wrap_angle(self.state.eta_2)

        # Update the dictionary with the history of the states
        self.history['x'].append(self.state.eta_1[0])
        self.history['y'].append(self.state.eta_1[1])
        self.history['z'].append(self.state.eta_1[2])

        self.history['x_dot'].append(self.state.eta_1_dot[0])
        self.history['y_dot'].append(self.state.eta_1_dot[1])
        self.history['z_dot'].append(self.state.eta_1_dot[2])

        self.history['roll'].append(self.state.eta_2[0])
        self.history['pitch'].append(self.state.eta_2[1])
        self.history['yaw'].append(self.state.eta_2[2])

        self.history['p'].append(self.state.v_2[0])
        self.history['q'].append(self.state.v_2[1])
        self.history['r'].append(self.state.v_2[2])
