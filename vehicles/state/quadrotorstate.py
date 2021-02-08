from dsorlib.vehicles.state.state import State
from dsorlib.rotations import RPY_to_quaternion

from numpy import array, zeros


class QuadrotorState(State):
    """
    A Class that implements the state of an UAV quadrotor
    by inheriting state and extending it with quaternions (of the angles)
    and x_dot, y_dot, z_dot
    """

    def __init__(self,
                 eta_1: array = array([0.0, 0.0, 0.0]),
                 eta_2: array = array([0.0, 0.0, 0.0])):
        """
        Instantiate a State Object
        :param eta_1 = (x, y, z) - position of the origin of the Body Frame {B} expressed in the Inertial Frame {U}
        :param eta_2 = (roll, pitch, yaw) - orientation of the origin of the Body Frame {B} with respect to the Inertial Frame {U}
        v_1 = (u, v, w) - linear velocity of the origin of the Body Frame {B} with respect to the Inertial Frame {U} expressed in {B}
        v_2 = (p, q, r) - angular velocity of the origin of the Body Frame {B} with respect to the Inertial Frame {U} expressed in {B}
        """

        # Call the super constructor of the state
        super(QuadrotorState, self).__init__(eta_1, eta_2, zeros(3), zeros(3))

        # Create eta_1_dot (usually used in quadrotor modelling instead of the typically v_1=(u, v, w))
        self.eta_1_dot = zeros(3)

    def __str__(self):
        """
        Method to print the State to the terminal
        """
        return "QuadState[eta_1=" + str(self.eta_1) + ", eta_2=" + str(self.eta_2) + ", eta_1_dot=" + str(
            self.eta_1_dot) + ", v_2=" + str(self.v_2) + "]"

    def __copy__(self):
        """
        :return: A new state with a deep copy of each array
        """
        q_state = QuadrotorState(self.eta_1.__copy__(), self.eta_2.__copy__())
        q_state.v_1 = self.v_1.__copy__()
        q_state.v_2 = self.v_2.__copy__()
        q_state.eta_1_dot = self.eta_1_dot.__copy__()

        return q_state


