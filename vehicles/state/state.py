#  MIT License
#
#  Copyright (c) 2020 Marcelo Jacinto
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
from numpy import array


class State:
    """
    A Class that implements the state of a generic vehicle
    """

    def __init__(self,
                 eta_1: array = array([0.0, 0.0, 0.0]),
                 eta_2: array = array([0.0, 0.0, 0.0]),
                 v_1: array = array([0.0, 0.0, 0.0]),
                 v_2: array = array([0.0, 0.0, 0.0])):
        """
        Instantiate a State Object
        eta_1 = (x, y, z) - position of the origin of the Body Frame {B} expressed in the Inertial Frame {U}
        eta_2 = (roll, pitch, yaw) - orientation of the origin of the Body Frame {B} with respect to the Inertial Frame {U}
        v_1 = (u, v, w) - linear velocity of the origin of the Body Frame {B} with respect to the Inertial Frame {U} expressed in {B}
        v_2 = (p, q, r) - angular velocity of the origin of the Body Frame {B} with respect to the Inertial Frame {U} expressed in {B}
        """

        # Assign the values and validate the shape of each input
        self.eta_1 = array(eta_1).reshape((3,))
        self.eta_2 = array(eta_2).reshape((3,))
        self.v_1 = array(v_1).reshape((3,))
        self.v_2 = array(v_2).reshape((3,))

    def __str__(self):
        """
        Method to print the State to the terminal
        """
        return "State[eta_1=" + str(self.eta_1) + ", eta_2=" + str(self.eta_2) + ", v_1=" + str(
            self.v_1) + ", v_2=" + str(self.v_2) + "]"

    def __copy__(self):
        """
        :return: A new state with a deep copy of each array
        """
        return State(self.eta_1.__copy__(), self.eta_2.__copy__(), self.v_1.__copy__(), self.v_2.__copy__())
