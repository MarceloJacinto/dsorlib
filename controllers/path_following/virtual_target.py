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
from abc import ABC, abstractmethod
from dsorlib.utils import integrate


class VirtualTarget(ABC):

    def __init__(self, initial_state: float = 0.0):
        """
        Virtual Target is an abstract class that when implemented should
        derive the dynamics for a virtual target moving along a path

        :param initial_state: The initial position of the target (path parameter)
        """
        self._gamma = initial_state
        self._gamma_dot = 0.0
        self._gamma_ddot = 0.0

        if type(self) == VirtualTarget:
            raise Exception("<VirtualTarget> cannot be instantiated. It is an abstract class")

    @abstractmethod
    def update(self, dt: float = 0.01, *args, **kwargs):
        """
        Abstract method that when implemented should derive the update law for the
        virtual moving target
        :param dt: The time-step between the current and last iteration

        Updates the self.gamma state of the target
        """
        pass

    def reset_target(self, state: float = 0.0):
        """
        Resets the virtual target to a given state
        """
        self._gamma: float = state
        self._gamma_dot = 0.0
        self._gamma_ddot = 0.0

    @property
    def gamma(self):
        """
        Getter for the virtual target parameter position
        :return: The virtual target position
        """
        return self._gamma

    @property
    def gamma_dot(self):
        """
        Getter for the virtual target parameter velocity
        :return: The virtual target velocity
        """
        return self._gamma_dot

    @property
    def gama_ddot(self):
        """
        Getter for the virtual target parameter acceleration
        :return: The virtual target acceleration
        """
        return self._gamma_ddot


class SimpleVirtualTarget(VirtualTarget):
    """
    Implements a simple virtual target that moves along a path
    with a predefined speed
    """

    def __init__(self, initial_state: float = 0.0):
        """
        The constructor for the virtual target that moves at a specified velocity
        :param initial_state: The initial position of the virtual target in the parameterized path
        """

        # Initialized the Super Class with the initial state for the virtual target
        super().__init__(initial_state=initial_state)

    def update(self, desired_velocity: float, dt: float = 0.01):
        # The velocity of the target is the same as the desired velocity
        self._gamma_dot = desired_velocity

        # Integrate the gamma to get the current parameter of the virtual target
        self._gamma = integrate(x_dot=self._gamma_dot, x=self._gamma, dt=dt)


class AdaptiveVirtualTarget(VirtualTarget):

    def __init__(self, initial_state: float = 0.0, kz: float = 1.0):
        """
        The constructor for the virtual target that is adaptive as a function of both
        the desired velocity for the virtual target and the distance between the real
        vehicle and the virtual target

        :param initial_state: The initial position of the virtual target in the parameterized path
        :param kz: The gain for the control law (how much importance to give to the desired velocity to follow)
        """

        # Initialized the Super Class with the initial state for the virtual target
        super().__init__(initial_state=initial_state)

    def update(self, desired_velocity: float, dt: float = 0.01):
        # TODO
        pass
