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

from dsorlib.vehicles.state.state import State


class Vehicle(ABC):

    def __init__(self, initial_state: State, initial_state_dot: State = State(), dt: float = 0.1):
        """
        An Abstract class that every vehicle should inherit from.
        :param initial_state: State object with the initial state of the vehicle
        :param dt: The sampling time in seconds [s]
        """
        # Save the sampling period
        self.dt = float(dt)

        # Setup the initial state of the AUV vehicle
        # This setup is done as a copy of the State object so that
        # more vehicles can share the same initial state without sharing the actual state object (to avoid bugs)
        self.state = initial_state.__copy__()

        # Create a initial state dot to save the derivatives of each state as they are computed
        self.state_dot = initial_state_dot.__copy__()

        if type(self) == Vehicle:
            raise Exception("<Vehicle> cannot be instantiated. It is an abstract class")

    @abstractmethod
    def update(self, desired_input):
        """
        Abstract method that should be implemented by all the classes that inherit from AbstractVehicle.
        The goal of this method is that given a desired input to the vehicle, this method updates the current
        state of that same vehicle
        :param desired_input: The desired input to apply to the vehicle
        """
        pass

    def reset_vehicle(self, state: State = State()):
        """
        Method to reset the current state of the vehicle
        :param state: The new state of the vehicle (if not given, it will be all zeros)
        """

        # Reset the state to one pre-defined by the user
        self.state = state

        # Reset the derivatives of the state to become zero
        self.state_dot = State()
