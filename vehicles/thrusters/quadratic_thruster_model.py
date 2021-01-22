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

from numpy import array, zeros, sqrt, power, around, exp, multiply, maximum, minimum

from dsorlib.vehicles.thrusters.abstract_thruster_model import AbstractThrusterModel


# noinspection PyPep8Naming
class QuadraticThrusterModel(AbstractThrusterModel):
    """
    QuadraticThrusterModel is an implementation for the thruster models of the seabotics
    thrusters or any other thruster whose map between the (%RPM) units [-100, 100] (aka the input of the thruster model)
    and the input in [kgf] is achieved using quadratic models of the type:

    thrust [kgf] = a (%RPM)^2 + b (%RPM)

    Since the thruster dynamics class that this class inherits requires the conversion of the input of the motor
    from some unit to the S.I standard Newton, a conversion from [kgf] to [N] is further applied in each method

    Furthermore, this model assumes that the thruster dynamics are modelled by a first order model
    with a delay and a pole

    This class inherits the AbstractThrusterModel class. For more information refer to the documentation for AbstractThrusterModel
    """

    def __init__(self,
                 number_of_thrusters: int,  # The number of thrusters
                 B: array,  # The thruster allocation matrix
                 min_thrust: float,  # The minimum thrust that can be applied in Newton [N]
                 max_thrust: float,  # The maximum thrust that can be applied in Newton [N]
                 thruster_map_positive: array,
                 # The params for the quadratic mapping between %RPM [0, 100] and [kgf] units
                 thruster_map_negative: array,
                 # The params for the quadratic mapping between %RPM [-100, 0] and [kgf] units
                 tau: float,  # The delay in the continuous time model of the thrusters
                 thrusters_poles: array,  # The position of the pole (in continuous time), for each thruster
                 period: float):  # The sampling period for this discretized model
        """
        The Constructor for the seabotics model of the thrusters

        params:
            - number_of_thrusters: int - The number of thrusters
            - B: array - The thruster allocation matrix with size (6 x num_thrusters)
            - min_thrust: float - The minimum thrust that can be applied in Newton [N]
            - max_thrust: float - The maximum thrust that can be applied in Newton [N]
            - thruster_map_positive: array - Array with 2 params for the quadratic mapping between %RPM [0, 100] and [kgf] units
            - thruster_map_negative: array - Array with 2 params for the quadratic mapping between %RPM [-100, 0] and [kgf] units
            - tau: float - The delay in the continuous time model of the thrusters
            - thrusters_poles: array - Array with the position of the pole (in continuous time), for each thruster with size (num_thrusters)
        """

        # Initialized the Super Class elements
        super().__init__(number_of_thrusters, B, min_thrust, max_thrust)

        # Save the other parameters
        self.thruster_map_positive = array(thruster_map_positive)
        self.thruster_map_negative = array(thruster_map_negative)
        self.tau = float(tau)
        self.thrusters_poles = array(thrusters_poles).reshape((self.number_of_thrusters,))
        self.period = float(period)

        # Auxiliary variables for the dynamics calculations (save the outputs in previous time steps)
        # Compute the number of delays that we get from the conversion of the continuous time model to discrete difference equations model
        self.num_delays_input = int(round(self.tau / self.period))

        # Create a circular buffer to store the previous input of each thruster model for the corresponding delays
        # Save input in (k-1, k-2, ..., k-num_delays)
        self.circular_buffer = zeros((self.number_of_thrusters, self.num_delays_input))

        # Create a variable to save the previous output (in k-1 timestep)
        self.y_1 = zeros(self.number_of_thrusters)

    def __copy__(self):
        """
        Implements a deep copy of the QuadraticThrusterModel
        because in this class the variables in the buffers must be different for different AUVs
        therefore sharing the same thruster model/object with several AUV at the same time
        could introduce bugs
        :return: A deep copy of the QuadraticThrusterModel
        """
        return QuadraticThrusterModel(number_of_thrusters=self.number_of_thrusters,
                                      B=self.B,
                                      min_thrust=self.min_thrust,
                                      max_thrust=self.max_thrust,
                                      thruster_map_positive=self.thruster_map_positive,
                                      thruster_map_negative=self.thruster_map_negative,
                                      tau=self.tau,
                                      thrusters_poles=self.thrusters_poles,
                                      period=self.period)

    def thrust_to_input(self, thrust: array = array([0.0, 0.0, 0.0, 0.0])):
        """
        The goal of this method is to convert a given vector of thrusts for each individual motor
        expressed in [N] and convert it to an input signal to the actual motors - the input signal
        is expressed in (%RPM) -> [-100, 1000]

        Note: Our model of the thrusters is defined in the relation [kgm] -> [%RPM]
        so in this method we just do an extra conversion [N] -> [kgm] by dividing
        the thrust in [N] by 9.8

        params:
            thrust - a vector of the thrust for each individual motor, expressed in Newton [N]
        returns:
            a vector with the corresponding input signals to the motors (RPM%)
        """

        # Convert the Thrust vector to a numpy array
        thrust = array(thrust)  # [N]

        # Saturate between min_thrust and max_thrust
        thrust = maximum(thrust, self.min_thrust)
        thrust = minimum(thrust, self.max_thrust)

        # Divide the thrust by 9.8 to convert it to [kgm]
        # Note that our model was identified between [%RPM] and [kgm units]
        thrust = 1 / 9.8 * thrust  # [kgf]

        # Positive side of the model
        a_positive = self.thruster_map_positive[0]
        b_positive = self.thruster_map_positive[1]

        # Negative side of the model
        a_negative = self.thruster_map_negative[0]
        b_negative = self.thruster_map_negative[1]

        # Create an empty vector of thrusts in [RPM%]
        input = zeros(thrust.shape)

        # Calculate the values for the inputs that are >= 0 [kgf]
        input[thrust >= 0] = (-b_positive + sqrt(power(b_positive, 2) + 4 * a_positive * thrust[thrust >= 0])) / (
                2 * a_positive)

        # Calculate the values for the inputs that are < 0 [kgf]
        input[thrust < 0] = (-b_negative + sqrt(power(b_negative, 2) + 4 * a_negative * thrust[thrust < 0])) / (
                2 * a_negative)

        return input

    def input_to_thrust(self, input: array = array([0.0, 0.0, 0.0, 0.0])):
        """
        The goal of this method is to convert a given vector of inputs for each individual motor
        (RPM%) [-100, 100] and convert it to thrust in Newton [N]

        Note: Our model of the thrusters is defined in the relation [%RPM] -> [kgm]
        so in this method we just do an extra conversion [kgm] -> [N] by multiplying
        the result in [kgm] by 9.8

        params:
            input - a vector of the input signals to the motors (RPM%)

        returns:
            a vector with the corresponding thrust expressed in Newton [N]
        """

        # Convert the input to a numpy array
        input = array(input)

        # Saturate the input between [-100, 100] (RPM%)
        input = maximum(input, -100.0)
        input = minimum(input, 100.0)

        # Positive side of the model
        a_positive = self.thruster_map_positive[0]
        b_positive = self.thruster_map_positive[1]

        # Negative side of the model
        a_negative = self.thruster_map_negative[0]
        b_negative = self.thruster_map_negative[1]

        # Create an empty vector to store the thrusts in [kgm]
        output = zeros(input.shape)

        # Calculate the outputs for the inputs that are >= than 0%
        output[input >= 0] = a_positive * power(input[input >= 0], 2) + b_positive * input[input >= 0]  # [kgf]

        # Calculate the outputs for the inputs that are < than 0%
        output[input < 0] = a_negative * power(input[input < 0], 2) + b_negative * input[input < 0]  # [kgf]

        # Multiply the thrust by 9.8 to convert it to [N]
        # Note that our model was identified between [%RPM] and [kgm units]
        output = output * 9.8

        return output

    def thrusters_dynamic_model(self, input: array = array([0.0, 0.0, 0.0, 0.0])):
        """
        The goal of this method is given a vector of inputs of thrusts for each individual motor compute the actual
        applied input to the model (propagate its dynamics) and return the actual inputs applied to the system

        The input is assumed to be in %RPM [-100, 100]

        params:
            input - a vector of the input signals to the motors (RPM%)

        returns:
            a vector with the actual applied thrust expressed in the same units as the original input signal (%RPM)
        """

        # Saturate the input between [-100, 100] (RPM%) for safety
        input = maximum(input, -100.0)
        input = minimum(input, 100.0)

        # load buffer input = u[k-delay]
        u_old = self.circular_buffer[:, 0]

        # Calculate the y[k] in the difference equations
        y = multiply(exp(-self.thrusters_poles * self.period), self.y_1) + multiply(
            (-exp(-self.thrusters_poles * self.period) + 1), u_old)  # Calculate y[k]

        # Rotate the buffer (and discard the oldest sample)
        self.circular_buffer[:, 0:-1] = self.circular_buffer[:, 1:]

        # Insert the new input in the end of the circular buffer
        self.circular_buffer[:, -1] = input

        # Save the new output to used in the next iteration
        self.y_1 = y

        # Saturate the output between [-100, 100] (RPM%) for safety
        y = maximum(y, -100.0)
        y = minimum(y, 100.0)

        return around(y)
