from numpy import ndarray, array, zeros, power, sqrt, exp, clip

from dsorlib.vehicles.thrusters.thruster import Thruster


class SimplePoleThruster(Thruster):

    def __init__(self,
                 min_input: float,
                 max_input: float,
                 thruster_map_positive: ndarray,
                 thruster_map_negative: ndarray,
                 delay: float,
                 pole: float,
                 dt: float):
        """
        QuadraticThruster is an implementation for a Thruster model that:
        - Dynamics can be described by a pole and a delay (in continuous time)
        - The map between force (in Newtons) and the input unit (%RPM, RPM or rad/s) can be mapped by using half of
        quadratic equations for the negative and positive sides

        For more information regarding implementing your own Thruster model, check the abstract class Thruster

        :param min_input: The minimum value for the thruster (for example -4500 RPM)
        :param max_input: The maximum value for the thruster (for example 4500 RPM)
        :param thruster_map_positive: Values for the positive side of the quadratic mapping between force and input [a, b]
        :param thruster_map_negative: Values for the negative side of the quadratic mapping between force and input [a, b]
        :param delay: The delay in the continuous time model of the thrusters
        :param pole: The position of the pole (in continuous time) for the thruster
        :param dt: The sampling period for this discretized model
        """

        # Initialized the Super Class (Thruster) elements
        super().__init__(min_input, max_input)

        # Save the values for the quadratic that allow the mapping between the forces and input values
        self.thruster_map_positive: ndarray = array(thruster_map_positive).reshape(2)
        self.thruster_map_negative: ndarray = array(thruster_map_negative).reshape(2)

        # Save the delay
        self.delay: float = float(delay)
        if delay < 0:
            raise ValueError("The delay of the motor cannot be negative")

        # Save the pole location
        self.pole: float = float(pole)

        # Save the sampling period
        self._dt: float = float(dt)
        if dt <= 0:
            raise ValueError("The sampling time must be greater than 0")

        # Auxiliary variables for the dynamics calculations (save the outputs in previous time steps)
        # Compute the number of delays that we get from the conversion of the continuous time model to discrete difference equations model
        self.num_delays_input = int(round(self.tau / self.period))

        # Create a circular buffer to store the previous input of the thruster model for the corresponding delays
        # Save input in (k-1, k-2, ..., k-num_delays)
        self.circular_buffer: ndarray = zeros(self.num_delays_input)

        # Create a variable to save the previous output (in k-1 timestep)
        self.y_1: float = 0.0

    def force_to_input_unit(self, force_val: float):
        """
        Convert the force in Newton [N] to the unit of input of the thruster dynamics system (in %RPM, RPM or rad/s for example)

        :param force_val: The force applied in Newton [N]
        :return: The corresponding force in the input unit (in %RPM, RPM or rad/s for example)
        """

        # Check whether the force applied is negative or positive
        if force_val >= 0:
            a = self.thruster_map_positive[0]
            b = self.thruster_map_positive[1]
        else:
            a = self.thruster_map_negative[0]
            b = self.thruster_map_negative[1]

        # Calculate the values for the inputs that are >= 0 [kgf]
        return (-b + sqrt(power(b, 2) + 4 * a * force_val)) / (2 * a)

    def input_unit_to_force(self, input_val: float):
        """
        Convert the unit of input of the system to Newton

        :param input_val: The input value for the thruster (in %RPM, RPM or rad/s for example)
        :return: The corresponding force in Newton [N]
        """

        # Check whether the input value is positive or negative and choose the parameters of the curve accordingly
        if input_val >= 0:
            a = self.thruster_map_positive[0]
            b = self.thruster_map_positive[1]
        else:
            a = self.thruster_map_negative[0]
            b = self.thruster_map_negative[1]

        # Return the force float Force[N] = a * input^2 + b*input
        return (a * power(input_val, 2)) + (b * input_val)

    def dynamic_model(self, input_val: float) -> float:

        # load buffer input = u[k-delay]
        u_old = self.circular_buffer[0]

        # Calculate the y[k] in the difference equations
        y = (exp(-self.thrusters_poles * self.period) * self.y_1) + ((-exp(-self.thrusters_poles * self.period) + 1) * u_old)

        # Rotate the buffer (and discard the oldest sample)
        self.circular_buffer[0:-1] = self.circular_buffer[1:]

        # Insert the new input in the end of the circular buffer
        self.circular_buffer[-1] = input

        # Save the new output to used in the next iteration
        self.y_1 = y

        # Saturate the output between the minimum and maximum values
        y = clip(y, self.min_input, self.max_input)

        return y

