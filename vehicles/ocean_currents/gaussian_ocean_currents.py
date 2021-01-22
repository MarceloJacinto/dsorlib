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
from numpy import array, array_equal, zeros, maximum, minimum
from numpy.random import normal

from dsorlib.vehicles.ocean_currents.abstract_ocean_currents import AbstractOceanCurrents


class GaussianOceanCurrents(AbstractOceanCurrents):
    """
    GaussianOceanCurrents is a class that implements ocean currents following a random
    gaussian distribution centered around a specified mean vector
    """

    def __init__(self,
                 mean: array = array([0.0, 0.0, 0.0]),
                 sigma: array = array([0.0, 0.0, 0.0]),
                 min: array = array([0.0, 0.0, 0.0]),
                 max: array = array([0.0, 0.0, 0.0])):
        """
        Instantiate a Gaussian Ocean Currents object.
        It defines the mean linear velocity for the current -> mean=[v_x, v_y, v_z] in the Inertial Frame {U}
        It also defines the sigma=[sigma_x, sigma_y, sigma_z]
        """
        # Call the super class constructor
        super().__init__()

        # Define the mean velocity for the waves
        self.mean = array(mean).reshape((3,))

        # Define the std deviation for the velocity of the waves (noise)
        self.sigma = array(sigma).reshape((3,))

        # Define the boundaries for the values of the current velocities
        self.min = array(min).reshape((3,))
        self.max = array(max).reshape((3,))

    def get_currents(self):
        """
        Update the values of the waves following a random gaussian distribution

        returns:
            A numpy array with 3 elements (vx, vy, vz)
        """

        # Check if we have no std. dev. - In this case, the waves are "static"
        if array_equal(self.sigma, zeros(self.sigma.shape)):
            return self.mean

        # Generate the random numbers according to a gaussian distribution
        rand_current = normal(self.mean, self.sigma)

        # Check if the generated current is above the limits pre-defined
        rand_current = maximum(self.min, rand_current)
        rand_current = minimum(self.max, rand_current)

        return rand_current
