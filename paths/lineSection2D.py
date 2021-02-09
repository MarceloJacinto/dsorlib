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
from dsorlib.paths.pathSection import PathSection
import numpy as np


class LineSection2D(PathSection):
    """
    Class LineSection2D extends PathSection
    This path parameterizes a line with a parameter s that varies from 0 to 1
    This line is inside a 2D plane
    """

    def __init__(self, xs: float, ys: float, xe: float, ye: float, z: float=0.0):
        """
        Constructor for a LineSection
        Parameters for line constructions are the coordinates of the starting point (xs, ys)
        and the coordinates of the finish point (xe, ye)
        """
        super()
        self._xs = xs  # Start point of the path
        self._ys = ys
        self._xe = xe  # End point of the path
        self._ye = ye
        self._z = z

    def curvature(self, s: float) -> float:
        """
        @Override
        Returns 0.0, since this section is a line, and a line has 0 curvature
        """
        return float(0.0)

    def angle(self, s: float) -> float:
        """
        @Override
        Returns the angle made with the line. Since the Section is a line
        the angle is given by the arctan(slope). To preserve the quadrant, arctan2 is used.
        This angle varies between [-pi, pi]
        """
        return np.arctan2((self._ye - self._ys), (self._xe - self._xs))

    def getXYZFromS(self, s: float) -> (float, float):
        """
        @Override
        Returns the X, Y position in the inertial frame of the point parameterized by s.
        If s > 1 the return will be the equivalent for s = 1
        If s < 0 the return will be the equivalent for s = 0
        """
        if 0 <= s <= 1:
            X = ((self._xe - self._xs) * s) + self._xs
            Y = ((self._ye - self._ys) * s) + self._ys
            return X, Y, self._z
        elif s < 0:
            return self._xs, self._ys, self._z
        else:
            return self._xe, self._ye, self._z

    def __str__(self):
        """
        @Override
        Returns a string with the information the segment is a Line 
        and the initial and final coordinates of the line in the inertial frame
        """
        return "Line[ xs=" + str(self._xs) + ", ys=" + str(self._ys) + ", xe=" + str(self._xe) + ", ye=" + str(
            self._ye) + "]"
