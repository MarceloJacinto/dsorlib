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


class PathSection(ABC):
    """
    Abstract class PathSection
    Every PathSection should extend this class and implement the common methods: angle, getXYFromS and __str__
    """

    @abstractmethod
    def curvature(self, s: float) -> float:
        """
        Abstract method that given the parameterization of the path, s, varying from 0 to 1
        should return the curvature of the path in that point
        """
        pass

    @abstractmethod
    def angle(self, s: float) -> float:
        """
        Abstract method that given the parameterization of the path, s, varying from 0 to 1
        should return the angle of the tangent to that given point in radians from -pi to pi 
        """
        pass

    @abstractmethod
    def getXYFromS(self, s: float) -> (float, float):
        """
        Abstract method that given the parameterization of the path, s, varying from 0 to 1
        should return the X, Y coordinates in the inertial frame of that point in the path
        """
        pass

    def getPathSectionPoints(self, decimation=0.01) -> ([], []):
        """
        Method that given the decimation (default=0.01), gives ([x_coord],[y_coord]) of points
        making the path. Very usefull for ploting :)
        """
        s = 0
        pathPointsX = []
        pathPointsY = []
        while s <= 1:
            x, y = self.getXYFromS(s)
            pathPointsX.append(x)
            pathPointsY.append(y)
            s += decimation
        return (pathPointsX, pathPointsY)

    @abstractmethod
    def __str__(self):
        """
        Abstract method that should print which type of object this is
        """
        pass
