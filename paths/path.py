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
from typing import List
from dsorlib.paths.pathSection import PathSection

"""
This path parameterization algorithm was based on the one used in the MEDUSA class of vehicles (property of ISR/DSOR)

Create a path that may contain multiple path sections
The way this works is:
- the controler works with the parameterization s which can vary in an not limited way (lets say, it can go from -1 to 100 for example)
- but each section of the path only varies from 0 to 1
- The controller is not aware that the path has different segments. It just does its job, calculates an s and wants some information back.

For example: we have a path with 3 segments. Therefore s will vary from 0 to 3 when the vehicle is moving. But each segment only varies from 0 to 1, 
so based on the value of s, this class knows which segment to look for and to get the parameters from :)

This is a nice layer of abstraction, so we can have different types of paths and path sections parameterized with arcs, lines or something more complicated
and the controller does not need to know nothing about that. It only needs to know given the s, the curvature, desired angle and X,Y coordinates in the inertial frame
"""


class Path:
    """
    Class Path stores severall pathSections. This class provides methods to
    - add new sections to the Path, 
    - retrieve a pathSection given an s, 
    - get the points of the path (with a specified decimation), usefull for ploting :)
    - get the curvature of a point in the path, given s
    - get the angle of the tangent to the path, given s
    - get the position X, Y in the inertial frame, given s
    - get the bounds of s

    IMPORTANT NOTE:
    It is assumed that s can vary from -inf to +inf, but in practise it should vary from
    0 to the number of pathSections is contains. Therefore, if s < 0 or s > number of Path Sections
    every method should return the equivalent of s=0 or s=numberOfPathSections
    """

    def __init__(self):
        """
        Constructor - creates an empty Path
        """
        self._pathSections: List[PathSection] = []  # Empty List of pathSections

    def addPathSection(self, pathSection: PathSection):
        """
        Method to add PathSection, or objects that inherite pathSection to the Path
        """
        self._pathSections.append(pathSection)

    def getPathSection(self, s: float) -> PathSection:
        """
        Method to get a PathSection given the parameterization s
        """

        # Edges cases where we fall on places where we dont have a parameterized path (s goes out of bounds)
        if s < 0:
            return self._pathSections[0]
        elif s > len(self._pathSections):
            return self._pathSections[len(self._pathSections) - 1]
        else:
            return self._pathSections[int(s)]

    def getPathPoints(self, decimation=0.01) -> ([], []):
        """
        Method to get the points X,Y of the path (using a given decimation - default=0.01). Usefull
        for plotting :)
        """
        pathPointsX = []
        pathPointsY = []
        for section in self._pathSections:
            x, y = section.getPathSectionPoints(decimation)
            pathPointsX = pathPointsX + x
            pathPointsY = pathPointsY + y

        return (pathPointsX, pathPointsY)

    def getCurvature(self, s: float) -> float:
        """
        Method to return the curvature of a certain point on a path, given the parameterization s
        """
        return self.getPathSection(s).curvature(self.__getInternalS(s))

    def getAngle(self, s: float) -> float:
        """
        Method to return the angle between the tangent to the path and the x axis, given the parameterization s
        """
        return self.getPathSection(s).angle(self.__getInternalS(s))

    def getXYPos(self, s: float) -> (float, float):
        """
        Method to return the X,Y position of a point in the path in the inertial frame, given the parameterization s
        """
        return self.getPathSection(s).getXYFromS(self.__getInternalS(s))

    def getPathBound(self) -> (float, float):
        """
        Method to return the (min_value_of_s, max_value_of_s) - the bounds of the parameterization s
        """
        return (float(0.0), len(self._pathSections))

    # Auxiliary private method to get an s that varies from 0 to 1 from one that can vary from -inf to inf
    def __getInternalS(self, s: float):
        """
        Private Auxiliary method to get a parameterization s varying from 0 to 1, from an s that can vary from -inf to +inf.
        This is useful since each internal PathSection varies from 0 to 1

        Default behaviour is when s<0 it will default to s=0
        Default behaviour is when s>len it will default to s=len
        """
        # Make internal s vary from 0 to 1
        if s < 0:
            return 0
        elif s > len(self._pathSections):
            return len(self._pathSections)
        else:
            return s - int(s)

    def __str__(self):
        """
        Method to print the path and its elements to the terminal
        """
        aux: str = "Path["
        for section in self._pathSections:
            aux += section.__str__() + ", \n"
        aux += "]"
