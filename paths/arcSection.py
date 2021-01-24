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
import numpy as np
from dsorlib.paths.pathSection import PathSection


class ArcSection(PathSection):
    """
    Class ArcSection extends PathSection
    This path parameterizes an arc with a parameter s that varies from 0 to 1
    """

    def __init__(self, xs: float, ys: float, xc: float, yc: float, xe: float, ye: float, direction=1.0):
        """
        Constructor for an ArcSection
        Parameters for arc constructions are the coordinates of the starting point (xs, ys), the coordinates
        of the center of the arc (xc, yc), the coordinates of the finish point (xe, ye) and the direction to draw
        the arc from (default 1.0 - clockwise direction)
        The radius of the arc (R) is calculated based on the norm between the center point and the start point, therefore
        the end point is only used to specify the total angle that the arc does. If the end point does not lie perfectly
        within the specified arc, it does not matter (because of what is just mentioned)

        It is assumed that an arc section does not make a rotation for more than 2*pi. Be carefull, expecially when invoking angle()
        if you create an arc with more than 2*pi
        """
        super()
        self._xs = xs                           # Start point of the path
        self._ys = ys
        self._xc = xc                           # Center of the path
        self._yc = yc
        self._xe = xe                           # End point of the path
        self._ye = ye
        self._direction = direction             # Direction the arc should flow
        
        pc = np.array([self._yc, self._xc])
        ps = np.array([self._ys, self._xs])
        self._R = np.linalg.norm(pc - ps)       # Radius of the arc

    def curvature(self, s: float) -> float:
        return float(1/self._R)
    
    def angle(self, s: float) -> float:
        """
        @Override
        Returns the angle made with the arc. Since the Section is an arc
        the angle is given by the tangent to the arc in the point s
        with the x axis. 
        This angle varies between [-pi, pi]
        """
        pc = np.array([self._yc, self._xc])
        ps = np.array([self._ys, self._xs])
        pe = np.array([self._ye, self._xe])

        phi_0 = np.arctan2(ps[0] - pc[0], ps[1] - pc[1])        # The angle in the initial point of the arc
        phi_final = np.arctan2(pe[0] - pc[0], pe[1] - pc[1])    # The angle in the final point of the arc

        # Normalization factor so that s can vary between 0 and 1
        if ps[0] == pe[0] and ps[1] == pe[1]:
            factor = 2 * np.pi      # We want a full circle
        else:
            if self._direction == 1.0 and phi_final < phi_0:
                factor = 2*np.pi - (phi_0 - phi_final)
            elif self._direction == -1.0 and phi_final > phi_0:
                factor = 2*np.pi - (phi_final - phi_0)
            elif self._direction == 1.0 and phi_final > phi_0:
                factor = phi_final - phi_0
            else:
                factor = phi_0 - phi_final
        
        # Wrap the phi_0 angle between -pi and pi so that the curve is well parameterized
        phi_0 = Utils.wrapAngle(phi_0)

        #Since s is normalized between 0 and 1, then calculate X and Y accordingly
        aux_angle = Utils.wrapAngle(phi_0 + self._direction * s * factor)

        pd_dot_ = self._direction * np.array([np.cos(aux_angle), -np.sin(aux_angle)]);
        psid_ = np.arctan2(pd_dot_[0], pd_dot_[1])
        return psid_

    def getXYFromS(self, s: float) -> (float, float):
        """
        @Override
        Returns the X, Y position in the inertial frame of the point parameterized by s.
        If s > 1 the return will be the equivalent for s = 1
        If s < 0 the return will be the equivalent for s = 0
        """
        if s > 1:
            s = 1
        elif s < 0:
            s = 0
        
        pc = np.array([self._yc, self._xc])
        ps = np.array([self._ys, self._xs])
        pe = np.array([self._ye, self._xe])
        
        phi_0 = np.arctan2(ps[0] - pc[0], ps[1] - pc[1])      #Inital angle
        phi_final = np.arctan2(pe[0] - pc[0], pe[1] - pc[1])  #Final angle

        # Normalization factor so that s can vary between 0 and 1
        if ps[0] == pe[0] and ps[1] == pe[1]:
            factor = 2 * np.pi      # We want a full circle
        else:
            if self._direction == 1.0 and phi_final < phi_0:
                factor = 2*np.pi - (phi_0 - phi_final)
            elif self._direction == -1.0 and phi_final > phi_0:
                factor = 2*np.pi - (phi_final - phi_0)
            elif self._direction == 1.0 and phi_final > phi_0:
                factor = phi_final - phi_0
            else:
                factor = phi_0 - phi_final
        
        # Wrap the phi_0 angle between -pi and pi so that the curve is well parameterized
        phi_0 = Utils.wrapAngle(phi_0)
        #Since s is normalized between 0 and 1, then calculate X and Y accordingly
        aux_angle = Utils.wrapAngle(phi_0 + self._direction * s * factor)
        pd_ = pc + self._R * np.array([np.sin(aux_angle), np.cos(aux_angle)]);
        
        X = pd_[1]
        Y = pd_[0]
        return (X, Y)

    def __str__(self):
        """
        @Override
        Returns a string with the information the segment is an Arc 
        and the initial, center and final coordinates of the arc in the inertial frame and the direction
        """
        return "Line[ xs=" + str(self._xs) + ", ys=" + str(self._ys) + ", xc=" + str(self._xc) + ", yc=" + str(self._yc) + ", xe=" + str(self._xe) + ", ye=" + str(self._ye) + ", direction=" + str(self._direction) + "]"