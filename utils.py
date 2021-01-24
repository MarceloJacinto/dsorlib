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
from numpy import array, sin, cos, tan, maximum, minimum, pi


def integrate(x_dot, x, dt):
    """
    Computes the Integral using Euler's method
    :param x_dot: update on the state
    :param x: the previous state
    :param dt: the time delta

    :return The integral of the state x
    """
    return (dt * x_dot) + x


def saturate(value, min_val, max_val):
    """
    Computes the saturation of a value or an array of values
    and caps it between min_val and max_val

    :param value:
    :param min_val: The minimum accepted values
    :param max_val: The maximum accepted values

    :return: the clipped value or array of values
    """
    value = minimum(value, max_val)
    value = maximum(value, min_val)
    return value


def Smtrx(x: array):
    """
    Computes the skew-symmetric matrix S(x)
    :param x: a vector with 3 components x = [x1, x2, x3]
    :return: A 3x3 matrix with S(x) = ---        ---
                                     | 0   -x3  x2|
                                     | x3   0  -x1|
                                     | -x2  x1  0 |
                                     ---        ---
    """
    x = array(x).reshape((3,))
    return array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])


def ang_vel_rot_B_to_U(phi: float, theta: float, psi: float):
    """
    Transformation matrix for angular velocities from Body Frame to Inertial Frame
    :param phi: A float with the phi angle
    :param theta: A float with the theta angle
    :param psi: A float with the psi angle

    :return: a 3x3 matrix with the rotation from angular velocities of {B} with respect to {U} expressed in {B}
    to angular velocities of {B} with respect to {U} expressed in {U}
    """
    return array([[1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
                  [0, cos(phi), -sin(phi)],
                  [0, sin(phi) / cos(theta), cos(phi) / cos(theta)]])


def rot_matrix_B_to_U(phi: float, theta: float, psi: float):
    """
    Rotation matrix for linear velocities from Body Frame to Inertial Frame
    :param phi: A float with the phi angle
    :param theta: A float with the theta angle
    :param psi: A float with the psi angle

    :return a 3x3 matrix rotation matrix from linear velocities of {B} with respect to {U} expressed in {B}
    to linear velocities of {B} with respect to {U} expressed in {U}
    """
    return array([[cos(psi) * cos(theta), (-sin(psi) * cos(phi)) + (cos(psi) * sin(theta) * sin(phi)),
                   (sin(psi) * sin(phi)) + (cos(psi) * cos(phi) * sin(theta))],
                  [sin(psi) * cos(theta), (cos(psi) * cos(phi)) + (sin(phi) * sin(theta) * sin(psi)),
                   (-cos(psi) * sin(phi)) + (sin(theta) * sin(psi) * cos(phi))],
                  [-sin(theta), cos(theta) * sin(phi), cos(theta) * cos(phi)]])
