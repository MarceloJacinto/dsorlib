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
from numpy import array, zeros, eye, divide, sum, power, sqrt, dot, sign, sin, cos, arcsin, arctan2


def Rot_to_quaternion(r: array):
    """
    Compute a quaternion from a 3x3 rotation matrix

    :param r: A 3x3 rotation matrix
    :return: An array with the 4 elements of the quaternion

    Algorithm written by Daniel Mellinger (props to him!)
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.html
    """

    # Compute the trace of the rotation matrix
    tr = r[0, 0] + r[1, 1] + r[2, 2]

    if tr > 0:
        S = sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (r[2, 1] - r[1, 2]) / S
        qy = (r[0, 2] - r[2, 0]) / S
        qz = (r[1, 0] - r[0, 1]) / S
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        S = sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        qw = (r[2, 1] - r[1, 2]) / S
        qx = 0.25 * S
        qy = (r[0, 1] + r[1, 0]) / S
        qz = (r[0, 2] + r[2, 0]) / S
    elif r[1, 1] > r[2, 2]:
        S = sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        qw = (r[0, 2] - r[2, 0]) / S
        qx = (r[0, 1] + r[1, 0]) / S
        qy = 0.25 * S
        qz = (r[1, 2] + r[2, 1]) / S
    else:
        S = sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        qw = (r[1, 0] - r[0, 1]) / S
        qx = (r[0, 2] + r[2, 0]) / S
        qy = (r[1, 2] + r[2, 1]) / S
        qz = 0.25 * S

    q = array([qw, qx, qy, qz])
    q = q * sign(qw)

    return q


def RPY_to_quaternion(phi: float, theta: float, psi: float):
    """
    Convert 3 Euler angles (roll, pitch and yaw) to
    a quaternion

    :param phi: The roll angle (around x)
    :param theta: The pitch angle (around y)
    :param psi: The yaw angle (around z)
    :return: A numpy array with 4 elements composing the unit quaternion

    URL with the original algorithm:
    https://marc-b-reynolds.github.io/math/2017/04/18/TaitEuler.html
    Excellent resource to learn about quaternions!
    """

    quaternion = zeros(4)
    quaternion[0] = cos(phi / 2) * cos(theta / 2) * cos(psi / 2) + sin(phi / 2) * sin(theta / 2) * sin(psi / 2)
    quaternion[1] = sin(phi / 2) * cos(theta / 2) * cos(psi / 2) - cos(phi / 2) * sin(theta / 2) * sin(psi / 2)
    quaternion[2] = cos(phi / 2) * sin(theta / 2) * cos(psi / 2) + sin(phi / 2) * cos(theta / 2) * sin(psi / 2)
    quaternion[3] = cos(phi / 2) * cos(theta / 2) * sin(psi / 2) - sin(phi / 2) * sin(theta / 2) * cos(psi / 2)

    return quaternion


def quaternion_to_Rot(q: array):
    """
    Compute a 3x3 rotation matrix from a quaternion
    Rotation matrix from inertial frame {U} to body frame {B}

    :param q: An array with 4 elements composing the quaternion
    :return: A 3x3 rotation matrix

    Algorithm written by Daniel Mellinger (props to him!)
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.html
    """

    # Create a vector from the quaternion parameters (and check dimensions)
    q = array(q).reshape(4)

    # Normalize the quaternion
    q = divide(q, sqrt(sum(power(q, 2))))

    # Auxiliary matrix
    q_hat = zeros((3, 3))
    q_hat[0, 1] = -q[3]
    q_hat[0, 2] = q[2]
    q_hat[1, 2] = -q[1]
    q_hat[1, 0] = q[3]
    q_hat[2, 0] = -q[2]
    q_hat[2, 1] = q[1]

    # Return the rotation matrix
    return eye(3) + 2 * dot(q_hat, q_hat) + 2 * dot(q[0], q_hat)


def quaternion_to_RPY(q: array):
    """
    Compute the roll, pitch and yaw angles from q unit quaternion
    :param q: a numpy array with 4 elements
    :return: a tuple with roll, pitch and yaw angles

    Computations from
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """

    roll: float = arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - (2 * (power(q[1], 2) + power(q[2], 2))))
    pitch: float = arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw: float = arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - (2 * (power(q[2], 2) + power(q[3], 2))))

    return roll, pitch, yaw


def RPY_to_RotZYX(phi: float, theta: float, psi: float):
    """
    Compute the 3x3 rotation matrix that converts linear velocities of {B} with respect to the inertial frame {U}
    expressed in {B} to the linear velocities expressed in {U}

    All of these calculations follow the NED (North-East-Down) convention

    :param phi: The roll angle (around x)
    :param theta: The pitch angle (around y)
    :param psi: The yaw angle (around z)
    :return: a 3x3 numpy array with the rotation matrix (from body to inertial frame)
    """
    return array([[cos(psi) * cos(theta), (-sin(psi) * cos(phi)) + (cos(psi) * sin(theta) * sin(phi)),
                   (sin(psi) * sin(phi)) + (cos(psi) * cos(phi) * sin(theta))],
                  [sin(psi) * cos(theta), (cos(psi) * cos(phi)) + (sin(phi) * sin(theta) * sin(psi)),
                   (-cos(psi) * sin(phi)) + (sin(theta) * sin(psi) * cos(phi))],
                  [-sin(theta), cos(theta) * sin(phi), cos(theta) * cos(phi)]])


def RotZYX_to_RPY(r: array):
    """
    Compute the roll, pitch and yaw angles from a 3x3 rotation matrix in the form ZYX

    :param r: A 3x3 rotation matrix in the form ZYX
    :return: a tuple with the corresponding roll, pitch and yaw angles

    Computations are based on passing from rotation matrix -> quaternion -> euler angles
    """

    # Make sure that r is an array and reshape it to be a 3x3 matrix
    r = array(r).reshape((3, 3))

    # Compute the corresponding quaternions
    q = Rot_to_quaternion(r)

    # Compute the euler angles from the quaternion
    return quaternion_to_RPY(q)
