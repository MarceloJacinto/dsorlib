from numpy import array
from numpy.testing import assert_allclose
from dsorlib.rotations import Rot_to_quaternion, RPY_to_quaternion, quaternion_to_Rot, quaternion_to_RPY, RPY_to_RotZYX, \
    RotZYX_to_RPY

"""
The following tests are conducted for the rotations module
The following URL was used to obtained the standard values to compare against
https://www.andre-gaschler.com/rotationconverter/
"""

def test1_Rot_to_quaternion():
    """
    Test1 the method Rot_to_quaternion
    """
    r = array([[-0.0690026, -0.9716355, 0.2261928],
               [0.6932812, 0.1163334, 0.7112156],
               [-0.7173561, 0.2058909, 0.6655893]])

    q_obtained = Rot_to_quaternion(r)
    q_expected = array([0.65439287, -0.19305097, 0.36046729, 0.63605396])
    assert_allclose(q_obtained, q_expected, rtol=2e-07, verbose=True)

def test2_Rot_to_quaternion():
    """
    Test2 the method Rot_to_quaternion
    """
    r2 = array([[0.0516208, 0.0057865, 0.9986500],
                [-0.2212602, 0.9751976, 0.0057865],
                [-0.9738476, -0.2212602, 0.0516208]])

    q2_obtained = Rot_to_quaternion(r2)
    q2_expected = array([0.7208397, -0.0787438, 0.6840972, -0.0787438])
    assert_allclose(q2_obtained, q2_expected, rtol=4e-07, verbose=True)

def test1_RPY_to_quaternion():
    """
    Test1 the method RPY_to_quaternion
    """
    roll: float = 0.3
    pitch: float = 0.8
    yaw: float = 1.67

    q_obtained = RPY_to_quaternion(roll, pitch, yaw)
    q_expected = array([0.65439287, -0.19305097, 0.36046729, 0.63605396])
    assert_allclose(q_obtained, q_expected, rtol=2e-07, verbose=True)

def test2_RPY_to_quaternion():
    """
    Test2 the method RPY_to_quaternion
    """
    roll: float = 1.8
    pitch: float = 1.8
    yaw: float = 1.8

    q_obtained = RPY_to_quaternion(roll, pitch, yaw)
    q_expected = array([0.7208397, -0.0787438, 0.6840972, -0.0787438])
    assert_allclose(q_obtained, q_expected, rtol=4e-07, verbose=True)


def test1_quaternion_to_Rot():
    """
    Test1 the method quaternion_to_Rot
    """
    q = array([0.65439287, -0.19305097, 0.36046729, 0.63605396])

    rot_obtained = quaternion_to_Rot(q)
    rot_expected = array([[-0.0690026, -0.9716355, 0.2261928],
                          [ 0.6932812,  0.1163334, 0.7112156],
                          [-0.7173561,  0.2058909, 0.6655893]])

    assert_allclose(rot_obtained, rot_expected, rtol=2e-07, verbose=True)

def test2_quaternion_to_Rot():
    """
    Test2 the method quaternion_to_Rot
    """
    q = array([0.7208397, -0.0787438, 0.6840972, -0.0787438])

    rot_obtained = quaternion_to_Rot(q)
    rot_expected = array([[0.0516208, 0.0057865, 0.9986500],
                          [-0.2212602, 0.9751976, 0.0057865],
                          [-0.9738476, -0.2212602, 0.0516208]])

    assert_allclose(rot_obtained, rot_expected, rtol=3e-06, verbose=True)

def test1_quaternion_to_RPY():
    """
    Test1 the method quaternion_to_RPY
    """
    q = array([0.65439287, -0.19305097, 0.36046729, 0.63605396])

    roll_obtained, pitch_obtained, yaw_obtained = quaternion_to_RPY(q)
    RPY_obtained = array([roll_obtained, pitch_obtained, yaw_obtained])
    RPY_expected = array([0.3, 0.8, 1.67])

    assert_allclose(RPY_obtained, RPY_expected, rtol=2e-07, verbose=True)

def test2_quaternion_to_RPY():
    """
    Test2 the method quaternion_to_RPY
    """
    q = array([0.7208397, -0.0787438, 0.6840972, -0.0787438])

    roll_obtained, pitch_obtained, yaw_obtained = quaternion_to_RPY(q)
    RPY_obtained = array([roll_obtained, pitch_obtained, yaw_obtained])
    RPY_expected = array([-1.341592, 1.341593, -1.341592])

    assert_allclose(RPY_obtained, RPY_expected, rtol=2e-07, verbose=True)

def test1_RPY_to_RotZYX():
    """
    Test1 the method RPY_to_RotZYX
    """
    roll: float = 0.3
    pitch: float = 0.8
    yaw: float = 1.67

    rot_obtained = RPY_to_RotZYX(roll, pitch, yaw)
    rot_expected = array([[-0.0690026, -0.9716355, 0.2261928],
                          [0.6932812, 0.1163334, 0.7112156],
                          [-0.7173561, 0.2058909, 0.6655893]])

    assert_allclose(rot_obtained, rot_expected, rtol=7e-07, verbose=True)

def test2_RPY_to_RotZYX():
    """
    Test2 the method RPY_to_RotZYX
    """
    roll: float = 1.8
    pitch: float = 1.8
    yaw: float = 1.8

    rot_obtained = RPY_to_RotZYX(roll, pitch, yaw)
    rot_expected = array([[0.0516208, 0.0057865, 0.9986500],
                          [-0.2212602, 0.9751976, 0.0057865],
                          [-0.9738476, -0.2212602, 0.0516208]])

    assert_allclose(rot_obtained, rot_expected, rtol=4e-06, verbose=True)

def test1_RotZYX_to_RPY():
    """
    Test1 the method RotZYX_to_RPY
    """
    r = array([[-0.0690026, -0.9716355, 0.2261928],
               [0.6932812, 0.1163334, 0.7112156],
               [-0.7173561, 0.2058909, 0.6655893]])

    roll_obtained, pitch_obtained, yaw_obtained = RotZYX_to_RPY(r)
    RPY_obtained = array([roll_obtained, pitch_obtained, yaw_obtained])
    RPY_expected = array([0.3, 0.8, 1.67])

    assert_allclose(RPY_obtained, RPY_expected, rtol=3e-07, verbose=True)

def test2_RotZYX_to_RPY():
    """
    Test2 the method RotZYX_to_RPY
    """
    r = array([[0.0516208, 0.0057865, 0.9986500],
               [-0.2212602, 0.9751976, 0.0057865],
               [-0.9738476, -0.2212602, 0.0516208]])

    roll_obtained, pitch_obtained, yaw_obtained = RotZYX_to_RPY(r)
    RPY_obtained = array([roll_obtained, pitch_obtained, yaw_obtained])
    RPY_expected = array([-1.341593, 1.341593, -1.341593])

    assert_allclose(RPY_obtained, RPY_expected, rtol=4e-07, verbose=True)

def main():
    # Test the conversion from rotation matrix to quaternion
    test1_Rot_to_quaternion()
    test2_Rot_to_quaternion()

    # Test the conversion from euler angles to quaternion
    test1_RPY_to_quaternion()
    test2_RPY_to_quaternion()

    # Test the conversion from quaternion to rotation matrix
    test1_quaternion_to_Rot()
    test2_quaternion_to_Rot()

    # Test the conversion from quaternion to euler angles
    test1_quaternion_to_RPY()
    test2_quaternion_to_RPY()

    # Test the conversion from euler angles to rotation matrix
    test1_RPY_to_RotZYX()
    test2_RPY_to_RotZYX()

    # Test the conversion from rotation matrix to euler angles
    test1_RotZYX_to_RPY()
    test2_RotZYX_to_RPY()

if __name__ == "__main__":
    main()
