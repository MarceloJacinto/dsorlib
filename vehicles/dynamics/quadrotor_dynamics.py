from numpy import array, diag
from numpy.linalg import inv


class QuadrotorDynamics:

    def __init__(self,
                 m: float,  # Mass
                 inertia_tensor: array,  # The inertia tensor (a vector of 9 elements)
                 g: float):

        self.m: float = float(m)  # The mass of the vehicle
        self.inertia_tensor: array = array(inertia_tensor)  # The inertia of the vehicle
        self.g: float = float(g)  # The gravity acceleration [m/s^2]

        # Check if inertia tensor is given as a vector of 3 terms (diagonal matrix)
        # or a vector with 9 terms (the complete matrix which might be non-diagonal)
        if self.inertia_tensor.size == 3:
            self.inertia_matrix = diag(self.inertia_tensor).reshape((3, 3))
        else:
            self.inertia_matrix = self.inertia_tensor.reshape((3, 3))

        # Compute the inverse of the inertia matrix
        self.inertia_matrix_inv = inv(self.inertia_matrix)