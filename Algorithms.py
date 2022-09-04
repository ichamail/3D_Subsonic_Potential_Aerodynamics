import numpy as np
from numpy.linalg import inv
from scipy.stats import beta

def LeastSquares(A, b):
    transposed = A.T
    inverted = inv(transposed @ A)
    x = (inverted @ transposed) @ b
    return x

def DenserAtBoundaries(start, end, num_points, alpha):
    '''
    alpha exists in (-oo, +oo)
    when alpha is 1 => evenly spaced
    when alpha < 1 denser at boundaries
    when alpha > 1 denser at midpoint
    '''
    x = np.linspace(0, 1, num_points)
    a = b = 2-alpha
    return start + beta.cdf(x, a, b) * (end-start)

def cubic_function(x):
    
    """
    cubic function used for airfoil interpolations
    
    from
    May-Fun Liou, Hyoungjin Kim, ByungJoon Lee, and Meng-Sing Liou
    "Aerodynamic Design of the Hybrid Wing Body Propulsion-Airframe Integration"
    """
    return x**2 * (3 - 2*x)

if __name__ == "__main__":
    A = np.array([[0, 1],
                [1, 1],
                [2, 1],
                [3, 1]])
    b = np.array([-1, 0.2, 0.9, 2.1])

    x = np.linalg.lstsq(A, b, rcond=None)[0]

    print(x)
    
    b = np.array([[-1],
                  [0.2],
                  [0.9],
                  [2.1]])
    
    x = LeastSquares(A, b)
    print(x)
    
    