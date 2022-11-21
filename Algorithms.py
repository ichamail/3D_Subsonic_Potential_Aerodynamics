import numpy as np
from numpy.linalg import inv
from scipy.stats import beta
from vector_class import Vector
from matplotlib import pyplot as plt

def LeastSquares(A, b):
    transposed = A.T
    inverted = inv(transposed @ A)
    x = (inverted @ transposed) @ b
    return x

def DenserAtBoundaries(start, end, num_points, alpha):
    '''
    Beta distribution
    
    Cumulative distribution function of beta distribution
    
    alpha exists in (-oo, +oo)
    when alpha is 1 => evenly spaced
    when alpha < 1 denser at boundaries
    when alpha > 1 denser at midpoint
    '''
    x = np.linspace(0, 1, num_points)
    a = b = 2-alpha
    return start + beta.cdf(x, a, b) * (end-start)

def cosspace(start, end, num_points):
    mean = (start+end)/2
    amp = (end-start)/2
    return mean + amp * np.cos(np.linspace(np.pi, 0, num_points))

def test_spacing(start, end, num_points, alpha):


    x = np.linspace(start, end, num_points)
    y1 = np.linspace(start, end, num_points)
    y2 = DenserAtBoundaries(start, end, num_points, alpha)
    y3 = cosspace(start, end, num_points)

    plt.plot(x, y1, "r", label="linear spacing")
    plt.plot(x, y2, "b", label = "prob dens func Beta, α=" + str(alpha))
    plt.plot(x, y3, "m", label="cosine spacing")

    plt.legend()
    plt.show()

def cubic_function(x):
    
    """
    cubic function used for airfoil interpolations
    
    from
    May-Fun Liou, Hyoungjin Kim, ByungJoon Lee, and Meng-Sing Liou
    "Aerodynamic Design of the Hybrid Wing Body Propulsion-Airframe Integration"
    """
    return x**2 * (3 - 2*x)

def interpolation(root_value, tip_value, span_percentage, type:str="linear"):
    
    """
    linear interpolation:
    x = {(y_tip - y)*x_root + (y - y_root)*x_tip}/(y_tip - y_root)
    x = {y_tip*x_root + y*(x_tip - x_root)}/y_tip
    x = x_root + y/y_tip * (x_tip - x_root)
    """
    
    if type == "linear":
        section_value = root_value + (tip_value - root_value) * span_percentage
    elif type == "cubic":
        section_value = (
            root_value + (tip_value 
                          - root_value) * cubic_function(span_percentage)
            )
    
    return section_value

def light_vector(magnitude, alpha, beta):
    
    """
    alpha: angle [degs] of light vector with x-axis in x-z plane
    beta: angle [degs] of light vector with x-axis in x-y plane
    """
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    x_component = magnitude * np.cos(alpha) * np.cos(beta)
    y_component = magnitude * np.cos(alpha) * np.sin(beta)
    z_component = - magnitude * np.sin(alpha)
    light_vector = Vector((x_component, y_component, z_component))
    return light_vector

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
    
    test_spacing(0, 100, 100, alpha=-0.15)