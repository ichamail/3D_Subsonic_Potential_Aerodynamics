import numpy as np


class Vector:
    
    def __init__(self, components:tuple):
        self.x = components[0]
        self.y = components[1]
        self.z = components[2]
    
    def __str__(self):
        """Human-readable string representation of the vector."""
        return '({:g})ex + ({:g})ey + ({:g})ez'.format(self.x, self.y, self.z)
    
    def norm(self):
        norm = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return norm
        
    def transformation(self, Matrix:np.ndarray(shape=(3,3), dtype=float)):
        A = np.copy(Matrix)
        x_comp = A[0][0] * self.x + A[0][1] * self.y + A[0][2] * self.z
        y_comp = A[1][0] * self.x + A[1][1] * self.y + A[1][2] * self.z
        z_comp = A[2][0] * self.x + A[2][1] * self.y + A[2][2] * self.z
        components= (x_comp, y_comp, z_comp)
        transformed_vector = Vector(components)
        return transformed_vector
    
    @classmethod
    def addition(cls, vector1, vector2):
        x_comp = vector1.x + vector2.x
        y_comp = vector1.y + vector2.y
        z_comp = vector1.z + vector2.z
        components = (x_comp, y_comp, z_comp)
        vector3 = Vector(components)
        return vector3
    
    @classmethod
    def dot_product(cls, vector1, vector2):
        v1, v2 = vector1, vector2
        return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
    
    @classmethod
    def cross_product(cls, vector1, vector2):
        v1, v2 = vector1, vector2
        x_comp = v1.y*v2.z - v1.z*v2.y
        y_comp = - (v1.x*v2.z - v1.z*v2.x)
        z_comp = v1.x*v2.y - v1.y*v2.x
        components = (x_comp, y_comp, z_comp)
        vector3 = Vector(components)
        return vector3
    
    def __add__(self, vector):
        x = self.x + vector.x
        y = self.y + vector.y
        z = self.z + vector.z
        components = (x, y, z)
        return Vector(components)
       
    def __sub__(self, vector):
        x = self.x - vector.x
        y = self.y - vector.y
        z = self.z - vector.z
        components = (x, y, z)
        return Vector(components)
    
    def __mul__(self, vector_or_scalar):
        if type(vector_or_scalar) == Vector:
            vector = vector_or_scalar
            x1, x2 = self.x, vector.x
            y1, y2 = self.y, vector.y
            z1, z2 = self.z, vector.z
            dot_product = x1*x2 + y1*y2 + z1*z2
            return dot_product
        else:
            scalar = vector_or_scalar
            x = scalar * self.x 
            y = scalar * self.y 
            z = scalar * self.z
            components = (x, y, z)
            return Vector(components)
    
    def __rmul__(self, vector_or_scalar):
        if type(vector_or_scalar) == Vector:
            vector = vector_or_scalar
            x1, x2 = self.x, vector.x
            y1, y2 = self.y, vector.y
            z1, z2 = self.z, vector.z
            dot_product = x1*x2 + y1*y2 + z1*z2
            return dot_product
        else:
            scalar = vector_or_scalar
            x = scalar * self.x 
            y = scalar * self.y 
            z = scalar * self.z
            components = (x, y, z)
            return Vector(components)
    
    def __matmul__(self, vector):
        return self.x*vector.x + self.y*vector.y + self.z*vector.z
        
    def __truediv__(self, scalar):
        x = self.x / scalar
        y = self.y / scalar
        z = self.z / scalar
        components = (x, y, z)
        return Vector(components)
    
    def __pos__(self):
        return self
       
    def __neg__(self):
        x = - self.x 
        y = - self.y 
        z = - self.z
        components = (x, y, z)
        return Vector(components)
        
        
if __name__=='__main__':
    
    pass