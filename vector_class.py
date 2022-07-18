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
        norm = Vector.dot_product(self, self)
        norm = np.sqrt(norm)
        return norm
    
    def scalar_product(self, scalar:float):
        x_comp = scalar * self.x
        y_comp = scalar * self.y
        z_comp = scalar * self.z
        components = (x_comp, y_comp, z_comp)
        scaled_vector = Vector(components)
        
        # ίσως θα πρέπει να το ορίσω έτσι το βαθμωτό γινόμενο για να μην 
        # δημιουργώ συνέχεια νέα Vector Objects
        # self.x = scalar * self.x
        # self.y = scalar * self.y
        # self.z = scalar * self.z
        
        return scaled_vector
    
    def transformation(self, Matrix:np.ndarray(shape=(3,3), dtype=float)):
        A = Matrix.copy()
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

if __name__=='__main__':
    pass