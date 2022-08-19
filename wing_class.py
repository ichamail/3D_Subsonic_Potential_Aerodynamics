from math import gamma
import numpy as np
from plot_functions import set_axes_equal
from vector_class import Vector
from scipy.stats import beta
from matplotlib import pyplot as plt
from airfoil_class import Airfoil


class Wing:
    
    def __init__(self, root_airfoil:Airfoil, tip_airfoil:Airfoil,
                 semi_span:float, sweep:float = 0, dihedral:float = 0,
                 twist:float = 0):
        
        self.semi_span = semi_span
        self.sweep = sweep
        self.dihedral = dihedral
        self.twist = twist
        
        self.root_airfoil = self.set_airfoil(root_airfoil, 0)
        self.tip_airfoil = self.set_airfoil(tip_airfoil, semi_span)
        
        self.taper_ratio = self.tip_airfoil.chord/self.root_airfoil.chord
  
    def set_airfoil(self, airfoil, span_location):
        
        lamda = np.deg2rad(self.sweep)
        x_coords = airfoil.x_coords + span_location * np.tan(lamda)
        
        gamma = np.deg2rad(self.dihedral)
        y_coords = span_location * np.ones_like(x_coords)
        z_coords = - airfoil.y_coords
        z_coords = z_coords + y_coords * np.sin(gamma)
        y_coords = y_coords - y_coords*(1 - np.cos(gamma))
        
        name = airfoil.name
        chord = airfoil.chord
        new_airfoil = Airfoil(name, chord, x_coords, y_coords)
        new_airfoil.set_z_coords(z_coords)
        
        return new_airfoil
    
    def new_x_spacing(self, num_x_points, location="root and tip"):
        
        if location == "root and tip":
            self.root_airfoil.new_x_spacing(num_x_points, indexing='xz')
            self.tip_airfoil.new_x_spacing(num_x_points, indexing='xz')
        elif location == "root":
            self.root_airfoil.new_x_spacing(num_x_points, indexing='xz')
        elif location == "tip":
            self.tip_airfoil.new_x_spacing(num_x_points, indexing='xz')          
   
    def plot(self):
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.set_xlim(-0.5, self.root_airfoil.chord + 0.5)
        ax.set_ylim(-(0.5 + self.semi_span), self.semi_span + 0.5)
        ax.set_zlim(-0.5 * self.root_airfoil.chord, 0.5 * self.root_airfoil.chord)
        ax.view_init(180, -90)
        
        x = self.root_airfoil.x_coords
        y = self.root_airfoil.y_coords
        z = self.root_airfoil.z_coords
        name = self.root_airfoil.name
        plt.plot(x, y, z, 'b', label="root airfoil: " + name)
        
        x = self.tip_airfoil.x_coords
        y = self.tip_airfoil.y_coords
        z = self.tip_airfoil.z_coords
        name = self.tip_airfoil.name
        plt.plot(x, y, z, 'r', label="tip airfoil: " + name)
        
        x = self.tip_airfoil.x_coords
        y = - self.tip_airfoil.y_coords
        z = self.tip_airfoil.z_coords
        plt.plot(x, y, z, 'r')
        
        ax.legend()
        set_axes_equal(ax)
        plt.show()

    def generate_mesh(self, num_x_shells:int, num_y_shells:int):
        
        # Double node ids for trailing edge nodes
        # With double node ids, suction side and pressure side trailing edge
        # panels are not considered to be neighbours
        
        self.new_x_spacing(num_x_shells, location="root")
        
        x_root = self.root_airfoil.x_coords
        z_root = self.root_airfoil.z_coords
        # y = np.linspace(self.semi_span, -self.semi_span, num_y_shells+1)
        y = DenserAtBoundaries(self.semi_span, -self.semi_span, num_y_shells+1,
                               alpha=0.3)
        
        # nx = len(x)-1
        nx = len(x_root)
        ny = len(y)
        
        X = np.zeros((nx, ny))
        Y = np.zeros((nx, ny))
        Z = np.zeros((nx, ny))
        C_r = self.root_airfoil.chord
        lamda = self.taper_ratio
        half_span = self.semi_span
        
        for j in range(ny):
            C_y = C_r * ( 1 - abs(y[j])/half_span * (1 - lamda) )
            x = C_y/C_r * x_root
            z = C_y/C_r * z_root
            X[:, j] = x
            Y[:, j] = y[j]
            Z[:, j] = z
        
        nodes = []
        delta = np.deg2rad(self.sweep)
        gamma = np.deg2rad(self.dihedral)
        for i in range(nx):
            for j in range(ny):
                x_ij = X[i][j] + abs(Y[i][j]) * np.tan(delta)
                y_ij = Y[i][j] - Y[i][j] * (1 - np.cos(gamma))
                z_ij = Z[i][j] + abs(Y[i][j]) * np.sin(gamma)
                nodes.append([x_ij, y_ij, z_ij])
                
        shells = []
        # for j in range(nx):
        for j in range(nx-1):
            for i in range(ny-1):
                # shells.append([(i+j*ny),
                #             (i+j*ny)+1,
                #             ((i+j*ny)+1 + ny)%len(nodes),
                #             ((i+j*ny)+ny)%len(nodes)])
                
                shells.append([(i+j*ny),
                            (i+j*ny)+1,
                            ((i+j*ny)+1 + ny),
                            ((i+j*ny)+ny)])
        
        return nodes, shells
    
    def generate_mesh2(self, num_x_shells:int, num_y_shells:int):
        
        self.new_x_spacing(num_x_shells, location="root")
        
        x_root = self.root_airfoil.x_coords
        z_root = self.root_airfoil.z_coords
        # y = np.linspace(self.semi_span, -self.semi_span, num_y_shells+1)
        y = DenserAtBoundaries(self.semi_span, -self.semi_span, num_y_shells+1,
                               alpha=0.3)
        
        nx = len(x_root)-1
        # nx = len(x_root)
        ny = len(y)
        
        X = np.zeros((nx, ny))
        Y = np.zeros((nx, ny))
        Z = np.zeros((nx, ny))
        C_r = self.root_airfoil.chord
        lamda = self.taper_ratio
        half_span = self.semi_span
        
        for j in range(ny):
            C_y = C_r * ( 1 - abs(y[j])/half_span * (1 - lamda) )
            x = C_y/C_r * x_root
            z = C_y/C_r * z_root
            X[:, j] = x
            Y[:, j] = y[j]
            Z[:, j] = z
        
        nodes = []
        delta = np.deg2rad(self.sweep)
        gamma = np.deg2rad(self.dihedral)
        for i in range(nx):
            for j in range(ny):
                x_ij = X[i][j] + abs(Y[i][j]) * np.tan(delta)
                y_ij = Y[i][j] - Y[i][j] * (1 - np.cos(gamma))
                z_ij = Z[i][j] + abs(Y[i][j]) * np.sin(gamma)
                nodes.append([x_ij, y_ij, z_ij])
                
        shells = []
        # for j in range(nx):
        for j in range(nx-1):
            for i in range(ny-1):
                # shells.append([(i+j*ny),
                #             (i+j*ny)+1,
                #             ((i+j*ny)+1 + ny)%len(nodes),
                #             ((i+j*ny)+ny)%len(nodes)])
                
                shells.append([(i+j*ny),
                            (i+j*ny)+1,
                            ((i+j*ny)+1 + ny),
                            ((i+j*ny)+ny)])
        
        return nodes, shells    
                
    def generate_wake_mesh(self, num_x_shells:int, num_y_shells:int):
        
        y = DenserAtBoundaries(self.semi_span, -self.semi_span, num_y_shells+1,
                               alpha=0.3)
        
        X = np.zeros((num_x_shells + 1, num_y_shells + 1))
        Y = np.zeros((num_x_shells + 1, num_y_shells + 1))
        Z = np.zeros((num_x_shells + 1, num_y_shells + 1))
        
        C_r = self.root_airfoil.chord
        lamda = self.taper_ratio
        half_span = self.semi_span
        
        for j in range(num_y_shells + 1):
            
            C_y = C_r * ( 1 - abs(y[j])/half_span * (1 - lamda) )
            x = C_y/C_r * self.root_airfoil.x_coords
            z = C_y/C_r * self.root_airfoil.z_coords
            
            x1 = x[0] - x[1]
            y1 = 0
            z1 = z[0] - z[1]
            
            x2 = x[-1] - x[-2]
            y2 = 0
            z2 = z[-1] - z[-2]
            
            vec1 = Vector((x1, y1, z1))
            vec2 = Vector((x2, y2, z2))
            
            bisector = vec1 + vec2
            bisector = 1/bisector.norm() * bisector
            bisector = 10 * C_r * bisector
            
            X[:, j] = np.linspace(x[0], x[0] + bisector.x, num_x_shells+1)
            Y[:, j] = y[j]
            Z[:, j] = np.linspace(z[0], z[0]+ bisector.z, num_x_shells+1)
        
        wake_nodes = []
               
        delta = np.deg2rad(self.sweep)
        gamma = np.deg2rad(self.dihedral)
        for i in range(num_x_shells + 1):
            for j in range(num_y_shells + 1):
                x_ij = X[i][j] + abs(Y[i][j]) * np.tan(delta)
                y_ij = Y[i][j] - Y[i][j] * (1 - np.cos(gamma))
                z_ij = Z[i][j] + abs(Y[i][j]) * np.sin(gamma)
                wake_nodes.append([x_ij, y_ij, z_ij])
                
        wake_shells = []
        nx = num_x_shells + 1
        ny = num_y_shells + 1
        for j in range(nx-1):
            for i in range(ny-1):                
                wake_shells.append([(i+j*ny),
                            (i+j*ny)+1,
                            ((i+j*ny)+1 + ny),
                            ((i+j*ny)+ny)])
        
        return wake_nodes, wake_shells
      
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
        
    
if __name__=="__main__":
    from matplotlib import pyplot as plt
    airfoil = Airfoil(name="naca0012", chord_length=1)
    wing = Wing(root_airfoil=airfoil, tip_airfoil=airfoil, semi_span=5,
                sweep=5, dihedral=10)
    wing.new_x_spacing(9)
    
    wing.plot() 