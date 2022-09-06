import numpy as np
from vector_class import Vector
from Algorithms import DenserAtBoundaries, cubic_function
from airfoil_class import Airfoil


class Wing:
    
    def __init__(self, root_airfoil:Airfoil, tip_airfoil:Airfoil,
                 semi_span:float, sweep:float = 0, dihedral:float = 0,
                 twist:float = 0):
        
        self.semi_span = semi_span
        self.sweep = sweep
        self.dihedral = dihedral
        self.twist = twist
        self.root_airfoil = root_airfoil
        self.tip_airfoil = tip_airfoil
        self.taper_ratio = self.tip_airfoil.chord/self.root_airfoil.chord
        self.RefArea = 0
        self.set_referenceArea()
    
    def set_referenceArea(self):
        C_r = self.root_airfoil.chord
        C_t = self.tip_airfoil.chord
        half_span = self.semi_span
        
        # ref_area = 2 * ((C_r + C_t) * half_span)/2
        
        # precise way
        
        twist = np.deg2rad(self.twist)
        twist_location = (0.25*C_t, 0)
        x_tip, z_tip = self.tip_airfoil.x_coords, - self.tip_airfoil.y_coords
        x_tip_twist, z_tip_twist = self.rotate(x_tip, z_tip, twist_location,
                                               twist)
        C_t = x_tip_twist.max() - x_tip_twist.min() 
        
        Gamma = np.deg2rad(self.dihedral)
        half_span = half_span - half_span*(1-np.cos(Gamma))
        
        ref_area = 2 * ((C_r + C_t) * half_span)/2
        self.RefArea = ref_area
        
    def new_x_spacing(self, num_x_points):
        self.root_airfoil.new_x_spacing2(num_x_points)
        self.tip_airfoil.new_x_spacing2(num_x_points)       

    def generate_bodyMesh(self, num_x_shells:int, num_y_shells:int,
                          mesh_shell_type:str='quadrilateral'):
        
        # Double node ids for trailing edge nodes
        # With double node ids, suction side and pressure side trailing edge
        # panels are not considered to be neighbours
        
        self.new_x_spacing(num_x_shells)
        
        # wing's coordinate system:
        # origin: root airfoils leading edge
        # x-axis: chord wise direction from leading edge to trailing edge
        # y-axis: spanwise direction form root to tip
        # z-axis: from top to bottom
        
        x_root = self.root_airfoil.x_coords
        z_root = - self.root_airfoil.y_coords
        x_tip = self.tip_airfoil.x_coords
        z_tip = - self.tip_airfoil.y_coords
        # y = np.linspace(self.semi_span, -self.semi_span, num_y_shells+1)
        y = DenserAtBoundaries(self.semi_span, -self.semi_span, num_y_shells+1,
                               alpha=0.3)
        
        nx = len(x_root)
        ny = len(y)
        
        X = np.zeros((nx, ny))
        Y = np.zeros((nx, ny))
        Z = np.zeros((nx, ny))
        
        C_r = self.root_airfoil.chord
        C_t = self.tip_airfoil.chord
        lamda = self.taper_ratio
        half_span = self.semi_span
        
        # dimensionless coords
        x_root = x_root/C_r
        z_root = z_root/C_r
        x_tip = x_tip/C_t
        z_tip = z_tip/C_t
        y = y/half_span
        root_twist = 0
        tip_twist = np.deg2rad(self.twist)
        
        for j in range(ny):
            C_y = C_r * ( 1 - abs(y[j]) * (1 - lamda) )
            
            # linear interpolation
            # x = {(y_tip - y)*x_root + (y - y_root)*x_tip}/(y_tip - y_root)
            # x = {y_tip*x_root + y*(x_tip - x_root)}/y_tip
            # x = x_root + y/y_tip * (x_tip - x_root)
        
            x_coords = x_root + (x_tip-x_root) * abs(y[j]) 
            z_coords = z_root + (z_tip-z_root) * abs(y[j])
            twist = root_twist + (tip_twist - root_twist) * abs(y[j])
            
            # interpolation using cubic function
            # x_coords = x_root + (x_tip-x_root) * cubic_function(abs(y[j])) 
            # z_coords = z_root + (z_tip-z_root) * cubic_function(abs(y[j])) 
            # twist = root_twist + (tip_twist
            #                       - root_twist) * cubic_function(abs(y[j])) 
            
            x, z = self.rotate(x_coords, z_coords, (0.25, 0), twist)
            
            X[:, j] = x * C_y
            Y[:, j] = y[j] * half_span
            Z[:, j] = z * C_y
        
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
                     
        if mesh_shell_type=="quadrilateral":
            
            # wing tip shells
            for i in [0, ny-1]:
                for j in range(int((nx-1)/2)):
                    if i == 0:
                        if j==0:
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx-1 -j-1)*ny)])
                            
                        elif j==(int((nx-1)/2)-1):
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx-1-j)*ny)])
                        else:
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx-1 -j-1)*ny),
                                           (i+(nx-1 - j)*ny)])
                                      
                    else:
                        if j==0:
                            shells.append([(i+j*ny),
                                           (i+(nx-1 -j-1)*ny),
                                           (i+(j+1)*ny)])
                            
                        elif j==(int((nx-1)/2)-1):
                            shells.append([(i+j*ny),
                                           (i+(nx-1 - j)*ny),
                                           (i+(nx-1 -j-1)*ny)])
                        else:
                            shells.append([(i+j*ny),
                                           (i+(nx-1 - j)*ny),
                                           (i+(nx-1 -j-1)*ny),
                                           (i+(j+1)*ny)])
            
            # pressure and suction sides
            for j in range(nx-1):
                for i in range(ny-1):                    
                    shells.append([(i+j*ny),
                                   (i+j*ny)+1,
                                   ((i+j*ny)+1 + ny),
                                   ((i+j*ny)+ny)])
                    
        elif mesh_shell_type=="triangular":
            
            # wing tip shells 
            for i in [0, ny-1]:
                for j in range(int((nx-1)/2)):
                    if i == 0:
                        if j==0:
                            shells.append([(i+j*ny),
                                        (i+(j+1)*ny),
                                        (i+(nx-1 -j-1)*ny)])
                              
                        elif j==(int((nx-1)/2)-1):
                            shells.append([(i+j*ny),
                                        (i+(j+1)*ny),
                                        (i+(nx-1-j)*ny)])

                        else:
                            shells.append([(i+j*ny),
                                        (i+(j+1)*ny),
                                        (i+(nx-1 - j)*ny)])
                            
                            shells.append([(i+(j+1)*ny),
                                        (i+(nx-1 -j-1)*ny),
                                        (i+(nx-1 - j)*ny)])
                              
                    else:
                        if j==0:
                            shells.append([(i+j*ny),
                                           (i+(nx-1 -j-1)*ny),
                                           (i+(j+1)*ny)])
                            
                        elif j==(int((nx-1)/2)-1):
                            shells.append([(i+j*ny),
                                           (i+(nx-1 - j)*ny),
                                           (i+(nx-1 -j-1)*ny)])
                        else:
                            shells.append([(i+j*ny),
                                           (i+(nx-1 - j)*ny),
                                           (i+(j+1)*ny)])
                            
                            shells.append([(i+(nx-1 - j)*ny),
                                           (i+(nx-1 -j-1)*ny),
                                           (i+(j+1)*ny)])
            
            # pressure and suction sides
            for j in range(nx-1):
                for i in range(ny-1):
                    
                    shells.append([(i+j*ny),
                                   (i+j*ny)+1,
                                   ((i+j*ny)+1 + ny)])
    
                    shells.append([((i+j*ny)+1 + ny),
                                   ((i+j*ny)+ny),
                                   (i+j*ny)])
                            
        return nodes, shells
    
    def generate_bodyMesh2(self, num_x_shells:int, num_y_shells:int,
                           mesh_shell_type:str="quadrilateral"):
        
        """
        note:        
        generate_bodyMesh function uses double node ids for trailing edge nodes.
        This way trailing edge shells doesn't share TE nodes. Hence trailing shells in suction side and trailing shells on pressure side will not be considered neighbours
        In generate_bodyMesh2 function shells at trailing egde share trailing edge nodes. This way trailing shells on suction side and trailing shells on pressure side will be considered neighbours
        """
        
        self.new_x_spacing(num_x_shells)
        
        # wing's coordinate system:
        # origin: root airfoils leading edge
        # x-axis: chord wise direction from leading edge to trailing edge
        # y-axis: spanwise direction form root to tip
        # z-axis: from top to bottom
        
        x_root = self.root_airfoil.x_coords
        z_root = - self.root_airfoil.y_coords
        x_tip = self.tip_airfoil.x_coords
        z_tip = - self.tip_airfoil.y_coords
        # y = np.linspace(self.semi_span, -self.semi_span, num_y_shells+1)
        y = DenserAtBoundaries(self.semi_span, -self.semi_span, num_y_shells+1,
                               alpha=0.3)
        
        nx = len(x_root)-1
        ny = len(y)
        
        X = np.zeros((nx+1, ny))
        Y = np.zeros((nx+1, ny))
        Z = np.zeros((nx+1, ny))
        
        C_r = self.root_airfoil.chord
        C_t = self.tip_airfoil.chord
        lamda = self.taper_ratio
        half_span = self.semi_span
        
        # dimensionless coords
        x_root = x_root/C_r
        z_root = z_root/C_r
        x_tip = x_tip/C_t
        z_tip = z_tip/C_t
        y = y/half_span
        
        root_twist = 0
        tip_twist = np.deg2rad(self.twist)
        
        for j in range(ny):
            
            C_y = C_r * ( 1 - abs(y[j]) * (1 - lamda) )
            
            # linear interpolation
            # x = {(y_tip - y)*x_root + (y - y_root)*x_tip}/(y_tip - y_root)
            # x = {y_tip*x_root + y*(x_tip - x_root)}/y_tip
            # x = x_root + y/y_tip * (x_tip - x_root)
        
            x_coords = x_root + (x_tip-x_root) * abs(y[j]) 
            z_coords = z_root + (z_tip-z_root) * abs(y[j])
            twist = root_twist + (tip_twist - root_twist) * abs(y[j])
            
            # interpolation using cubic function
            # x_coords = x_root + (x_tip-x_root) * cubic_function(abs(y[j])) 
            # z_coords = z_root + (z_tip-z_root) * cubic_function(abs(y[j])) 
            # twist = root_twist + (tip_twist
            #                       - root_twist) * cubic_function(abs(y[j]))
            
            x, z = self.rotate(x_coords, z_coords, (0.25, 0), twist)
            
            X[:, j] = x * C_y
            Y[:, j] = y[j] * half_span
            Z[:, j] = z * C_y
        
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
        if mesh_shell_type=="quadrilateral":
            
            # wing tip shells 
            for i in [0, ny-1]:
                for j in range(int(nx/2)):
                    if i == 0:
                        if j==0:
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx -j-1)*ny)])
                            
                        elif j==(int(nx/2)-1):
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx-j)*ny)])
                        else:
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx -j-1)*ny),
                                           (i+(nx - j)*ny)])
                                      
                    else:
                        if j==0:
                            shells.append([(i+j*ny),
                                           (i+(nx -j-1)*ny),
                                           (i+(j+1)*ny)])
                            
                        elif j==(int(nx/2)-1):
                            shells.append([(i+j*ny),
                                           (i+(nx - j)*ny),
                                           (i+(nx -j-1)*ny)])
                        else:
                            shells.append([(i+j*ny),
                                           (i+(nx - j)*ny),
                                           (i+(nx -j-1)*ny),
                                           (i+(j+1)*ny)])
                            
            # pressure and suction sides
            for j in range(nx):
                for i in range(ny-1):
                    shells.append([(i+j*ny),
                                   (i+j*ny)+1,
                                   ((i+j*ny)+1 + ny)%len(nodes),
                                   ((i+j*ny)+ny)%len(nodes)])
            
        elif mesh_shell_type=="triangular":
            
            # wing tips 
            for i in [0, ny-1]:
                for j in range(int(nx/2)):
                    if i == 0:
                        if j==0:
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx-j-1)*ny)])
                              
                        elif j==(int(nx/2)-1):
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx-j)*ny)])

                        else:
                            shells.append([(i+j*ny),
                                           (i+(j+1)*ny),
                                           (i+(nx-j)*ny)])
                            
                            shells.append([(i+(j+1)*ny),
                                           (i+(nx-j-1)*ny),
                                           (i+(nx-j)*ny)])
                              
                    else:
                        if j==0:
                            shells.append([(i+j*ny),
                                           (i+(nx-j-1)*ny),
                                           (i+(j+1)*ny)])
                            
                        elif j==(int(nx/2)-1):
                            shells.append([(i+j*ny),
                                           (i+(nx-j)*ny),
                                           (i+(nx-j-1)*ny)])
                        else:
                            shells.append([(i+j*ny),
                                           (i+(nx-j)*ny),
                                           (i+(j+1)*ny)])
                            
                            shells.append([(i+(nx-j)*ny),
                                           (i+(nx-j-1)*ny),
                                           (i+(j+1)*ny)])
            
            # pressure and suction sides
            for j in range(nx):
                for i in range(ny-1):
                    
                    shells.append([(i+j*ny),
                                   (i+j*ny)+1,
                                   ((i+j*ny)+1 + ny)%len(nodes)])
                    
                    shells.append([((i+j*ny)+1 + ny)%len(nodes),
                                   ((i+j*ny)+ny)%len(nodes),
                                   (i+j*ny)])
                       
        return nodes, shells         
                        
    def generate_wakeMesh(self, num_x_shells:int, num_y_shells:int,
                          mesh_shell_type:str='quadrilateral'):
        
        # wing's coordinate system:
        # origin: root airfoils leading edge
        # x-axis: chord wise direction from leading edge to trailing edge
        # y-axis: spanwise direction form root to tip
        # z-axis: from top to bottom
        
        x_root = self.root_airfoil.x_coords
        z_root = - self.root_airfoil.y_coords
        x_tip = self.tip_airfoil.x_coords
        z_tip = - self.tip_airfoil.y_coords
        
        y = DenserAtBoundaries(self.semi_span, -self.semi_span, num_y_shells+1,
                               alpha=0.3)
        
        X = np.zeros((num_x_shells + 1, num_y_shells + 1))
        Y = np.zeros((num_x_shells + 1, num_y_shells + 1))
        Z = np.zeros((num_x_shells + 1, num_y_shells + 1))
        
        C_r = self.root_airfoil.chord
        C_t = self.tip_airfoil.chord
        lamda = self.taper_ratio
        half_span = self.semi_span
        
        # dimensionless coords
        x_root = x_root/C_r
        z_root = z_root/C_r
        x_tip = x_tip/C_t
        z_tip = z_tip/C_t
        y = y/half_span
        root_twist = 0
        tip_twist = np.deg2rad(self.twist)
        
        for j in range(num_y_shells + 1):
            
            C_y = C_r * ( 1 - abs(y[j]) * (1 - lamda) )
            
            # linear interpolation
            # x = {(y_tip - y)*x_root + (y - y_root)*x_tip}/(y_tip - y_root)
            # x = {y_tip*x_root + y*(x_tip - x_root)}/y_tip
            # x = x_root + y/y_tip * (x_tip - x_root)
        
            x_coords = x_root + (x_tip-x_root) * abs(y[j]) 
            z_coords = z_root + (z_tip-z_root) * abs(y[j])
            twist = root_twist + (tip_twist - root_twist) * abs(y[j])
            
            # interpolation using cubic function
            # x_coords = x_root + (x_tip-x_root) * cubic_function(abs(y[j])) 
            # z_coords = z_root + (z_tip-z_root) * cubic_function(abs(y[j])) 
            # twist = root_twist + (tip_twist
            #                       - root_twist) * cubic_function(abs(y[j]))
            
            x, z = self.rotate(x_coords, z_coords, (0.25, 0), twist)
            
            x = x * C_y
            y[j] = y[j] * half_span
            z = z * C_y
            
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
                        
        if mesh_shell_type=="quadrilateral":
            
            for j in range(nx-1):
                for i in range(ny-1):
                                   
                    wake_shells.append([(i+j*ny),
                                        ((i+j*ny)+ny),
                                        ((i+j*ny)+1 + ny),
                                        (i+j*ny)+1])
                    
        elif mesh_shell_type=="triangular":
            
            for j in range(nx-1):
                for i in range(ny-1):
                    
                    wake_shells.append([(i+j*ny),
                                        ((i+j*ny)+ny),
                                        (i+j*ny)+1])
                    
                    wake_shells.append([((i+j*ny)+ny),
                                        ((i+j*ny)+1 + ny),
                                        (i+j*ny)+1])
                
        return wake_nodes, wake_shells
    
    def generate_wakeMesh2(self, V_fs:Vector,
                           num_x_shells:int, 
                           num_y_shells:int,
                           mesh_shell_type:str='quadrilateral'):
        
        """
        generate_wakeMesh generates a steady wake in the same direction as the bisector vector of trailing edge
        
        generate_wakeMesh2 generates a plane steady wake in free stream's vector direction
        """
        
        wake_direction_unit_vec = V_fs/V_fs.norm()
        vec = 10 * self.root_airfoil.chord * wake_direction_unit_vec
        
        # wing's coordinate system:
        # origin: root airfoils leading edge
        # x-axis: chord wise direction from leading edge to trailing edge
        # y-axis: spanwise direction form root to tip
        # z-axis: from top to bottom
        
        x_root = self.root_airfoil.x_coords
        z_root = - self.root_airfoil.y_coords
        x_tip = self.tip_airfoil.x_coords
        z_tip = - self.tip_airfoil.y_coords
        
        y = DenserAtBoundaries(self.semi_span, -self.semi_span, num_y_shells+1,
                               alpha=0.3)
        
        X = np.zeros((num_x_shells + 1, num_y_shells + 1))
        Y = np.zeros((num_x_shells + 1, num_y_shells + 1))
        Z = np.zeros((num_x_shells + 1, num_y_shells + 1))
        
        C_r = self.root_airfoil.chord
        C_t = self.tip_airfoil.chord
        lamda = self.taper_ratio
        half_span = self.semi_span
        
        # dimensionless coords
        x_root = x_root/C_r
        z_root = z_root/C_r
        x_tip = x_tip/C_t
        z_tip = z_tip/C_t
        y = y/half_span
        root_twist = 0
        tip_twist = np.deg2rad(self.twist)
        
        for j in range(num_y_shells + 1):
            
            C_y = C_r * ( 1 - abs(y[j]) * (1 - lamda) )
            
            # linear interpolation
            # x = {(y_tip - y)*x_root + (y - y_root)*x_tip}/(y_tip - y_root)
            # x = {y_tip*x_root + y*(x_tip - x_root)}/y_tip
            # x = x_root + y/y_tip * (x_tip - x_root)
        
            x_coords = x_root + (x_tip-x_root) * abs(y[j]) 
            z_coords = z_root + (z_tip-z_root) * abs(y[j])
            twist = root_twist + (tip_twist - root_twist) * abs(y[j])
            
            # interpolation using cubic function
            # x_coords = x_root + (x_tip-x_root) * cubic_function(abs(y[j])) 
            # z_coords = z_root + (z_tip-z_root) * cubic_function(abs(y[j])) 
            # twist = root_twist + (tip_twist
            #                       - root_twist) * cubic_function(abs(y[j]))
            
            x, z = self.rotate(x_coords, z_coords, (0.25, 0), twist)
            
            x = x * C_y
            y[j] = y[j] * half_span
            z = z * C_y
                        
            X[:, j] = np.linspace(x[0], x[0] + vec.x, num_x_shells+1)
            Y[:, j] = y[j]
            Z[:, j] = np.linspace(z[0], z[0]+ vec.z, num_x_shells+1)
        
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
                        
        if mesh_shell_type=="quadrilateral":
            
            for j in range(nx-1):
                for i in range(ny-1):
                                   
                    wake_shells.append([(i+j*ny),
                                        ((i+j*ny)+ny),
                                        ((i+j*ny)+1 + ny),
                                        (i+j*ny)+1])
                    
        elif mesh_shell_type=="triangular":
            
            for j in range(nx-1):
                for i in range(ny-1):
                    
                    wake_shells.append([(i+j*ny),
                                        ((i+j*ny)+ny),
                                        (i+j*ny)+1])
                    
                    wake_shells.append([((i+j*ny)+ny),
                                        ((i+j*ny)+1 + ny),
                                        (i+j*ny)+1])
                
        return wake_nodes, wake_shells
        
        pass
    
    def generate_mesh(self, num_x_bodyShells,
                      num_x_wakeShells, num_y_Shells,
                      mesh_shell_type:str="quadrilateral"):
        
        # body_nodes, body_shells = self.generate_bodyMesh(num_x_bodyShells,
        #                                                  num_y_Shells,
        #                                                  mesh_shell_type)
        
        body_nodes, body_shells = self.generate_bodyMesh2(num_x_bodyShells,
                                                         num_y_Shells,
                                                         mesh_shell_type)
        
        wake_nodes, wake_shells = self.generate_wakeMesh(num_x_wakeShells,
                                                         num_y_Shells,
                                                         mesh_shell_type)
        
        for i, wake_shell in enumerate(wake_shells):
            for j in range(len(wake_shell)):
                wake_shells[i][j] = wake_shells[i][j] + len(body_nodes)
    
        nodes = [*body_nodes, *wake_nodes]
        shells = [*body_shells, *wake_shells]
        
        return nodes, shells
    
    def generate_mesh2(self, V_fs:Vector, num_x_bodyShells,
                       num_x_wakeShells, num_y_Shells,
                       mesh_shell_type:str="quadrilateral"):
        
        # body_nodes, body_shells = self.generate_bodyMesh(num_x_bodyShells,
        #                                                  num_y_Shells,
        #                                                  mesh_shell_type)
        
        body_nodes, body_shells = self.generate_bodyMesh2(num_x_bodyShells,
                                                         num_y_Shells,
                                                         mesh_shell_type)
        
        wake_nodes, wake_shells = self.generate_wakeMesh2(V_fs,
                                                          num_x_wakeShells,
                                                          num_y_Shells,
                                                          mesh_shell_type)
        
        for i, wake_shell in enumerate(wake_shells):
            for j in range(len(wake_shell)):
                wake_shells[i][j] = wake_shells[i][j] + len(body_nodes)
    
        nodes = [*body_nodes, *wake_nodes]
        shells = [*body_shells, *wake_shells]
        
        return nodes, shells
    
    @staticmethod
    def give_shells_id_dict(num_x_bodyShells, num_x_wakeShells,
                            num_y_Shells, mesh_shell_type="quadrilateral"):
        
        # x shells on Suction side + x shells on Pressure side
        num_x_bodyShells = 2 * num_x_bodyShells
        
        if mesh_shell_type == "quadrilateral":
            num_UpperLower_Shells = num_x_bodyShells * num_y_Shells
            num_wing_tip_Shells = num_x_bodyShells
            num_bodyShells = num_UpperLower_Shells + num_wing_tip_Shells
            num_wakeShells = num_x_wakeShells * num_y_Shells
            
        elif mesh_shell_type == "triangular":
            num_UpperLower_Shells = num_x_bodyShells * num_y_Shells * 2
            num_wing_tip_Shells = 2 * num_x_bodyShells - 4
            num_bodyShells = num_UpperLower_Shells + num_wing_tip_Shells
            num_wakeShells = num_x_wakeShells * num_y_Shells * 2
            
        
        
        bodyShells_id_list = []
        wakeShells_id_list = []
                
        for id in range(num_bodyShells + num_wakeShells):
            
            if id < (num_bodyShells):
                bodyShells_id_list.append(id)
            else:
                wakeShells_id_list.append(id)

        shells_id_dict = {"body": bodyShells_id_list,
                          "wake": wakeShells_id_list}
        return shells_id_dict
        
    @staticmethod
    def give_TrailingEdge_Shells_id(num_x_bodyShells:int, num_y_Shells:int,
                                    mesh_shell_type:str="quadrilateral"):
        
        # x shells on Suction side + x shells on Pressure side
        num_x_bodyShells = 2 * num_x_bodyShells 
        
        if mesh_shell_type == "quadrilateral":
            c1 = 0
            c2 = 1
             
        elif mesh_shell_type == "triangular":
            c1 = 1
            c2 = 2
        
        num_WingTipShells = num_x_bodyShells * c2 - 4*c1
        
        SS_TE_shell_id_list = []
        PS_TE_shell_id_list = []

        for i in range(num_y_Shells):
            id = i * c2 + num_WingTipShells
            SS_TE_shell_id_list.append(id)
            
            id = (num_x_bodyShells - 1) * num_y_Shells * c2 + id + c1
            PS_TE_shell_id_list.append(id)
        
        TrailingEdge = {"suction side" : SS_TE_shell_id_list,
                        "pressure side" : PS_TE_shell_id_list}
        
        return TrailingEdge
    
    @staticmethod
    def give_wake_sheddingShells(num_x_wakeShells:int, TrailingEdge: dict,
                                 mesh_shell_type:str="quadrilateral"):
        
        if mesh_shell_type == "quadrilateral":
            c=1
        elif mesh_shell_type == "triangular":
            c=2
        
        num_y_Shells = len(TrailingEdge["suction side"])
        last_id = TrailingEdge["pressure side"][-1] + 1
        
        # wake_sheddingShells_id = {}
        # for j in range(num_y_Shells):
        #     id_list = []
        #     for k in range(c):
        #         id = last_id + j + k
        #         for i in range(num_x_wakeShells):
        #             id_list.append(id + c*num_y_Shells*i)
        
        wake_sheddingShells_id = {}
        for j in range(num_y_Shells):
            id_list = []
            for i in range(num_x_wakeShells):
                for k in range(c):
                    id = last_id + j + k
                    id_list.append(id + c*num_y_Shells*i)
            last_id = last_id + k
            
            wake_sheddingShells_id[TrailingEdge["suction side"][j]] = id_list
            wake_sheddingShells_id[TrailingEdge["pressure side"][j]] = id_list
        
        return wake_sheddingShells_id

    @staticmethod
    def rotate(x_coords, z_coords, rotate_location, rotate_angle):

        x_c4 = rotate_location[0]
        z_c4 = rotate_location[1]
        # angle = rotate_angle
        angle = -rotate_angle
        x = (
            (x_coords - x_c4) * np.cos(angle)
            + (z_coords - z_c4) * np.sin(angle)
            + x_c4 
        )
        
        z = (
            -(x_coords - x_c4) * np.sin(angle)
            + (z_coords - z_c4) * np.cos(angle)
            + z_c4                 
        )
        
        return x, z
    
    
      
if __name__=="__main__":
    from mesh_class import PanelMesh
    
    
    airfoil = Airfoil(name="naca0012", chord_length=1)
    wing = Wing(root_airfoil=airfoil, tip_airfoil=airfoil, semi_span=5,
                sweep=5, dihedral=10)
    wing.new_x_spacing(9)
    
    root_airfoil = Airfoil(name="naca0012", chord_length=1)
    tip_airfoil = Airfoil(name="naca0012", chord_length=0.8)
    wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=10, dihedral=0,
                twist=10)
    
    body_nodes, body_shells = wing.generate_bodyMesh2(4, 4, "quadrilateral")

    # for node_id, node in enumerate(body_nodes):
    #     print(node_id, node)

    # for shell_id, shell in enumerate(body_shells):
    #     print(shell_id, shell)

    body_mesh = PanelMesh(body_nodes, body_shells)
    # for shell_id, neighbours in enumerate(mesh.shell_neighbours):
    #     print(shell_id, neighbours)
        
    body_mesh.plot_panels(elevation=-150, azimuth=-120)


    ######### wing mesh with wake ###############
    wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=10, dihedral=10,
                twist=2)
    num_x_bodyShells = 4
    num_x_wakeShells = 4
    num_y_Shells = 4

    # body_nodes, body_shells = wing.generate_bodyMesh(num_x_bodyShells,
    #                                               num_y_Shells)
    # wake_nodes, wake_shells = wing.generate_wakeMesh(num_x_wakeShells,
    #                                                   num_y_Shells)

    # for wake_node_id, wake_node in enumerate(wake_nodes):
    #     print(wake_node_id, wake_node)

    # for wake_shell_id, wake_shell in enumerate(wake_shells):
    #     print(wake_shell_id, wake_shell)

    # for i, wake_shell in enumerate(wake_shells):
    #     for j in range(len(wake_shell)):
    #         wake_shells[i][j] = wake_shells[i][j] + len(body_nodes)
        
    # nodes = [*body_nodes, *wake_nodes]
    # shells = [*body_shells, *wake_shells]

    nodes, shells = wing.generate_mesh(num_x_bodyShells, num_x_wakeShells,
                                       num_y_Shells, mesh_shell_type="triangular")
    
    # for node_id, node in enumerate(nodes):
    #     print(node_id, node)

    # for shell_id, shell in enumerate(shells):
    #     print(shell_id, shell)

    mesh = PanelMesh(nodes, shells)
    mesh.plot_panels(elevation=-150, azimuth=-120)

    shells_id = wing.give_shells_id_dict(num_x_bodyShells,
                                         num_x_wakeShells,
                                         num_y_Shells, "triangular")
    # print(shells_id["body"])
    # print(shells_id["wake"])


    TrailingEdge = wing.give_TrailingEdge_Shells_id(num_x_bodyShells,
                                                    num_y_Shells,
                                                    mesh_shell_type="triangular")

    wake_sheddingShells = wing.give_wake_sheddingShells(num_x_wakeShells,
                                                        TrailingEdge,
                                                        mesh_shell_type="triangular")

    # for trailing_edge_shell_id in wake_sheddingShells:
    #     print(trailing_edge_shell_id,
    #           wake_sheddingShells[trailing_edge_shell_id])

    # for shell_id in TrailingEdge["pressure side"]:
    #     print(wake_sheddingShells[shell_id])
    
    # mesh = PanelMesh(nodes, shells, shells_id, TrailingEdge,
    #                  wake_sheddingShells)
    # print(mesh.panels_id)
    # print(mesh.TrailingEdge)    
    # print(mesh.wake_sheddingShells)
    
    # for id in mesh.panels_id["body"]:
    #     print(mesh.panels[id].id, id)
    #     if mesh.panels[id].id in mesh.TrailingEdge["suction side"]:
    #         print(True)
    #         print(mesh.wake_sheddingShells[id])
    
    pass