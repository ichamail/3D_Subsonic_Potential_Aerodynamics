import numpy as np
from vector_class import Vector
from Algorithms import DenserAtBoundaries, interpolation
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
                           mesh_shell_type:str="quadrilateral"):
        
        """
        note:
        In generate_bodyMesh function shells at trailing egde share trailing edge nodes. This way trailing shells on suction side and trailing shells on pressure side will be considered neighbours        
        generate_bodyMesh2 function uses double node ids for trailing edge nodes.
        This way trailing edge shells doesn't share TE nodes. Hence trailing shells in suction side and trailing shells on pressure side will not be considered neighbours
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
        
        y_left = DenserAtBoundaries(self.semi_span, 0, num_y_shells + 1,
                                    alpha=0.3)
        y_right = DenserAtBoundaries(0,-self.semi_span, num_y_shells + 1,
                                     alpha=0.3)
        y = np.array([*y_left, *y_right[1:]])
        
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
        span_percentage = y/half_span
        root_twist = 0
        tip_twist = np.deg2rad(self.twist)
        delta = np.deg2rad(self.sweep)
        gamma = np.deg2rad(self.dihedral)
        
        for j in range(ny):
                        
            C_y = interpolation(C_r, C_t, abs(span_percentage[j]))
            
            x_coords = interpolation(x_root, x_tip, abs(span_percentage[j]))
            z_coords = interpolation(z_root, z_tip, abs(span_percentage[j]))
            twist = interpolation(root_twist, tip_twist,
                                  abs(span_percentage[j]))
            
            x_coords = x_coords * C_y
            y_coords = span_percentage[j] * half_span * np.ones_like(x_coords)
            span_location = y_coords[0]
            z_coords = z_coords * C_y
            
            
            x, y, z = self.move_airfoil_section(x_coords, y_coords, z_coords,
                                                span_location, C_y,
                                                twist, delta, gamma)           
            
            X[:, j] = x 
            Y[:, j] = y
            Z[:, j] = z 
        
        nodes = []
        for i in range(nx):
            for j in range(ny):
                node = (X[i][j], Y[i][j], Z[i][j])                    
                nodes.append(node)
               
        shells = []
        if mesh_shell_type=="quadrilateral":
                                       
            # pressure and suction sides
            for j in range(nx):
                for i in range(ny-1):
                    shells.append([(i+j*ny),
                                   (i+j*ny)+1,
                                   ((i+j*ny)+1 + ny)%len(nodes),
                                   ((i+j*ny)+ny)%len(nodes)])
            
        elif mesh_shell_type=="triangular":
            
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
        
    def generate_bodyMesh2(self, num_x_shells:int, num_y_shells:int,
                          mesh_shell_type:str='quadrilateral'):
        
        """
        note:
        In generate_bodyMesh function shells at trailing egde share trailing edge nodes. This way trailing shells on suction side and trailing shells on pressure side will be considered neighbours        
        generate_bodyMesh2 function uses double node ids for trailing edge nodes.
        This way trailing edge shells doesn't share TE nodes. Hence trailing shells in suction side and trailing shells on pressure side will not be considered neighbours
        """
        
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
        
        y_left = DenserAtBoundaries(self.semi_span, 0, num_y_shells + 1,
                                    alpha=0.3)
        y_right = DenserAtBoundaries(0,-self.semi_span, num_y_shells + 1,
                                     alpha=0.3)
        y = np.array([*y_left, *y_right[1:]])
        
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
        span_percentage = y/half_span
        root_twist = 0
        tip_twist = np.deg2rad(self.twist)
        delta = np.deg2rad(self.sweep)
        gamma = np.deg2rad(self.dihedral)
        
        for j in range(ny):
            
            C_y = interpolation(C_r, C_t, abs(span_percentage[j]))
            
            x_coords = interpolation(x_root, x_tip, abs(span_percentage[j]))
            z_coords = interpolation(z_root, z_tip, abs(span_percentage[j]))
            twist = interpolation(root_twist, tip_twist,
                                  abs(span_percentage[j]))
            
            x_coords = x_coords * C_y
            y_coords = span_percentage[j] * half_span * np.ones_like(x_coords)
            span_location = y_coords[0]
            z_coords = z_coords * C_y
            
            
            x, y, z = self.move_airfoil_section(x_coords, y_coords, z_coords,
                                                span_location, C_y,
                                                twist, delta, gamma)           
            
            X[:, j] = x 
            Y[:, j] = y
            Z[:, j] = z 
        
        nodes = []
        for i in range(nx):
            for j in range(ny):
                node = (X[i][j], Y[i][j], Z[i][j])                    
                nodes.append(node)
                
        shells = []
                     
        if mesh_shell_type=="quadrilateral":
            
            # pressure and suction sides
            for j in range(nx-1):
                for i in range(ny-1):                    
                    shells.append([(i+j*ny),
                                   (i+j*ny)+1,
                                   ((i+j*ny)+1 + ny),
                                   ((i+j*ny)+ny)])
                    
        elif mesh_shell_type=="triangular":
            
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
       
    def generate_wingTipMesh(self, num_x_shells,
                             mesh_shell_type:str="quadrilateral"):
        
        self.new_x_spacing(num_x_shells)
                
        Ct = self.tip_airfoil.chord
        
        x_upper, z_upper = self.tip_airfoil.give_suctionSide()
        x_chord = x_upper.copy()
        z_chord = np.zeros_like(x_chord)
        y_chord = self.semi_span * np.ones_like(x_chord)
        
        x_tip = self.tip_airfoil.x_coords.copy()
        z_tip = -self.tip_airfoil.y_coords.copy()
        y_tip = self.semi_span * np.ones_like(z_tip)
        
        twist = np.deg2rad(self.twist)
        delta = np.deg2rad(self.sweep)
        gamma = np.deg2rad(self.dihedral)
        
        x_chord, y_chord, z_chord = self.move_airfoil_section(x_chord, y_chord,
                                                              z_chord, self.semi_span, Ct,twist, delta, gamma)
        
        x_tip, y_tip, z_tip = self.move_airfoil_section(x_tip, y_tip, z_tip,
                                                        self.semi_span, Ct,
                                                        twist, delta, gamma)
        index = len(x_upper)
        x_tip_upper, x_tip_lower = x_tip[0:index], np.flip(x_tip[index:])
        y_tip_upper, y_tip_lower = y_tip[0:index], np.flip(y_tip[index:])
        z_tip_upper, z_tip_lower = z_tip[0:index], np.flip(z_tip[index:])
        
        
        # generate nodes
        # left wingtip
        x_first, y_first, z_first = x_chord[0], y_chord[0], z_chord[0]
        x_last, y_last, z_last = x_chord[-1], y_chord[-1], z_chord[-1]
        nodes = []                      
        nodes.append((x_first, y_first, z_first))
        for i in range(1, len(x_upper)-1):
            nodes.append((x_tip_upper[i], y_tip_upper[i], z_tip_upper[i]))
            
            nodes.append((x_chord[i], y_chord[i], z_chord[i]))
            
            nodes.append((x_tip_lower[i], y_tip_lower[i], z_tip_lower[i]))
                
        nodes.append((x_last, y_last, z_last))
        
        # right wingtip
        x_first, y_first, z_first = x_chord[0], -y_chord[0], z_chord[0]
        x_last, y_last, z_last = x_chord[-1], -y_chord[-1], z_chord[-1]
                     
        nodes.append((x_first, y_first, z_first))
        for i in range(1, len(x_upper)-1):
            
            nodes.append((x_tip_upper[i], -y_tip_upper[i], z_tip_upper[i]))
            
            nodes.append((x_chord[i], -y_chord[i], z_chord[i]))
            
            nodes.append((x_tip_lower[i], -y_tip_lower[i], z_tip_lower[i]))
                
        nodes.append((x_last, y_last, z_last))
        
        # generate shells
        shells = []
        ni = len(x_upper)  # x_tip_upper + first + last nodes
        nj = 3
        
        if mesh_shell_type == "quadrilateral":       
            # left tip
            const = -2   
            for i in range(ni-1):
                for j in range(nj-1):
            
                    if i==0:
                        shells.append([0, j+1, j+2])
                    elif i == ni-2:
                        shells.append([(const + i*nj + j),
                                    (const + (i+1)*nj),
                                    (const + i*nj + j+1)])
                    else:
                        shells.append([(const + i*nj + j),
                                    (const + (i+1)*nj +j),
                                    (const + (i+1)*nj + j+1),
                                    (const + i*nj + j+1)])
            
            # right tip
            const = const + int(len(nodes)/2)
            for i in range(ni-1):
                for j in range(nj-1):
                            
                    if i==0:
                        shells.append([const+2, const+2 + j+2, const+2 + j+1])
                    elif i == ni-2:
                        shells.append([(const + i*nj + j),
                                    (const + i*nj + j+1),
                                    (const + (i+1)*nj)])
                    else:
                        shells.append([(const + i*nj + j),
                                    (const + i*nj + j+1),
                                    (const + (i+1)*nj + j+1),
                                    (const + (i+1)*nj +j)])
                                
        elif mesh_shell_type == "triangular":
            # left tip
            const = -2   
            for i in range(ni-1):
                for j in range(nj-1):
            
                    if i==0:
                        shells.append([0, j+1, j+2])
                    elif i == ni-2:
                        shells.append([(const + i*nj + j),
                                    (const + (i+1)*nj),
                                    (const + i*nj + j+1)])
                    else:
                        shells.append([(const + i*nj + j),
                                    (const + (i+1)*nj +j),
                                    (const + i*nj + j+1)])
                        
                        shells.append([(const + (i+1)*nj +j),
                                    (const + (i+1)*nj + j+1),
                                    (const + i*nj + j+1)])
            
            # right tip
            const = const + int(len(nodes)/2)
            for i in range(ni-1):
                for j in range(nj-1):
                            
                    if i==0:
                        shells.append([const+2, const+2 + j+2, const+2 + j+1])
                    elif i == ni-2:
                        shells.append([(const + i*nj + j),
                                    (const + i*nj + j+1),
                                    (const + (i+1)*nj)])
                    else:
                        shells.append([(const + i*nj + j),
                                    (const + i*nj + j+1),
                                    (const + (i+1)*nj +j)])
                        
                        shells.append([(const + i*nj + j+1),
                                    (const + (i+1)*nj + j+1),
                                    (const + (i+1)*nj +j)])
        
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
        
        y_left = DenserAtBoundaries(self.semi_span, 0, num_y_shells + 1,
                                    alpha=0.3)
        y_right = DenserAtBoundaries(0,-self.semi_span, num_y_shells + 1,
                                     alpha=0.3)
        y = np.array([*y_left, *y_right[1:]])
        
        nx, ny = num_x_shells + 1, len(y)
        X = np.zeros((nx, ny))
        Y = np.zeros_like(X)
        Z = np.zeros_like(X)
        
        C_r = self.root_airfoil.chord
        C_t = self.tip_airfoil.chord
        half_span = self.semi_span
        
        # dimensionless coords
        x_root = x_root/C_r
        z_root = z_root/C_r
        x_tip = x_tip/C_t
        z_tip = z_tip/C_t
        span_percentage = y/half_span
        root_twist = 0
        tip_twist = np.deg2rad(self.twist)
        delta = np.deg2rad(self.sweep)
        gamma = np.deg2rad(self.dihedral)
        
        for j in range(ny):
            
            C_y = interpolation(C_r, C_t, abs(span_percentage[j]))
            
            x_coords = interpolation(x_root, x_tip, abs(span_percentage[j]))
            z_coords = interpolation(z_root, z_tip, abs(span_percentage[j]))
            twist = interpolation(root_twist, tip_twist,
                                  abs(span_percentage[j]))
            
            x_coords = x_coords * C_y
            y_coords = span_percentage[j] * half_span * np.ones_like(x_coords)
            span_location = y_coords[0]
            z_coords = z_coords * C_y
                        
            x, y, z = self.move_airfoil_section(x_coords, y_coords, z_coords,
                                                span_location, C_y,
                                                twist, delta, gamma)
                       
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
               
        for i in range(nx):
            for j in range(ny):
                node = (X[i][j], Y[i][j], Z[i][j])                    
                wake_nodes.append(node)
                
        wake_shells = []
                                
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
        
        nx, ny = num_x_shells + 1, len(y)
        X = np.zeros((nx, ny))
        Y = np.zeros_like(X)
        Z = np.zeros_like(X)
        
        C_r = self.root_airfoil.chord
        C_t = self.tip_airfoil.chord
        half_span = self.semi_span
        
        # dimensionless coords
        x_root = x_root/C_r
        z_root = z_root/C_r
        x_tip = x_tip/C_t
        z_tip = z_tip/C_t
        span_percentage = y/half_span
        root_twist = 0
        tip_twist = np.deg2rad(self.twist)
        delta = np.deg2rad(self.sweep)
        gamma = np.deg2rad(self.dihedral)
        
        for j in range(ny):
                        
            C_y = interpolation(C_r, C_t, abs(span_percentage[j]))
            
            x_coords = interpolation(x_root, x_tip, abs(span_percentage[j]))
            z_coords = interpolation(z_root, z_tip, abs(span_percentage[j]))
            twist = interpolation(root_twist, tip_twist,
                                  abs(span_percentage[j]))
            
            x_coords = x_coords * C_y
            y_coords = span_percentage[j] * half_span * np.ones_like(x_coords)
            span_location = y_coords[0]
            z_coords = z_coords * C_y
                        
            x, y, z = self.move_airfoil_section(x_coords, y_coords, z_coords,
                                                span_location, C_y,
                                                twist, delta, gamma)
                        
            X[:, j] = np.linspace(x[0], x[0] + vec.x, num_x_shells+1)
            Y[:, j] = y[j]
            Z[:, j] = np.linspace(z[0], z[0]+ vec.z, num_x_shells+1)
        
        wake_nodes = []
        for i in range(nx):
            for j in range(ny):
                node = (X[i][j], Y[i][j], Z[i][j])
                wake_nodes.append(node)
                
                     
        wake_shells = []
                               
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
    
    def generate_wingMesh(self, num_x_bodyShells:int, num_y_Shells:int,
                          mesh_shell_type:str="quadrilateral",
                          return_WingTip=False):
        
        # body_nodes, body_shells = self.generate_bodyMesh2(num_x_bodyShells,
        #                                                  num_y_Shells,
        #                                                  mesh_shell_type)
        
        body_nodes, body_shells = self.generate_bodyMesh(num_x_bodyShells,
                                                         num_y_Shells,
                                                         mesh_shell_type)
        
        wingTip_nodes, wingTip_shells = self.generate_wingTipMesh(
                                                            num_x_bodyShells,
                                                            mesh_shell_type)
                
        for id, wingTip_shell in enumerate(wingTip_shells):
            for j in range(len(wingTip_shell)):
                wingTip_shells[id][j] = wingTip_shells[id][j] + len(body_nodes)
                
        nodes = [*body_nodes, *wingTip_nodes]
        shells = [*body_shells, *wingTip_shells]
        
        if return_WingTip:
            id_start_left = len(body_shells)
            id_end_left = id_start_left + int(len(wingTip_shells)/2) - 1
            id_start_right = id_end_left + 1
            id_end_right = id_start_right  + int(len(wingTip_shells)/2) - 1
            
            WingTip = {"left" : [id for id in range(id_start_left,
                                                   id_end_left+1)],
                       "right" : [id for id in range(id_start_right,
                                                     id_end_right + 1)] }
            
            return nodes, shells, WingTip
        
        return nodes, shells
    
    def generate_mesh(self, num_x_bodyShells:int,
                      num_x_wakeShells:int, num_y_Shells:int,
                      mesh_shell_type:str="quadrilateral",
                      return_id_dicts=False):
        
        wing_nodes, wing_shells, WingTip = self.generate_wingMesh(
            num_x_bodyShells, num_y_Shells, mesh_shell_type,return_WingTip=True)
        
        wake_nodes, wake_shells = self.generate_wakeMesh(num_x_wakeShells,
                                                         num_y_Shells,
                                                         mesh_shell_type)
               
        for i, wake_shell in enumerate(wake_shells):
            for j in range(len(wake_shell)):
                wake_shells[i][j] = wake_shells[i][j] + len(wing_nodes)
                                                         
                
        nodes = [*wing_nodes, *wake_nodes]
        shells = [*wing_shells, *wake_shells]
        
        
        if return_id_dicts:
            nodes_id = {"body": [id for id in range(len(wing_nodes))],
                        "wake": [id for id in
                                 range(len(wing_nodes), len(nodes))]}
        
            shells_id = {"body": [id for id in range(len(wing_shells))],
                         "wake": [id for id in
                                  range(len(wing_shells), len(shells))]}
            
            return nodes, shells, nodes_id, shells_id, WingTip
        
        return nodes, shells
           
    def generate_mesh2(self, V_fs:Vector, num_x_bodyShells,
                       num_x_wakeShells, num_y_Shells,
                       mesh_shell_type:str="quadrilateral",
                       return_id_dicts=False):
        
        """
        uses generate_wakeMesh2 method for wake meshing
        """
        
        wing_nodes, wing_shells, WingTip = self.generate_wingMesh(
            num_x_bodyShells, num_y_Shells, mesh_shell_type, return_WingTip=True)
        
        wake_nodes, wake_shells = self.generate_wakeMesh2(V_fs,
                                                          num_x_wakeShells,
                                                          num_y_Shells,
                                                          mesh_shell_type)
                
        for i, wake_shell in enumerate(wake_shells):
            for j in range(len(wake_shell)):
                wake_shells[i][j] = wake_shells[i][j] + len(wing_nodes)
                                                         
                
        nodes = [*wing_nodes,*wake_nodes]
        shells = [*wing_shells,*wake_shells]
        
        if return_id_dicts:
            nodes_id = {"body": [id for id in range(len(wing_nodes))],
                        "wake": [id for id in
                                 range(len(wing_nodes), len(nodes))]}
        
            shells_id = {"body": [id for id in range(len(wing_shells))],
                         "wake": [id for id in
                                  range(len(wing_shells), len(shells))]}
            
            return nodes, shells, nodes_id, shells_id, WingTip
        
        return nodes, shells
         
    def generate_mesh3(self, num_x_bodyShells,
                      num_x_wakeShells, num_y_Shells,
                      mesh_shell_type:str="quadrilateral"):
        
        """
        wing mesh without wingtip meshing
        """
        
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
      
    @staticmethod
    def give_TrailingEdge_Shells_id(num_x_bodyShells:int, num_y_Shells:int,
                                    mesh_shell_type:str="quadrilateral"):
        
        # x shells on Suction side + x shells on Pressure side
        num_x_bodyShells = 2 * num_x_bodyShells
        # ny shells on positive y axis + y shells on negative y axis
        num_y_Shells = 2 * num_y_Shells
        
        if mesh_shell_type == "quadrilateral":
            c1 = 0
            c2 = 1
             
        elif mesh_shell_type == "triangular":
            c1 = 1
            c2 = 2
        
        # num_WingTipShells = num_x_bodyShells * c2 - 4*c1
        num_WingTipShells = 2*num_x_bodyShells * c2 - 8*c1
        num_WingTipShells = 0
        
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
            c1 = 0
            c2 = 1
             
        elif mesh_shell_type == "triangular":
            c1 = 1
            c2 = 2
        
        num_y_Shells = len(TrailingEdge["suction side"])
        num_x_bodyShells = (TrailingEdge["pressure side"][-1] + 1)/num_y_Shells
        num_x_bodyShells = int(num_x_bodyShells/c2)
        num_WingTipShells = 2*num_x_bodyShells * c2 - 8*c1
        
        last_id = TrailingEdge["pressure side"][-1] + 1 + num_WingTipShells
        
        wake_sheddingShells_id = {}
        for j in range(num_y_Shells):
            id_list = []
            for i in range(num_x_wakeShells):
                for k in range(c2):
                    id = last_id + j + k 
                    id_list.append(id + c2*num_y_Shells*i)
            last_id = last_id + k
            
            wake_sheddingShells_id[TrailingEdge["suction side"][j]] = id_list
            wake_sheddingShells_id[TrailingEdge["pressure side"][j]] = id_list
        
        return wake_sheddingShells_id

    @staticmethod
    def sweep_airfoil_section(x_coords, span_location, sweep_angle):
        x_coords = x_coords + abs(span_location) * np.tan(sweep_angle)
        return x_coords
          
    @staticmethod
    def rotate(x_coords, y_coords, rotate_location, rotate_angle):

        x_c4 = rotate_location[0]
        y_c4 = rotate_location[1]
        angle = rotate_angle
        x = (
            (x_coords - x_c4) * np.cos(angle)
            + (y_coords - y_c4) * np.sin(angle)
            + x_c4 
        )
        
        y = (
            -(x_coords - x_c4) * np.sin(angle)
            + (y_coords - y_c4) * np.cos(angle)
            + y_c4                 
        )
        
        return x, y
    
    def twist_airfoil_section(self, x_coords, z_coords, chord, twist_angle):
        
        z_coords, x_coords = self.rotate(z_coords, x_coords,
                                        (0, 0.25*chord), twist_angle)
        
        return x_coords, z_coords
       
    def rotate_airfoil_section(self, y_coords, z_coords, span_location,
                               dihedral_angle):
        
        if span_location < 0:
            dihedral_angle = - dihedral_angle
        
        # y_coords, z_coords = self.rotate(y_coords, z_coords,
        #                                  (0, 0), dihedral_angle)
        
        # or xflr5's style to avoid  airfoil sections near root to intersect
        root_gamma = 0
        tip_gamma = dihedral_angle
        span_percentage = abs(span_location)/self.semi_span
        
        section_gamma = interpolation(root_gamma, tip_gamma, span_percentage)
                
        y_coords, z_coords = self.rotate(y_coords, z_coords,
                                         (span_location, 0), section_gamma)
        
        z_coords = z_coords - span_location * np.tan(tip_gamma)
        
        return y_coords, z_coords
           
    def move_airfoil_section(self, x_coords, y_coords, z_coords, span_location,
                             chord, twist_angle, sweep_angle, dihedral_angle):
        # angles in rads
                    
        x_coords, z_coords = self.twist_airfoil_section(x_coords,
                                                        z_coords,
                                                        chord, twist_angle)
        
        x_coords = self.sweep_airfoil_section(x_coords, span_location,
                                              sweep_angle)
        
        y_coords, z_coords = self.rotate_airfoil_section(y_coords, z_coords,
                                                         span_location, dihedral_angle)        
        return x_coords, y_coords, z_coords
    
      
if __name__=="__main__":
    from mesh_class import PanelMesh
      
    root_airfoil = Airfoil(name="naca0018", chord_length=1)
    tip_airfoil = Airfoil(name="naca0012", chord_length=0.8)
    wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0,
                twist=0)
    
    num_x_bodyShells = 10
    num_y_Shells = 10
        
    wing_nodes, wing_shells = wing.generate_wingMesh(num_x_bodyShells,
                                                     num_y_Shells,
                                                     "quadrilateral")    
    wing_mesh = PanelMesh(wing_nodes, wing_shells)
    # wing_mesh.plot_shells(elevation=-150, azimuth=-120)
    # wing_mesh.plot_panels(elevation=-150, azimuth=-120)  
    
    
    wing_mesh.set_body_fixed_frame_origin(0, 0, 0)
    roll, pitch, yaw = np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)
    wing_mesh.set_body_fixed_frame_orientation(roll, pitch, yaw)
    
    wing_mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120)
    wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120)
    
    omega = Vector((np.deg2rad(10), 0, 0))
    Vo = Vector((-1, 0, 0))
    V_wind = Vector((0, 0, 0))
    dt = 1
    wing_mesh.set_angular_velocity(omega)
    wing_mesh.set_origin_velocity(Vo)
    
    for i in range(5):
        
        wing_mesh.move_body(dt)
        wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120)
        
        wing_mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120)

    
    pass