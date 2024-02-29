import numpy as np
from vector_class import Vector
from Algorithms import DenserAtBoundaries, interpolation, cosspace, DenserAtWingTips, DenserAtWingRoot, logspace
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
        
    def new_x_spacing(self, num_x_points, spacing="cosine"):
        self.root_airfoil.repanel(num_x_points+1, spacing)
        self.tip_airfoil.repanel(num_x_points+1, spacing) 
        
    def generate_mesh(self, num_x_shells:int, num_y_shells:int,
                           mesh_shell_type:str="quadrilateral",
                           chord_wise_spacing = "cosine",
                           span_wise_spacing = "uniform",
                           mesh_main_surface=True, mesh_tips=True, mesh_wake=True,
                           num_x_wake_shells = 1,
                           wake_length_in_chords = 30,
                           triangular_wake_mesh = False,
                           standard_mesh_format =True):
        
        
        if span_wise_spacing == "uniform":
            space = np.linspace
        elif span_wise_spacing == "cosine":
            space = cosspace
        elif span_wise_spacing == "beta distribution":
            space = lambda start, end, steps: DenserAtBoundaries(start, end,
                                                                 steps, -0.15)
        elif span_wise_spacing == "denser at wingtips":
            space = lambda start, end, steps: DenserAtWingTips(start, end, steps, factor=5)
        elif span_wise_spacing == "denser at root":
            space = lambda start, end, steps: DenserAtWingRoot(start, end, steps, factor=5)
        elif span_wise_spacing == "logarithmic":
            space = logspace
                
        self.new_x_spacing(num_x_shells, spacing = chord_wise_spacing)
        
        # wing's coordinate system:
        # origin: root airfoils leading edge
        # x-axis: chord wise direction from leading edge to trailing edge
        # y-axis: spanwise direction form root to tip
        # z-axis: from top to bottom
                
        x_root = self.root_airfoil.x_coords 
        z_root = - self.root_airfoil.y_coords
        x_tip = self.tip_airfoil.x_coords
        z_tip = - self.tip_airfoil.y_coords
        
        # remove double node at leading edge
        x_root, z_root = x_root[0:-1], z_root[0:-1]
        x_tip, z_tip = x_tip[0:-1], z_tip[0:-1]
                
        y_right = space(-self.semi_span, 0, num_y_shells + 1)
        y_left = space(0, self.semi_span, num_y_shells + 1)
        
        y = np.array([*y_right, *y_left[1:]])
        
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
        
        def node_id(chord_wise_index, span_wise_index):
            
            global j_max, i_max
            j_max = ny-1
            i_max = nx  # i_max = nx-1 when double node at trailing edge
            i = chord_wise_index
            j = span_wise_index
            node_id = (j + i*ny)%(nx*ny)
            return node_id
         
        def add_shell(*node_ids, reverse_order=False):
            # node_id_list should be in counter clock wise order

            if reverse_order:
                node_ids = list(node_ids)
                node_ids.reverse()
                
            if len(node_ids) == 4:
                if mesh_shell_type=="quadrilateral":
                    shells.append(list(node_ids))
                elif mesh_shell_type=="triangular":
                    index = node_ids
                    shells.append([index[0], index[1], index[2]])
                    shells.append([index[2], index[3], index[0]])
            
            elif len(node_ids) == 3:
                shells.append(list(node_ids))
        
        node_id(0, 0)  # call node_id() to access i_max, and j_max
        shells = []
        
        if mesh_main_surface:
            
            if mesh_shell_type=="quadrilateral":
                                        
                # pressure and suction sides
                for i in range(i_max):
                    for j in range(j_max):
                        
                        add_shell(
                            node_id(i, j),
                            node_id(i+1, j),
                            node_id(i+1, j+1),
                            node_id(i, j+1)
                        )
                                
            elif mesh_shell_type=="triangular":
                
                # symmetrical meshing 
                for i in range(i_max):
                    
                    for j in range(j_max):
                        
                        # suction side
                        if i < i_max//2:
                            
                            # right side
                            if j < j_max//2:
                                add_shell(    
                                    node_id(i, j),
                                    node_id(i+1, j),
                                    node_id(i+1, j+1),
                                    node_id(i, j+1)
                                )
                            
                            # left side    
                            else:
                                add_shell(    
                                    node_id(i+1, j),
                                    node_id(i+1, j+1),
                                    node_id(i, j+1),
                                    node_id(i, j)
                                )

                        # suction side
                        else:
                            
                            # right side
                            if j < j_max//2:
                                add_shell(    
                                    node_id(i+1, j),
                                    node_id(i+1, j+1),
                                    node_id(i, j+1),
                                    node_id(i, j)
                                )
                            
                            # left side
                            else:
                                add_shell(    
                                    node_id(i, j),
                                    node_id(i+1, j),
                                    node_id(i+1, j+1),
                                    node_id(i, j+1)
                                )
        
        
        if mesh_tips:
            
            last_id = len(nodes) -1  # last node id
            id = last_id
            
            for j in [0, j_max]:
                # j=0 root or right tip
                # j-j_max yip or left tip
                
                if j==0:
                    add_face = add_shell
                elif j==j_max:
                    from collections import deque
                    def add_face(*node_ids):
                        node_ids = deque(node_ids)
                        node_ids.rotate(-1)
                        return add_shell(*node_ids, reverse_order=True)
                
                # root or right tip   
                id = id + 1
                   
                # trailing edge   
                i = 0
                     
                vec = (Vector(nodes[node_id(i+1, j)])
                        + Vector(nodes[node_id(i_max-i-1, j)])) / 2
                node = (vec.x, vec.y, vec.z)
                nodes.append(node)
                
                add_face(
                    node_id(i, j),
                    id,
                    node_id(i+1, j),
                )
                
                add_face(
                    node_id(i, j),
                    node_id(i_max - i - 1, j),
                    id
                )
                
                for i in range(1, i_max//2 - 1):
                    id = id+1
                    
                    vec = (Vector(nodes[node_id(i+1, j)])
                        + Vector(nodes[node_id(i_max-i-1, j)])) / 2
                    node = (vec.x, vec.y, vec.z)
                    nodes.append(node)
                                
                    add_face(
                        node_id(i, j),
                        id-1,
                        id,
                        node_id(i+1, j)   
                    )
                    
                    add_face(
                        id-1,
                        node_id(i_max-i, j),
                        node_id(i_max - i - 1, j),
                        id    
                    )
                    
                # leading edge
                i = i+1           
                add_face(
                    node_id(i, j),
                    id,
                    node_id(i+1, j)
                )
                
                add_face(
                    id,
                    node_id(i+2, j),  # node_id(i_max - i, j),
                    node_id(i+1, j),  # node_id(i_max - i - 1, j)     
                )
            
            extra_wing_tip_node_ids = [(last_id+1)+i for i in range(i_max-2)]
        
        
        if mesh_wake:
            
            x = np.zeros((num_x_wake_shells+1, j_max+1))
            y = np.zeros_like(x)
            z = np.zeros_like(x)
                        
            C_root  = self.root_airfoil.chord
            
            for j in range(j_max+1):
                
                vec1 = (
                    Vector(nodes[node_id(0, j)]) - Vector(nodes[node_id(1, j)])
                )
                                
                vec2 = (
                    Vector(nodes[node_id(i_max, j)]) 
                    - Vector(nodes[node_id(i_max - 1, j)])
                )
                                
                bisector = vec1+vec2
                
                bisector = bisector/bisector.norm()
                
                bisector = bisector * C_root * wake_length_in_chords
                
                (x0, y0, z0) = nodes[node_id(0, j)]
                x[:, j] = np.linspace(x0, x0 + bisector.x, num_x_wake_shells+1)
                y[:, j] = np.linspace(y0, y0 + bisector.y, num_x_wake_shells+1)
                z[:, j] = np.linspace(z0, z0 + bisector.z, num_x_wake_shells+1)
            
            
            num_body_nodes = len(nodes)
            for i in range(1, num_x_wake_shells+1):
                for j in range(j_max + 1):
                    node = (x[i][j], y[i][j], z[i][j])
                    nodes.append(node)
            
            def wake_node_id(chord_wise_index, span_wise_index):
                nx = num_x_wake_shells + 1

                global i_max_wake
                i_max_wake = nx - 1
                i = chord_wise_index
                j = span_wise_index
                
                if i==0:
                    id = node_id(i, j)
                
                else:
                                   
                    id = j + i*ny + (num_body_nodes - ny)
                
                return id
            
            # call wake_node_id() so i_max_wake can be accessed
            wake_node_id(0, 0)
            if triangular_wake_mesh == True : mesh_shell_type = "triangular"
            if mesh_shell_type=="quadrilateral":
                
                for i in range(i_max_wake):
                    for j in range(j_max):
                        add_shell(
                            wake_node_id(i, j),
                            wake_node_id(i, j+1),
                            wake_node_id(i+1, j+1),
                            wake_node_id(i+1, j)
                        )
                    
            elif mesh_shell_type=="triangular":
                
                for i in range(i_max_wake):
                    for j in range(j_max):
                        # right side
                        if j < j_max//2:
                            add_shell(
                                    wake_node_id(i, j),
                                    wake_node_id(i, j+1),
                                    wake_node_id(i+1, j+1),
                                    wake_node_id(i+1, j)
                            )
                        # left side
                        else:
                            add_shell(
                                    wake_node_id(i, j+1),
                                    wake_node_id(i+1, j+1),
                                    wake_node_id(i+1, j),
                                    wake_node_id(i, j)
                            )            
        
        if standard_mesh_format:
            
            return nodes, shells
        
        else:
            
            if mesh_main_surface:
                
                main_surface_nodes_ids = [
                    node_id(i, j) for i in range(i_max) for j in range(j_max+1)
                ]
                
                suction_side_nodes_ids = [
                    node_id(i, j) 
                    for i in range(i_max//2 + 1) for j in range(j_max + 1)
                ]
                
                pressure_side_nodes_ids = [
                    node_id(i, j) 
                    for i in range(i_max//2, i_max + 1)
                    for j in range(j_max + 1)
                ]
                
                trailing_edge_nodes_ids = [
                    node_id(0, j) for j in range(j_max+1)
                ]
            else:
                main_surface_nodes_ids = []
                suction_side_nodes_ids = []
                pressure_side_nodes_ids = []
                trailing_edge_nodes_ids = []
            
            
            if mesh_tips:
                right_tip_nodes_ids = [node_id(i, 0) for i in range(i_max)] \
                + extra_wing_tip_node_ids[0:len(extra_wing_tip_node_ids)//2]

                left_tip_nodes_ids = [node_id(i, j_max) for i in range(i_max)] \
                    + extra_wing_tip_node_ids[len(extra_wing_tip_node_ids)//2:]
                    
                wing_tips_nodes_ids = right_tip_nodes_ids + left_tip_nodes_ids 
            
            else:
                extra_wing_tip_node_ids = []
                right_tip_nodes_ids = []
                left_tip_nodes_ids = []
                wing_tips_nodes_ids = []   
                            
            
            if mesh_wake:
                wake_nodes_ids = [
                    wake_node_id(i, j) 
                    for i in range(i_max_wake + 1) for j in range(j_max + 1)
                ]
                
                wake_lines = [
                    [wake_node_id(i, j) for i in range(i_max_wake + 1)]
                    for j in range(j_max + 1)
                ]
                
                wake_lines = np.array(wake_lines)
            
            else:
                wake_nodes_ids = []
                wake_lines = []
            
            
            body_nodes_ids = main_surface_nodes_ids + extra_wing_tip_node_ids
            
            nodes_ids_dict = {
            "body": body_nodes_ids,
            "main surface" : main_surface_nodes_ids,
            "suction side": suction_side_nodes_ids,
            "pressure side": pressure_side_nodes_ids,
            "wing tips": wing_tips_nodes_ids,
            "right wing tip": right_tip_nodes_ids,
            "left wing tip": left_tip_nodes_ids,
            "trailing edge": trailing_edge_nodes_ids,
            "wake": wake_nodes_ids,
            "wake lines": wake_lines
        }
                                                     
            return nodes, shells, nodes_ids_dict
    
    def generate_mesh2(self, num_x_shells:int, num_y_shells:int,
                           mesh_shell_type:str="quadrilateral",
                           chord_wise_spacing="cosine",
                           span_wise_spacing='uniform',
                           mesh_main_surface=True, mesh_tips=True, mesh_wake=True,
                           triangular_wake_mesh = False,
                           num_x_wake_shells = 1,
                           V_fs = Vector((1, 0, 0)),
                           wake_length_in_chords = 30,
                           standard_mesh_format =True):
        
        """
        generate_mesh (if chosen to) generates a steady wake in the same direction as the bisector vector of trailing edge
        
        generate_mesh2 (if chosen to) generates a plane steady wake in free stream's vector direction
        """
                
        if span_wise_spacing == "uniform":
            space = np.linspace
        elif span_wise_spacing == "cosine":
            space = cosspace
        elif span_wise_spacing == "beta distribution":
            space = lambda start, end, steps: DenserAtBoundaries(start, end,
                                                                 steps, -0.15)
        elif span_wise_spacing == "denser at wingtips":
            space = lambda start, end, steps: DenserAtWingTips(start, end, steps, factor=5)
        elif span_wise_spacing == "denser at root":
            space = lambda start, end, steps: DenserAtWingRoot(start, end, steps, factor=5)
        elif span_wise_spacing == "logarithmic":
            space = logspace
              
        self.new_x_spacing(num_x_shells, spacing=chord_wise_spacing)
        
        # wing's coordinate system:
        # origin: root airfoils leading edge
        # x-axis: chord wise direction from leading edge to trailing edge
        # y-axis: spanwise direction form root to tip
        # z-axis: from top to bottom
                
        x_root = self.root_airfoil.x_coords 
        z_root = - self.root_airfoil.y_coords
        x_tip = self.tip_airfoil.x_coords
        z_tip = - self.tip_airfoil.y_coords
        
        # remove double node at leading edge
        x_root, z_root = x_root[0:-1], z_root[0:-1]
        x_tip, z_tip = x_tip[0:-1], z_tip[0:-1]
                
        y_right = space(-self.semi_span, 0, num_y_shells + 1)
        y_left = space(0, self.semi_span, num_y_shells + 1)
        
        y = np.array([*y_right, *y_left[1:]])
        
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
        
        def node_id(chord_wise_index, span_wise_index):
            
            global j_max, i_max
            j_max = ny-1
            i_max = nx  # i_max = nx-1 when double node at trailing edge
            i = chord_wise_index
            j = span_wise_index
            node_id = (j + i*ny)%(nx*ny)
            return node_id
         
        def add_shell(*node_ids, reverse_order=False):
            # node_id_list should be in counter clock wise order

            if reverse_order:
                node_ids = list(node_ids)
                node_ids.reverse()
                
            if len(node_ids) == 4:
                if mesh_shell_type=="quadrilateral":
                    shells.append(list(node_ids))
                elif mesh_shell_type=="triangular":
                    index = node_ids
                    shells.append([index[0], index[1], index[2]])
                    shells.append([index[2], index[3], index[0]])
            
            elif len(node_ids) == 3:
                shells.append(list(node_ids))
        
        node_id(0, 0)  # call node_id() to access i_max, and j_max
        shells = []
        
        if mesh_main_surface:
            
            if mesh_shell_type=="quadrilateral":
                                        
                # pressure and suction sides
                for i in range(i_max):
                    for j in range(j_max):
                        
                        add_shell(
                            node_id(i, j),
                            node_id(i+1, j),
                            node_id(i+1, j+1),
                            node_id(i, j+1)
                        )
                                
            elif mesh_shell_type=="triangular":
                
                # symmetrical meshing 
                for i in range(i_max):
                    
                    for j in range(j_max):
                        
                        # suction side
                        if i < i_max//2:
                            
                            # right side
                            if j < j_max//2:
                                add_shell(    
                                    node_id(i, j),
                                    node_id(i+1, j),
                                    node_id(i+1, j+1),
                                    node_id(i, j+1)
                                )
                            
                            # left side    
                            else:
                                add_shell(    
                                    node_id(i+1, j),
                                    node_id(i+1, j+1),
                                    node_id(i, j+1),
                                    node_id(i, j)
                                )

                        # suction side
                        else:
                            
                            # right side
                            if j < j_max//2:
                                add_shell(    
                                    node_id(i+1, j),
                                    node_id(i+1, j+1),
                                    node_id(i, j+1),
                                    node_id(i, j)
                                )
                            
                            # left side
                            else:
                                add_shell(    
                                    node_id(i, j),
                                    node_id(i+1, j),
                                    node_id(i+1, j+1),
                                    node_id(i, j+1)
                                )
        
        
        if mesh_tips:
            
            last_id = len(nodes) -1  # last node id
            id = last_id
            
            for j in [0, j_max]:
                # j=0 root or right tip
                # j-j_max yip or left tip
                
                if j==0:
                    add_face = add_shell
                elif j==j_max:
                    from collections import deque
                    def add_face(*node_ids):
                        node_ids = deque(node_ids)
                        node_ids.rotate(-1)
                        return add_shell(*node_ids, reverse_order=True)
                
                # root or right tip   
                id = id + 1
                   
                # trailing edge   
                i = 0
                     
                vec = (Vector(nodes[node_id(i+1, j)])
                        + Vector(nodes[node_id(i_max-i-1, j)])) / 2
                node = (vec.x, vec.y, vec.z)
                nodes.append(node)
                
                add_face(
                    node_id(i, j),
                    id,
                    node_id(i+1, j),
                )
                
                add_face(
                    node_id(i, j),
                    node_id(i_max - i - 1, j),
                    id
                )
                
                for i in range(1, i_max//2 - 1):
                    id = id+1
                    
                    vec = (Vector(nodes[node_id(i+1, j)])
                        + Vector(nodes[node_id(i_max-i-1, j)])) / 2
                    node = (vec.x, vec.y, vec.z)
                    nodes.append(node)
                                
                    add_face(
                        node_id(i, j),
                        id-1,
                        id,
                        node_id(i+1, j)   
                    )
                    
                    add_face(
                        id-1,
                        node_id(i_max-i, j),
                        node_id(i_max - i - 1, j),
                        id    
                    )
                    
                # leading edge
                i = i+1           
                add_face(
                    node_id(i, j),
                    id,
                    node_id(i+1, j)
                )
                
                add_face(
                    id,
                    node_id(i+2, j),  # node_id(i_max - i, j),
                    node_id(i+1, j),  # node_id(i_max - i - 1, j)     
                )
            
            extra_wing_tip_node_ids = [(last_id+1)+i for i in range(i_max-2)]
        
        
        if mesh_wake:
            
            wake_direction_unit_vec = V_fs/V_fs.norm()
            vec = wake_direction_unit_vec * wake_length_in_chords * self.root_airfoil.chord 
            
            x = np.zeros((num_x_wake_shells+1, j_max+1))
            y = np.zeros_like(x)
            z = np.zeros_like(x)
            
            for j in range(j_max+1):
                                
                (x0, y0, z0) = nodes[node_id(0, j)]
                x[:, j] = np.linspace(x0, x0 + vec.x, num_x_wake_shells+1)
                y[:, j] = np.linspace(y0, y0 + vec.y, num_x_wake_shells+1)
                z[:, j] = np.linspace(z0, z0 + vec.z, num_x_wake_shells+1)
            
            
            num_body_nodes = len(nodes)
            for i in range(1, num_x_wake_shells+1):
                for j in range(j_max + 1):
                    node = (x[i][j], y[i][j], z[i][j])
                    nodes.append(node)
            
            def wake_node_id(chord_wise_index, span_wise_index):
                nx = num_x_wake_shells + 1

                global i_max_wake
                i_max_wake = nx - 1
                i = chord_wise_index
                j = span_wise_index
                
                if i==0:
                    id = node_id(i, j)
                
                else:
                                   
                    id = j + i*ny + (num_body_nodes - ny)
                
                return id
            
            # call wake_node_id() so i_max_wake can be accessed
            wake_node_id(0, 0)
            if triangular_wake_mesh == True : mesh_shell_type = "triangular"
            if mesh_shell_type=="quadrilateral":
                
                for i in range(i_max_wake):
                    for j in range(j_max):
                        
                        add_shell(
                            wake_node_id(i, j),
                            wake_node_id(i, j+1),
                            wake_node_id(i+1, j+1),
                            wake_node_id(i+1, j)
                        )
                    
            elif mesh_shell_type=="triangular":
                
                for i in range(i_max_wake):
                    for j in range(j_max):
                        
                        # right side
                        if j < j_max//2:
                            add_shell(
                                    wake_node_id(i, j),
                                    wake_node_id(i, j+1),
                                    wake_node_id(i+1, j+1),
                                    wake_node_id(i+1, j)
                            )
                            
                        # left side
                        else:
                            add_shell(
                                    wake_node_id(i, j+1),
                                    wake_node_id(i+1, j+1),
                                    wake_node_id(i+1, j),
                                    wake_node_id(i, j)
                            )            
        
        
        if standard_mesh_format:
            
            return nodes, shells
        
        else:
            
            if mesh_main_surface:
                
                main_surface_nodes_ids = [
                    node_id(i, j) for i in range(i_max) for j in range(j_max+1)
                ]
                
                suction_side_nodes_ids = [
                    node_id(i, j) 
                    for i in range(i_max//2 + 1) for j in range(j_max + 1)
                ]
                
                pressure_side_nodes_ids = [
                    node_id(i, j) 
                    for i in range(i_max//2, i_max + 1)
                    for j in range(j_max + 1)
                ]
                
                trailing_edge_nodes_ids = [
                    node_id(0, j) for j in range(j_max+1)
                ]
            else:
                main_surface_nodes_ids = []
                suction_side_nodes_ids = []
                pressure_side_nodes_ids = []
                trailing_edge_nodes_ids = []
            
            
            if mesh_tips:
                right_tip_nodes_ids = [node_id(i, 0) for i in range(i_max)] \
                + extra_wing_tip_node_ids[0:len(extra_wing_tip_node_ids)//2]

                left_tip_nodes_ids = [node_id(i, j_max) for i in range(i_max)] \
                    + extra_wing_tip_node_ids[len(extra_wing_tip_node_ids)//2:]
                    
                wing_tips_nodes_ids = right_tip_nodes_ids + left_tip_nodes_ids 
            
            else:
                extra_wing_tip_node_ids = []
                right_tip_nodes_ids = []
                left_tip_nodes_ids = []
                wing_tips_nodes_ids = []   
                            
            
            if mesh_wake:
                wake_nodes_ids = [
                    wake_node_id(i, j) 
                    for i in range(i_max_wake + 1) for j in range(j_max + 1)
                ]
                
                wake_lines = [
                    [wake_node_id(i, j) for i in range(i_max_wake + 1)]
                    for j in range(j_max + 1)
                ]
                
                wake_lines = np.array(wake_lines)
            
            else:
                wake_nodes_ids = []
                wake_lines = []
            
            
            body_nodes_ids = main_surface_nodes_ids + extra_wing_tip_node_ids
            
            nodes_ids_dict = {
            "body": body_nodes_ids,
            "main surface" : main_surface_nodes_ids,
            "suction side": suction_side_nodes_ids,
            "pressure side": pressure_side_nodes_ids,
            "wing tips": wing_tips_nodes_ids,
            "right wing tip": right_tip_nodes_ids,
            "left wing tip": left_tip_nodes_ids,
            "trailing edge": trailing_edge_nodes_ids,
            "wake": wake_nodes_ids,
            "wake lines": wake_lines
        }
                                                     
            return nodes, shells, nodes_ids_dict
        
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
    from mesh_class import PanelMesh, PanelAeroMesh
     
    root_airfoil = Airfoil(name="naca0012 sharp", chord_length=1)
    tip_airfoil = Airfoil(name="naca0012 sharp", chord_length=1)
    wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0,
                twist=0)
    
    nodes, shells, nodes_ids = wing.generate_mesh(
        num_x_shells=30, num_y_shells=17, mesh_shell_type="quadrilateral",
        mesh_main_surface=True, mesh_tips=True, mesh_wake=False, wake_length_in_chords=30, chord_wise_spacing="cosine", span_wise_spacing="denser at wingtips",
        standard_mesh_format=False
    )
    
    wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)
    wing_mesh.set_body_fixed_frame_origin(1, 1.5, -1)
    wing_mesh.set_body_fixed_frame_orientation(roll=np.deg2rad(20), pitch=np.deg2rad(20), yaw=0)
    
    
    # wing_mesh.set_body_fixed_frame_origin(1, 0, -1)
    wing_mesh.plot_mesh_inertial_frame(elevation=-90, azimuth=180)
    wing_mesh.plot_mesh_inertial_frame(elevation=-151, azimuth=-56, plot_wake=True)