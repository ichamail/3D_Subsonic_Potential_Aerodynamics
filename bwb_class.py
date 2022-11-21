from airfoil_class import Airfoil
from vector_class import Vector
from Algorithms import DenserAtBoundaries, cosspace
import numpy as np


class WingCrossSection:
    
    def __init__(self, r_leadingEdge:Vector,
                 chord:float, twist:float, airfoil:Airfoil):
        
        self.r_leadingEdge = r_leadingEdge
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil
    
    def give_RotationMatrix(self, wing_cross_section_prev,
                            wing_cross_section_next):
        
        
        # local frame of reference before twist
        ex_local = Vector((1, 0, 0))
        
        if wing_cross_section_next == None:
            
            vec01 = (self.r_leadingEdge
                    - wing_cross_section_prev.r_leadingEdge)
            vec01_yz_project = Vector((0, vec01.y, vec01.z))
            ey_local = vec01_yz_project/vec01_yz_project.norm()
            
            z_scale = 1
            
        elif wing_cross_section_prev == None:
            
            vec12 = (wing_cross_section_next.r_leadingEdge
                    - self.r_leadingEdge)
            
            vec12_yz_project = Vector((0, vec12.y, vec12.z))
            ey_local = vec12_yz_project/vec12_yz_project.norm()
            
            z_scale = 1
            
        else:
            vec01 = (self.r_leadingEdge
                    - wing_cross_section_prev.r_leadingEdge)
            vec01_yz_project = Vector((0, vec01.y, vec01.z))
            vec_prev = vec01_yz_project/vec01_yz_project.norm()
            
            vec12 = (wing_cross_section_next.r_leadingEdge
                    - self.r_leadingEdge)
            
            vec12_yz_project = Vector((0, vec12.y, vec12.z))
            vec_next = vec12_yz_project/vec12_yz_project.norm()
            
            span_vec = (vec_prev+vec_next)/2
            
            ey_local = span_vec/span_vec.norm()
            
            z_scale = np.sqrt(2/(vec_prev.dot(vec_next) + 1))           
        
        ez_local = ex_local.cross(ey_local) * z_scale
        
        
        # local frame of reference after twist
        twist = np.deg2rad(self.twist)
        
        # first method
        
        # Rotation matrix from axis and angle (https://en.wikipedia.org/wiki/Rotation_matrix)
        
        
        c, s = np.cos(twist), np.sin(twist)
        ux, uy, uz = ey_local.x, ey_local.y, ey_local.z
        
        R_twist = [
            [c + ux ** 2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy ** 2 * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz ** 2 * (1 - c)]
            ]
        
        R_twist = np.array(R_twist)
        
        # Ά τρόπος
        # ex_local = ex_local.transformation(R_twist)
        # # ey_local = ey_local.transformation(R_twist) 
        # ez_local = ez_local.transformation(R_twist)
        
        # R_twisted = np.array([[ex_local.x, ey_local.x, ez_local.x],
        #                       [ex_local.y, ey_local.y, ez_local.y],
        #                       [ex_local.z, ey_local.z, ez_local.z]])
        
        #΄ Β τρόπος
        R_untwisted = np.array([[ex_local.x, ey_local.x, ez_local.x],
                              [ex_local.y, ey_local.y, ez_local.y],
                              [ex_local.z, ey_local.z, ez_local.z]])
        
        R_twisted = R_twist@R_untwisted
        
        
        # Rotate local frame of reference so z axis points downward
        theta = np.deg2rad(-90)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        
        RotationMatrix = R_twisted @ Rx        
                  
        return RotationMatrix

    def give_coords(self, x_percent, y_percent,
                    wing_cross_section_prev, wing_cross_section_next):
        
        r_le = self.r_leadingEdge
        R = self.give_RotationMatrix(wing_cross_section_prev,
                                     wing_cross_section_next)
        chord = self.chord
        r = Vector((x_percent*chord, y_percent*chord, 0))
        r_p = r_le + r.transformation(R)
        
        return (r_p.x, r_p.y, r_p.z)


class BWB:
    
    def __init__(self, name, wingXsection_list):
        self.name = name
        self.wingXsection_list = wingXsection_list 
    
    def compute_wingXsection_coord(self, id, x_perc, y_perc):
        wing_cross_section = self.wingXsection_list[id]
        
        if id == 0:
            wing_cross_section_prev = None
            wing_cross_section_next = self.wingXsection_list[id + 1]
            R = wing_cross_section.give_RotationMatrix(wing_cross_section_prev,
                                                       wing_cross_section_next)
            
        elif id == len(self.wingXsection_list)-1 or id==-1:
            wing_cross_section_prev = self.wingXsection_list[id - 1]
            wing_cross_section_next = None
            R = wing_cross_section.give_RotationMatrix(wing_cross_section_prev,
                                                       wing_cross_section_next)
        else:
            wing_cross_section_prev = self.wingXsection_list[id - 1]
            wing_cross_section_next = self.wingXsection_list[id + 1]
            R = wing_cross_section.give_RotationMatrix(wing_cross_section_prev,
                                                       wing_cross_section_next)
        
        r_le = wing_cross_section.r_leadingEdge
        chord = wing_cross_section.chord
        r = Vector((x_perc*chord, y_perc*chord, 0))
        r_p = r_le + r.transformation(R)
        print(R)
        return (r_p.x, r_p.y, r_p.z)
    
    def mesh_line(self, x_percent_list, y_percent_list, resolution,
                  spacing="uniform"):
        
        """
        meshes the line that connects the i-th node of every wing cross section
        
        returns line nodes = [[x0 y0 z0]
                              [x1 y1 z1]
                              [x2 y2 z2]]
        """
        if len(x_percent_list) != len(self.wingXsection_list):
            print("number of x-coords must \
                  be equal to number of wing cross sections ")

        if len(y_percent_list) != len(self.wingXsection_list):
            print("number of y-coords must \
                  be equal to number of wing cross sections ")
        
        if spacing == "uniform":
            space = np.linspace
        elif spacing == "cosine":
            space = cosspace
        elif spacing == "beta distribution":
            space = lambda start, end, steps: DenserAtBoundaries(start, end,
                                                                 steps, -0.15)
                    
        wingXsectionS_nodes = []
        
        for id, wingXsection in enumerate(self.wingXsection_list):
            
            x_perc = x_percent_list[id]
            y_perc = y_percent_list[id]

            if id == 0:
                wingXsection_prev=None
                wingXsection_next=self.wingXsection_list[id + 1]
            
            elif id == len(self.wingXsection_list)-1 or id==-1:
                wingXsection_prev = self.wingXsection_list[id - 1]
                wingXsection_next = None
            
            else:
                wingXsection_prev = self.wingXsection_list[id - 1]
                wingXsection_next=self.wingXsection_list[id + 1]
                
            
            wingXsection_node = wingXsection.give_coords(
                x_perc, y_perc, wingXsection_prev, wingXsection_next
            )
            
            wingXsectionS_nodes.append(wingXsection_node)
            
        wingSpanwiseSectionS_nodes = []
        for i in range(len(wingXsectionS_nodes) - 1):
            wingSpanwiseSection_nodes = np.stack(
                [ space(wingXsectionS_nodes[i][dim],
                        wingXsectionS_nodes[i+1][dim],
                        resolution) for dim in [0, 1, 2] ], axis=1 )
            
            # avoid last node of i-th spanwise section to be the first 
            # node of the i-th + 1 spanwise section
            if not i== len(wingXsectionS_nodes)-2:
                wingSpanwiseSection_nodes = wingSpanwiseSection_nodes[0:-1, :]
            
            wingSpanwiseSectionS_nodes.append(wingSpanwiseSection_nodes)
        
        line_nodes = np.concatenate(wingSpanwiseSectionS_nodes)
        
        return line_nodes

    def mesh_body(self, ChordWise_resolution, SpanWise_resolution,
                  SpanWise_spacing="uniform", shellType="quads",
                  mesh_main_surface=True, mesh_tips=True, standard_mesh=True):
        
        for wingXsection in self.wingXsection_list:
            wingXsection.airfoil.new_x_spacing2(ChordWise_resolution)
        
        x_perc = np.array([wingXsection.airfoil.x_coords
                           for wingXsection
                           in self.wingXsection_list]).T
        
        y_perc = np.array([wingXsection.airfoil.y_coords
                           for wingXsection
                           in self.wingXsection_list]).T
        
        x_perc = x_perc[0:-1] # removedouble node at trailing edge
        y_perc = y_perc[0:-1]
        
        nodes_ofSpanWiseLineS = []
        for line_x_perc, line_y_perc in zip(x_perc, y_perc):
            nodes_ofSpanWiseLine = self.mesh_line(line_x_perc, line_y_perc,
                                                  resolution = SpanWise_resolution + 1,
                                                  spacing=SpanWise_spacing)
            nodes_ofSpanWiseLineS.append(nodes_ofSpanWiseLine)
        
        nodes = np.concatenate(nodes_ofSpanWiseLineS)       
        
        def node_id(chord_wise_index, span_wise_index):
            ny = SpanWise_resolution * (len(self.wingXsection_list)-1) + 1
            nx = len(nodes_ofSpanWiseLineS)
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
                if shellType == "quads":
                    shells.append(list(node_ids))
                elif shellType == "trias":
                    index = node_ids
                    shells.append([index[0], index[1], index[2]])
                    shells.append([index[2], index[3], index[0]])
            
            elif len(node_ids) == 3:
                shells.append(list(node_ids))
            
        
        # call node_id() so i_max and j_max can be accessed
        node_id(0, 0)
        
        shells = []
        if mesh_main_surface:
            
            for i in range(i_max):
                for j in range(j_max):
                    add_shell(
                        node_id(i, j),
                        node_id(i+1, j),
                        node_id(i+1, j+1),
                        node_id(i, j+1)
                    )

                if i == 0:
                    suction_side_trailing_edge_shell_ids = [
                        id for id in range(len(shells))
                    ]
                if i == i_max-2:
                    id_start = len(shells)
            id_end = len(shells)-1
            pressure_side_trailing_edge_shell_ids=[
                id for id in range(id_start, id_end+1)
            ]
            main_surface_shell_ids = [id for id in range(0, id_end+1)]
                    
        # if mesh_tips:
            
        #     # root or right tip
        #     # trailing edge
        #     add_shell(
        #         node_id(0, 0),
        #         node_id(i_max - 1, 0),
        #         node_id(1, 0)
        #     )
        #     # leading edge
        #     add_shell(
        #         node_id(i_max//2 + 1, 0),
        #         node_id(i_max//2, 0),
        #         node_id(i_max//2 - 1, 0)   
        #     )
            
        #     # tip or left tip
        #     # trailing edge
        #     add_shell(
        #         node_id(0, j_max),
        #         node_id(1, j_max),
        #         node_id(i_max-1, j_max)    
        #     )
        #     # leading edge
        #     add_shell(
        #         node_id(i_max//2 - 1, j_max),
        #         node_id(i_max//2, j_max),
        #         node_id(i_max//2 + 1, j_max)   
        #     )
            
        #     for i in range(1, i_max//2-1):
                
        #         # root or right tip
        #         add_shell(
        #             node_id(i, 0),
        #             node_id(i_max-i, 0),
        #             node_id(i_max - i - 1, 0),
        #             node_id(i+1, 0)   
        #         )
                
        #         # tip or left tip
        #         add_shell(
        #             node_id(i, j_max),
        #             node_id(i+1, j_max),
        #             node_id(i_max - i - 1, j_max),
        #             node_id(i_max-i, j_max)    
        #         )
        
        extra_wing_tip_node_ids = []
        if mesh_tips:
            id = len(nodes) -1  # last node id
            
            for j in [0, j_max]:
                # j=0 root or right tip
                # j-j_max yip or left tip
                
                if j==0:
                    add_face = add_shell
                elif j==j_max:
                    add_face = lambda *node_ids: add_shell(*node_ids,
                                                            reverse_order=True)
                
                # root or right tip   
                id = id + 1
                extra_wing_tip_node_ids.append(id)
                   
                # trailing edge   
                i = 0      
                node = (nodes[node_id(i+1, j)]
                        + nodes[node_id(i_max-i-1, j)]) / 2
                nodes = np.vstack([nodes, node])
                
                add_face(
                    node_id(i, j),
                    node_id(i_max - i - 1, j),
                    id
                )
                
                add_face(
                    node_id(i, j),
                    id,
                    node_id(i+1, j),
                )
                
                for i in range(1, i_max//2 - 1):
                    id = id+1
                    extra_wing_tip_node_ids.append(id)
                    
                    node = (nodes[node_id(i+1, j)]
                            + nodes[node_id(i_max-i-1, j)]) / 2
                                
                    nodes = np.vstack([nodes, node])
                                
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

                if j == 0:
                    id_start = len(main_surface_shell_ids)
                    id_end = len(shells)-1
                    right_tip_shell_ids = [id for id in range(id_start,
                                                              id_end+1)]
                if j == j_max:
                    id_start = id_end + 1
                    id_end = len(shells)-1
                    left_tip_shell_ids = [id for id in range(id_start,
                                                             id_end+1)]
        
        # node and shell id dicts      
        main_surface_node_ids = [node_id(i, j)
                                 for i in range(i_max)
                                 for j in range(j_max+1)]

        body_node_ids = main_surface_node_ids + extra_wing_tip_node_ids
        
        right_tip_node_ids = [node_id(i, 0) for i in range(i_max)] \
            + extra_wing_tip_node_ids[0:len(extra_wing_tip_node_ids)//2]

        left_tip_node_ids = [node_id(i, j_max) for i in range(i_max)] \
            + extra_wing_tip_node_ids[len(extra_wing_tip_node_ids)//2:]
        
        wing_tips_node_ids = right_tip_node_ids + left_tip_node_ids
        
        trailing_edge_node_ids = [node_id(0, j) for j in range(j_max+1)]
        
        
        nodes_id_dict = {
            "body": body_node_ids,
            "main surface" : main_surface_node_ids,
            "wing tips": wing_tips_node_ids,
            "right wing tip": right_tip_node_ids,
            "left wing tip": left_tip_node_ids,
            "trailing edge": trailing_edge_node_ids
        }
        
        body_shell_ids = main_surface_shell_ids \
            + left_tip_shell_ids + right_tip_shell_ids
        
        shells_id_dict = {
            "body": body_shell_ids,
            "main surface": main_surface_shell_ids,
            "wing tips": left_tip_shell_ids + right_tip_shell_ids,
            "right tip": right_tip_shell_ids,
            "left tip": left_tip_shell_ids,
            "trailing edge": suction_side_trailing_edge_shell_ids \
                + pressure_side_trailing_edge_shell_ids,
            "suction side trailing edge": suction_side_trailing_edge_shell_ids,
            "pressure side trailing edge": pressure_side_trailing_edge_shell_ids
        }
        
        nodes = [(node[0], node[1], node[2]) for node in nodes]
        
        if standard_mesh:
            return nodes, shells
        else:
            return nodes, shells, nodes_id_dict, shells_id_dict


class Wing(BWB):
           
    def __init__(self,name:str, root_airfoil:Airfoil, tip_airfoil:Airfoil,
                 half_span:float, sweep_angle:float = 0,
                 dihedral_angle:float = 0, twist_angle:float = 0):
        
        
        self.half_span = half_span
        self.sweep_angle = sweep_angle
        self.dihedral_angle = dihedral_angle
        self.twist_angle = twist_angle
        self.root_wingXsection = None
        self.left_tip_wingXsection = None
        self.right_tip_wingXsection = None        
        super().__init__(name, [self.right_tip_wingXsection,
                                self.root_wingXsection,
                                self.left_tip_wingXsection])
        self.set_wingXsection_list(root_airfoil, tip_airfoil)
        
    def set_root_wingXsection(self, root_airfoil):
        chord = root_airfoil.chord
        x_coords = root_airfoil.x_coords/chord
        y_coords = root_airfoil.y_coords/chord
        name = root_airfoil.name
        airfoil = Airfoil(name, chord_length=1,
                          x_coords=x_coords, y_coords=y_coords)
        r_leadingEdge = Vector((0, 0, 0))
        self.root_wingXsection = WingCrossSection(r_leadingEdge = r_leadingEdge,
                                                  chord = chord,
                                                  twist=0,
                                                  airfoil = airfoil)
    
    def set_tip_wingXsections(self, tip_airfoil):
        chord = tip_airfoil.chord
        x_coords = tip_airfoil.x_coords
        y_coords = tip_airfoil.y_coords
        name = tip_airfoil.name
        airfoil = Airfoil(name, chord_length=1,
                          x_coords=x_coords, y_coords=y_coords)
        
        # left wing tip
        x_le, y_le, z_le = 0, self.half_span, 0
        x_le = self.sweep(x_le, self.half_span, np.deg2rad(self.sweep_angle))
        y_le, z_le = self.rotate(y_le, z_le, (0, 0),
                                 np.deg2rad(self.dihedral_angle))
        
        r_leadingEdge = Vector((x_le, y_le, z_le))
        self.left_tip_wingXsection = WingCrossSection(r_leadingEdge,
                                                      chord=chord,
                                                      twist = self.twist_angle,
                                                      airfoil=airfoil)
        
        # right wing tip
        y_le, z_le = -self.half_span, 0
        y_le, z_le = self.rotate(y_le, z_le, (0, 0),
                                 np.deg2rad(-self.dihedral_angle))
        
        r_leadingEdge = Vector((x_le, y_le, z_le))
        self.right_tip_wingXsection = WingCrossSection(r_leadingEdge,
                                                       chord=chord,
                                                       twist = self.twist_angle,
                                                       airfoil=airfoil)
    
    def set_wingXsection_list(self, root_airfoil, tip_airfoil):
        self.set_root_wingXsection(root_airfoil)
        self.set_tip_wingXsections(tip_airfoil)
        self.wingXsection_list = [self.right_tip_wingXsection,
                                  self.root_wingXsection,
                                  self.left_tip_wingXsection]
    
    @staticmethod
    def rotate(x, y, rotate_location, rotate_angle):
        
        # this fucntion rotates a point(x, y) about z-axis
        # to rotate a point (y, z) about x-axis:
        # y, z = rotate(y, z, (yo, zo), angle)
        # to rotate a point (x, z) about y-axis:
        # z, x = rotate(z, x, (zo, xo), angle)
        
        x_rot = rotate_location[0]
        y_rot = rotate_location[1]
        angle = rotate_angle
        x = (
            (x - x_rot) * np.cos(angle)
            + (y - y_rot) * np.sin(angle)
            + x_rot 
        )
        
        y = (
            -(x - x_rot) * np.sin(angle)
            + (y - y_rot) * np.cos(angle)
            + y_rot                 
        )
        
        return x, y   
    
    @staticmethod
    def sweep(x, span_location, sweep_angle):
        x = x + abs(span_location) * np.tan(sweep_angle)
        return x 
     

if __name__=="__main__":
    from mesh_class import PanelMesh
    
    
    bwb = BWB(name = "first try",
              wingXsection_list = [
                  WingCrossSection(
                  r_leadingEdge=Vector((0, -1, 0)),
                  chord=1,
                  twist=0,
                  airfoil=Airfoil(name="naca0018_new")),
                  
                  WingCrossSection(
                  r_leadingEdge=Vector((0, -0.5, 0)),
                  chord=1,
                  twist=0,
                  airfoil=Airfoil(name="naca0018_new")),
                  
                  WingCrossSection(
                  r_leadingEdge=Vector((0, 0.5, 0)),
                  chord=1,
                  twist=0,
                  airfoil=Airfoil(name="naca0018_new")),
                  
                  WingCrossSection(
                  r_leadingEdge=Vector((0, 1, 0)),
                  chord=1,
                  twist=0,
                  airfoil=Airfoil(name="naca0018_new"))
                                        ]
              )
    
    nodes, shells = bwb.mesh_body(5, 1)
    
    bwb_mesh = PanelMesh(nodes, shells)
    bwb_mesh.plot_mesh_inertial_frame(elevation=-150,azimuth=-120)
    
    wing = Wing(name="random",
                root_airfoil=Airfoil(name="naca0012_new", chord_length=1),
                tip_airfoil=Airfoil(name="naca0012_new", chord_length=1),
                half_span=1, sweep_angle=0, dihedral_angle=0, twist_angle=0)
    
    nodes, shells = wing.mesh_body(10, 10, SpanWise_spacing="beta distribution")
    wing_mesh = PanelMesh(nodes, shells)
    wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120)