from turtle import st
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Algorithms import light_vector
from plot_functions import set_axes_equal
import numpy as np
from vector_class import Vector
from panel_class import Panel, triPanel, quadPanel
from numba import typed
from copy import deepcopy


class Mesh:
    
    def __init__(self, nodes:list, shells:list,
                 ):
        self.nodes = nodes
        self.shells = shells
        self.node_num = len(nodes)
        self.shell_num = len(shells)

        self.shell_neighbours = self.find_shell_neighbours()
        
        
        
        ### unsteady features ###
        
        self.origin = (0, 0, 0)  # body-fixed frame origin
        
        # position vector of body-fixed frame origin
        self.ro = Vector(self.origin) 
        
        # velocity vector of body-fixed frame origin
        self.Vo = Vector((0, 0, 0))
        
        # orientation of body-fixed frame : (yaw-angle, pitch-angle, roll-angle)
        self.orientation = (0, 0, 0)
        
        self.theta = Vector(self.orientation)
        
        # Rotation Matrix
        self.R = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
                
        # body-fixed frame's angular velocity vector
        self.omega = Vector((0, 0, 0))

    @staticmethod
    def do_intersect(shell_i, shell_j):
        intersections = 0
        for node_id in shell_i:
            if node_id in shell_j:
                intersections = intersections + 1
        if intersections > 1 :
            return True
        else:
            return False
    
    def find_shell_neighbours(self):
        shells = self.shells
        neighbours=[]
        for i, shell_i in enumerate(shells):
            neighbours.append([])
            for j, shell_j in enumerate(shells):
                if i != j and self.do_intersect(shell_i, shell_j):
                    neighbours[-1].append(j)
        
        return neighbours

    
        
        ro = self.ro + self.Vo*dt
        theta = self.theta + self.omega*dt
        
        self.set_body_fixed_frame_origin(ro.x, ro.y, ro.z)
        self.set_body_fixed_frame_orientation(theta.x, theta.y, theta.z)

    def add_extra_neighbours(self):
         
        old_shell_neighbours = {}
        for id_i in range(self.shell_num):
            old_shell_neighbours[id_i] = self.shell_neighbours[id_i].copy()
            for id_j in self.shell_neighbours[id_i]:
                old_shell_neighbours[id_j] = self.shell_neighbours[id_j].copy()
        
        for id_i in range(self.shell_num):
            for id_j in old_shell_neighbours[id_i]:
                for id_k in old_shell_neighbours[id_j]: 
                    if id_k!=id_i and id_k not in self.shell_neighbours[id_i]:
                        self.shell_neighbours[id_i].append(id_k)
    
    def plot_shells(self, elevation=30, azimuth=-60):
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elevation, azimuth)
        for shell in self.shells:
            x, y, z = [], [], []
            for id in shell:
                [x_i, y_i, z_i] = self.nodes[id]
                x.append(x_i)
                y.append(y_i)
                z.append(z_i)
            
            id = shell[0]
            [x_i, y_i, z_i] = self.nodes[id]
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)    
            ax.plot3D(x, y, z, color='k')    
        
        set_axes_equal(ax)
        plt.show()
    
    
    ### unsteady features ###
    
    def set_body_fixed_frame_origin(self, xo, yo, zo):
        self.origin = (xo, yo, zo)
        self.ro = Vector(self.origin)
    
    def set_body_fixed_frame_orientation(self, roll, pitch, yaw):
        self.orientation = (roll, pitch, yaw)
        
        self.theta = Vector(self.orientation)
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        R = Rz @ Ry @ Rx
        
        self.R = R       
        
    def set_origin_velocity(self, Vo:Vector):
        self.Vo = Vo
    
    def set_angular_velocity(self, omega:Vector):
        self.omega = omega
    
    def move_body(self, dt):
        
        """
        We chose to work in the body-fixed frame of reference (and not in the inertial frame of reference). Any movement of the body in space isn't observed by someone that stands on the bodyfixed frame. Hence, every coordinate or position vector of the rigid body mesh doesn't change, unless it can move relative to the body-fixed frame (e.g. wake nodes, wake panels).
        
        This method change only the position vector of body-fixed frame's origin and the orintetation of the body-fixed frame.
        Note that origin's position vector and body-fixed frame's orientation are observed from the inertial frame of reference.
        """
        
        ro = self.ro + self.Vo*dt
        theta = self.theta + self.omega*dt
        
        self.set_body_fixed_frame_origin(ro.x, ro.y, ro.z)
        self.set_body_fixed_frame_orientation(theta.x, theta.y, theta.z) 
    
    def move_node(self, node_id, v_rel, dt):
        
        """
        v_rel = velocity relative to the inertial frame of reference. (e.g. v_rel = wind speed => v_rel + v_body = v_freestream)    
                
        Ιδιότητα Πρόσθεσης:
        ω_f"/f = ω_f"/f' + ω_f'/f
        
        1)
        f"=f  =>  ω_f"/f' = ω_f/f' και ω_f"/f = 0 
        => 0 = ω_f/f' + ω_f'/f
        => ω_f/f' = - ω_f'/f
        
        2)
        F:αδρανειακό συστημα αναφοράς, F:OXYZ
        f:μεταφερόμενο σύστημα αναφοράς, f:O'xyz
        f:προσδεδεμένο σύστημα αναφοράς στο στερεό, f':O'x'y'z'
        
        Από ιδιότητα πρόσθεσης: 
        ω_f'/F = ω_f'/f + ω_f/F
        (ω_f/F=0) => ω_f'/F = ω_f'/f
        
        από 1) ω_f/f' = - ω_f'/f => ω_f'/F = - ω_F/f'
        (ω_F/F = ω_F/f' + ω_f'/F => 0 = ω_F/f' + ω_f'/F => ω_f'/F = - ω_F/f')
        
        3)
        Μεταφορά παραγώγου διανύσματος από το σύστημα αναφοράς f' στο σύστημα ανφοράς f:
        (r_dot)f = (r_dot)f' + (ω)f'/f X r
        
        
        r = ro' + r' => r_dot = ro'_dot + r'_dot =>
        (r_dot)F = (ro'_dot)F + (r'_dot)F => ...
        (r'_dot)f' = (r_dot)F - [ (ro'_dot)F + (ω)f'/F X r']
        r':position vector of node measured from bodyfixed frame of reference f'
        r: position vector of node measured from inertial frame of reference F
        ω: angular velocity of body-fixed frame observed from inetial frame of reference F (or from frame of refernce f)
         
        (Δες και χειρόγραφες σημειώσεις για περισσότερες λεπτομέρειες)        
        """
        node = self.nodes[node_id]
        
        # r': position vector of node measured from bodyfixed frame of reference f'
        r = Vector(node)  
        
        # (r'_dot)f' 
        # r':position vector of node measured from bodyfixed frame of reference f'
        # v = v_rel - (self.Vo + Vector.cross_product(self.omega, r))
        v = v_rel - (self.Vo + self.omega.cross(r))
        
        
        dr = v*dt
        dr = dr.transformation(self.R.T)
        r = r+dr
        
        self.nodes[node_id] = (r.x, r.y, r.z)

    def move_nodes(self, node_id_list, v_rel, dt):
        
        for node_id in node_id_list:
            self.move_node(node_id, v_rel, dt)
    
    def convect_node(self, node_id, velocity:Vector, dt):
        
        node = self.nodes[node_id]
        r = Vector(node)
        dr = velocity * dt
        r = r + dr
        self.nodes[node_id] = (r.x, r.y, r.z)
    
    def convect_nodes(self, node_id_list, velocity_list, dt):
        for i in range(len(node_id_list)):
            node_id = node_id_list[i]
            velocity = velocity_list[i]
            self.convect_node(node_id, velocity, dt)
        
        
class AeroMesh(Mesh):
    
    def __init__(self, nodes: list, shells: list,
                 nodes_id:dict = {},
                 shells_id:dict = {},
                 TrailingEdge:dict={},
                 wake_sheddingShells:dict={},
                 WingTip:dict={}):
        super().__init__(nodes, shells)
        
        self.nodes_id = nodes_id
        self.shells_id = shells_id
        self.TrailingEdge = TrailingEdge
        self.wake_sheddingShells = wake_sheddingShells
        self.WingTip = WingTip
    
    def free_TrailingEdge(self):
        """
        if trailing edge shells (or panels) share trailing edge nodes then
        the method "find_shell_neighbours" will assume that suction side trailing shells and pressure side trailing shells are neighbours.
        In panel methods we assume that traling edge is a free edge.
        "free_TrailingEdge" method will remove false neighbour ids from the attribute shell_neighbour
        """
        
        for id in self.TrailingEdge["suction side"]:
            for neighbour_id in self.shell_neighbours[id]:
                if neighbour_id in self.TrailingEdge["pressure side"]:
                    self.shell_neighbours[id].remove(neighbour_id)
    
        for id in self.TrailingEdge["pressure side"]:
            for neighbour_id in self.shell_neighbours[id]:
                if neighbour_id in self.TrailingEdge["suction side"]:
                    self.shell_neighbours[id].remove(neighbour_id)

    def add_extra_neighbours(self):
         
        old_shell_neighbours = {}
        for id_i in self.shells_id["body"]:
            old_shell_neighbours[id_i] = self.shell_neighbours[id_i].copy()
            for id_j in self.shell_neighbours[id_i]:
                old_shell_neighbours[id_j] = self.shell_neighbours[id_j].copy()
        
        for id_i in self.shells_id["body"]:
            for id_j in old_shell_neighbours[id_i]:
                for id_k in old_shell_neighbours[id_j]: 
                    if id_k!=id_i and id_k not in self.shell_neighbours[id_i]:
                        self.shell_neighbours[id_i].append(id_k)      

    def give_near_root_shells_id(self):
        
        near_root_nodes_id = []
        for node_id, node in enumerate(self.nodes):
            if node[1] == 0:
                near_root_nodes_id.append(node_id)
        
        if self.shells_id:
            body_shells_id = self.shells_id["body"]
        else:
            body_shells_id = np.arange(len(self.shells))
        
        near_root_shells_id = []
        for shell_id in body_shells_id:
            for node_id in self.shells[shell_id]:
                if node_id in near_root_nodes_id:
                    near_root_shells_id.append(shell_id)
                    break
        
        return near_root_shells_id


    ### unsteady features  ###
    def initialize_wake_nodes(self):
        num_TrailingEdge_nodes = len(self.TrailingEdge["pressure side"]) + 1
        last_body_node_id = self.nodes_id["body"][-1]
        first_wake_node_id = last_body_node_id + 1
        
        for id in range(num_TrailingEdge_nodes):
                self.nodes.append(self.nodes[id])
                id = first_wake_node_id + id
                self.nodes_id["wake"].append(id)
    
    def add_wakeNodes(self):
        
        num_TrailingEdge_nodes = len(self.TrailingEdge["pressure side"]) + 1
        for id in range(num_TrailingEdge_nodes):
            self.nodes.append(self.nodes[id])
            id = self.nodes_id["wake"][-1] + 1
            self.nodes_id["wake"].append(id)
        
    def add_wakeShells(self, type="quadrilateral"):
        
        num_TrailingEdge_nodes = len(self.TrailingEdge["pressure side"]) + 1
        first_id = self.nodes_id["wake"][-1] - 2*num_TrailingEdge_nodes + 1 
        ny = num_TrailingEdge_nodes
        for i in range(ny-1):
            
            if type=="quadrilateral":
                shell = [i + first_id,
                        (i + 1) + first_id,
                        (i + 1 + ny) + first_id,
                        (i + ny) + first_id]
                
                self.shells.append(shell)
                
                if self.shells_id["wake"] == []:
                    id = self.shells_id["body"][-1] + 1
                else:
                    id = self.shells_id["wake"][-1] + 1
                
                # Προσοχή θα ανανεωθεί και το λεξικό self.panels_id
                self.shells_id["wake"].append(id)
                
                
                value_list_id = id
                
                key_id = self.TrailingEdge["pressure side"][i]
                self.wake_sheddingShells[key_id].append(value_list_id)
                
                key_id = self.TrailingEdge["suction side"][i]
                self.wake_sheddingShells[key_id].append(value_list_id)
                
            elif type=="triangular":
                shell = [i + first_id,
                        (i + 1) + first_id,
                        (i + 1 + ny) + first_id]
                
                self.shells.append(shell)
                
                shell = [i + first_id,
                        (i + 1 + ny) + first_id,
                        (i + ny) + first_id]
                
                self.shells.append(shell)
                
                if self.shells_id["wake"] == []:
                    id = self.shells_id["body"][-1] + 1
                else:
                    id = self.shells_id["wake"][-1] + 1
                
                # Προσοχή θα ανανεωθεί και το λεξικό self.panels_id
                self.shells_id["wake"].append(id)
                self.shells_id["wake"].append(id+1)
                
                
                value_list_id = id
                
                key_id = self.TrailingEdge["pressure side"][i]
                self.wake_sheddingShells[key_id].append(value_list_id)
                self.wake_sheddingShells[key_id].append(value_list_id + 1)
                
                key_id = self.TrailingEdge["suction side"][i]
                self.wake_sheddingShells[key_id].append(value_list_id)
                self.wake_sheddingShells[key_id].append(value_list_id+1)
    
    def shed_wake(self, v_rel, dt, wake_shed_factor=1, type="quadrilateral"):
                
        if self.nodes_id["wake"] == []:
            self.initialize_wake_nodes()
                      
        self.move_nodes(self.nodes_id["wake"], v_rel, dt*wake_shed_factor)    
        self.add_wakeNodes()
        self.add_wakeShells(type)

    def nodes_to_convect(self):
        num_TrailingEdge_nodes = len(self.TrailingEdge["suction side"]) + 1
        id_start = self.nodes_id["wake"][0]
        id_end = self.nodes_id["wake"][-1] - num_TrailingEdge_nodes
        
        node_id_list = []
        for id in range(id_start, id_end+1):
            node_id_list.append(id)
        
        return node_id_list
            
    def convect_wake(self, velocity_list, dt):
        node_id_list = self.nodes_to_convect()
        self.convect_nodes(node_id_list, velocity_list, dt)
        
                          
class PanelMesh(Mesh):
    def __init__(self, nodes:list, shells:list):
        super().__init__(nodes, shells)
        self.panels = None
        self.panels_num = None
        self.panel_neighbours = self.shell_neighbours
        # self.CreatePanels()
        self.CreatePanels_jit()         
        
    def CreatePanels(self):
        panels = []
        for shell_id, shell in enumerate(self.shells):
            vertex = []
            for node_id in shell:
                node = self.nodes[node_id]
                vertex.append(Vector(node))

            if len(vertex) == 3:
                panels.append(triPanel(vertex[0], vertex[1], vertex[2]))
            elif len(vertex) == 4:
                panels.append(quadPanel(vertex[0], vertex[1],
                                        vertex[2], vertex[3]))
            
            panels[-1].id = shell_id
                
        self.panels = panels
        self.panels_num = len(panels)

    def CreatePanels_jit(self):
        panels = []
        for shell_id, shell in enumerate(self.shells):
            vertex_list = []
            for node_id in shell:
                [x, y, z] = self.nodes[node_id]
                vertex_list.append((x, y, z))
            
            vertex_array = np.array(vertex_list)
            
            panels.append(Panel(vertex_array))
            
            panels[-1].id = shell_id
                
        self.panels = panels
        self.panels_num = len(panels)
    
    def give_neighbours(self, panel):
        
        neighbours_id_list = self.panel_neighbours[panel.id]
        
        # neighbours_list = []
        # for id in neighbours_id_list:
        #     neighbours_list.append(self.panels[id])
         
        neighbours_list = [self.panels[id] for id in neighbours_id_list]
        
        return neighbours_list
    
    def plot_panels(self, elevation=30, azimuth=-60):
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elevation, azimuth)
        
        for panel in self.panels:
            
            r_vertex = panel.r_vertex
            
            # plot panels
            if panel.num_vertices == 3:
                x = [r_vertex[0].x, r_vertex[1].x, r_vertex[2].x, r_vertex[0].x]
                y = [r_vertex[0].y, r_vertex[1].y, r_vertex[2].y, r_vertex[0].y]
                z = [r_vertex[0].z, r_vertex[1].z, r_vertex[2].z, r_vertex[0].z]
                ax.plot3D(x, y, z, color='k')
                
            elif panel.num_vertices == 4:
                
                x = [r_vertex[0].x, r_vertex[1].x, r_vertex[2].x, r_vertex[3].x,
                    r_vertex[0].x]
                y = [r_vertex[0].y, r_vertex[1].y, r_vertex[2].y, r_vertex[3].y,
                    r_vertex[0].y]
                z = [r_vertex[0].z, r_vertex[1].z, r_vertex[2].z, r_vertex[3].z,
                    r_vertex[0].z]
                ax.plot3D(x, y, z, color='k') 
                
            # plot normal vectors
            r_cp = panel.r_cp
            n = panel.n
            scale = 0.1
            n = n * scale
            ax.scatter(r_cp.x, r_cp.y, r_cp.z, color='k', s=5)
            ax.quiver(r_cp.x, r_cp.y, r_cp.z, n.x, n.y, n.z, color='r')
                
                
            
        set_axes_equal(ax)
        plt.show()

    def plot_mesh(self, elevation=30, azimuth=-60):
        shells = []
        vert_coords = []
        
        for panel in self.panels:
            shell=[]
            for r_vertex in panel.r_vertex:
                shell.append((r_vertex.x, r_vertex.y, r_vertex.z))
                vert_coords.append([r_vertex.x, r_vertex.y, r_vertex.z])
            shells.append(shell)
        
        
        light_vec = light_vector(magnitude=1, alpha=-45, beta=-45)
        face_normals = [panel.n for panel in self.panels]
        # dot_prods = [-light_vec * face_normal for face_normal in face_normals]
        dot_prods = [-light_vec.dot(face_normal) 
                     for face_normal in face_normals]
        min = np.min(dot_prods)
        max = np.max(dot_prods)
        target_min = 0.2 # darker gray
        target_max = 0.6 # lighter gray
        shading = [(dot_prod - min)/(max - min) *(target_max - target_min) 
                   + target_min
                   for dot_prod in dot_prods]
        facecolor = plt.cm.gray(shading)
        
        ax = plt.axes(projection='3d')
        poly3 = Poly3DCollection(shells, facecolor=facecolor)
        ax.add_collection(poly3)
        ax.view_init(elevation, azimuth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        vert_coords = np.array(vert_coords)
        x, y, z = vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2]
        ax.set_xlim3d(x.min(), x.max())
        ax.set_ylim3d(y.min(), y.max())
        ax.set_zlim3d(z.min(), z.max())
        set_axes_equal(ax)
        
        plt.show()
    
    
    ### unsteady features  ###
    def move_panel(self, panel_id, v_rel, dt):
        
        panel = self.panels[panel_id]
        panel.move(v_rel, self.Vo, self.omega, self.R, dt)
        self.panels[panel_id] = panel
    
    def move_panels(self, panel_id_list, v_rel, dt):
        
        for id in panel_id_list:
            self.move_panel(id, v_rel, dt)      
    
    def plot_mesh_inertial_frame(self, elevation=30, azimuth=-60):
        shells = []
        vert_coords = []
                
        for panel in self.panels:
            shell=[]
            for r_vertex in panel.r_vertex:
                r = self.ro + r_vertex.transformation(self.R)
                shell.append((r.x, r.y, r.z))
                vert_coords.append([r.x, r.y, r.z])
            shells.append(shell)
        
        
        light_vec = light_vector(magnitude=1, alpha=-45, beta=-45)
        face_normals = [panel.n.transformation(self.R) for panel in self.panels]
        # dot_prods = [-light_vec * face_normal for face_normal in face_normals]
        dot_prods = [-light_vec.dot(face_normal) 
                     for face_normal in face_normals]
        min = np.min(dot_prods)
        max = np.max(dot_prods)
        target_min = 0.2 # darker gray
        target_max = 0.6 # lighter gray
        shading = [(dot_prod - min)/(max - min) *(target_max - target_min) 
                   + target_min
                   for dot_prod in dot_prods]
        facecolor = plt.cm.gray(shading)
        
        ax = plt.axes(projection='3d')
        poly3 = Poly3DCollection(shells, facecolor=facecolor)
        ax.add_collection(poly3)
        ax.view_init(elevation, azimuth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        
        # Body-fixed frame of reference f'
        ex = Vector((1, 0, 0))  # ex'
        ey = Vector((0, 1, 0))  # ey'
        ez = Vector((0, 0, 1))  # ez'
        
        ex = ex.transformation(self.R)
        ey = ey.transformation(self.R)
        ez = ez.transformation(self.R)
        
        ro = self.ro
        ax.quiver(ro.x, ro.y, ro.z, ex.x, ex.y, ex.z, color='b', label="ex'")
        ax.quiver(ro.x, ro.y, ro.z, ey.x, ey.y, ey.z, color='g', label="ey'")
        ax.quiver(ro.x, ro.y, ro.z, ez.x, ez.y, ez.z, color='r', label="ez'")
        
        
        # Inertial frame of reference
        ex = Vector((1, 0, 0))  # eX
        ey = Vector((0, 1, 0))  # eY
        ez = Vector((0, 0, 1))  # eZ
        
        ax.quiver(0, 0, 0, ex.x, ex.y, ex.z, color='b', label='ex')
        ax.quiver(0, 0, 0, ey.x, ey.y, ey.z, color='g', label='ey')
        ax.quiver(0, 0, 0, ez.x, ez.y, ez.z, color='r', label='ez')
        
        vert_coords.append([ex.x, ex.y, ex.z])
        vert_coords.append([ey.x, ey.y, ey.z])
        vert_coords.append([ez.x, ez.y, ez.z])
       
       
        vert_coords = np.array(vert_coords)
        x, y, z = vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2]
        ax.set_xlim3d(x.min(), x.max())
        ax.set_ylim3d(y.min(), y.max())
        ax.set_zlim3d(z.min(), z.max())
        
        set_axes_equal(ax)
        
        plt.show()
    
    def plot_mesh_bodyfixed_frame(self, elevation=30, azimuth=-60):
        shells = []
        vert_coords = []
                
        for panel in self.panels:
            shell=[]
            for r_vertex in panel.r_vertex:
                shell.append((r_vertex.x, r_vertex.y, r_vertex.z))
                vert_coords.append([r_vertex.x, r_vertex.y, r_vertex.z])
            shells.append(shell)
        
        
        light_vec = light_vector(magnitude=1, alpha=-45, beta=-45)
        face_normals = [panel.n for panel in self.panels]
        # dot_prods = [-light_vec * face_normal for face_normal in face_normals]
        dot_prods = [-light_vec.dot(face_normal) 
                     for face_normal in face_normals]
        
        min = np.min(dot_prods)
        max = np.max(dot_prods)
        target_min = 0.2 # darker gray
        target_max = 0.6 # lighter gray
        shading = [(dot_prod - min)/(max - min) *(target_max - target_min) 
                   + target_min
                   for dot_prod in dot_prods]
        facecolor = plt.cm.gray(shading)
        
        ax = plt.axes(projection='3d')
        poly3 = Poly3DCollection(shells, facecolor=facecolor)
        ax.add_collection(poly3)
        ax.view_init(elevation, azimuth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        
        # Body-fixed frame of reference f'
        ex = Vector((1, 0, 0))  # ex'
        ey = Vector((0, 1, 0))  # ey'
        ez = Vector((0, 0, 1))  # ez'
        
        ax.quiver(0, 0, 0, ex.x, ex.y, ex.z, color='b', label="ex'")
        ax.quiver(0, 0, 0, ey.x, ey.y, ey.z, color='g', label="ey'")
        ax.quiver(0, 0, 0, ez.x, ez.y, ez.z, color='r', label="ez'")
        
        
        # Inertial frame of reference F
        ro = -self.ro  # ro: r_oo' -> r_o'o = -roo'  
        ex = Vector((1, 0, 0))  # eX
        ey = Vector((0, 1, 0))  # eY
        ez = Vector((0, 0, 1))  # eZ
        
        ro = ro.transformation(self.R.T)
        ex = ex.transformation(self.R.T)
        ey = ey.transformation(self.R.T)
        ez = ez.transformation(self.R.T)
        
        
        ax.quiver(ro.x, ro.y, ro.z, ex.x, ex.y, ex.z, color='b', label='ex')
        ax.quiver(ro.x, ro.y, ro.z, ey.x, ey.y, ey.z, color='g', label='ey')
        ax.quiver(ro.x, ro.y, ro.z, ez.x, ez.y, ez.z, color='r', label='ez')
        
        vert_coords.append([(ro+ex).x, (ro+ex).y, (ro+ex).z])
        vert_coords.append([(ro+ey).x, (ro+ey).y, (ro+ey).z])
        vert_coords.append([(ro+ez).x, (ro+ez).y, (ro+ez).z])
                    
        
        vert_coords = np.array(vert_coords)
        x, y, z = vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2]
        ax.set_xlim3d(x.min(), x.max())
        ax.set_ylim3d(y.min(), y.max())
        ax.set_zlim3d(z.min(), z.max())
        set_axes_equal(ax)
        
        plt.show()        
         

class PanelAeroMesh(AeroMesh, PanelMesh):
    
    def __init__(self, nodes: list, shells: list,
                 nodes_id: dict = {},
                 shells_id: dict = {},
                 TrailingEdge: dict = {},
                 wake_sheddingShells: dict = {},
                 WingTip: dict = {}):
        super().__init__(nodes, shells, nodes_id, shells_id, TrailingEdge, wake_sheddingShells, WingTip)
        
        self.panels_id = self.shells_id
        self.wake_sheddingPanels = self.wake_sheddingShells
    
    def free_TrailingEdge(self):
        super().free_TrailingEdge()
        self.panel_neighbours = self.shell_neighbours
      
    def give_near_root_panels(self):
        near_root_shells_id = super().give_near_root_shells_id()
        near_root_panels = []
        for id in near_root_shells_id:
            near_root_panels.append(self.panels[id])
        
        return near_root_panels
    
    def give_leftSide_near_root_panels(self):
        
        near_root_panels = self.give_near_root_panels()
        
        for panel in near_root_panels:
            if panel.r_cp.y < 0:
                near_root_panels.remove(panel)
        
        return near_root_panels

    
    ### unsteady features ###
    
    def add_wakePanels(self, type="quadrilateral"):

        num_TrailingEdge_panels = len(self.TrailingEdge["pressure side"])
        id_end = self.shells_id["wake"][-1]
        if type == "quadrilateral":
            id_start = id_end - num_TrailingEdge_panels + 1
        elif type == "triangular":
            id_start = id_end - 2*num_TrailingEdge_panels + 1
                
        for shell_id in range(id_start, id_end+1):
            vertex = []
            for node_id in self.shells[shell_id]:
                node = self.nodes[node_id]
                vertex.append(Vector(node))
            
            if len(vertex) == 3:
                self.panels.append(triPanel(vertex[0], vertex[1], vertex[2]))
            elif len(vertex) == 4:
                self.panels.append(quadPanel(vertex[0], vertex[1],
                                        vertex[2], vertex[3]))
            
            self.panels[-1].id = shell_id
    
    def add_wakePanels_jit(self, type="quadrilateral"):
        num_TrailingEdge_panels = len(self.TrailingEdge["pressure side"])
        id_end = self.shells_id["wake"][-1]
        if type == "quadrilateral":
            id_start = id_end - num_TrailingEdge_panels + 1
        elif type == "triangular":
            id_start = id_end - 2*num_TrailingEdge_panels + 1
                
        for shell_id in range(id_start, id_end+1):
            vertex_list = []
            for node_id in self.shells[shell_id]:
                node = self.nodes[node_id]
                vertex_list.append(node)
            
            vertex_array = np.array(vertex_list)
            
            self.panels.append(Panel(vertex_array))
            
            self.panels[-1].id = shell_id
        
    def shed_wake(self, v_rel, dt, wake_shed_factor=1, type="quadrilateral"):
        if self.panels_id["wake"] == []:
            super().shed_wake(v_rel, dt, wake_shed_factor, type)
            # self.add_wakePanels(type)
            self.add_wakePanels_jit(type)
        else:
            self.move_panels(self.panels_id["wake"], v_rel, dt*wake_shed_factor)
            super().shed_wake(v_rel, dt, wake_shed_factor, type)        
            # self.add_wakePanels(type)
            self.add_wakePanels_jit(type)
   
    def convect_wake(self, induced_velocity_function, dt):
        
        # create velocity list 
        node_id_list = self.nodes_to_convect()
        velocity_list = []
        body_panels = [self.panels[id] for id in self.panels_id["body"]]
        wake_panels = [self.panels[id] for id in self.panels_id["wake"]]
        for node_id in node_id_list:
            r_p = Vector(self.nodes[node_id])
            induced_velocity = induced_velocity_function(r_p, body_panels,
                                                         wake_panels)
            velocity_list.append(induced_velocity)
        
        # convect wake
        super().convect_wake(velocity_list, dt)
        
        # update panel vertices' location
        for shell_id in self.shells_id["wake"]:
            shell = self.shells[shell_id]          
            vertex_list = []
            for node_id in shell:
                node = self.nodes[node_id]
                vertex = Vector(node)
                vertex_list.append(vertex)
                
            vertex_list = typed.List(vertex_list)
            
            self.panels[shell_id].update_vertices_location(vertex_list)
           
    def plot_mesh_inertial_frame(self, elevation=30, azimuth=-60,
                                 plot_wake=False):
        body_shells = []
        wake_shells = []
        vert_coords = []
        
        if self.shells_id:
            body_panels = [self.panels[id] for id in self.shells_id["body"]]
            if plot_wake:
                wake_panels = [self.panels[id] for id in self.shells_id["wake"]]
        else:
            body_panels = self.panels
            
        for panel in body_panels:
            shell=[]
            for r_vertex in panel.r_vertex:
                r = self.ro + r_vertex.transformation(self.R)
                shell.append((r.x, r.y, r.z))
                vert_coords.append([r.x, r.y, r.z])
            body_shells.append(shell)
            
        if plot_wake:
            for panel in wake_panels:
                shell=[]
                for r_vertex in panel.r_vertex:
                    r = self.ro + r_vertex.transformation(self.R)
                    shell.append((r.x, r.y, r.z))
                    vert_coords.append([r.x, r.y, r.z])
                wake_shells.append(shell)
            
        
        
        light_vec = light_vector(magnitude=1, alpha=-45, beta=-45)
        face_normals = [panel.n.transformation(self.R) for panel in body_panels]
        # dot_prods = [-light_vec * face_normal for face_normal in face_normals]
        dot_prods = [-light_vec.dot(face_normal) 
                     for face_normal in face_normals]
        min = np.min(dot_prods)
        max = np.max(dot_prods)
        target_min = 0.2 # darker gray
        target_max = 0.6 # lighter gray
        shading = [(dot_prod - min)/(max - min) *(target_max - target_min) 
                   + target_min
                   for dot_prod in dot_prods]
        facecolor = plt.cm.gray(shading)
        
        ax = plt.axes(projection='3d')
        body_collection = Poly3DCollection(body_shells, facecolor=facecolor)
        ax.add_collection(body_collection)
        
        if plot_wake:
            wake_collection = Poly3DCollection(wake_shells, alpha=0.1)
            wake_collection.set_edgecolors('k')
            ax.add_collection(wake_collection)
            
        ax.view_init(elevation, azimuth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        
        
        # Body-fixed frame of reference f'
        ex = Vector((1, 0, 0))  # ex'
        ey = Vector((0, 1, 0))  # ey'
        ez = Vector((0, 0, 1))  # ez'
        
        ex = ex.transformation(self.R)
        ey = ey.transformation(self.R)
        ez = ez.transformation(self.R)
        
        ro = self.ro
        ax.quiver(ro.x, ro.y, ro.z, ex.x, ex.y, ex.z, color='b', label="ex'")
        ax.quiver(ro.x, ro.y, ro.z, ey.x, ey.y, ey.z, color='g', label="ey'")
        ax.quiver(ro.x, ro.y, ro.z, ez.x, ez.y, ez.z, color='r', label="ez'")
        
        
        # Inertial frame of reference
        ex = Vector((1, 0, 0))  # eX
        ey = Vector((0, 1, 0))  # eY
        ez = Vector((0, 0, 1))  # eZ
        
        ax.quiver(0, 0, 0, ex.x, ex.y, ex.z, color='b', label='ex')
        ax.quiver(0, 0, 0, ey.x, ey.y, ey.z, color='g', label='ey')
        ax.quiver(0, 0, 0, ez.x, ez.y, ez.z, color='r', label='ez')
        
        vert_coords.append([ex.x, ex.y, ex.z])
        vert_coords.append([ey.x, ey.y, ey.z])
        vert_coords.append([ez.x, ez.y, ez.z])
        
        vert_coords = np.array(vert_coords)
        x, y, z = vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2]
        ax.set_xlim3d(x.min(), x.max())
        ax.set_ylim3d(y.min(), y.max())
        ax.set_zlim3d(z.min(), z.max())
        
        set_axes_equal(ax)
        
        plt.show()
    
    def plot_mesh_bodyfixed_frame(self, elevation=30, azimuth=-60,
                                  plot_wake=False):
        body_shells = []
        wake_shells = []
        vert_coords = []
        
        if self.shells_id:
            body_panels = [self.panels[id] for id in self.shells_id["body"]]
            if plot_wake:
                wake_panels = [self.panels[id] for id in self.shells_id["wake"]]
        else:
            body_panels = self.panels
        
        for panel in body_panels:
            shell=[]
            for r_vertex in panel.r_vertex:
                shell.append((r_vertex.x, r_vertex.y, r_vertex.z))
                vert_coords.append([r_vertex.x, r_vertex.y, r_vertex.z])
            body_shells.append(shell)
        
        if plot_wake:     
            for panel in wake_panels:
                shell=[]
                for r_vertex in panel.r_vertex:
                    shell.append((r_vertex.x, r_vertex.y, r_vertex.z))
                    vert_coords.append([r_vertex.x, r_vertex.y, r_vertex.z])
                wake_shells.append(shell)
            
        
        
        
        light_vec = light_vector(magnitude=1, alpha=-45, beta=-45)
        face_normals = [panel.n for panel in body_panels]
        # dot_prods = [-light_vec * face_normal for face_normal in face_normals]
        dot_prods = [-light_vec.dot(face_normal) 
                     for face_normal in face_normals]
        min = np.min(dot_prods)
        max = np.max(dot_prods)
        target_min = 0.2 # darker gray
        target_max = 0.6 # lighter gray
        shading = [(dot_prod - min)/(max - min) *(target_max - target_min) 
                   + target_min
                   for dot_prod in dot_prods]
        facecolor = plt.cm.gray(shading)
        
        ax = plt.axes(projection='3d')
        body_collection = Poly3DCollection(body_shells, facecolor=facecolor)
        ax.add_collection(body_collection)
        
        if plot_wake:
            wake_collection = Poly3DCollection(wake_shells, alpha=0.1)
            wake_collection.set_edgecolors('k')
            ax.add_collection(wake_collection)
            
        ax.view_init(elevation, azimuth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        
        # Body-fixed frame of reference f'
        ex = Vector((1, 0, 0))  # ex'
        ey = Vector((0, 1, 0))  # ey'
        ez = Vector((0, 0, 1))  # ez'
        
        ax.quiver(0, 0, 0, ex.x, ex.y, ex.z, color='b', label="ex'")
        ax.quiver(0, 0, 0, ey.x, ey.y, ey.z, color='g', label="ey'")
        ax.quiver(0, 0, 0, ez.x, ez.y, ez.z, color='r', label="ez'")
        
        
        # Inertial frame of reference F
        ro = -self.ro  # ro: r_oo' -> r_o'o = -roo'  
        ex = Vector((1, 0, 0))  # eX
        ey = Vector((0, 1, 0))  # eY
        ez = Vector((0, 0, 1))  # eZ
        
        ro = ro.transformation(self.R.T)
        ex = ex.transformation(self.R.T)
        ey = ey.transformation(self.R.T)
        ez = ez.transformation(self.R.T)
        
        ax.quiver(ro.x, ro.y, ro.z, ex.x, ex.y, ex.z, color='b', label='ex')
        ax.quiver(ro.x, ro.y, ro.z, ey.x, ey.y, ey.z, color='g', label='ey')
        ax.quiver(ro.x, ro.y, ro.z, ez.x, ez.y, ez.z, color='r', label='ez')
        
        
        vert_coords.append([(ro+ex).x, (ro+ex).y, (ro+ex).z])
        vert_coords.append([(ro+ey).x, (ro+ey).y, (ro+ey).z])
        vert_coords.append([(ro+ez).x, (ro+ez).y, (ro+ez).z])
               
        
        vert_coords = np.array(vert_coords)
        x, y, z = vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2]
        ax.set_xlim3d(x.min(), x.max())
        ax.set_ylim3d(y.min(), y.max())
        ax.set_zlim3d(z.min(), z.max())
        set_axes_equal(ax)
        
        plt.show()        

    def copy(self):
        nodes = deepcopy(self.nodes)
        shells = deepcopy(self.shells)
        nodes_id = deepcopy(self.nodes_id)
        shells_id = deepcopy(self.shells_id)
        TrailingEdge = deepcopy(self.TrailingEdge)
        wake_sheddingShells = deepcopy(self.wake_sheddingShells)
        WingTip = deepcopy(self.WingTip)
        
        mesh = PanelAeroMesh(nodes, shells, nodes_id, shells_id, TrailingEdge,
                             wake_sheddingShells, WingTip)
        
        (xo, yo, zo) = self.origin
        mesh.set_body_fixed_frame_origin(xo, yo, zo)
        
        (roll, pitch, yaw) = self.orientation
        mesh.set_body_fixed_frame_orientation(roll, pitch, yaw)
        
        mesh.set_origin_velocity(self.Vo)
        mesh.set_angular_velocity(self.omega)
        
        return mesh


        
if __name__=='__main__':
    from matplotlib import pyplot as plt
    from sphere import sphere
    nodes, shells = sphere(1, 10, 10, mesh_shell_type='quadrilateral')
    sphere_mesh = PanelMesh(nodes, shells)    
    sphere_mesh.plot_panels()
    sphere_mesh.plot_mesh_bodyfixed_frame()
    
    