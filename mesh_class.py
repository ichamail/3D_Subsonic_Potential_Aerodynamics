from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Algorithms import light_vector
from plot_functions import set_axes_equal
import numpy as np
from vector_class import Vector
from panel_class import Panel, triPanel, quadPanel


class Mesh:
    
    def __init__(self, nodes:list, shells:list,
                 shells_id:dict = {},
                 TrailingEdge:dict={},
                 wake_sheddingShells:dict={},
                 WingTip:dict={}):
        self.nodes = nodes
        self.shells = shells
        self.node_num = len(nodes)
        self.shell_num = len(shells)

        self.shell_neighbours = self.find_shell_neighbours()
        
        self.shells_id = shells_id
        self.TrailingEdge = TrailingEdge
        self.wake_sheddingShells = wake_sheddingShells
        self.WingTip = WingTip

        
        ### unsteady features ###
        
        self.origin = (0, 0, 0)  # body-fixed frame origin
        
        # position vector of body-fixed frame origin
        self.ro = Vector(self.origin) 
        
        # velocity vector of body-fixed frame origin
        self.Vo = Vector((0, 0, 0))
        
        # orientation of body-fixed frame : (yaw-angle, pitch-angle, roll-angle)
        self.orientation = (0, 0, 0)
        
        # Rotation Matrix
        self.R = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
                
        # body-fixed frame's angular velocity vector
        self.omega = Vector((0, 0, 0))              
        
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
    
    # unsteady features
    def set_body_fixed_frame_origin(self, xo, yo, zo):
        self.origin = (xo, yo, zo)
        self.ro = Vector(self.origin)
    
    def set_body_fixed_frame_orientation(self, yaw, pitch, roll):
        self.orientation = (yaw, pitch, roll)
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rx = np.array([[1, 0, 1],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        R = Rz @ Ry @ Rx
        
        self.R = R       
        
    def set_origin_velocity(self, Vx, Vy, Vz):
        self.Vo = Vector((Vx, Vy, Vz))
    
    def set_angular_velocity(self, omega_x, omega_y, omega_z):
        self.omega = Vector((omega_x, omega_y, omega_z))
    
                
class PanelMesh(Mesh):
    def __init__(self, nodes:list, shells:list,
                 shells_id:dict = {},
                 TrailingEdge:dict={},
                 wake_sheddingShells:dict={},
                 WingTip:dict={}):
        super().__init__(nodes, shells,
                         shells_id, TrailingEdge, wake_sheddingShells, WingTip)
        self.panels = None
        self.panels_num = None
        self.panel_neighbours = self.shell_neighbours
        self.panels_id = self.shells_id
        self.wake_sheddingPanels = self.wake_sheddingShells
        self.CreatePanels()
    
    def CreatePanels(self):
        panels = []
        for shell_id, shell in enumerate(self.shells):
            vertex = []
            for node_id in shell:
                [x, y, z] = self.nodes[node_id]
                vertex.append(Vector((x, y, z)))

            if len(vertex) == 3:
                panels.append(triPanel(vertex[0], vertex[1], vertex[2]))
            elif len(vertex) == 4:
                panels.append(quadPanel(vertex[0], vertex[1],
                                        vertex[2], vertex[3]))
            
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
    
    def free_TrailingEdge(self):
        super().free_TrailingEdge()
        self.panel_neighbours = self.shell_neighbours
    
    def WingTip_add_extra_neighbours(self):
        super().WingTip_add_extra_neighbours()
        self.panel_neighbours = self.shell_neighbours
      
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
            n = n.scalar_product(scale)
            ax.scatter(r_cp.x, r_cp.y, r_cp.z, color='k', s=5)
            ax.quiver(r_cp.x, r_cp.y, r_cp.z, n.x, n.y, n.z, color='r')
                
                
            
        set_axes_equal(ax)
        plt.show()

    def plot_mesh(self, elevation=30, azimuth=-60):
        shells = []
        vert_coords = []
        
        if self.shells_id:
            body_panels = [self.panels[id] for id in self.shells_id["body"]]
        else:
            body_panels = self.panels
        
        for panel in body_panels:
            shell=[]
            for r_vertex in panel.r_vertex:
                shell.append((r_vertex.x, r_vertex.y, r_vertex.z))
                vert_coords.append([r_vertex.x, r_vertex.y, r_vertex.z])
            shells.append(shell)
        
        
        light_vec = light_vector(magnitude=1, alpha=-45, beta=-45)
        face_normals = [panel.n for panel in body_panels]
        dot_prods = [-light_vec * face_normal for face_normal in face_normals]
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


    # unsteady features
    
    def plot_mesh_inertial_frame(self, elevation=30, azimuth=-60):
        shells = []
        vert_coords = []
        
        if self.shells_id:
            body_panels = [self.panels[id] for id in self.shells_id["body"]]
        else:
            body_panels = self.panels
        
        for panel in body_panels:
            shell=[]
            for r_vertex in panel.r_vertex:
                r = self.ro + r_vertex.transformation(self.R)
                shell.append((r.x, r.y, r.z))
                vert_coords.append([r.x, r.y, r.z])
            shells.append(shell)
        
        
        light_vec = light_vector(magnitude=1, alpha=-45, beta=-45)
        face_normals = [panel.n.transformation(self.R) for panel in body_panels]
        dot_prods = [-light_vec * face_normal for face_normal in face_normals]
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
           
         
if __name__=='__main__':
    from matplotlib import pyplot as plt
    from sphere import sphere
    nodes, shells = sphere(1, 10, 10, mesh_shell_type='quadrilateral')
    sphere_mesh = PanelMesh(nodes, shells)    
    sphere_mesh.plot_panels()
    sphere_mesh.plot_mesh()
    
    yaw, pitch, roll = np.deg2rad(20), np.deg2rad(30), np.deg2rad(40)
    sphere_mesh.set_body_fixed_frame_orientation(yaw, pitch, roll)
    sphere_mesh.set_R2(yaw, pitch, roll)
    print(sphere_mesh.R)
    print(sphere_mesh.R2)