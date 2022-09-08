from matplotlib import pyplot as plt
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
    
    def plot_shells(self):
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(0, 0)
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

    def WingTip_add_extra_neighbours(self):
         
        old_shell_neighbours = {}
        for id_i in self.WingTip["left"] + self.WingTip["right"]:
            old_shell_neighbours[id_i] = self.shell_neighbours[id_i].copy()
            for id_j in self.shell_neighbours[id_i]:
                old_shell_neighbours[id_j] = self.shell_neighbours[id_j].copy()
        
        for id_i in self.WingTip["left"] + self.WingTip["right"]:
            for id_j in old_shell_neighbours[id_i]:
                for id_k in old_shell_neighbours[id_j]: 
                    if id_k!=id_i and id_k not in self.shell_neighbours[id_i]:
                        self.shell_neighbours[id_i].append(id_k)      
        
                
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
        self.wake_sheddingShells = self.wake_sheddingShells
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
 

      
if __name__=='__main__':
    from matplotlib import pyplot as plt
    from sphere import sphere
    nodes, shells = sphere(1, 10, 10, mesh_shell_type='quadrilateral')
    sphere_mesh = PanelMesh(nodes, shells)    
    sphere_mesh.plot_panels()
    