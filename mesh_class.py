from vector_class import Vector
from panel_class import Panel, triPanel, quadPanel


class Mesh:
    
    def __init__(self, nodes, shells):
        self.nodes = nodes
        self.shells = shells
        self.node_num = len(nodes)
        self.shell_num = len(shells)

        self.shell_neighbours = self.find_shell_neighbours()
    
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

      
class PanelMesh(Mesh):
    def __init__(self, nodes, shells):
        super().__init__(nodes, shells)
        self.panels = None
        self.panels_num = None
        self.panel_neighbours = None
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
        self.panel_neighbours = self.shell_neighbours
        self.panels_num = len(panels)

    def give_neighbours(self, panel):
        
        neighbours_id_list = self.panel_neighbours[panel.id]
        
        # neighbours_list = []
        # for id in neighbours_id_list:
        #     neighbours_list.append(self.panels[id])
         
        neighbours_list = [self.panels[id] for id in neighbours_id_list]
        
        return neighbours_list

    def plot_panels(self):
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.view_init(0, 0)
        
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
            scale = 0.2
            n = n.scalar_product(scale)
            ax.scatter(r_cp.x, r_cp.y, r_cp.z, color='k', s=5)
            ax.quiver(r_cp.x, r_cp.y, r_cp.z, n.x, n.y, n.z, color='r')
                
                
            
        
        plt.show()
        
 
      
if __name__=='__main__':
    from matplotlib import pyplot as plt
    from sphere import sphere
    nodes, shells = sphere(1, 10, 10, mesh_shell_type='quadrilateral')
    sphere_mesh = PanelMesh(nodes, shells)    
    sphere_mesh.plot_panels()
    