from vector_class import Vector
from wing_class import Wing
from mesh_class import PanelMesh


def generate_WingPanelMesh(wing:Wing, num_x_bodyShells:int,
                           num_x_wakeShells:int, num_y_Shells:int,
                           mesh_shell_type:str="quadrilateral")->PanelMesh:

    nodes, shells, nodes_id, shells_id, WingTip = wing.generate_mesh(
        num_x_bodyShells, num_x_wakeShells, num_y_Shells, mesh_shell_type,
        return_id_dicts=True)

    TrailingEdge = wing.give_TrailingEdge_Shells_id(num_x_bodyShells,
                                                    num_y_Shells,
                                                    mesh_shell_type)

    wake_sheddingShells = wing.give_wake_sheddingShells(num_x_wakeShells,
                                                    TrailingEdge,
                                                    mesh_shell_type)
    
    

    wing_mesh = PanelMesh(nodes, shells, nodes_id, shells_id, TrailingEdge,
                          wake_sheddingShells, WingTip)
      
    wing_mesh.free_TrailingEdge()
    
    return wing_mesh

def generate_WingPanelMesh2(V_fs:Vector, wing:Wing, num_x_bodyShells:int,
                           num_x_wakeShells:int, num_y_Shells:int,
                           mesh_shell_type:str="quadrilateral")->PanelMesh:

    nodes, shells, nodes_id, shells_id, WingTip = wing.generate_mesh2(
        V_fs, num_x_bodyShells, num_x_wakeShells, num_y_Shells, mesh_shell_type,
        return_id_dicts=True)

    TrailingEdge = wing.give_TrailingEdge_Shells_id(num_x_bodyShells,
                                                    num_y_Shells,
                                                    mesh_shell_type)

    wake_sheddingShells = wing.give_wake_sheddingShells(num_x_wakeShells,
                                                    TrailingEdge,
                                                    mesh_shell_type)

    wing_mesh = PanelMesh(nodes, shells, nodes_id, shells_id, TrailingEdge,
                          wake_sheddingShells, WingTip)
    
    wing_mesh.free_TrailingEdge()
    return wing_mesh

if __name__=="__main__":
    from airfoil_class import Airfoil
    
    root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
    tip_airfoil = Airfoil(name="naca0012_new", chord_length=0.8)
    wing = Wing(root_airfoil, tip_airfoil,
                semi_span=1, sweep=20, dihedral=20, twist=10)

    num_x_bodyShells = 4
    num_x_wakeShells = 4
    num_y_Shells = 2
    
    wing_mesh = generate_WingPanelMesh(wing, num_x_bodyShells,
                                       num_x_wakeShells, num_y_Shells,
                                       mesh_shell_type="quadrilateral")
    
    # V_fs = Vector((1, 0, 0))
    # wing_mesh = generate_WingPanelMesh2(V_fs, wing, num_x_bodyShells,
    #                                    num_x_wakeShells, num_y_Shells,
    #                                    mesh_shell_type="quadrilateral")
    
    wing_mesh.plot_panels()
    
    # print(wing_mesh.panels_id)
    # print(wing_mesh.TrailingEdge)    
    # print(wing_mesh.wake_sheddingShells)
        
    # for id in wing_mesh.panels_id["body"]:
    #     print(id, wing_mesh.panel_neighbours[id])
    pass    