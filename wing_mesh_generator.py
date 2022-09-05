from wing_class import Wing
from mesh_class import PanelMesh


def generate_WingPanelMesh(wing:Wing, num_x_bodyShells:int,
                           num_x_wakeShells:int, num_y_Shells:int,
                           mesh_shell_type:str="quadrilateral")->PanelMesh:

    nodes, shells = wing.generate_mesh(num_x_bodyShells,
                                    num_x_wakeShells,
                                    num_y_Shells,
                                    mesh_shell_type)

    shells_id = wing.give_shells_id_dict(num_x_bodyShells,
                                        num_x_wakeShells,
                                        num_y_Shells, mesh_shell_type)

    TrailingEdge = wing.give_TrailingEdge_Shells_id(num_x_bodyShells,
                                                    num_y_Shells,
                                                    mesh_shell_type)

    wake_sheddingShells = wing.give_wake_sheddingShells(num_x_wakeShells,
                                                    TrailingEdge,
                                                    mesh_shell_type)

    wing_mesh = PanelMesh(nodes, shells, shells_id, TrailingEdge,
                          wake_sheddingShells)
    
    return wing_mesh


if __name__=="__main__":
    from airfoil_class import Airfoil
    
    
    root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
    tip_airfoil = Airfoil(name="naca0012_new", chord_length=0.8)
    wing = Wing(root_airfoil, tip_airfoil,
                semi_span=1, sweep=15, dihedral=10, twist=0)

    num_x_bodyShells = 4
    num_x_wakeShells = 4
    num_y_Shells = 4
    
    wing_mesh = generate_WingPanelMesh(wing, num_x_bodyShells,
                                         num_x_wakeShells, num_y_Shells,
                                         mesh_shell_type="quadrilateral")
    wing_mesh.plot_panels()
    
    # print(wing_mesh.panels_id)
    # print(wing_mesh.TrailingEdge)    
    # print(wing_mesh.wake_sheddingShells)
    
    # for id in wing_mesh.panels_id["body"] + wing_mesh.panels_id["wake"]:
    #     print(wing_mesh.panels[id].id == id)
    
    # for id_i in (wing_mesh.TrailingEdge["suction side"]
    #              + wing_mesh.TrailingEdge["pressure side"]):
    #     print(wing_mesh.panels[id_i].id == id_i)
    #     for id_j in wing_mesh.wake_sheddingShells[id_i]:
    #         print(wing_mesh.panels[id_j].id == id_j)
            
    pass