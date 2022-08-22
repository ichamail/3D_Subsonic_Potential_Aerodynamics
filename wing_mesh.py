from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelMesh


def Wing_panelMesh_generator(wing:Wing, num_x_bodyShells, num_x_wakeShells,
                             num_y_Shells)->PanelMesh:

    nodes, shells = wing.generate_mesh(num_x_bodyShells,
                                    num_x_wakeShells,
                                    num_y_Shells)

    shells_id = wing.give_shells_id_dict(num_x_bodyShells,
                                        num_x_wakeShells,
                                        num_y_Shells)

    TrailingEdge = wing.TrailingEdge_Shells_id_dict(num_x_bodyShells,
                                                        num_y_Shells)

    shed_wakeShells = wing.wake_shells_shed_from_TrailingEdge(num_x_wakeShells,
                                                            TrailingEdge)

    wing_mesh = PanelMesh(nodes, shells, shells_id, TrailingEdge, shed_wakeShells)
    
    return wing_mesh


if __name__=="__main__":
    
    root_airfoil = Airfoil(name="naca0012", chord_length=1)
    tip_airfoil = Airfoil(name="naca0012", chord_length=0.8)
    wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=15, dihedral=10)

    num_x_bodyShells = 8
    num_x_wakeShells = 15
    num_y_Shells = 4
    
    wing_mesh = Wing_panelMesh_generator(wing, num_x_bodyShells,
                                         num_x_wakeShells, num_y_Shells)
    # wing_mesh.plot_panels()
    
    # print(wing_mesh.panels_id)
    # print(wing_mesh.TrailingEdge)    
    # print(wing_mesh.shed_wakePanels)
    
    # for id in wing_mesh.panels_id["body"]:
    #     print(wing_mesh.panels[id].id, id)
    #     if wing_mesh.panels[id].id in wing_mesh.TrailingEdge["suction side"]:
    #         print(True)
    #         print(wing_mesh.shed_wakePanels[id])
    
    pass