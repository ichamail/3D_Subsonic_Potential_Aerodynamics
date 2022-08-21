from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelMesh


# airfoil = Airfoil(name="naca0012", chord_length=1)
# wing = Wing(root_airfoil=airfoil, tip_airfoil=airfoil, semi_span=1,
#             sweep=10, dihedral=20)

root_airfoil = Airfoil(name="naca0012", chord_length=1)
tip_airfoil = Airfoil(name="naca0012", chord_length=0.8)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=15, dihedral=10)

body_nodes, body_shells = wing.generate_bodyMesh(8, 4)

# for node_id, node in enumerate(body_nodes):
#     print(node_id, node)

# for shell_id, shell in enumerate(body_shells):
#     print(shell_id, shell)

mesh = PanelMesh(body_nodes, body_shells)
# for shell_id, neighbours in enumerate(mesh.shell_neighbours):
#     print(shell_id, neighbours)
    
mesh.plot_panels()


######### wing mesh with wake ###############

num_x_bodyShells = 8
num_x_wakeShells = 15
num_y_Shells = 4

# body_nodes, body_shells = wing.generate_bodyMesh(num_x_bodyShells,
#                                               num_y_Shells)
# wake_nodes, wake_shells = wing.generate_wakeMesh(num_x_wakeShells,
#                                                   num_y_Shells)

# for wake_node_id, wake_node in enumerate(wake_nodes):
#     print(wake_node_id, wake_node)

# for wake_shell_id, wake_shell in enumerate(wake_shells):
#     print(wake_shell_id, wake_shell)

# for i, wake_shell in enumerate(wake_shells):
#     for j in range(len(wake_shell)):
#         wake_shells[i][j] = wake_shells[i][j] + len(body_nodes)
    
# nodes = [*body_nodes, *wake_nodes]
# shells = [*body_shells, *wake_shells]

nodes, shells = wing.generate_mesh(num_x_bodyShells, num_x_wakeShells,
                                   num_y_Shells)
# for node_id, node in enumerate(nodes):
#     print(node_id, node)

# for shell_id, shell in enumerate(shells):
#     print(shell_id, shell)

mesh = PanelMesh(nodes, shells)
mesh.plot_panels()

shells_id = wing.give_shells_id_dict(num_x_bodyShells,
                                          num_x_wakeShells,
                                          num_y_Shells)
print(shells_id["body"])
print(shells_id["wake"])


TrailingEdge = wing.TrailingEdge_Shells_id_dict(num_x_bodyShells, num_y_Shells)

shed_wakeShells = wing.wake_shells_shed_from_TrailingEdge(4, TrailingEdge)


# for trailing_edge_shell_id in shed_wake_shells:
#     print(trailing_edge_shell_id, shed_wake_shells[trailing_edge_shell_id])

# for shell_id in TrailingEdge["pressure side"]:
#     print(shed_wake_shells[shell_id])