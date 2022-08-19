from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelMesh


# airfoil = Airfoil(name="naca0012", chord_length=1)
# wing = Wing(root_airfoil=airfoil, tip_airfoil=airfoil, semi_span=1,
#             sweep=10, dihedral=20)

root_airfoil = Airfoil(name="naca0012", chord_length=1)
tip_airfoil = Airfoil(name="naca0012", chord_length=0.8)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=15, dihedral=10)

nodes, shells = wing.generate_mesh(8, 4)

# for node_id, node in enumerate(nodes):
#     print(node_id, node)

# for shell_id, shell in enumerate(shells):
#     print(shell_id, shell)

mesh = PanelMesh(nodes, shells)
# for shell_id, neighbours in enumerate(mesh.shell_neighbours):
#     print(shell_id, neighbours)
    
mesh.plot_panels()


######### wing mesh with wake ###############

num_x_bodyShells = 8
num_x_wakeShells = 15
num_y_shells = 4

body_nodes, body_shells = wing.generate_mesh(num_x_bodyShells,
                                              num_y_shells)
wake_nodes, wake_shells = wing.generate_wake_mesh(num_x_wakeShells,
                                                  num_y_shells)

# for wake_node_id, wake_node in enumerate(wake_nodes):
#     print(wake_node_id, wake_node)

# for wake_shell_id, wake_shell in enumerate(wake_shells):
#     print(wake_shell_id, wake_shell)

for i, wake_shell in enumerate(wake_shells):
    for j in range(len(wake_shell)):
        wake_shells[i][j] = wake_shells[i][j] + len(body_nodes)
    
nodes = [*body_nodes, *wake_nodes]
shells = [*body_shells, *wake_shells]

# for node_id, node in enumerate(nodes):
#     print(node_id, node)

# for shell_id, shell in enumerate(shells):
#     print(shell_id, shell)

mesh = PanelMesh(nodes, shells)
mesh.plot_panels()

body_panels_id_list = []
wake_panels_id_list = []

for panel in mesh.panels:
    if panel.id > len(body_shells):
        wake_panels_id_list.append(panel.id)
    else:
        body_panels_id_list.append(panel.id)

# print(wake_panels_id_list)
# print(body_panels_id_list)