import stl
import numpy as np
from vector_class import Vector
from mesh_class import PanelMesh, PanelAeroMesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Algorithms import light_vector
from plot_functions import set_axes_equal


def plot_mesh(mesh, shell_id_list, node_id_list, elevation=-150, azimuth=-120):
    shells = []
    vert_coords = []
    
    for panel in mesh.panels:
        shell=[]
        for r_vertex in panel.r_vertex:
            shell.append((r_vertex.x, r_vertex.y, r_vertex.z))
            vert_coords.append([r_vertex.x, r_vertex.y, r_vertex.z])
        shells.append(shell)
    
    
    light_vec = light_vector(magnitude=1, alpha=-45, beta=-45)
    face_normals = [panel.n for panel in mesh.panels]
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
    
    # change shell color to red
    for id in shell_id_list:
        facecolor[id] = [1, 0, 0, 1]
    
    ax = plt.axes(projection='3d')
    poly3 = Poly3DCollection(shells, facecolor=facecolor)
    ax.add_collection(poly3)
    ax.view_init(elevation, azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    x = [mesh.nodes[id][0] for id in node_id_list]
    y = [mesh.nodes[id][1] for id in node_id_list]
    z = [mesh.nodes[id][2] for id in node_id_list]
    ax.scatter(x, y, z, c="r")

    vert_coords = np.array(vert_coords)
    x, y, z = vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2]
    ax.set_xlim3d(x.min(), x.max())
    ax.set_ylim3d(y.min(), y.max())
    ax.set_zlim3d(z.min(), z.max())
    set_axes_equal(ax)    
    plt.show()
    
        
suction_side_stl_file = open("coord_seligFmt/suction side.stl", "r")
pressure_side_stl_file = open("coord_seligFmt/pressure side.stl", "r")
right_tip_stl_file = open('coord_seligFmt/right tip.stl', "r")
left_tip_stl_file = open("coord_seligFmt/left tip.stl", "r")
suction_side_solid = stl.read_ascii_file(suction_side_stl_file)
pressure_side_solid = stl.read_ascii_file(pressure_side_stl_file)
right_tip_solid = stl.read_ascii_file(right_tip_stl_file)
left_tip_solid = stl.read_ascii_file(left_tip_stl_file)

nodes = []
shells = []

wing_solid_list = [suction_side_solid,
                   pressure_side_solid,
                   right_tip_solid,
                   left_tip_solid]

for solid in wing_solid_list:
    for facet in solid.facets:
        for vertex in facet.vertices:
            node = (vertex[0], vertex[1], vertex[2])
            is_node_in_main_wing_node = False
            for node_else in nodes:
                dx = abs(node[0] - node_else[0])
                dy = abs(node[1] - node_else[1])
                dz = abs(node[2] - node_else[2])
                if dx < 10**(-6) and dy < 10**(-6) and dz < 10**(-6):
                    is_node_in_main_wing_node = True
                    break
            if not is_node_in_main_wing_node:
                nodes.append(node)

        

for solid in wing_solid_list:
    for facet in solid.facets:
        shell = []
        for vertex in facet.vertices:
            node = (vertex[0], vertex[1], vertex[2])
            
            for node_id, node_else in enumerate(nodes):
                dx = abs(node[0] - node_else[0])
                dy = abs(node[1] - node_else[1])
                dz = abs(node[2] - node_else[2])
                if dx < 10**(-6) and dy < 10**(-6) and dz < 10**(-6):
                    break
                
            shell.append(node_id)
            
        shells.append(shell)
    


mesh = PanelMesh(nodes, shells)
mesh.set_body_fixed_frame_orientation(roll=np.deg2rad(-90), pitch=0, yaw=0)
mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120)

Trailing_edge_shell_id_list = []
Trailing_egde_node_id_list = []

node_x_list = [node[0] for node in mesh.nodes]
x_max = max(node_x_list)

for node_id, node in enumerate(mesh.nodes):
    if node[0] == x_max:
        Trailing_egde_node_id_list.append(node_id)

for shell_id, shell in enumerate(mesh.shells):
    num_nodes_in_trailing_edge = 0
    for node_id in shell:
        if node_id in Trailing_egde_node_id_list:
            num_nodes_in_trailing_edge = num_nodes_in_trailing_edge + 1
    
    if num_nodes_in_trailing_edge > 1:
        Trailing_edge_shell_id_list.append(shell_id)



nodes_id = {"body": [node_id for node_id in range(len(nodes))],
            "wake": [], "trailing edge": Trailing_egde_node_id_list}
shells_id = {"body": [shell_id for shell_id in range(len(shells))],
             "wake": []}

TrailingEdge = {"suction side": [], "pressure side": []}

for shell_id in Trailing_edge_shell_id_list:
    panel = mesh.panels[shell_id]
    if panel.r_cp.y < 0:
        TrailingEdge["pressure side"].append(shell_id)
    elif panel.r_cp.y > 0:
        TrailingEdge["suction side"].append(shell_id)


wake_sheddingShells = {}
for j in range(len(TrailingEdge["suction side"])):          
    wake_sheddingShells[TrailingEdge["suction side"][j]] = []
    wake_sheddingShells[TrailingEdge["pressure side"][j]] = []


wing_mesh = PanelAeroMesh(nodes, shells, nodes_id, shells_id, TrailingEdge,
                          wake_sheddingShells)


plot_mesh(mesh, wing_mesh.TrailingEdge["pressure side"],
          wing_mesh.nodes_id["trailing edge"])


wing_mesh.set_body_fixed_frame_origin(xo=3, yo=-1, zo=0)
wing_mesh.set_body_fixed_frame_orientation(roll=np.deg2rad(-90), pitch=0, yaw=0)

# wing_mesh.plot_shells(elevation=-150, azimuth=-120)
# wing_mesh.plot_panels(elevation=-150, azimuth=-120)
wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120, plot_wake=False)



omega = Vector((np.deg2rad(0), 0, 0))
Vo = Vector((-1, 0, 0))
V_wind = Vector((0, 0, 0))
dt = 0.5
wing_mesh.set_angular_velocity(omega)
wing_mesh.set_origin_velocity(Vo)

# induced_velocity_function = ( lambda r_p,
#                              body_panels,
#                              wake_panels : Vector((0, 0, -r_p.x * 0.1 )) )

for i in range(5):
    
    wing_mesh.move_body(dt)
    wing_mesh.shed_wake(V_wind, dt, wake_shed_factor=0.3, type="triangular")
    # wing_mesh.convect_wake(induced_velocity_function, dt)
    # wing_mesh.plot_shells(elevation=-150, azimuth=-120)
    wing_mesh.plot_panels(elevation=-150, azimuth=-120)
    wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120,
                                       plot_wake=True)
