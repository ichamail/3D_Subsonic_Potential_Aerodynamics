from airfoil_class import Airfoil
from mesh_class import PanelAeroMesh
from wing_class import Wing
from panel_method_class import PanelMethod, Steady_Wakeless_PanelMethod
from vector_class import Vector
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours


# create wing object   
root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
tip_airfoil = Airfoil(name="naca0012_new", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0)

# generate wing mesh
num_x_bodyShells = 10
num_y_Shells = 10

nodes, shells, nodes_ids = wing.generate_mesh(
    num_x_shells=num_x_bodyShells, num_y_shells=num_y_Shells,
    mesh_shell_type="quadrilateral",
    mesh_main_surface=True, mesh_tips=True, mesh_wake=False,
    standard_mesh_format=False 
)

wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)


V_fs = Vector((1, 0, 0))
panel_method = Steady_Wakeless_PanelMethod(V_fs)
panel_method.solve(wing_mesh)

# Cp plot        
near_root_panels = wing_mesh.give_leftSide_near_root_panels()

Cp = [panel.Cp for panel in near_root_panels]
x = [panel.r_cp.x for panel in near_root_panels]
plt.plot(x, Cp, 'ks--', markerfacecolor='r', label='Panel Method')
plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.show()

# Surface Contour plot
plot_Cp_SurfaceContours(wing_mesh.panels, elevation=-150, azimuth=-120)