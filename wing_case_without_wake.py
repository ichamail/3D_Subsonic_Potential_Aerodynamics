from airfoil_class import Airfoil
from mesh_class import PanelMesh
from wing_class import Wing
from wing_mesh_generator import generate_WingPanelMesh
from panel_method_class import PanelMethod, Steady_Wakeless_PanelMethod
from vector_class import Vector
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours


# create wing object   
root_airfoil = Airfoil(name="naca0012", chord_length=1)
tip_airfoil = Airfoil(name="naca0012", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0)

# generate wing mesh
num_x_bodyShells = 10
num_y_Shells = 10

nodes, shells = wing.generate_wingMesh(num_x_bodyShells, num_y_Shells)
TrailingEdge = wing.give_TrailingEdge_Shells_id(num_x_bodyShells, num_y_Shells)
wing_mesh = PanelMesh(nodes, shells, TrailingEdge=TrailingEdge)
wing_mesh.free_TrailingEdge()

# wing_mesh.plot_panels(elevation=-150, azimuth=-120)


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