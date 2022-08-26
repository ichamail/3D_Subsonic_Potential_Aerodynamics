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
num_x_bodyShells = 20
num_y_Shells = 20

nodes, shells = wing.generate_bodyMesh2(num_x_bodyShells, num_y_Shells)
wing_mesh = PanelMesh(nodes, shells)

wing_mesh.plot_panels(elevation=-150, azimuth=-120)


V_fs = Vector((1, 0, 0))
panel_method = Steady_Wakeless_PanelMethod(V_fs)
panel_method.solve(wing_mesh)


saved_ids = []
for panel in wing_mesh.panels:
    if 0 < panel.r_cp.y < 0.5:
        saved_ids.append(panel.id)
Cp = []
x = []
for id in saved_ids:
    Cp.append(wing_mesh.panels[id].Cp)
    x.append([wing_mesh.panels[id].r_cp.x])


plt.plot(x, Cp, 'ks', markerfacecolor='r', label='Panel Method')
plt.legend()
plt.grid()
plt.show()

# Surface Contour plot
body_panels = [wing_mesh.panels[id] for id in wing_mesh.panels_id["body"]]
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-120)