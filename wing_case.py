from airfoil_class import Airfoil
from wing_class import Wing
from wing_mesh_generator import generate_WingPanelMesh, generate_WingPanelMesh2
from panel_method_class import PanelMethod, Steady_PanelMethod
from vector_class import Vector
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours

# create wing object   
root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
tip_airfoil = Airfoil(name="naca0012_new", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0)

# generate wing mesh
num_x_bodyShells = 10
num_x_wakeShells = 15
num_y_Shells = 20

wing_mesh = generate_WingPanelMesh(wing, num_x_bodyShells,
                                   num_x_wakeShells, num_y_Shells,
                                   mesh_shell_type="quadrilateral")
wing_mesh.plot_panels(elevation=-150, azimuth=-120)


V_fs = Vector((1, 0, 0))
panel_method = Steady_PanelMethod(V_fs)
panel_method.set_V_fs(1, AngleOfAttack=10, SideslipAngle=0)

# wing_mesh = generate_WingPanelMesh2(panel_method.V_fs, wing, num_x_bodyShells,
#                                    num_x_wakeShells, num_y_Shells,
#                                    mesh_shell_type="quadrilateral")

# wing_mesh.plot_panels(elevation=-150, azimuth=-120)

# search for panels near wing's root
saved_ids = []
for panel in [wing_mesh.panels[id] for id in wing_mesh.panels_id["body"]]:
    if 0 < panel.r_cp.y < 0.1:
        saved_ids.append(panel.id)
        
panel_method.solve(wing_mesh)
Cp = []
x = []
for id in saved_ids:
    Cp.append(wing_mesh.panels[id].Cp)
    x.append((wing_mesh.panels[id].r_cp.x)/(wing.root_airfoil.chord))

plt.plot(x, Cp, 'ks--', markerfacecolor='r', label='Panel Method')
plt.xlabel("x/c")
plt.ylabel("Cp")
plt.title("Pressure Coefficient plot")
plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.show()


# Surface Contour plot
body_panels = [wing_mesh.panels[id] for id in wing_mesh.panels_id["body"]]
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-120)

CL = panel_method.LiftCoeff(wing_mesh.panels, wing.RefArea)
CD = panel_method.inducedDragCoeff(wing_mesh.panels, wing.RefArea)

print("CL = " + str(CL))
print("CD = " + str(CD))

print([panel.Cp for panel in [wing_mesh.panels[id]
                              for id in wing_mesh.panels_id["body"]]])