from airfoil_class import Airfoil
from wing_class import Wing
from wing_mesh_generator import generate_WingPanelMesh, generate_WingPanelMesh2
from panel_method_class import PanelMethod, Steady_PanelMethod
from vector_class import Vector
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours
import time

# create wing object   
root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
tip_airfoil = Airfoil(name="naca0012_new", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0)

# generate wing mesh
num_x_bodyShells = 10
num_x_wakeShells = 15
num_y_Shells = 10

wing_mesh = generate_WingPanelMesh(wing, num_x_bodyShells,
                                   num_x_wakeShells, num_y_Shells,
                                   mesh_shell_type="quadrilateral")
# wing_mesh.plot_panels(elevation=-150, azimuth=-120)


V_fs = Vector((1, 0, 0))
panel_method = Steady_PanelMethod(V_fs)
panel_method.set_V_fs(1, AngleOfAttack=0, SideslipAngle=0)

# wing_mesh = generate_WingPanelMesh2(panel_method.V_fs, wing, num_x_bodyShells,
#                                    num_x_wakeShells, num_y_Shells,
#                                    mesh_shell_type="quadrilateral")

# wing_mesh.plot_panels(elevation=-150, azimuth=-120)

t_start = time.time()        
panel_method.solve(wing_mesh)
t_end = time.time()
solution_time = t_end-t_start
print("solution time = " + str(solution_time))

################ Results ###############

# Pressure Coefficient Distribution across root's airfoil section 
near_root_panels = wing_mesh.give_leftSide_near_root_panels()
Cp = [panel.Cp for panel in near_root_panels]
x = [panel.r_cp.x/wing.root_airfoil.chord for panel in near_root_panels]

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

CL = panel_method.LiftCoeff(body_panels, wing.RefArea)
CD = panel_method.inducedDragCoeff(body_panels, wing.RefArea)

print("CL = " + str(CL))
print("CD = " + str(CD))
