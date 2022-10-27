from airfoil_class import Airfoil
from wing_class import Wing
from wing_mesh_generator import generate_WingPanelMesh
from panel_method_class import UnSteady_PanelMethod
from vector_class import Vector
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours
import numpy as np
import time

# create wing object   
root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
tip_airfoil = Airfoil(name="naca0012_new", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0)

# generate wing mesh
num_x_bodyShells = 10
num_x_wakeShells = 0
num_y_Shells = 10

wing_mesh = generate_WingPanelMesh(wing, num_x_bodyShells,
                                   num_x_wakeShells, num_y_Shells,
                                   mesh_shell_type="quadrilateral")

wing_mesh.set_body_fixed_frame_origin(xo=0, yo=0, zo=0)
wing_mesh.set_body_fixed_frame_orientation(roll=0, pitch=np.deg2rad(-10), yaw=0)

omega = Vector((0, 0, 0))
wing_mesh.set_angular_velocity(omega)

Vo = Vector((-1, 0, 0))
wing_mesh.set_origin_velocity(Vo)


V_wind = Vector((0, 0, 0))
panel_method = UnSteady_PanelMethod(V_wind)
panel_method.solve(wing_mesh, 0.5, 20)

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


