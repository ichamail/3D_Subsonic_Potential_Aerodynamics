from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelAeroMesh
from panel_method_class import PanelMethod, Steady_PanelMethod
from vector_class import Vector
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours
from time import perf_counter

# create wing object   
root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
tip_airfoil = Airfoil(name="naca0012_new", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0)

# generate wing mesh
num_x_bodyShells = 10
num_x_wakeShells = 15
num_y_Shells = 10

nodes, shells, nodes_ids = wing.generate_mesh(
    num_x_shells=num_x_bodyShells, num_y_shells=num_y_Shells,
    mesh_shell_type="quadrilateral",
    mesh_main_surface=True, mesh_tips=True, mesh_wake=True, 
    num_x_wake_shells=num_x_wakeShells, standard_mesh_format=False 
)

wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

# wing_mesh.plot_panels(elevation=-150, azimuth=-120)


V_fs = Vector((1, 0, 0))
panel_method = Steady_PanelMethod(V_fs)
panel_method.set_V_fs(1, AngleOfAttack=0, SideslipAngle=0)

# nodes, shells, nodes_ids = wing.generate_mesh2(
#     num_x_shells=num_x_bodyShells, num_y_shells=num_y_Shells,
#     mesh_shell_type="quadrilateral",
#     mesh_main_surface=True, mesh_tips=True, mesh_wake=True, 
#     num_x_wake_shells=num_x_wakeShells, V_fs=V_fs,
#     standard_mesh_format=False 
# )

# wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)
# wing_mesh.plot_panels(elevation=-150, azimuth=-120)

t_start = perf_counter()        
panel_method.solve(wing_mesh)
t_end = perf_counter()
solution_time = t_end-t_start
print("solution time + compile time = " + str(solution_time))

t_start = perf_counter()        
panel_method.solve(wing_mesh)
t_end = perf_counter()
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
body_panels = [wing_mesh.panels[id] for id in wing_mesh.panels_ids["body"]]
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-120)

CL = panel_method.LiftCoeff(body_panels, wing.RefArea)
CD = panel_method.inducedDragCoeff(body_panels, wing.RefArea)

print("CL = " + str(CL))
print("CD = " + str(CD))
