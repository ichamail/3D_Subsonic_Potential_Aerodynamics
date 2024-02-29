from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelAeroMesh
from panel_method_class import PanelMethod, Steady_PanelMethod, Trefftz_Plane_Analysis, local_Cl, local_CL
from vector_class import Vector
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours
from time import perf_counter

# create wing object
wing = Wing(
    root_airfoil=Airfoil(
        name="naca0012 sharp",
        chord_length=1
    ),
    tip_airfoil=Airfoil(
        name="naca0012 sharp",
        chord_length=1
    ),
    semi_span=1,
    sweep=0,
    dihedral=0
)

# # generate wing mesh
# nodes, shells, nodes_ids = wing.generate_mesh(
#     num_x_shells=30,
#     num_y_shells=37,
#     num_x_wake_shells=1,
#     mesh_shell_type="quadrilateral",
#     chord_wise_spacing="cosine",
#     span_wise_spacing="cosine",
#     mesh_main_surface=True,
#     mesh_tips=True,
#     mesh_wake=True,
#     triangular_wake_mesh=False,
#     wake_length_in_chords=30,
#     standard_mesh_format=False
# )

# wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)
# wing_mesh.plot_mesh_bodyfixed_frame(
#     elevation=-150, azimuth=-120, plot_wake=True
# )


V_fs = Vector((1, 0, 0))
panel_method = Steady_PanelMethod(V_fs)
panel_method.set_V_fs(1, AngleOfAttack=10, SideslipAngle=0)

# generate wing mesh with wake in the free stream direction
nodes, shells, nodes_ids = wing.generate_mesh2(
    V_fs=panel_method.V_fs,
    num_x_shells=30,
    num_y_shells=17,
    num_x_wake_shells=50,
    mesh_shell_type="quadrilateral",
    chord_wise_spacing ="cosine",
    span_wise_spacing="denser at wingtips",
    mesh_main_surface=True,
    mesh_tips=True,
    mesh_wake=True,
    triangular_wake_mesh=True,
    wake_length_in_chords=2,
    standard_mesh_format=False
)

wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)
wing_mesh.plot_mesh_bodyfixed_frame(
    elevation=-150, azimuth=-120, plot_wake=True
)

# t_start = perf_counter()        
# panel_method.solve(wing_mesh.copy())
# t_end = perf_counter()
# solution_time = t_end-t_start
# print("solution time + compile time = " + str(solution_time))

t_start = perf_counter()        
# panel_method.solve(wing_mesh)
# panel_method.solve_with_pressure_kutta(wing_mesh)
panel_method.solve_iteratively(wing_mesh, wing.RefArea, 8,
                               convergence_value=10**(-10))
t_end = perf_counter()
solution_time = t_end-t_start
print("solution time = " + str(solution_time))


################ Results ###############

# Pressure Coefficient Distribution across root's airfoil section 
near_root_panels = wing_mesh.give_leftSide_near_root_panels()

Cp = [panel.Cp for panel in near_root_panels]
x = [panel.r_cp.x/wing.root_airfoil.chord for panel in near_root_panels]
y = near_root_panels[0].r_cp.y/wing.semi_span

plt.plot(x, Cp, 'ks--', markerfacecolor='r', label='Panel Method')
plt.xlabel("x/c")
plt.ylabel("Cp")
plt.title("Pressure Coefficient plot at " + str(round(y,2)) + "of semi span")
plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.show()

near_tip_panels = wing_mesh.give_leftSide_near_tip_panels()
Cp = [panel.Cp for panel in near_tip_panels]
x = [panel.r_cp.x/wing.tip_airfoil.chord for panel in near_tip_panels]
y = near_tip_panels[0].r_cp.y/wing.semi_span

plt.plot(x, Cp, 'ks--', markerfacecolor='r', label='Panel Method')
plt.xlabel("x/c")
plt.ylabel("Cp")
plt.title("Pressure Coefficient plot at " + str(round(y,2)) + "of semi span")
plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.show()


# Surface Contour plot
body_panels = [wing_mesh.panels[id] for id in wing_mesh.panels_ids["body"]]
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-120)
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-70)
plot_Cp_SurfaceContours(body_panels, elevation=150, azimuth=-70)
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-50)
plot_Cp_SurfaceContours(body_panels, elevation=150, azimuth=-50)
plot_Cp_SurfaceContours(body_panels, elevation=180, azimuth=-90)

CL = panel_method.LiftCoeff(body_panels, wing.RefArea)
CD = panel_method.inducedDragCoeff(body_panels, wing.RefArea)

print("CL = " + str(CL))
print("CD = " + str(CD))

CL_trefftz, CD_trefftz = Trefftz_Plane_Analysis(
    wing_mesh, panel_method.V_fs, wing.RefArea
)

print("trefftz plane CL = " + str(CL_trefftz))
print("trefftz plane CD = " + str(CD_trefftz))

Cp = [panel.Cp for panel in wing_mesh.panels]
print(max(Cp), min(Cp))

Cp = [wing_mesh.panels[id].Cp for id in wing_mesh.panels_ids["wing tips"]]
print(max(Cp), min(Cp))




local_lift = local_Cl(wing_mesh, panel_method.V_fs)
x = local_lift[:, 0]
y = local_lift[:, 1]
plt.plot(x, y)
plt.show()


local_Lift = local_CL(wing_mesh, panel_method.V_fs, MAC=1)
x = local_Lift[:, 0]
y = local_Lift[:, 1]
plt.plot(x, y)
plt.show()

# import csv, numpy as np

# with open('local_CL.csv', 'w', newline='') as file:
#     writer = csv.writer(file, delimiter=" ")
#     writer.writerows(local_lift)



wing_mesh.plot_mesh_inertial_frame(elevation=180, azimuth= 0,
                                   plot_wake = True)
wing_mesh.plot_mesh_inertial_frame(elevation=-175, azimuth=180,
                                   plot_wake = True)
wing_mesh.plot_mesh_inertial_frame(elevation=-160, azimuth= -30,
                                   plot_wake = True)
wing_mesh.plot_mesh_inertial_frame(elevation=-180, azimuth= -90,
                                   plot_wake = True)