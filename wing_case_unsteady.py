from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelAeroMesh
from panel_method_class import Center_of_Pressure, Cm_about_point,UnSteady_PanelMethod
from vector_class import Vector
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours
import numpy as np
from time import perf_counter

# create wing object   
root_airfoil = Airfoil(name="naca0012 sharp", chord_length=1)
tip_airfoil = Airfoil(name="naca0012 sharp", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0)

# generate wing mesh
num_x_bodyShells = 30
num_y_Shells = 17

nodes, shells, nodes_ids = wing.generate_mesh(
    num_x_shells=num_x_bodyShells,
    num_y_shells=num_y_Shells,
    chord_wise_spacing="cosing",
    span_wise_spacing="denser at wingtips",
    mesh_shell_type="quadrilateral",
    mesh_main_surface=True, mesh_tips=True, mesh_wake=False,
    standard_mesh_format=False,
)

wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

wing_mesh.set_body_fixed_frame_origin(xo=0, yo=0, zo=0)
wing_mesh.set_body_fixed_frame_orientation(roll=0, pitch=np.deg2rad(-10), yaw=0)

omega = Vector((0, 0, 0))
wing_mesh.set_angular_velocity(omega)

Vo = Vector((-1, 0, 0))
wing_mesh.set_origin_velocity(Vo)


V_wind = Vector((0, 0, 0))
panel_method = UnSteady_PanelMethod(V_wind)
panel_method.set_wakePanelType("triangular")
panel_method.set_WakeShedFactor(1)

t_start = perf_counter()        
# panel_method.solve(wing_mesh, dt=0.15, iters=60)
panel_method.solve_steady(wing_mesh, wing.RefArea, dt=0.2, max_iters=50, convergence_value=10**(-6))
t_end = perf_counter()
solution_time = t_end-t_start
print("solution time + compile time = " + str(solution_time))


################ Results ###############

wing_mesh.plot_mesh_bodyfixed_frame(elevation=-150,azimuth=-120, plot_wake=True)
wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120, plot_wake=True)

wing_mesh.plot_mesh_bodyfixed_frame(elevation=-160, azimuth= -30,
                                   plot_wake = True)
wing_mesh.plot_mesh_bodyfixed_frame(elevation=-180, azimuth= -90,
                                   plot_wake = True)
wing_mesh.plot_mesh_inertial_frame(elevation=-160, azimuth= -30,
                                   plot_wake = True)
wing_mesh.plot_mesh_inertial_frame(elevation=-180, azimuth= -90,
                                   plot_wake = True)


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

CL = panel_method.LiftCoeff(wing_mesh, wing.RefArea)
CD = panel_method.inducedDragCoeff(wing_mesh, wing.RefArea)

print("CL = " + str(CL))
print("CD = " + str(CD))

r_c4 = Vector((wing.root_airfoil.chord, 0, 0))
Cm = Cm_about_point(r_c4, body_panels, wing.RefArea)

r_CoP = Center_of_Pressure(body_panels, wing.RefArea)

print("CMx = " + str(Cm.x) + ", CMy = " + str(Cm.y) + ", CMz = " + str(Cm.z))
print("r_CoP = " + str(r_CoP.x)+"ex + " + str(r_CoP.y) + "ey + " 
      + str(r_CoP.z) + "ez")