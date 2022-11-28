from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelAeroMesh
from panel_method_class import Center_of_Pressure, Cm_about_point,UnSteady_PanelMethod
from vector_class import Vector
import numpy as np
import csv

# create wing object

# root airfoil
root_airfoil_name = "naca0012_new" #"naca0018"
root_chord = 1
root_airfoil = Airfoil(name=root_airfoil_name, chord_length=root_chord)

# tip airfoil
tip_airfoil_name = "naca0012_new" #"naca0012"
tip_chord = 1 # 0.5
tip_airfoil = Airfoil(name=tip_airfoil_name, chord_length=tip_chord)

# wing
span = 2 #4
sweep = 0 #25
dihedral = 0 #-10
twist = 0 #-3
wing = Wing(root_airfoil=root_airfoil,
            tip_airfoil = tip_airfoil,
            semi_span=0.5*span,
            sweep = sweep,
            dihedral = dihedral,
            twist = twist)

# generate wing mesh
shell_type = "quadrilateral" # "quadrilateral" "triangular"
num_x_shells = 10 #15
num_y_shells = 10 #25
nodes, shells, nodes_ids = wing.generate_mesh(
    num_x_shells=num_x_shells, num_y_shells=num_y_shells,
    mesh_shell_type="quadrilateral", span_wise_spacing="uniform",
    mesh_main_surface=True, mesh_tips=True, mesh_wake=False,
    standard_mesh_format=False 
)

wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

# wing position and orientation in space
xo, yo, zo = 5, 0, -2
roll, pitch, yaw = 0, 0, 0
wing_mesh.set_body_fixed_frame_origin(xo=xo, yo=yo, zo=zo)
wing_mesh.set_body_fixed_frame_orientation(roll=np.deg2rad(roll),
                                           pitch=np.deg2rad(pitch),
                                           yaw=np.deg2rad(yaw))
# plot mesh
wing_mesh.plot_mesh(elevation=-150, azimuth=-120)
wing_mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120)
wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120)


# Aerodynamic Analysis Parameters
AoA_start = -20
AoA_end = 20
AoA_step = 2.5

AoA_list = []
for i in range(int((AoA_end-AoA_start)/AoA_step)+1):
    AoA_list.append(AoA_start+i*AoA_step)


FreeStreamVelocity = 1

Vox = -FreeStreamVelocity
Voy = 0
Voz = 0
Vo = Vector((Vox, Voy, Voz))
wing_mesh.set_origin_velocity(Vo)

omega_roll = 0
omega_pitch = 0
omega_yaw = 0
omega = Vector((omega_roll, omega_pitch, omega_yaw))
wing_mesh.set_angular_velocity(omega)


# Panel Method Parameters
dt = 0.5 # 0.2
iterations = 60 # 50 
wake_shed_factor = 0.3
wake_panel_type = "quadrilateral"  # "quadrilateral" "triangular"
V_wind = Vector((0, 0, 0))

unsteady_panel_method = UnSteady_PanelMethod(V_wind)
unsteady_panel_method.set_wakePanelType(wake_panel_type)
unsteady_panel_method.set_WakeShedFactor(wake_shed_factor)

csv_params_row_list = [["Wing"],
                       ["root airfoil", root_airfoil_name],
                       ["root chord length", root_chord],
                       ["tip airfoil", tip_airfoil_name],
                       ["tip chord length", tip_chord],
                       ["span", span],
                       ["sweep", sweep],
                       ["dihedral", dihedral],
                       ["twist", twist], 
                       [],
                       ["Mesh"],
                       ["x shells", num_x_shells],
                       ["y shells", num_y_shells],
                       ["shell type of wing", shell_type],
                       ["wing's number of shells", wing_mesh.shell_num],
                       [],
                       ["Panel Method"],
                       ["dt", dt],
                       ["iterations", iterations],
                       ["wake shell type", wake_panel_type],
                       ["wake shed factor", wake_shed_factor],
                       ["V_wind", V_wind.norm(), "V_wind_x", V_wind.x,
                        "V_wind_y",V_wind.y, "V_wind_z", V_wind.z],
                       ["V_origin", Vo.norm(), "V_origin_x", Vo.x,
                        "V_origin_y", Vo.y, "V_origin_z", Vo.z],
                       ["V_freestream", (V_wind - Vo).norm()],
                       []]

csv_results_row_list = [["AoA", "CL", "CD",
                 "CM_root_c/4_x", "CM_root_c/4_y", "CM_root_c/4_z", "CM_root_tip_x", "CM_root_tip_y", "CM_root_tip_z",
                 "CoP_x", "CoP_y", "CoP_z"]]
for AoA in AoA_list:
    print(AoA)
    mesh = wing_mesh.copy()
    mesh.set_body_fixed_frame_orientation(0, np.deg2rad(-AoA), 0)
    unsteady_panel_method.solve(mesh, dt, iterations)
    
    body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
    r_c4 = Vector((0.25*wing.root_airfoil.chord, 0, 0))
    r_o = Vector((0, 0, 0)) 
    
    CL = unsteady_panel_method.LiftCoeff(mesh, wing.RefArea)
    CD = unsteady_panel_method.inducedDragCoeff(mesh, wing.RefArea)
    Cm_c4 = Cm_about_point(r_c4, body_panels, wing.RefArea)
    Cm_o = Cm_about_point(r_o, body_panels, wing.RefArea)
    CoP = Center_of_Pressure(body_panels, wing.RefArea)
    
    csv_results_row_list.append([AoA, CL, CD,
                         Cm_c4.x, Cm_c4.y, Cm_c4.z,
                         Cm_o.x, Cm_o.y, Cm_o.z,
                         CoP.x, CoP.y, CoP.z])

with open('Aerodynamic Analysis Results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_params_row_list)
    writer.writerows(csv_results_row_list)

mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120, plot_wake=True)