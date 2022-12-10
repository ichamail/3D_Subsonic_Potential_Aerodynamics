from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelAeroMesh
from panel_method_class import Center_of_Pressure, Cm_about_point,Steady_PanelMethod
from vector_class import Vector
import numpy as np
import csv

# create wing object

# root airfoil
root_airfoil_name = "naca0018_new" #"naca0018"
root_chord = 1
root_airfoil = Airfoil(name=root_airfoil_name, chord_length=root_chord)

# tip airfoil
tip_airfoil_name = "naca0012_new" #"naca0012"
tip_chord = 0.5 # 0.5
tip_airfoil = Airfoil(name=tip_airfoil_name, chord_length=tip_chord)

# wing
span = 4 #4
sweep = 25 #25
dihedral = -10 #-10
twist = -3 #-3
wing = Wing(root_airfoil=root_airfoil,
            tip_airfoil = tip_airfoil,
            semi_span=0.5*span,
            sweep = sweep,
            dihedral = dihedral,
            twist = twist)

# generate wing mesh
shell_type = "quadrilateral" # "quadrilateral" "triangular"
num_x_shells = 10 #15
num_y_shells = 20 #25
num_x_wakeShells = 100
nodes, shells, nodes_ids = wing.generate_mesh(
    num_x_shells=num_x_shells, num_y_shells=num_y_shells,
    mesh_shell_type="quadrilateral", span_wise_spacing="uniform",
    mesh_main_surface=True, mesh_tips=True, mesh_wake=True, 
    num_x_wake_shells=num_x_wakeShells, standard_mesh_format=False 
)

wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

# plot mesh
wing_mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120,
                                    plot_wake=True)
wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120,
                                   plot_wake=False)


# Aerodynamic Analysis Parameters
AoA_start = -20
AoA_end = 20
AoA_step = 2.5

AoA_list = []
for i in range(int((AoA_end-AoA_start)/AoA_step)+1):
    AoA_list.append(AoA_start+i*AoA_step)


# Panel Method Parameters
SideSlipAngle = 0
V_fs = 1
steady_panel_method = Steady_PanelMethod(V_freestream=Vector((0, 0, 0)))

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
                       ["x bodyShells", num_x_shells],
                       ["y bodyShells", num_y_shells],
                       ["x wakeShells", num_x_wakeShells],
                       ["shell type", shell_type],
                       ["number of body shells",
                        len(wing_mesh.shells_ids["body"])],
                       ["number of wake shells",
                        len(wing_mesh.shells_ids["wake"])],
                       ["number of shells", wing_mesh.shell_num],
                       [],
                       ["Panel Method"],
                       ["V_freestream", V_fs],
                       ["side slip angle", SideSlipAngle],
                       []]

csv_results_row_list = [["AoA", "CL", "CD",
                 "CM_root_c/4_x", "CM_root_c/4_y", "CM_root_c/4_z", "CM_root_leading_edge_x", "CM_root_leading_edge_y", "CM_root_leading_edge_z",
                 "CoP_x", "CoP_y", "CoP_z"]]

for AoA in AoA_list:
    mesh = wing_mesh.copy()
    steady_panel_method.set_V_fs(V_fs, AoA, SideSlipAngle)
    steady_panel_method.solve(mesh)
    
    body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
    r_c4 = Vector((0.25*wing.root_airfoil.chord, 0, 0))
    r_o = Vector((0, 0, 0)) 
    
    CL = steady_panel_method.LiftCoeff(body_panels, wing.RefArea)
    CD = steady_panel_method.inducedDragCoeff(body_panels, wing.RefArea)
    Cm_c4 = Cm_about_point(r_c4, body_panels, wing.RefArea)
    Cm_o = Cm_about_point(r_o, body_panels, wing.RefArea)
    CoP = Center_of_Pressure(body_panels, wing.RefArea)
    
    csv_results_row_list.append([AoA, CL, CD,
                         Cm_c4.x, Cm_c4.y, Cm_c4.z,
                         Cm_o.x, Cm_o.y, Cm_o.z,
                         CoP.x, CoP.y, CoP.z])

with open('Steady Aerodynamic Analysis Results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_params_row_list)
    writer.writerows(csv_results_row_list)
