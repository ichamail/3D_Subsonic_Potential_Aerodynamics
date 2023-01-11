import numpy as np
from airfoil_class import Airfoil
from vector_class import Vector
from bwb_class import WingCrossSection, BWB, BWB_reader
from mesh_class import PanelAeroMesh
from panel_method_class import UnSteady_PanelMethod, Cm_about_point, Center_of_Pressure
import csv
from time import perf_counter


data_dict = BWB_reader(filePath="BWB/" , fileName= "BWB_X_sections_info")

# Change Airfoil's class, class atribute
Airfoil.filePath = "BWB/"


# create BWB object
bwb = BWB(
    name="RX3",
    wingXsection_list=[
        WingCrossSection(
            Vector(data_dict["leading edge coords"][id]),
            chord=data_dict["chord"][id],
            twist=data_dict["twist"][id],
            airfoil=Airfoil(
                name=data_dict["airfoil name"][id]
            )
        )
        
        for id in range(len(data_dict["airfoil name"]))
            
    ]
)

# C_t = 0.5267, λ=C_t/C_r = 0.5, b/2 = 3.1
RefArea = 4.898

# bwb.subdivide_spanwise_sections(1, interpolation_type="cubic")

# inspect bwb mesh
n_x = 20
n_y = 1
spacing = "uniform"
shell_type = "quads"

nodes, shells, nodes_ids = bwb.mesh_body(
    ChordWise_resolution=n_x,
    SpanWise_resolution=n_y,
    SpanWise_spacing=spacing,
    shellType=shell_type,
    mesh_main_surface=True,
    mesh_tips=True,
    mesh_wake=False,
    standard_mesh_format=False
)

bwb_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

bwb_mesh.plot_mesh_bodyfixed_frame(
    elevation=-120, azimuth=-150, plot_wake=False
)


# wing position and orientation in space
xo, yo, zo = 5, 0, -2
roll, pitch, yaw = 0, 0, 0
bwb_mesh.set_body_fixed_frame_origin(xo=xo, yo=yo, zo=zo)
bwb_mesh.set_body_fixed_frame_orientation(roll=np.deg2rad(roll),
                                           pitch=np.deg2rad(pitch),
                                           yaw=np.deg2rad(yaw))
# plot mesh
bwb_mesh.plot_mesh(elevation=-150, azimuth=-120)
bwb_mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120)
bwb_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120)


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
bwb_mesh.set_origin_velocity(Vo)

omega_roll = 0
omega_pitch = 0
omega_yaw = 0
omega = Vector((omega_roll, omega_pitch, omega_yaw))
bwb_mesh.set_angular_velocity(omega)


# Panel Method Parameters
dt = 0.075 # 0.2
max_iters = 250 # 50
convergence_value = 10**-(4)
wake_shed_factor = 0.3
wake_panel_type = "triangular"  # "quadrilateral" "triangular"
V_wind = Vector((0, 0, 0))

unsteady_panel_method = UnSteady_PanelMethod(V_wind)
unsteady_panel_method.set_wakePanelType(wake_panel_type)
unsteady_panel_method.set_WakeShedFactor(wake_shed_factor)

csv_params_row_list = [
                        ["BWB"],
                        ["name:", bwb.name],
                        ["number of input Xsections", len(bwb.wingXsection_list)],
                        ["span:", (bwb.wingXsection_list[-1].r_leadingEdge.y
                                - bwb.wingXsection_list[0].r_leadingEdge.y)],
                        ["Reference Area:", RefArea],
                        [],
                        ["Mesh"],
                        ["x body-shells:", n_x, "spacing:", "cosine"],
                        ["y shells between Xsections:", n_y, "spacing:", spacing],
                        ["shell type:", shell_type],
                        ["number of body shells:",
                        len(bwb_mesh.shells_ids["body"])],
                        ["number of wake shells:",
                        len(bwb_mesh.shells_ids["wake"])],
                        ["number of shells:", bwb_mesh.shell_num],
                        [],
                        ["Panel Method"],
                        ["dt", dt],
                        ["maximum iterations", max_iters],
                        ["convergence_value", convergence_value],
                        ["wake shell type", wake_panel_type],
                        ["wake shed factor", wake_shed_factor],
                        ["V_wind", V_wind.norm(), "V_wind_x", V_wind.x,
                        "V_wind_y",V_wind.y, "V_wind_z", V_wind.z],
                        ["V_origin", Vo.norm(), "V_origin_x", Vo.x,
                        "V_origin_y", Vo.y, "V_origin_z", Vo.z],
                        ["V_freestream", (V_wind - Vo).norm()],
                        []]

csv_results_row_list = [
    [
        "AoA", "CL", "CD", "trefftz-CL", "trefftz-CD",
        "CM_root_leading_edge_x", "CM_root_leading_edge_y", "CM_root_leading_edge_z",
        "CoP_x", "CoP_y", "CoP_z"
    ]
]

t = perf_counter()
for AoA in AoA_list:
    print(
        "Operation Point: "+ str(AoA_list.index(AoA)+1) +"/"+ str(len(AoA_list))
    )
    
    mesh = bwb_mesh.copy()
    mesh.set_body_fixed_frame_orientation(0, np.deg2rad(-AoA), 0)
    unsteady_panel_method.solve_steady(
        mesh, RefArea, dt, max_iters, convergence_value)
    
    body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
    r_o = Vector((0, 0, 0)) 
    
    CL = unsteady_panel_method.LiftCoeff(mesh, RefArea)
    CD = unsteady_panel_method.inducedDragCoeff(mesh, RefArea)
    Cm_o = Cm_about_point(r_o, body_panels, RefArea)
    CoP = Center_of_Pressure(body_panels, RefArea)
    
    csv_results_row_list.append([AoA, CL, CD,
                         Cm_o.x, Cm_o.y, Cm_o.z,
                         CoP.x, CoP.y, CoP.z])

t = perf_counter()-t
print("total simulation time: " + str(t) + " seconds")

with open('Steady Aerodynamic Analysis Results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_params_row_list)
    writer.writerows([["total simulation time [sec]:", t], []])
    writer.writerows(csv_results_row_list)
