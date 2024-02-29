from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelAeroMesh
from panel_method_class import Center_of_Pressure, Cm_about_point,Steady_PanelMethod, Trefftz_Plane_Analysis
from vector_class import Vector
import csv
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
    dihedral=0,
    twist=0
)


# wing mesh info
n_x = 10
n_y = 10
nw_x = 100
spacing = "cosine"
shell_type = "quadrilateral"

# inspect wing mesh
nodes, shells, nodes_ids = wing.generate_mesh(
    num_x_shells=n_x,
    num_y_shells=n_y,
    num_x_wake_shells=nw_x,
    span_wise_spacing=spacing,
    mesh_shell_type=shell_type,
    mesh_main_surface=True,
    mesh_tips=True,
    mesh_wake=True,
    standard_mesh_format=False    
)

wing_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

# wing_mesh.plot_mesh_bodyfixed_frame(
#     elevation=-150, azimuth=-120, plot_wake=True
# )

# wing_mesh.plot_mesh_inertial_frame(
#     elevation=-150, azimuth=-120, plot_wake=False
# )


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
dt = 0.1
max_iters = 200
convergence_value = 10**(-5)
steady_panel_method = Steady_PanelMethod(V_freestream=Vector((0, 0, 0)))


csv_params_row_list = [
    ["Wing"],
    ["root airfoil:", wing.root_airfoil.name],
    ["root chord length:", wing.root_airfoil.chord],
    ["tip airfoil:", wing.tip_airfoil.name],
    ["tip chord length:", wing.tip_airfoil.chord],
    ["span:", wing.semi_span*2],
    ["sweep:", wing.sweep],
    ["dihedral:", wing.dihedral],
    ["twist:", wing.twist], 
    [],
    ["Mesh"],
    ["x body-shells:", n_x, "spacing:", "cosine"],
    ["y shells:", n_y, "spacing:", spacing],
    ["x wake-shells:", nw_x],
    ["shell type:", shell_type],
    ["number of body shells:",
    len(wing_mesh.shells_ids["body"])],
    ["number of wake shells:",
    len(wing_mesh.shells_ids["wake"])],
    ["number of shells:", wing_mesh.shell_num],
    [],
    ["Panel Method"],
    ["type:", "steady", "wake roll-up:", "True"],
    ["dt:", dt],
    ["max number of iterations:", max_iters],
    ["convergence value:", convergence_value],
    ["V_freestream:", V_fs],
    ["side slip angle:", SideSlipAngle],
    [],
    ["Results:"],
    []
]

csv_results_row_list = [
    [
        "AoA", "CL", "CD", "trefftz-CL", "trefftz-CD",
        "CM_root_c/4_x", "CM_root_c/4_y", "CM_root_c/4_z", "CM_root_leading_edge_x", "CM_root_leading_edge_y", "CM_root_leading_edge_z",
        "CoP_x", "CoP_y", "CoP_z"
    ]
]

t = perf_counter()
for AoA in AoA_list:
    print(
        "Operation Point: "+ str(AoA_list.index(AoA)+1) +"/"+ str(len(AoA_list))
    )
    
    steady_panel_method.set_V_fs(V_fs, AoA, SideSlipAngle)
    
    mesh = wing_mesh.copy()
    
    steady_panel_method.solve_iteratively(
        mesh, wing.RefArea, dt, max_iters, convergence_value
    )
    
    body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
    r_c4 = Vector((0.25*wing.root_airfoil.chord, 0, 0))
    r_o = Vector((0, 0, 0)) 
    
    CL = steady_panel_method.LiftCoeff(body_panels, wing.RefArea)
    CD = steady_panel_method.inducedDragCoeff(body_panels, wing.RefArea)
    
    trefftz_CL, trefftz_CD = Trefftz_Plane_Analysis(
        mesh, steady_panel_method.V_fs, wing.RefArea
    ) 
    
    Cm_c4 = Cm_about_point(r_c4, body_panels, wing.RefArea)
    Cm_o = Cm_about_point(r_o, body_panels, wing.RefArea)
    CoP = Center_of_Pressure(body_panels, wing.RefArea)
    
    csv_results_row_list.append(
        [
            AoA, CL, CD, trefftz_CL, trefftz_CD, Cm_c4.x, Cm_c4.y, Cm_c4.z,
            Cm_o.x, Cm_o.y, Cm_o.z, CoP.x, CoP.y, CoP.z
        ]
    )

print("total simulation time: " + str(perf_counter()-t) + " seconds")

with open('Steady Aerodynamic Analysis Results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_params_row_list)
    writer.writerows(csv_results_row_list)