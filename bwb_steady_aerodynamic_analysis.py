from airfoil_class import Airfoil
from vector_class import Vector
from bwb_class import WingCrossSection, BWB, BWB_reader
from mesh_class import PanelAeroMesh
from panel_method_class import Steady_PanelMethod, Trefftz_Plane_Analysis, Cm_about_point, Center_of_Pressure
import csv
from time import perf_counter


# data_dict = BWB_reader(filePath="BWB concept/" , fileName= "BWB_X_sections_info")

# # Change Airfoil's class, class atribute
# Airfoil.filePath = "BWB concept/Airfoils/"

# # BWB40_Sweep40deg
# filePath = "BWB/BWB40_Sweep40deg/"
# fileName = "BWB40_Sweep40deg_54_X_sections_info"
# data_dict = BWB_reader(filePath, fileName, scale=0.001)

# # Change Airfoil's class, class atribute
# Airfoil.filePath = "BWB/BWB40_Sweep40deg/Airfoils/"

# BWB40_Sweep40deg
filePath = "BWB/BWB40_Sweep40deg_flow5/"
fileName = "BWB40_Sweep40deg_flow5_54_X_sections_info"
data_dict = BWB_reader(filePath, fileName, scale=0.001)

# Change Airfoil's class, class atribute
Airfoil.filePath = "BWB/BWB40_Sweep40deg_flow5/Airfoils/"

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
# RefArea = 4.898

RefArea = 4.56

# inspect bwb mesh
n_x = 30
n_y = 1
nw_x = 50
chordwise_spacing = "cosine"
spanwise_spacing = "uniform"
shell_type = "quads"
triangular_wake_mesh = True
wake_length_in_chords = 10

nodes, shells, nodes_ids = bwb.mesh_body(
    ChordWise_resolution=n_x,
    SpanWise_resolution=n_y,
    wake_resolution=nw_x,
    ChordWise_spacing=chordwise_spacing,
    SpanWise_spacing=spanwise_spacing,
    shellType=shell_type,
    mesh_main_surface=True,
    mesh_tips=True,
    mesh_wake=True,
    triangular_wake_mesh=triangular_wake_mesh,
    wake_length_in_chords=wake_length_in_chords,
    planar_wake=True,
    V_fs=Vector((1, 0, 0)),
    standard_mesh_format=False
)

bwb_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

bwb_mesh.plot_mesh_bodyfixed_frame(
    elevation=-120, azimuth=-150, plot_wake=False
)



# Aerodynamic Analysis Parameters
AoA_start = -4
AoA_end = 12
AoA_step = 4

AoA_list = []
for i in range(int((AoA_end-AoA_start)/AoA_step)+1):
    AoA_list.append(AoA_start+i*AoA_step)

# Panel Method Parameters
SideSlipAngle = 0
V_fs = 1
steady_panel_method = Steady_PanelMethod(V_freestream=Vector((0, 0, 0)))
wake_rollup = True


csv_params_row_list = [
    ["BWB"],
    ["name:", bwb.name],
    ["number of input Xsections", len(bwb.wingXsection_list)],
    ["span:", (bwb.wingXsection_list[-1].r_leadingEdge.y
               - bwb.wingXsection_list[0].r_leadingEdge.y)],
    ["Reference Area:", RefArea],
    [],
    ["Mesh"],
    ["x body-shells:", n_x, "spacing", chordwise_spacing],
    ["y shells between Xsections:", n_y, "spacing:", spanwise_spacing],
    ["body shells type:", shell_type],
    ["number of body shells:",
    len(bwb_mesh.shells_ids["body"])],
    ["wake length in chords:", wake_length_in_chords],
    ["x wake-shells:", nw_x],
    ["triangular wake mesh:", triangular_wake_mesh],
    ["number of wake shells:",
    len(bwb_mesh.shells_ids["wake"])],
    ["number of shells:", bwb_mesh.shell_num],
    [],
    ["Panel Method"],
    ["type:", "steady", "wake roll-up:", wake_rollup],
    ["V_freestream:", V_fs],
    ["side slip angle:", SideSlipAngle],
    [],
    ["Results:"],
    []
]

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
    
    steady_panel_method.set_V_fs(V_fs, AoA, SideSlipAngle)
    
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
    
    nodes, shells, nodes_ids = bwb.mesh_body(
        ChordWise_resolution=n_x,
        SpanWise_resolution=n_y,
        wake_resolution=nw_x,
        ChordWise_spacing=chordwise_spacing,
        SpanWise_spacing=spanwise_spacing,
        shellType=shell_type,
        mesh_main_surface=True,
        mesh_tips=True,
        mesh_wake=True,
        triangular_wake_mesh=triangular_wake_mesh,
        wake_length_in_chords=wake_length_in_chords,
        planar_wake=True,
        V_fs=steady_panel_method.V_fs,
        standard_mesh_format=False
        )

    bwb_mesh = PanelAeroMesh(nodes, shells, nodes_ids)
    
    if wake_rollup:
        steady_panel_method.solve_iteratively(bwb_mesh, 4.56, 12, 10**(-6))
    else:
        # steady_panel_method.solve(bwb_mesh)
        steady_panel_method.solve_with_pressure_kutta(bwb_mesh)
    
    
    body_panels = [bwb_mesh.panels[id] for id in bwb_mesh.panels_ids["body"]]
    
    r_o = Vector((0, 0, 0)) 
    
    CL = steady_panel_method.LiftCoeff(body_panels, RefArea)
    CD = steady_panel_method.inducedDragCoeff(body_panels, RefArea)
    
    trefftz_CL, trefftz_CD = Trefftz_Plane_Analysis(
        bwb_mesh, steady_panel_method.V_fs, RefArea
    ) 
    
    Cm_o = Cm_about_point(r_o, body_panels, RefArea)
    CoP = Center_of_Pressure(body_panels, RefArea)
    
    csv_results_row_list.append(
        [
            AoA, CL, CD, trefftz_CL, trefftz_CD,
            Cm_o.x, Cm_o.y, Cm_o.z, CoP.x, CoP.y, CoP.z
        ]
    )

t = perf_counter()-t
print("total simulation time: " + str(t) + " seconds")

with open('Steady Aerodynamic Analysis Results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_params_row_list)
    writer.writerows([["total simulation time [sec]:", t], []])
    writer.writerows(csv_results_row_list)