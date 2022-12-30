import numpy as np
from airfoil_class import Airfoil
from vector_class import Vector
from bwb_class import WingCrossSection, BWB, BWB_reader
from mesh_class import PanelAeroMesh
from panel_method_class import UnSteady_PanelMethod, Trefftz_Plane_Analysis
from plot_functions import plot_Cp_SurfaceContours
from time import perf_counter


data_dict = BWB_reader(filePath="BWB/" , fileName= "BWB_X_sections_info")

# Change Airfoil's class, class atribute
Airfoil.filePath = "BWB/"

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
    ChordWise_resolution=20,
    SpanWise_resolution=1,
    SpanWise_spacing="uniform",
    shellType="quads",
    mesh_main_surface=True,
    mesh_tips=True,
    mesh_wake=False,
    standard_mesh_format=False
)

bwb_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

bwb_mesh.plot_mesh_bodyfixed_frame(
    elevation=-120, azimuth=-150, plot_wake=False
)

bwb_mesh.plot_mesh_inertial_frame(
    elevation=-120, azimuth=-150, plot_wake=False
)


bwb_mesh.set_body_fixed_frame_origin(xo=0, yo=0, zo=0)
bwb_mesh.set_body_fixed_frame_orientation(roll=0, pitch=np.deg2rad(0), yaw=0)

omega_roll, omega_pitch, omega_yaw = 0, 0, 0
bwb_mesh.set_angular_velocity(
    omega = Vector((omega_roll, omega_pitch, omega_yaw))
)
Vox, Voy, Voz = -1, 0, 0
bwb_mesh.set_origin_velocity(Vo=Vector((Vox, Voy, Voz)))


V_wind = Vector((0, 0, 0))
panel_method = UnSteady_PanelMethod(V_wind)
panel_method.set_wakePanelType("triangular")
panel_method.set_WakeShedFactor(0.3)

t_start = perf_counter()        
panel_method.solve(bwb_mesh, dt=0.15, iters=200)
# panel_method.solve_steady(bwb_mesh, RefArea=4.898, dt=0.15, max_iters=200)
t_end = perf_counter()
solution_time = t_end-t_start
print("solution time + compile time = " + str(solution_time))


# Surface Contour plot
body_panels = [bwb_mesh.panels[id] for id in bwb_mesh.panels_ids["body"]]
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-120)

CL = panel_method.LiftCoeff(body_panels, ReferenceArea=4.898)
CD = panel_method.inducedDragCoeff(body_panels, ReferenceArea=4.898)

print("CL = " + str(CL))
print("CD = " + str(CD))

CL_trefftz, CD_trefftz = Trefftz_Plane_Analysis(
    bwb_mesh, panel_method.V_fs, RefArea = 4.898
)

print("trefftz plane CL = " + str(CL_trefftz))
print("trefftz plane CD = " + str(CD_trefftz))