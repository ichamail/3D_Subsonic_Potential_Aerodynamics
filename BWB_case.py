import numpy as np
import csv
from airfoil_class import Airfoil
from vector_class import Vector
from bwb_class import WingCrossSection, BWB, BWB_reader
from mesh_class import PanelMesh, PanelAeroMesh
from panel_method_class import Steady_PanelMethod, Trefftz_Plane_Analysis
from plot_functions import plot_Cp_SurfaceContours
from matplotlib import pyplot as plt


data_dict = BWB_reader(filePath="BWB/" , fileName= "BWB_X_sections_info")

# Change Airfoil's class, class atribute
Airfoil.filePath = "BWB/"

RX3 = BWB(
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

panel_method = Steady_PanelMethod(V_freestream=Vector((0, 0, 0)))

panel_method.set_V_fs(Velocity=1, AngleOfAttack=0, SideslipAngle=0)

nodes, shells, nodes_ids = RX3.mesh_body(
    ChordWise_resolution=20,
    SpanWise_resolution=1,
    SpanWise_spacing="uniform",
    shellType="quads",
    mesh_main_surface=True,
    mesh_tips=True,
    mesh_wake=True,
    wake_resolution=1,
    planar_wake=True,
    V_fs=panel_method.V_fs,
    standard_mesh_format=False
)

rx3_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

rx3_mesh.plot_mesh_bodyfixed_frame(
    elevation=-120, azimuth=-150, plot_wake=False
)

panel_method.solve(rx3_mesh)

# Surface Contour plot
body_panels = [rx3_mesh.panels[id] for id in rx3_mesh.panels_ids["body"]]
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-120)

CL = panel_method.LiftCoeff(body_panels, ReferenceArea=6.8)
CD = panel_method.inducedDragCoeff(body_panels, ReferenceArea=6.8)

print("CL = " + str(CL))
print("CD = " + str(CD))

CL_trefftz, CD_trefftz = Trefftz_Plane_Analysis(
    rx3_mesh, panel_method.V_fs, RefArea = 6.8
)

print("trefftz plane CL = " + str(CL_trefftz))
print("trefftz plane CD = " + str(CD_trefftz))