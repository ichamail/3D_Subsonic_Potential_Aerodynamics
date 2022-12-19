import numpy as np
import csv
from airfoil_class import Airfoil
from vector_class import Vector
from bwb_class import WingCrossSection, BWB
from mesh_class import PanelMesh, PanelAeroMesh
from panel_method_class import Steady_PanelMethod, Trefftz_Plane_Analysis
from plot_functions import plot_Cp_SurfaceContours
from matplotlib import pyplot as plt


def load_airfoil(filePath, fileName, header_lines=1):
        
    # Load the data from the text file
    fileName = filePath + fileName
    dataBuffer = np.loadtxt(fileName, delimiter=' ', skiprows=header_lines)
    
    # Extract data from the loaded dataBuffer array
    dataX = np.asarray(dataBuffer[:,0])
    dataY = np.asarray(dataBuffer[:,1])
    
    return dataX, dataY

def BWB_reader(filePath, fileName):
    fileName = filePath + fileName


    with open("BWB\BWB_X_sections_info") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        
        data_list = [row for row in csv_reader]

    for row_id in range(len(data_list)):
        for col_id in range(len(data_list[row_id])):
            if row_id !=0:
                if col_id == 0:
                    data_list[row_id][col_id] = int(data_list[row_id][col_id])
                elif col_id == 1:
                    pass
                else:
                    data_list[row_id][col_id] = float(data_list[row_id][col_id])
    
    data_list = data_list[1:]
    
    data_dict = {
        "airfoil name": [data_list[i][1] for i in range(len(data_list))],
        "chord": [10**(-3) * data_list[i][2] for i in range(len(data_list))],
        "leading edge coords": [
            (10**(-3) * data_list[i][3], 10**(-3) * data_list[i][4], 10**(-3) * data_list[i][5])
            for i in range(len(data_list))],
        "twist": [data_list[i][6] for i in range(len(data_list))]
    }            
    
    return data_dict

data_dict = BWB_reader(filePath="BWB/" , fileName= "BWB_X_sections_info")

Airfoil.filePath = "BWB/"

RX3 = BWB(
    name="RX3",
    wingXsection_list=[
        WingCrossSection(
            Vector(data_dict["leading edge coords"][id]),
            chord=data_dict["chord"][id],
            twist=data_dict["twist"][id],
            airfoil=Airfoil(
                name=data_dict["airfoil name"][id],
                CCW_order=False
            )
        )
        
        for id in range(len(data_dict["airfoil name"]))
             
        if (data_dict["airfoil name"][id] != "Airfoil_13" 
            and data_dict["airfoil name"][id] != "Airfoil_17" 
            and data_dict["airfoil name"][id] != "Airfoil_19" 
            and data_dict["airfoil name"][id] != "Airfoil_20")
        
         
    ]
)


panel_method = Steady_PanelMethod(V_freestream=Vector((0, 0, 0)))

panel_method.set_V_fs(Velocity=1, AngleOfAttack=0, SideslipAngle=0)

nodes, shells, nodes_ids = RX3.mesh_body(
    ChordWise_resolution=15,
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

# rx3_mesh = PanelMesh(nodes, shells) 

rx3_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

rx3_mesh.plot_mesh_bodyfixed_frame(elevation=-120, azimuth=-150, plot_wake=True)

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