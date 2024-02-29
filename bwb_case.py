from airfoil_class import Airfoil
from vector_class import Vector
from bwb_class import WingCrossSection, BWB, BWB_reader
from mesh_class import PanelAeroMesh
from panel_method_class import Steady_PanelMethod, Trefftz_Plane_Analysis, local_Cl, local_CL
from plot_functions import plot_Cp_SurfaceContours
from matplotlib import pyplot as plt



# # RX3 conceptual
# filePath="BWB/BWB concepts/" 
# fileName= "BWB_X_sections_info"

# # Change Airfoil's class, class atribute
# Airfoil.filePath = "BWB/BWB concepts/Airfoils"

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


# bwb.subdivide_spanwise_sections(3, interpolation_type="linear")

panel_method = Steady_PanelMethod(V_freestream=Vector((0, 0, 0)))

panel_method.set_V_fs(Velocity=1, AngleOfAttack=-4, SideslipAngle=0)

nodes, shells, nodes_ids = bwb.mesh_body(
    ChordWise_resolution=30,
    SpanWise_resolution=1,
    ChordWise_spacing="cosine",
    SpanWise_spacing="uniform",
    shellType="quads",
    mesh_main_surface=True,
    mesh_tips=True,
    mesh_wake=True,
    triangular_wake_mesh=True,
    wake_resolution=50,
    planar_wake=True, 
    V_fs=panel_method.V_fs,
    wake_length_in_chords=10,
    standard_mesh_format=False
)


bwb_mesh = PanelAeroMesh(nodes, shells, nodes_ids)

bwb_mesh.plot_mesh_bodyfixed_frame(
    elevation=-120, azimuth=-150, plot_wake=False
)

bwb_mesh.plot_mesh_bodyfixed_frame(
    elevation=-120, azimuth=-150, plot_wake=True
)

# panel_method.solve_with_pressure_kutta(bwb_mesh)
# panel_method.solve(bwb_mesh)
panel_method.solve_iteratively(bwb_mesh, 4.56, 7, 10**(-6))



# Surface Contour plot
body_panels = [bwb_mesh.panels[id] for id in bwb_mesh.panels_ids["body"]]
plot_Cp_SurfaceContours(body_panels, elevation=-90, azimuth=0)
plot_Cp_SurfaceContours(body_panels, elevation=90, azimuth=180)
plot_Cp_SurfaceContours(body_panels, elevation=-178, azimuth=-120)
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-120)
plot_Cp_SurfaceContours(body_panels, elevation=-150, azimuth=-70)
plot_Cp_SurfaceContours(body_panels, elevation=150, azimuth=-70)

CL = panel_method.LiftCoeff(body_panels, ReferenceArea=4.56)
CD = panel_method.inducedDragCoeff(body_panels, ReferenceArea=4.56)

print("CL = " + str(CL))
print("CD = " + str(CD))

CL_trefftz, CD_trefftz = Trefftz_Plane_Analysis(
    bwb_mesh, panel_method.V_fs, RefArea = 4.56
)

print("trefftz plane CL = " + str(CL_trefftz))
print("trefftz plane CD = " + str(CD_trefftz))

################ Results ###############

# Pressure Coefficient Distribution across root's airfoil section 
near_root_panels = bwb_mesh.give_leftSide_near_root_panels()

Cp = [panel.Cp for panel in near_root_panels]
x = [panel.r_cp.x for panel in near_root_panels]


plt.plot(x, Cp, 'ks--', markerfacecolor='r', label='Panel Method')
plt.xlabel("x/c")
plt.ylabel("Cp")
plt.title("Pressure Coefficient plot at root")
plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.show()

near_tip_panels = bwb_mesh.give_leftSide_near_tip_panels()
Cp = [panel.Cp for panel in near_tip_panels]
x = [panel.r_cp.x for panel in near_tip_panels]

plt.plot(x, Cp, 'ks--', markerfacecolor='r', label='Panel Method')
plt.xlabel("x/c")
plt.ylabel("Cp")
plt.title("Pressure Coefficient plot at tip")
plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.show()


local_lift = local_Cl(bwb_mesh, panel_method.V_fs)
x = local_lift[:, 0]
y = local_lift[:, 1]
plt.plot(x, y, 'ks--', markerfacecolor='r')
plt.xlabel("span percentage")
plt.ylabel("cl")
plt.title("cl spanwise distribution")
plt.show()


local_Lift = local_CL(bwb_mesh, panel_method.V_fs, MAC=bwb.MAC)
x = local_Lift[:, 0]
y = local_Lift[:, 1]
plt.plot(x, y)
plt.plot(x, y, 'ks--', markerfacecolor='r')
plt.xlabel("span percentage")
plt.ylabel("cl * c/MAC")
plt.title("cl * c/MAC spanwise distribution")
plt.show()

bwb_mesh.plot_mesh_inertial_frame(elevation=180, azimuth= 0,
                                   plot_wake = True)
bwb_mesh.plot_mesh_inertial_frame(elevation=-175, azimuth=180,
                                   plot_wake = True)
bwb_mesh.plot_mesh_inertial_frame(elevation=-160, azimuth= -30,
                                   plot_wake = True)
bwb_mesh.plot_mesh_inertial_frame(elevation=-180, azimuth= -90,
                                   plot_wake = True)