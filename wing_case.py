from airfoil_class import Airfoil
from wing_class import Wing
from wing_mesh_generator import generate_WingPanelMesh
from panel_method_class import PanelMethod
from vector_class import Vector

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# create wing object   
root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
tip_airfoil = Airfoil(name="naca0012_new", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0)

# generate wing mesh
num_x_bodyShells = 20
num_x_wakeShells = 40
num_y_Shells = 20

wing_mesh = generate_WingPanelMesh(wing, num_x_bodyShells,
                                        num_x_wakeShells, num_y_Shells)
wing_mesh.plot_panels(elevation=-150, azimuth=-120)


V_fs = Vector((1, 0, 0))
panel_method = PanelMethod(V_fs)



saved_ids = []
for panel in [wing_mesh.panels[id] for id in wing_mesh.panels_id["body"]]:
    if 0 < panel.r_cp.y < 0.1:
        saved_ids.append(panel.id)
        
panel_method.solve2(wing_mesh)
Cp = []
x = []
for id in saved_ids:
    Cp.append(wing_mesh.panels[id].Cp)
    x.append([wing_mesh.panels[id].r_cp.x])


plt.plot(x, Cp, 'ks', markerfacecolor='r', label='Panel Method')
plt.legend()
plt.grid()
plt.show()


shells = []
for panel in [wing_mesh.panels[id] for id in wing_mesh.panels_id["body"]]:
    shell=[]
    for r_vertex in panel.r_vertex:
        shell.append((r_vertex.x, r_vertex.y, r_vertex.z))
    
    shells.append(shell)

C = [panel.Cp for panel in [wing_mesh.panels[id]
                            for id in wing_mesh.panels_id["body"]]]

facecolor = plt.cm.coolwarm(C/np.max(C))

# print(facecolor)
    
ax = plt.axes(projection='3d')
poly3 = Poly3DCollection(shells, facecolor=facecolor)

ax.add_collection(poly3)
ax.set_xlabel('X')
ax.set_xlim3d(-0.5, 1.5)
ax.set_ylabel('Y')
ax.set_ylim3d(-1.5, 1.5)
ax.set_zlabel('Z')
ax.set_zlim3d(-1, 1)
plt.show()