import numpy as np
from vector_class import Vector
from panel_method_class import PanelMethod, Steady_Wakeless_PanelMethod
from matplotlib import pyplot as plt
from plot_functions import plot_Cp_SurfaceContours
from mesh_class import PanelMesh, Mesh
from sphere import sphere


radius = 1
num_longitude, num_latitude = 21, 20
nodes, shells = sphere(radius, num_longitude, num_latitude,
                                    mesh_shell_type='quadrilateral')

mesh = PanelMesh(nodes, shells)

# mesh = PanelMesh.generate_from_stl_file("Sphere_0080")


V_fs = Vector((1, 0, 0))
panel_method = Steady_Wakeless_PanelMethod(V_fs)
panel_method.solve(mesh)


# ax = plt.axes(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.view_init(0, 0)

# for panel in mesh.panels:
        
#     r_vertex = panel.r_vertex
    
#     # plot panels
#     if panel.num_vertices == 3:
#         x = [r_vertex[0].x, r_vertex[1].x, r_vertex[2].x, r_vertex[0].x]
#         y = [r_vertex[0].y, r_vertex[1].y, r_vertex[2].y, r_vertex[0].y]
#         z = [r_vertex[0].z, r_vertex[1].z, r_vertex[2].z, r_vertex[0].z]
#         ax.plot3D(x, y, z, color='k')
        
#     elif panel.num_vertices == 4:
        
#         x = [r_vertex[0].x, r_vertex[1].x, r_vertex[2].x, r_vertex[3].x,
#             r_vertex[0].x]
#         y = [r_vertex[0].y, r_vertex[1].y, r_vertex[2].y, r_vertex[3].y,
#             r_vertex[0].y]
#         z = [r_vertex[0].z, r_vertex[1].z, r_vertex[2].z, r_vertex[3].z,
#             r_vertex[0].z]
#         ax.plot3D(x, y, z, color='k') 
        
#     # plot normal vectors
#     r_cp = panel.r_cp
#     n = panel.n
#     scale = 0.2
#     n = n.scalar_product(scale)
#     if abs(r_cp.z) <= 10**(-5):
#         ax.scatter(r_cp.x, r_cp.y, r_cp.z, color='b', s=5)
#         ax.quiver(r_cp.x, r_cp.y, r_cp.z, n.x, n.y, n.z, color='b')
        
#     else:
#         ax.scatter(r_cp.x, r_cp.y, r_cp.z, color='k', s=5)
#         ax.quiver(r_cp.x, r_cp.y, r_cp.z, n.x, n.y, n.z, color='r')
        
# plt.show()


saved_ids = []
for panel in mesh.panels:
    if abs(panel.r_cp.z) <= 10**(-3):
        saved_ids.append(panel.id)

r = 1 # sphere radius
x0, y0 = 0, 0 # center of sphere
analytical_theta = np.linspace(-np.pi, np.pi, 200)
analytical_cp = 1 - (3/2*np.sin(analytical_theta))**2
fig = plt.figure()
plt.plot(analytical_theta*(180/np.pi), analytical_cp ,'b-',
            label='Analytical - sphere')
analytical_cp = 1 - 4 * np.sin(analytical_theta)**2
plt.plot(analytical_theta*(180/np.pi), analytical_cp ,'g-',
            label='Analytical - cylinder')

thetas = []
Cp = []
for id in saved_ids:
    # print(mesh.panels[id].r_cp)
    theta = np.arctan2(mesh.panels[id].r_cp.y, mesh.panels[id].r_cp.x)
    thetas.append(np.rad2deg(theta))
    Cp.append(mesh.panels[id].Cp)
    
plt.plot(thetas, Cp, 'ks', markerfacecolor='r',
            label='Panel Method - Sphere')
plt.xlabel("angle [degrees]")
plt.ylabel("Cp")
plt.title("Cp distribution along the great circle of a sphere")
plt.legend()
plt.grid()
plt.show()

# Surface Contour plot
plot_Cp_SurfaceContours(mesh.panels, elevation=20, azimuth=45)