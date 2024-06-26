from re import S
import numpy as np
from vector_class import Vector
from panel_class import Panel, quadPanel, triPanel
from disturbance_velocity_functions import Src_disturb_velocity
from matplotlib import pyplot as plt

vertex1 = Vector((1, 1, 0))
vertex2 = Vector((-1, 1, 0))
vertex3 = Vector((-1, -1, 0))
vertex4 = Vector((1, -1, 0))

# Quadrilateral panel
panel = quadPanel(vertex1, vertex2, vertex3, vertex4)
panel.sigma = 100


# # Triangular panel
# panel = triPanel(vertex1, vertex2, vertex3)
# panel.sigma = 700

# Panel plot

ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.view_init(90, 0)
x = []
y = []
z = []
for i in range(panel.num_vertices+1):
    i = i % panel.num_vertices
    x.append(panel.r_vertex[i].x)
    y.append(panel.r_vertex[i].y)
    z.append(panel.r_vertex[i].z)
    
ax.plot3D(x, y, z, color='k', label='panel')

ax.quiver(panel.r_cp.x, panel.r_cp.y, panel.r_cp.z,
            panel.n.x, panel.n.y, panel.n.z,
            color='r', label='normal vector n')
ax.quiver(panel.r_cp.x, panel.r_cp.y, panel.r_cp.z,
            panel.l.x, panel.l.y, panel.l.z,
            color='b', label='longidutinal vector l')
ax.quiver(panel.r_cp.x, panel.r_cp.y, panel.r_cp.z,
            panel.m.x, panel.m.y, panel.m.z,
            color='g', label='transverse vector m')

ax.legend()



# velocity field plot

x = np.linspace(-3, 3, 5)
y = np.linspace(-3, 3, 5)
z = np.linspace(-3, 3, 5)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

nx, ny, nz = X.shape

# for i in range(nx):
#     for j in range(ny):
#         for k in range(nz):
#             r_p = Vector((X[i,j,k], Y[i,j,k], Z[i, j, k]))
#             v = Src_disturb_velocity(r_p, panel)
#             ax.quiver(r_p.x, r_p.y, r_p.z,
#                         v.x, v.y, v.z,
#                         color='m')


V = np.empty_like(X, dtype=Vector)
V = np.empty_like(X, dtype=Vector)
u = np.zeros_like(X, dtype=float)
v = np.zeros_like(X, dtype=float)
w = np.zeros_like(X, dtype=float)

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            r_p = Vector((X[i,j,k], Y[i,j,k], Z[i, j, k]))
            V[i, j, k] = Src_disturb_velocity(r_p, panel)
            u[i, j, k] = V[i, j, k].x
            v[i, j, k] = V[i, j, k].y
            w[i, j, k] = V[i, j, k].z

ax.quiver(X, Y, Z, u, v, w, normalize=True, color='m')

plt.show()



"""
Data provided from the paper "A Comprehensive and Practical Guide to the Hess and Smith Constant Source and Dipole Panel" of Lothar Birk
"""

from influence_coefficient_functions import Src_influence_coeff

vertex1 = Vector((-0.5, -0.5, 0))
vertex2 = Vector((0.5, -0.5, 0))
vertex3 = Vector((0.5, 0.5, 0))
vertex4 = Vector((-0.5, 0.5, 0))

# Quadrilateral panel
panel = quadPanel(vertex1, vertex2, vertex3, vertex4)
panel.sigma = 1
r_p_list = [Vector((0, 0, 2)),
            Vector((0, 0, -0.5)),
            Vector((0.5, 1, 0)),
            Vector((-0.25, 0.25, 0.5)),
            Vector((0.1, 0.4, 8))]

print("Panel's information \n...........\n"
      + "position vectors of vertices: \n"
      + "r1 = " + str(panel.r_vertex[0]) + "\n"
      + "r2 = " + str(panel.r_vertex[1]) + "\n"
      + "r3 = " + str(panel.r_vertex[2]) + "\n"
      + "r4 = " + str(panel.r_vertex[3]) + "\n"
      + "Source strength sigma = " + str(panel.sigma) + "\n\n")
for r_p in r_p_list:
    V, phi = Src_disturb_velocity(r_p, panel), Src_influence_coeff(r_p, panel)
    print(
        "rp = " + str(r_p) + " : \n" 
        + "V = " + str(V) + "\n" 
        + "phi = " + str(phi) + "\n" 
    )