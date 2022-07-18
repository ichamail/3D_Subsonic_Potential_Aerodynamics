import numpy as np
from matplotlib import pyplot as plt
from panel_class import Panel, quadPanel, triPanel, Vector
    

num_theta = 20
num_phi = 20
thetas = np.linspace(0, 2*np.pi, num_theta)
phis = np.linspace(0, np.pi, num_phi)
r = 1
coords = np.empty((num_theta, num_phi), dtype=tuple)

for i, theta in enumerate(thetas):
    for j, phi in enumerate(phis):
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta) 
        z = r * np.cos(phi)
        coords[i][j] = (x, y, z)

panels = []
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.view_init(0, 0)


# quadrilateral panels

# for i in range(num_theta):
#     for j in range(num_phi-1):
        
#         if j == 0:
#             vertex0 = Point(coords[i][j])
#             vertex1 = Point(coords[i][j+1])
#             vertex2 = Point(coords[(i+1)%num_theta][j+1])
#             panels.append(triPanel(vertex0, vertex1, vertex2))
            
#             x = [vertex0.x, vertex1.x, vertex2.x, vertex0.x]
#             y = [vertex0.y, vertex1.y, vertex2.y, vertex0.y]
#             z = [vertex0.z, vertex1.z, vertex2.z, vertex0.z]
#             ax.plot3D(x, y, z, color='k')
#             plt.draw()
#             plt.pause(0.1)
            
#         elif j == num_phi-1:
#             vertex0 = Point(coords[i][j])
#             vertex1 = Point(coords[(i+1)%num_theta][j+1])
#             vertex2 = Point(coords[(i+1)%num_theta][j])
#             panels.append(triPanel(vertex0, vertex1, vertex2))
            
#             x = [vertex0.x, vertex1.x, vertex2.x, vertex0.x]
#             y = [vertex0.y, vertex1.y, vertex2.y, vertex0.y]
#             z = [vertex0.z, vertex1.z, vertex2.z, vertex0.z]
#             ax.plot3D(x, y, z, color='k')
#             plt.draw()
#             plt.pause(0.01)
        
#         else:           
#             vertex0 = Point(coords[i][j])
#             vertex1 = Point(coords[i][j+1])
#             vertex2 = Point(coords[(i+1)%num_theta][j+1])
#             vertex3 = Point(coords[(i+1)%num_theta][j])
#             panels.append(quadPanel(vertex0, vertex1, vertex2, vertex3))
#             x = [vertex0.x, vertex1.x, vertex2.x, vertex3.x, vertex0.x]
#             y = [vertex0.y, vertex1.y, vertex2.y, vertex3.y, vertex0.y]
#             z = [vertex0.z, vertex1.z, vertex2.z, vertex3.z, vertex0.z]
            
#             ax.plot3D(x, y, z, color='k')
#             plt.draw()
#             plt.pause(0.01)
# plt.show()


### triangular panels

for i in range(num_theta):
    for j in range(num_phi-1):
        if j == 0:
            vertex0 = Vector(coords[i][j])
            vertex1 = Vector(coords[i][j+1])
            vertex2 = Vector(coords[(i+1)%num_theta][j+1])
            panels.append(triPanel(vertex0, vertex1, vertex2))
            
            x = [vertex0.x, vertex1.x, vertex2.x, vertex0.x]
            y = [vertex0.y, vertex1.y, vertex2.y, vertex0.y]
            z = [vertex0.z, vertex1.z, vertex2.z, vertex0.z]
            ax.plot3D(x, y, z, color='k')
            plt.draw()
            plt.pause(0.1)
        
        elif j == num_phi-1:
            vertex0 = Vector(coords[i][j])
            vertex1 = Vector(coords[(i+1)%num_theta][j+1])
            vertex2 = Vector(coords[(i+1)%num_theta][j])
            panels.append(triPanel(vertex0, vertex1, vertex2))
            
            x = [vertex0.x, vertex1.x, vertex2.x, vertex0.x]
            y = [vertex0.y, vertex1.y, vertex2.y, vertex0.y]
            z = [vertex0.z, vertex1.z, vertex2.z, vertex0.z]
            ax.plot3D(x, y, z, color='k')
            plt.draw()
            plt.pause(0.005)
           
        else:   
            vertex0 = Vector(coords[i][j])
            vertex1 = Vector(coords[i][j+1])
            vertex2 = Vector(coords[(i+1)%num_theta][j+1])
            vertex3 = Vector(coords[(i+1)%num_theta][j])
            
            panels.append(triPanel(vertex0, vertex1, vertex2))
            x = [vertex0.x, vertex1.x, vertex2.x, vertex0.x]
            y = [vertex0.y, vertex1.y, vertex2.y, vertex0.y]
            z = [vertex0.z, vertex1.z, vertex2.z, vertex0.z]
            ax.plot3D(x, y, z, color='k')
            plt.draw()
            plt.pause(0.01)
            
            panels.append(triPanel(vertex2, vertex3, vertex0))
            x = [vertex2.x, vertex3.x, vertex0.x, vertex2.x]
            y = [vertex2.y, vertex3.y, vertex0.y, vertex2.y]
            z = [vertex2.z, vertex3.z, vertex0.z, vertex2.z]
            ax.plot3D(x, y, z, color='k')
            plt.draw()
            plt.pause(0.01)        
        
        
plt.show()