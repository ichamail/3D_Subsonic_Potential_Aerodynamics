import numpy as np
from panel_class import Panel
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_Cp_SurfaceContours(panel_list, elevation=30, azimuth=-60):
  shells = []
  vert_coords = []
  for panel in panel_list:
      shell=[]
      for r_vertex in panel.r_vertex:
          shell.append((r_vertex.x, r_vertex.y, r_vertex.z))
          vert_coords.append([r_vertex.x, r_vertex.y, r_vertex.z])
      shells.append(shell)
  
  Cp = [panel.Cp for panel in panel_list]
  Cp_norm = [(float(Cp_i)-min(Cp))/(max(Cp)-min(Cp)) for Cp_i in Cp]
  facecolor = plt.cm.coolwarm(Cp_norm)
  
  ax = plt.axes(projection='3d')
  poly3 = Poly3DCollection(shells, facecolor=facecolor)
  ax.add_collection(poly3)
  ax.view_init(elevation, azimuth)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  vert_coords = np.array(vert_coords)
  x, y, z = vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2]
  ax.set_xlim3d(x.min(), x.max())
  ax.set_ylim3d(y.min(), y.max())
  ax.set_zlim3d(z.min(), z.max())
  set_axes_equal(ax)
  
  plt.show()