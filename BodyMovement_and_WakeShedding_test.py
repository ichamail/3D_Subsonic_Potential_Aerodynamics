import numpy as np
from vector_class import Vector
from airfoil_class import Airfoil
from wing_class import Wing
from mesh_class import PanelMesh, PanelAeroMesh
from wing_mesh_generator import generate_WingPanelMesh

     
root_airfoil = Airfoil(name="naca0012_new", chord_length=1)
tip_airfoil = Airfoil(name="naca0012_new", chord_length=1)
wing = Wing(root_airfoil, tip_airfoil, semi_span=1, sweep=0, dihedral=0,
            twist=0)

num_x_bodyShells = 4
num_y_Shells = 2
num_x_wakeShells = 0


wing_mesh = generate_WingPanelMesh(wing, num_x_bodyShells, num_x_wakeShells,
                                   num_y_Shells)


wing_mesh.set_body_fixed_frame_origin(0, 0, 0)
roll, pitch, yaw = np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)
wing_mesh.set_body_fixed_frame_orientation(roll, pitch, yaw)

# wing_mesh.plot_shells(elevation=-150, azimuth=-120)
# wing_mesh.plot_panels(elevation=-150, azimuth=-120)
wing_mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120,
                                    plot_wake=False)
wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120, plot_wake=False)

wing_mesh.set_body_fixed_frame_origin(xo=0, yo=0, zo=0)
wing_mesh.set_body_fixed_frame_orientation(roll=0, pitch=0, yaw=0)

omega = Vector((np.deg2rad(10), 0, 0))
Vo = Vector((-1, 0, 0))
V_wind = Vector((0, 0, 0))
dt = 1
wing_mesh.set_angular_velocity(omega)
wing_mesh.set_origin_velocity(Vo)

# induced_velocity_function = ( lambda r_p,
#                              body_panels,
#                              wake_panels : Vector((0, 0, -r_p.x * 0.1 )) )

for i in range(5):
    
    wing_mesh.move_body(dt)
    wing_mesh.shed_wake(V_wind, dt)
    # wing_mesh.convect_wake(induced_velocity_function, dt)
    # wing_mesh.plot_shells(elevation=-150, azimuth=-120)
    # wing_mesh.plot_panels(elevation=-150, azimuth=-120)
    wing_mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120,
                                       plot_wake=True)    
    wing_mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120,
                                        plot_wake=True)
    
  
    