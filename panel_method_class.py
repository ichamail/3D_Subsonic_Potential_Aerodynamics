import numpy as np
from vector_class import Vector
from influence_coefficient_functions import Src_influence_coeff, Dblt_influence_coeff
from disturbance_velocity_functions import Src_disturb_velocity, Dblt_disturb_velocity, Vrtx_ring_disturb_velocity
from mesh_class import PanelMesh
from least_squares_method import LeastSquares


class PanelMethod:
    
    def __init__(self, V_freestream):
        self.V_fs = V_freestream
        
    
    def set_V_fs(self, Velocity, AoA):
        AoA = np.deg2rad(AoA)
        Vx = Velocity * np.cos(AoA)
        Vy = Velocity * np.sin(AoA)
        Vz = 0
        self.V_fs = Vector((Vx, Vy, Vz))
        
    def solve(self, mesh:PanelMesh):
        
        
        for panel in mesh.panels:
            panel.sigma = source_strength(panel, self.V_fs) 
        
        B, C = influence_coeff_matrices(mesh.panels)
        
        RHS = right_hand_side(mesh.panels, B)
        
        doublet_strengths = np.linalg.solve(C, RHS)
        
        for panel_id, panel in enumerate(mesh.panels):
            panel.mu = doublet_strengths[panel_id]
        

        # compute Velocity and pressure coefficient at panels' control points
        V_fs_norm = self.V_fs.norm()
        
        for panel in mesh.panels:
            
            # Velocity caclulation with least squares approach (faster)
            
            panel_neighbours = mesh.give_neighbours(panel)
            panel.Velocity = panel_velocity(panel, panel_neighbours, self.V_fs)
            
            
            # Velocity calculation with disturbance velocity functions
            
            # Δεν δουλεύει αυτή η μέθοδος. Δεν μπορώ να καταλάβω γιατ΄ί
            # Είναι πιο straight forward (σε αντίθεση με την παραπάνω μέθοδο
            # που απαιτεί προσεγγιστική επίλυση των gradients της έντασης μ)
            # παρ' ότι πολύ πιο αργή
            
            # panel.Velocity = Velocity(panel.r_cp, mesh.panels, self.V_fs)
            
            
            # pressure coefficient calculation
            panel.Cp = 1 - (panel.Velocity.norm()/V_fs_norm)**2
                   

def influence_coeff_matrices(panels):
    
    # Compute Influence coefficient matrices
    # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)
    
    n = len(panels)
    B = np.zeros((n, n))
    C = np.zeros_like(B)
    
    # loop all over panels' control points
    for i, panel_i in enumerate(panels):
        
        r_cp = panel_i.r_cp
        
        # loop all over panels
        for j, panel_j in enumerate(panels):
            B[i][j] = Src_influence_coeff(r_cp, panel_j)
            C[i][j] = Dblt_influence_coeff(r_cp, panel_j)
            
    return B, C
    

def source_strength(panel, V_fs):
    # Katz & Plotkin eq 9.12
    # Katz & Plotkin defines normal vector n with opposite direction (inward)
    source_strength = - (panel.n * V_fs)
    return source_strength

   
def right_hand_side(panels, B):
    
    # Calculate right-hand-side 
    # Katz & Plotkin eq(12.35, 12.36)
        
    n = len(panels)
    RHS = np.zeros(n)
    
    for i in range(n):
        RHS[i] = 0
        for j, panel in enumerate(panels):
            RHS[i] = RHS[i] - panel.sigma * B[i][j]

    return RHS


def panel_velocity(panel, panel_neighbours, V_fs):
    """
    V = Vx*ex + Vy*ey + Vz*ez or u*i + V*j  + w*k (global coordinate system)
    
    V = Vl*l + Vm*m + Vn*n (panel's local coordinate system)
    
    (r_ij * nabla)μ = μ_j - μ_i
    
    (r_ij * nabla)μ = Δl_ij*dμ/dl + Δm_ij*dμ/dm + Δn_ij*dμ/dn 
                    =~ Δl_ij*dμ/dl + Δm_ij*dμ/dm
    
    Vl = - dμ/dl, Vm = - dμ/dm, Vn = σ    (Katz & Plotkin eq.9.26 or eq.12.37)
    
    (r_ij * nabla)μ = Δl_ij(-Vl) + Δm_ij(-Vm) = μ_j - μ_i
    
    [[Δl_i1 , Δm_i1]                  [[μ_1 - μ_i]
     [Δl_i2 , Δm_i2]      [[-Vl]       [μ_2 - μ_i]
     [Δl_i3 , Δm_i3]  =    [-Vm]]  =   [μ_3 - μ_i]
          ....                            ....
     [Δl_iN , Δm_iN]]                  [μ_4 - μ_i]]
     
    least squares method ---> Vl, Vm
    """
    
    
    n = len(panel_neighbours)
    A = np.zeros((n, 2))
    b = np.zeros((n, 1))
    
    for j, neighbour in enumerate(panel_neighbours):
        r_ij = neighbour.r_cp - panel.r_cp
        r_ij = r_ij.transformation(panel.R)
        A[j][0] = r_ij.x
        A[j][1] = r_ij.y
        b[j][0] = neighbour.mu - panel.mu
    
    del_mu = LeastSquares(A, b)
    components = (-del_mu[0][0], -del_mu[1][0], panel.sigma)
    

    V_disturb = Vector(components)
    V_disturb = V_disturb.transformation(panel.R.T)
    
    V = V_fs + V_disturb
    
    return V

              
def Velocity(r_p, panels, V_fs):
    # Velocity calculation with disturbance velocity functions
    
    velocity = V_fs
    
    for panel in panels:
        
        # Doublet panels
        velocity = (velocity
                    + Src_disturb_velocity(r_p, panel)
                    + Dblt_disturb_velocity(r_p, panel))
        
        # Vortex ring panels
        # velocity = (velocity
        #             + Src_disturb_velocity(r_p, panel)
        #             + Vrtx_ring_disturb_velocity(r_p, panel))
    
    
    return velocity


               
        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mesh_class import PanelMesh
    from sphere import sphere
    
    radius = 1
    num_longitude, num_latitude = 21, 20
    nodes, shells = sphere(radius, num_longitude, num_latitude,
                                     mesh_shell_type='quadrilateral')
    mesh = PanelMesh(nodes, shells)
    mesh.CreatePanels()
    
    V_fs = Vector((1, 0, 0))
    panel_method = PanelMethod(V_fs)
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
    
    plt.legend()
    plt.grid()
    plt.show()
    