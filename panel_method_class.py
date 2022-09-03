import numpy as np
from vector_class import Vector
from influence_coefficient_functions import Src_influence_coeff, Dblt_influence_coeff
from disturbance_velocity_functions import Src_disturb_velocity, Dblt_disturb_velocity, Vrtx_ring_disturb_velocity
from mesh_class import PanelMesh
from Algorithms import LeastSquares


class PanelMethod:
    
    def __init__(self, V_freestream):
        self.V_fs = V_freestream  
    
    def set_V_fs(self, Velocity, AngleOfAttack, SideslipAngle):
        alpha = np.deg2rad(AngleOfAttack)
        beta = np.deg2rad(SideslipAngle)
        Vx = Velocity * np.cos(alpha) * np.cos(beta)
        Vy = Velocity * np.cos(alpha) * np.sin(beta)
        Vz = - Velocity * np.sin(alpha)
        self.V_fs = Vector((Vx, Vy, Vz))


class Steady_Wakeless_PanelMethod(PanelMethod):
    
    def __init__(self, V_freestream):
        super().__init__(V_freestream)
    
    def solve(self, mesh:PanelMesh):
        
        for panel in mesh.panels:
            panel.sigma = source_strength(panel, self.V_fs) 
        
        B, C = self.influence_coeff_matrices(mesh.panels)
        
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
    
    @staticmethod
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


class Steady_PanelMethod(PanelMethod):
    
    def __init__(self, V_freestream):
        super().__init__(V_freestream)
    
    def solve(self, mesh:PanelMesh):
            
        body_panels = [mesh.panels[id] for id in mesh.panels_id["body"]]
        
        for panel in body_panels:
            panel.sigma = source_strength(panel, self.V_fs)
               
        A, B, C = self.influence_coeff_matrices(mesh)
        
        RHS = right_hand_side(body_panels, B)
        
        doublet_strengths = np.linalg.solve(A, RHS)
        
        for panel_i in (body_panels):
            
            panel_i.mu = doublet_strengths[panel_i.id]
            
            if panel_i.id in mesh.TrailingEdge["suction side"]:
                for id_j in mesh.wake_sheddingShells[panel_i.id]:
                    panel_j = mesh.panels[id_j]
                    panel_j.mu = panel_j.mu + doublet_strengths[panel_i.id]
                    
            elif panel_i.id in mesh.TrailingEdge["pressure side"]:
                for id_j in mesh.wake_sheddingShells[panel_i.id]:
                    panel_j = mesh.panels[id_j]
                    panel_j.mu = panel_j.mu - doublet_strengths[panel_i.id]
        

        # compute Velocity and pressure coefficient at panels' control points
        V_fs_norm = self.V_fs.norm()
        
        for panel in body_panels:
            
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

    @staticmethod
    def influence_coeff_matrices(mesh:PanelMesh):
        
        # Compute Influence coefficient matrices
        # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)
        
        Nb = len(mesh.panels_id["body"])
        Nw = len(mesh.panels_id["wake"])
        B = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb+Nw))
        A = np.zeros_like(B)
        
        # loop all over panels' control points
        for id_i in mesh.panels_id["body"]:
            panel_i = mesh.panels[id_i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for id_j in mesh.panels_id["body"]:
                
                panel_j = mesh.panels[id_j]
                B[id_i][id_j] = Src_influence_coeff(r_cp, panel_j)
                C[id_i][id_j] = Dblt_influence_coeff(r_cp, panel_j)
                A[id_i][id_j] = C[id_i][id_j]
                
                if id_j in mesh.TrailingEdge["suction side"]:
                    for id_k in mesh.wake_sheddingShells[id_j]:
                        panel_k = mesh.panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
                        
                elif id_j in mesh.TrailingEdge["pressure side"]:
                    for id_k in mesh.wake_sheddingShells[id_j]:
                        panel_k = mesh.panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                
        return A, B, C

   
# function definitions for functions used in solve method

def source_strength(panel, V_fs):
    # Katz & Plotkin eq 9.12
    # Katz & Plotkin defines normal vector n with opposite direction (inward)
    source_strength = - (panel.n * V_fs)
    return source_strength

def right_hand_side(body_panels, B):
    
    # Calculate right-hand-side 
    # Katz & Plotkin eq(12.35, 12.36)
        
    Nb = len(body_panels)
    RHS = np.zeros(Nb)
    
    # loop all over panels' control points
    for panel_i in body_panels:
        id_i = panel_i.id
        RHS[id_i] = 0
        
        # loop all over panels
        for panel_j in body_panels:
            id_j = panel_j.id
            RHS[id_i] = RHS[id_i] - panel_j.sigma * B[id_i][id_j]
            
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
    from mesh_class import PanelMesh
    from sphere import sphere
    from plot_functions import plot_Cp_SurfaceContours
    
    radius = 1
    num_longitude, num_latitude = 21, 20
    nodes, shells = sphere(radius, num_longitude, num_latitude,
                                     mesh_shell_type='quadrilateral')
    mesh = PanelMesh(nodes, shells)
    V_fs = Vector((1, 0, 0))
    panel_method = Steady_Wakeless_PanelMethod(V_fs)
    panel_method.solve(mesh)
    plot_Cp_SurfaceContours(mesh.panels)
        