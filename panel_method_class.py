import numpy as np
from vector_class import Vector
from influence_coefficient_functions import Src_influence_coeff, Dblt_influence_coeff, influence_coeff
from disturbance_velocity_functions import Src_disturb_velocity, Dblt_disturb_velocity, Vrtx_ring_disturb_velocity
from mesh_class import PanelMesh, PanelAeroMesh
from Algorithms import LeastSquares
from numba import typed, jit, prange
from python_object_to_numba_object import pyObjToNumbaObj


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

    def LiftCoeff(self, panels, ReferenceArea):
        C_force = AerodynamicForce(panels, ReferenceArea)
        CL_vec = LiftCoefficient(C_force, self.V_fs)
        CL = CL_vec.norm()
        return CL

    def inducedDragCoeff(self, panels, ReferenceArea):
        C_force = AerodynamicForce(panels, ReferenceArea)
        CD_vec = inducedDragCoefficient(C_force, self.V_fs)
        CD = CD_vec.norm()
        return CD
    

class Steady_Wakeless_PanelMethod(PanelMethod):
    
    def __init__(self, V_freestream):
        super().__init__(V_freestream)
    
    def solve(self, mesh:PanelMesh):
        
        for panel in mesh.panels:
            panel.sigma = source_strength(panel, self.V_fs) 
        
        # B, C = self.influence_coeff_matrices(mesh.panels)
        B, C = self.influence_coeff_matrices(typed.List(mesh.panels))
        
        # RHS = right_hand_side(mesh.panels, B)
        RHS = right_hand_side(typed.List(mesh.panels), B)
        
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
            
            # panel.Velocity = Velocity(self.V_fs, panel.r_cp, self.panels)
            
            
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
                # B[i][j] = Src_influence_coeff(r_cp, panel_j)
                # C[i][j] = Dblt_influence_coeff(r_cp, panel_j)
                B[i][j], C[i][j] = influence_coeff(r_cp, panel_j)
                
        return B, C

    @staticmethod
    @jit(nopython=True)
    def influence_coeff_matrices(panels):
        
        # Compute Influence coefficient matrices
        # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)
        
        n = len(panels)
        B = np.zeros((n, n))
        C = np.zeros_like(B)
        
        # loop all over panels' control points
        for i in prange(n):
            panel_i = panels[i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for j in prange(n):
                panel_j = panels[j]
                # B[i][j] = Src_influence_coeff(r_cp, panel_j)
                # C[i][j] = Dblt_influence_coeff(r_cp, panel_j)
                B[i][j], C[i][j] = influence_coeff(r_cp, panel_j)
                
        return B, C

    
class Steady_PanelMethod(PanelMethod):
    
    def __init__(self, V_freestream):
        super().__init__(V_freestream)
    
    def solve(self, mesh:PanelAeroMesh):
            
        body_panels = [mesh.panels[id] for id in mesh.panels_id["body"]]
        wake_panels = [mesh.panels[id] for id in mesh.panels_id["wake"]]
        
        for panel in body_panels:
            panel.sigma = source_strength(panel, self.V_fs)
               
        A, B, C = self.influence_coeff_matrices(mesh)
        
        # RHS = right_hand_side(body_panels, B)
        RHS = right_hand_side(typed.List(body_panels), B)
        
        doublet_strengths = np.linalg.solve(A, RHS)
        
        for panel_i in (body_panels):
            
            panel_i.mu = doublet_strengths[panel_i.id]
            
            if panel_i.id in mesh.TrailingEdge["suction side"]:
                for id_j in mesh.wake_sheddingPanels[panel_i.id]:
                    panel_j = mesh.panels[id_j]
                    panel_j.mu = panel_j.mu + doublet_strengths[panel_i.id]
                    
            elif panel_i.id in mesh.TrailingEdge["pressure side"]:
                for id_j in mesh.wake_sheddingPanels[panel_i.id]:
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
            
            # panel.Velocity = Velocity(self.V_fs, panel.r_cp, body_panels,
            #                           wake_panels)
            
            
            # pressure coefficient calculation
            panel.Cp = 1 - (panel.Velocity.norm()/V_fs_norm)**2

    @staticmethod
    def influence_coeff_matrices(mesh:PanelAeroMesh):
        
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
                # B[id_i][id_j] = Src_influence_coeff(r_cp, panel_j)
                # C[id_i][id_j] = Dblt_influence_coeff(r_cp, panel_j)
                B[id_i][id_j], C[id_i][id_j] = influence_coeff(r_cp, panel_j)
                A[id_i][id_j] = C[id_i][id_j]
                
                if id_j in mesh.TrailingEdge["suction side"]:
                    for id_k in mesh.wake_sheddingPanels[id_j]:
                        panel_k = mesh.panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
                        
                elif id_j in mesh.TrailingEdge["pressure side"]:
                    for id_k in mesh.wake_sheddingPanels[id_j]:
                        panel_k = mesh.panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                
        return A, B, C

    def influence_coeff_matrices(self, mesh:PanelMesh):
        
        # Compute Influence coefficient matrices
        # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)
              
        panels = typed.List(mesh.panels)
        panels_id = pyObjToNumbaObj(mesh.panels_id)
        TrailingEdge = pyObjToNumbaObj(mesh.TrailingEdge)
        wake_sheddingShells = pyObjToNumbaObj(mesh.wake_sheddingShells)
                
        A, B, C = self.jit_influence_coeff_matrices(panels, panels_id,
                                                TrailingEdge, wake_sheddingShells)
        
        return A, B, C

    @staticmethod
    @jit(nopython=True, parallel=True)
    def jit_influence_coeff_matrices(panels:list, panels_id:dict,
                                TrailingEdge:dict, wake_sheddingShells:dict):
        
        Nb = len(panels_id["body"])
        Nw = len(panels_id["wake"])
        B = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb+Nw))
        A = np.zeros_like(B)
        
        # loop all over panels' control points
        for i in prange(Nb):
            id_i = panels_id["body"][i]
            panel_i = panels[id_i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for j in prange(Nb):
                id_j = panels_id["body"][j]
                panel_j = panels[id_j]
                
                # B[id_i][id_j] = Src_influence_coeff(r_cp, panel_j)
                # C[id_i][id_j] = Dblt_influence_coeff(r_cp, panel_j)
                B[id_i][id_j], C[id_i][id_j] = influence_coeff(r_cp, panel_j)
                A[id_i][id_j] = C[id_i][id_j]
                
                if id_j in TrailingEdge["suction side"]:
                    for id_k in wake_sheddingShells[id_j]:
                        panel_k = panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
                            
                elif id_j in TrailingEdge["pressure side"]:
                    for id_k in wake_sheddingShells[id_j]:
                        panel_k = panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                    
        return A, B, C


class UnSteady_PanelMethod(PanelMethod):
    
    def __init__(self, V_wind=Vector((0, 0, 0))):
        # V_wind: wind velocity observed from inertial frame of reference F
        
        super().__init__(V_freestream=Vector((0, 0, 0)))
        self.V_wind = V_wind
        self.set_V_fs(Vo=Vector((0, 0, 0)), V_wind=self.V_wind)
        self.wake_shed_factor = 0.3
        self.dt = 0.1
    
    def set_V_wind(self, Velocity, alpha, beta):
        # alpha: angle between X-axis & Y-axis of inertial frame of reference F
        # beta: angle between X-axis & Y-axis of inertial frame of reference F
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        Vx = Velocity * np.cos(alpha) * np.cos(beta)
        Vy = Velocity * np.cos(alpha) * np.sin(beta)
        Vz = - Velocity * np.sin(alpha)
        self.V_wind = Vector((Vx, Vy, Vz))
    
    def set_V_fs(self, Vo, V_wind):
        # Vo: Velocity vector of body-fixed frame's of reference (f') origin, observed from inertial frame of reference F
        # V_wind: wind velocity vector observed from inertial frame of reference F
        self.V_wind = V_wind
        self.V_fs = V_wind - Vo
            
    def set_WakeShedFactor(self, wake_shed_factor):
        self.wake_shed_factor = wake_shed_factor
    
    def advance_solution(self, mesh:PanelAeroMesh):
                    
        body_panels = [mesh.panels[id] for id in mesh.panels_id["body"]]
        wake_panels = [mesh.panels[id] for id in mesh.panels_id["wake"]]
        body_panels = typed.List(body_panels)
        wake_panels = typed.List(wake_panels)
                
        for panel in body_panels:
            
            v_rel = self.V_wind  # velocity relative to inertial frame (V_wind)
            r_cp = panel.r_cp
            # v = v_rel - (mesh.Vo + Vector.cross_product(mesh.omega, r_cp))
            v = v_rel - (mesh.Vo + mesh.omega.cross(r_cp))
            # V_wind - Vo = V_fs
            # v = self.V_fs - Vector.cross_product(mesh.omega, r_cp)
            v = v.transformation(mesh.R.T)           
            panel.sigma = source_strength(panel, v)
               
        A, B, C = self.influence_coeff_matrices(mesh)
        
        RHS = right_hand_side(body_panels, B)
        RHS = RHS + additional_right_hand_side(body_panels, wake_panels, C)
                
        doublet_strengths = np.linalg.solve(A, RHS)
        
        doublet_strength_old = np.zeros(len(body_panels))
        for panel_i in body_panels:
            
            doublet_strength_old[panel_i.id] = panel_i.mu
            
            panel_i.mu = doublet_strengths[panel_i.id]
            
            if panel_i.id in mesh.TrailingEdge["suction side"]:
                id_j = mesh.wake_sheddingPanels[panel_i.id][-1]
                panel_j = mesh.panels[id_j]
                panel_j.mu = panel_j.mu + doublet_strengths[panel_i.id]
                    
            elif panel_i.id in mesh.TrailingEdge["pressure side"]:
                id_j = mesh.wake_sheddingPanels[panel_i.id][-1]
                panel_j = mesh.panels[id_j]
                panel_j.mu = panel_j.mu - doublet_strengths[panel_i.id]
        

        # compute Velocity and pressure coefficient at panels' control points
               
        for panel in body_panels:
            
            v_rel = self.V_wind  # velocity relative to inertial frame (V_wind)
            r_cp = panel.r_cp
            # v = v_rel - (mesh.Vo + Vector.cross_product(mesh.omega, r_cp))
            v = v_rel - (mesh.Vo + mesh.omega.cross(r_cp))
            # V_wind - Vo = V_fs
            # v = self.V_fs - Vector.cross_product(mesh.omega, r_cp)
            v = v.transformation(mesh.R.T)
            
            # Velocity caclulation with least squares approach (faster)
            
            panel_neighbours = mesh.give_neighbours(panel)
            panel.Velocity = panel_velocity(panel, panel_neighbours, v)
            
          
            # Velocity calculation with disturbance velocity functions
            
            # Δεν δουλεύει αυτή η μέθοδος. Δεν μπορώ να καταλάβω γιατ΄ί
            # Είναι πιο straight forward (σε αντίθεση με την παραπάνω μέθοδο
            # που απαιτεί προσεγγιστική επίλυση των gradients της έντασης μ)
            # παρ' ότι πολύ πιο αργή
            
            # panel.Velocity = Velocity(v, panel.r_cp, body_panels,
            #                           wake_panels)
            
            # pressure coefficient calculation
            # Katz & Plotkin eq. 13.168
            
            panel.Cp = 1 - (panel.Velocity.norm()/v.norm())**2
            
            # dφ/dt = dμ/dt
            phi_dot = (panel.mu - doublet_strength_old[panel.id])/self.dt
            
            panel.Cp = panel.Cp - 2 * phi_dot
    
    def solve(self, mesh:PanelAeroMesh, dt, iters):
        self.dt = dt
        self.set_V_fs(mesh.Vo, self.V_wind)
        
        for i in range(iters):
            mesh.move_body(dt)
            mesh.shed_wake(self.V_wind, dt, self.wake_shed_factor)
            self.advance_solution(mesh)
            mesh.convect_wake(induced_velocity, dt)
            
            # mesh.convect_wake(jit_induced_velocity, dt)
            
            
            # mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120,
            #                                plot_wake=True)
            # mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120,
            #                               plot_wake=True)
            
        
        # mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120,
        #                                plot_wake=True)
        # mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120,
        #                               plot_wake=True)
           
    @staticmethod
    def influence_coeff_matrices(mesh:PanelAeroMesh):
        
        # Compute Influence coefficient matrices
        # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)
        
        Nb = len(mesh.panels_id["body"])
        Nw = len(mesh.panels_id["wake"])
        Nte = len(mesh.TrailingEdge["suction side"])
        B = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb + Nw))
        A = np.zeros_like(B)
        
        # loop all over panels' control points
        for id_i in mesh.panels_id["body"]:
            panel_i = mesh.panels[id_i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for id_j in mesh.panels_id["body"]:
                
                panel_j = mesh.panels[id_j]
                # B[id_i][id_j] = Src_influence_coeff(r_cp, panel_j)
                # C[id_i][id_j] = Dblt_influence_coeff(r_cp, panel_j)
                B[id_i][id_j], C[id_i][id_j] = influence_coeff(r_cp, panel_j)
                A[id_i][id_j] = C[id_i][id_j]
                
                if id_j in mesh.TrailingEdge["suction side"]:
                    for id_k in mesh.wake_sheddingPanels[id_j]:
                        panel_k = mesh.panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        if id_k == mesh.wake_sheddingPanels[id_j][-1]:
                            A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
                        
                elif id_j in mesh.TrailingEdge["pressure side"]:
                    for id_k in mesh.wake_sheddingPanels[id_j]:
                        panel_k = mesh.panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        if id_k == mesh.wake_sheddingPanels[id_j][-1]:
                            A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                
        return A, B, C

    def influence_coeff_matrices(self, mesh:PanelAeroMesh):
        
        # Compute Influence coefficient matrices
        # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)      
        
        panels = typed.List(mesh.panels)
        panels_id = pyObjToNumbaObj(mesh.panels_id)
        TrailingEdge = pyObjToNumbaObj(mesh.TrailingEdge)
        wake_sheddingPanels = pyObjToNumbaObj(mesh.wake_sheddingPanels)
        
        A, B, C = self.jit_influence_coeff_matrices(panels, panels_id,
                                                    TrailingEdge, wake_sheddingPanels)
                
        return A, B, C
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def jit_influence_coeff_matrices(panels:list, panels_id:dict,
                                TrailingEdge:dict, wake_sheddingPanels:dict):
        
        Nb = len(panels_id["body"])
        Nw = len(panels_id["wake"])
        B = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb + Nw))
        A = np.zeros_like(B)
        
    
        # loop all over panels' control points
        for id_i in panels_id["body"]:
            panel_i = panels[id_i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for id_j in panels_id["body"]:
                
                panel_j = panels[id_j]
                # B[id_i][id_j] = Src_influence_coeff(r_cp, panel_j)
                # C[id_i][id_j] = Dblt_influence_coeff(r_cp, panel_j)
                B[id_i][id_j], C[id_i][id_j] = influence_coeff(r_cp, panel_j)
                A[id_i][id_j] = C[id_i][id_j]
                
                if id_j in TrailingEdge["suction side"]:
                    for id_k in wake_sheddingPanels[id_j]:
                        panel_k = panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        if id_k == wake_sheddingPanels[id_j][-1]:
                            A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
                        
                elif id_j in TrailingEdge["pressure side"]:
                    for id_k in wake_sheddingPanels[id_j]:
                        panel_k = panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        if id_k == wake_sheddingPanels[id_j][-1]:
                            A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                
        return A, B, C

    
# function definitions for functions used in solve method

def source_strength(panel, V_fs):
    # Katz & Plotkin eq 9.12
    # Katz & Plotkin defines normal vector n with opposite direction (inward)
    # source_strength = - (panel.n * V_fs)
    source_strength = - (V_fs.dot(panel.n))
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

@jit(nopython=True, parallel = True)
def right_hand_side(body_panels, B):
    
    # Calculate right-hand-side 
    # Katz & Plotkin eq(12.35, 12.36)
        
    Nb = len(body_panels)
    RHS = np.zeros(Nb)
    
    # loop all over panels' control points
    for i in prange(Nb):
        panel_i = body_panels[i]
        id_i = panel_i.id
        RHS[id_i] = 0
        
        # loop all over panels
        for j in prange(Nb):
            panel_j = body_panels[j]
            id_j = panel_j.id
            RHS[id_i] = RHS[id_i] - panel_j.sigma * B[id_i][id_j]
            
    return RHS

def additional_right_hand_side(body_panels, wake_panels, C):
    
    Nb = len(body_panels)
    RHS = np.zeros(Nb)
    
    # loop all over panels' control points
    for panel_i in body_panels:
        id_i = panel_i.id
        
        # loop all over wake panels
        for panel_j in wake_panels:
            """
            Κανονικά πρέπει η <<λούπα>> πρέπει να περιλαμβάνει μόνο τα πάνελ του απόρρου της προγούμενη επανάλληψης καθώς μόνο γι αυτά έχει υπολογιστεί η τιμή της έντασης τους. Παρ' όλα αυτά αφού στα νέα πάνελ ορίζεται μηδενική τιμή έντασης κατά την ορισμό τους, δεν θα συμβάλουν στο δεξί μέλος
            """
            id_j = panel_j.id
            RHS[id_i] = RHS[id_i] - panel_j.mu * C[id_i][id_j]
            
    
    # Nb, _ = C.shape
    # RHS = np.zeros(Nb)
    
    # # loop all over body panels' control points
    # for id_i in range(Nb):
        
    #     # loop all over wake panels 
    #     for panel_j in wake_panels:
    #         id_j = panel_j.id
    #         RHS[id_i] = RHS[id_j] - panel_j.mu * C[id_i][id_j]
    
    return RHS

@jit(nopython=True, parallel=True)
def additional_right_hand_side(body_panels, wake_panels, C):
    
    Nb = len(body_panels)
    Nw = len(wake_panels)
    RHS = np.zeros(Nb)
    
    # loop all over panels' control points
    for i in prange(Nb):
        panel_i = body_panels[i]
        id_i = panel_i.id
        
        # loop all over wake panels
        for j in prange(Nw):
            """
            Κανονικά πρέπει η <<λούπα>> πρέπει να περιλαμβάνει μόνο τα πάνελ του απόρρου της προγούμενη επανάλληψης καθώς μόνο γι αυτά έχει υπολογιστεί η τιμή της έντασης τους. Παρ' όλα αυτά αφού στα νέα πάνελ ορίζεται μηδενική τιμή έντασης κατά την ορισμό τους, δεν θα συμβάλουν στο δεξί μέλος
            """ 
            panel_j = wake_panels[j]
            id_j = panel_j.id
            RHS[id_i] = RHS[id_i] - panel_j.mu * C[id_i][id_j]
            
    
    # without using body_panels list as function argument
    # Nb, _ = C.shape
    # Nw = len(wake_panels)
    # RHS = np.zeros(Nb)
    
    # # loop all over body panels' control points
    # for id_i in prange(Nb):
        
    #     # loop all over wake panels 
    #     for j in prange(Nw):
    #         panel_j = wake_panels[j] 
    #         id_j = panel_j.id
    #         RHS[id_i] = RHS[id_j] - panel_j.mu * C[id_i][id_j]
    
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

def body_induced_velocity(r_p, body_panels):
    # Velocity calculation with disturbance velocity functions
    
    velocity = Vector((0, 0, 0))
    
    for panel in body_panels:
        
        # Doublet panels
        velocity = (velocity
                    + Src_disturb_velocity(r_p, panel)
                    + Dblt_disturb_velocity(r_p, panel))
        
        # Vortex ring panels
        # velocity = (velocity
        #             + Src_disturb_velocity(r_p, panel)
        #             + Vrtx_ring_disturb_velocity(r_p, panel))
        
    
    return velocity

@jit(nopython=True, parallel=False)
def body_induced_velocity(r_p, body_panels):
    # Velocity calculation with disturbance velocity functions
    
    velocity = Vector((0, 0, 0))
    
    for i in prange(len(body_panels)):
        panel = body_panels[i]
        # Doublet panels
        velocity = (velocity
                    + Src_disturb_velocity(r_p, panel)
                    + Dblt_disturb_velocity(r_p, panel))
        
        # velocity += Src_disturb_velocity(r_p, panel)
        # velocity += Dblt_disturb_velocity(r_p, panel)
        
        # Vortex ring panels
        # velocity = (velocity
        #             + Src_disturb_velocity(r_p, panel)
        #             + Vrtx_ring_disturb_velocity(r_p, panel))
        
    
    return velocity

def wake_induce_velocity(r_p, wake_panels):
    """
    Vortex ring panels handle distortion. In unsteady solution method, because of the wake roll-up, hence the distortion of the wake panels, we can use vortex rings or triangular panels to surpass this problem
    """
    velocity = Vector((0, 0, 0))
    
    for panel in wake_panels:
        
        # Doublet panels
        # velocity = velocity + Dblt_disturb_velocity(r_p, panel)

        # Vortex ring panels
    
        velocity = velocity + Vrtx_ring_disturb_velocity(r_p, panel)
    
    return velocity

@jit(nopython=True, parallel=False)
def wake_induce_velocity(r_p, wake_panels):
    """
    Vortex ring panels handle distortion. In unsteady solution method, because of the wake roll-up, hence the distortion of the wake panels, we can use vortex rings or triangular panels to surpass this problem
    """
    velocity = Vector((0, 0, 0))
    
    for i in prange(len(wake_panels)):
        panel = wake_panels[i]
        
        # Doublet panels
        # velocity = velocity + Dblt_disturb_velocity(r_p, panel)

        # Vortex ring panels
        velocity = velocity + Vrtx_ring_disturb_velocity(r_p, panel)
        # velocity += Vrtx_ring_disturb_velocity(r_p, panel)
        
    return velocity

def induced_velocity(r_p, body_panels, wake_panels=[]):
    
    body_panels = typed.List(body_panels)
    
    if wake_panels == []:
        velocity = body_induced_velocity(r_p, body_panels)
    else:
        wake_panels = typed.List(wake_panels)
        
        velocity = (body_induced_velocity(r_p, body_panels) 
                    + wake_induce_velocity(r_p, wake_panels))
    
    return velocity
          
def Velocity(V_fs, r_p, body_panels, wake_panels=[]):
    # Velocity calculation with disturbance velocity functions
    
    velocity = V_fs + induced_velocity(r_p, body_panels, wake_panels)
      
    return velocity

@jit(nopython=True, parallel=True)
def jit_body_induce_velocity(r_p, body_panels):
    velocity_array = np.zeros((3, len(body_panels)))
            
    for i in prange(len(body_panels)):
        velocity = Src_disturb_velocity(r_p, body_panels[i]) + \
            Vrtx_ring_disturb_velocity(r_p, body_panels[i])
        velocity_array[0][i] = velocity.x
        velocity_array[1][i] = velocity.y
        velocity_array[2][i] = velocity.z
        
    vx = np.sum(velocity_array[0])
    vy = np.sum(velocity_array[1])
    vz = np.sum(velocity_array[2])
    velocity = Vector((vx, vy, vz))
    return velocity

@jit(nopython=True, parallel=True)
def jit_wake_induced_velocity(r_p, wake_panels):
    velocity_array = np.zeros((3, len(wake_panels)))
            
    for i in prange(len(wake_panels)):
        velocity = Vrtx_ring_disturb_velocity(r_p, wake_panels[i])
        velocity_array[0][i] = velocity.x
        velocity_array[1][i] = velocity.y
        velocity_array[2][i] = velocity.z
    
    vx = np.sum(velocity_array[0])
    vy = np.sum(velocity_array[1])
    vz = np.sum(velocity_array[2])
    velocity = Vector((vx, vy, vz))
    
    return velocity

def jit_induced_velocity(r_p, body_panels, wake_panels=[]):
    
    velocity = Vector((0, 0, 0))
    
    if wake_panels == []:
        body_panels = typed.List(body_panels)
        
        velocity = jit_body_induce_velocity(r_p, body_panels)
    else:
        
        body_panels = typed.List(body_panels)
        wake_panels = typed.List(wake_panels)
        
        velocity = jit_body_induce_velocity(r_p, body_panels) \
            + jit_wake_induced_velocity(r_p, wake_panels)
    
    return velocity
   
def AerodynamicForce(panels, ReferenceArea):
    ref_area = ReferenceArea
    C_force = Vector((0, 0, 0))
    for panel in panels:
        # C_force = C_force + (-panel.Cp*panel.area/ref_area)*panel.n
        C_force = C_force + panel.n * (-panel.Cp*panel.area/ref_area)
    return C_force

def inducedDragCoefficient(AerodynamicForce, V_fs):
    C_force = AerodynamicForce
    # CD_vector = (C_force * V_fs/V_fs.norm()) * V_fs/V_fs.norm()
    CD_vector = V_fs/V_fs.norm() * (C_force.dot(V_fs/V_fs.norm()))
    return CD_vector

def LiftCoefficient(AerodynamicForce, V_fs):
    C_force = AerodynamicForce
    CD_vector = inducedDragCoefficient(C_force, V_fs)
    CL_vector = C_force - CD_vector
    return CL_vector

        
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
        