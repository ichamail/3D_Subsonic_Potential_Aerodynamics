import numpy as np
from vector_class import Vector
from influence_coefficient_functions import Src_influence_coeff, Dblt_influence_coeff, influence_coeff
from NASA4023_VSAERO_theory import Src_NASA4023, Dblt_NASA4023
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
        if CL_vec.z <=0:
            CL = CL_vec.norm()
        else:
            CL = -CL_vec.norm()
        return CL

    def inducedDragCoeff(self, panels, ReferenceArea):
        C_force = AerodynamicForce(panels, ReferenceArea)
        CD_vec = inducedDragCoefficient(C_force, self.V_fs)
        if CD_vec.x >= 0:
            CD = CD_vec.norm()
        else:
            CD = -CD_vec.norm()    
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
            
            # panel.Velocity = Velocity(self.V_fs, panel.r_cp, mesh.panels)
            
            
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
            
        body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
        wake_panels = [mesh.panels[id] for id in mesh.panels_ids["wake"]]
        
        for panel in body_panels:
            panel.sigma = source_strength(panel, self.V_fs)
               
        A, B, C = self.influence_coeff_matrices(mesh)
        
        # RHS = right_hand_side(body_panels, B)
        RHS = right_hand_side(typed.List(body_panels), B)
        
        doublet_strengths = np.linalg.solve(A, RHS)
        
        for panel_i in (body_panels):
            panel_i.mu = doublet_strengths[panel_i.id]
                    
        for id_i in mesh.TrailingEdge["suction side"]:
            for id_j in mesh.wake_sheddingPanels[id_i]:
                panel_j = mesh.panels[id_j]
                panel_j.mu = panel_j.mu + doublet_strengths[id_i]
        
        for id_i in mesh.TrailingEdge["pressure side"]:
            for id_j in mesh.wake_sheddingPanels[id_i]:
                panel_j = mesh.panels[id_j]
                panel_j.mu = panel_j.mu - doublet_strengths[id_i]
                

        # compute Velocity and pressure coefficient at panels' control points
        # V_fs_norm = self.V_fs.norm()
        
        # for panel in body_panels:
            
        #     # Velocity caclulation with least squares approach (faster)
            
        #     panel_neighbours = mesh.give_neighbours(panel)
        #     panel.Velocity = panel_velocity(panel, panel_neighbours, self.V_fs)
            
          
        #     # Velocity calculation with disturbance velocity functions
            
        #     # Δεν δουλεύει αυτή η μέθοδος. Δεν μπορώ να καταλάβω γιατ΄ί
        #     # Είναι πιο straight forward (σε αντίθεση με την παραπάνω μέθοδο
        #     # που απαιτεί προσεγγιστική επίλυση των gradients της έντασης μ)
        #     # παρ' ότι πολύ πιο αργή
            
        #     # panel.Velocity = Velocity(self.V_fs, panel.r_cp, body_panels,
        #     #                           wake_panels)
            
            
        #     # pressure coefficient calculation
        #     panel.Cp = 1 - (panel.Velocity.norm()/V_fs_norm)**2
            
        # on-body Analysis based on "Program VSAERO Theory Document"
        VSAERO_onbody_analysis(self.V_fs, mesh)

    @staticmethod
    def influence_coeff_matrices(mesh:PanelAeroMesh):
        
        # Compute Influence coefficient matrices
        # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)
        
        Nb = len(mesh.panels_ids["body"])
        Nw = len(mesh.panels_ids["wake"])
        B = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb+Nw))
        A = np.zeros_like(B)
        
        # loop all over panels' control points
        for id_i in mesh.panels_ids["body"]:
            panel_i = mesh.panels[id_i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for id_j in mesh.panels_ids["body"]:
                
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
        panels_ids = pyObjToNumbaObj(mesh.panels_ids)
        TrailingEdge = pyObjToNumbaObj(mesh.TrailingEdge)
        wake_sheddingShells = pyObjToNumbaObj(mesh.wake_sheddingShells)
                
        A, B, C = self.jit_influence_coeff_matrices(panels, panels_ids,
                                                TrailingEdge, wake_sheddingShells)
        
        return A, B, C

    @staticmethod
    @jit(nopython=True, parallel=True)
    def jit_influence_coeff_matrices(panels:list, panels_ids:dict,
                                TrailingEdge:dict, wake_sheddingShells:dict):
        
        Nb = len(panels_ids["body"])
        Nw = len(panels_ids["wake"])
        B = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb+Nw))
        A = np.zeros_like(B)
        
        # loop all over panels' control points
        for i in prange(Nb):
            id_i = panels_ids["body"][i]
            panel_i = panels[id_i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for j in prange(Nb):
                id_j = panels_ids["body"][j]
                panel_j = panels[id_j]
                
                # B[id_i][id_j] = Src_influence_coeff(r_cp, panel_j)
                # C[id_i][id_j] = Dblt_influence_coeff(r_cp, panel_j)
                B[id_i][id_j], C[id_i][id_j] = influence_coeff(r_cp, panel_j)
                A[id_i][id_j] = C[id_i][id_j]
                        
            for j in prange(len(TrailingEdge["suction side"])):
                id_j = TrailingEdge["suction side"][j]
                for id_k in wake_sheddingShells[id_j]:
                    panel_k = panels[id_k]
                    C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                    A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
            
            for j in prange(len(TrailingEdge["pressure side"])):
                id_j = TrailingEdge["pressure side"][j]
                for id_k in wake_sheddingShells[id_j]:
                    panel_k = panels[id_k]
                    C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                    A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                    
        return A, B, C

    def solve_iteratively(self, mesh:PanelAeroMesh, RefArea, dt, max_iters,
                          convergence_value = 10**(-5)):
                
        ny, nx = mesh.nodes_ids["wake lines"].shape
        CL_prev, CD_prev = 0, 0
        
        for it in range(max_iters-1):
            
            print("iteration: ", it)
            
            self.solve(mesh)
            
            # Parallel = False
            # body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
            # wake_panels = [mesh.panels[id] for id in mesh.panels_ids["wake"]]
            # old_nodes = mesh.nodes.copy()
            # for j in range(nx-1):
            #     for i in range(ny):
            #         node_id = mesh.nodes_ids["wake lines"][i][j]
            #         r = Vector(old_nodes[node_id])
            #         v = Velocity(self.V_fs, r, body_panels, wake_panels)
            #         dr = v * dt
            #         r = r + dr
            #         node_id = mesh.nodes_ids["wake lines"][i][j+1]
            #         mesh.nodes[node_id] = (r.x, r.y, r.z)
            
            
            # Parallel = True
            r_list = [
                Vector(mesh.nodes[mesh.nodes_ids["wake lines"][i][j]])
                for j in range(nx-1) for i in range(ny)
            ]

            velocity_list = jit_induced_velocity_function(r_list, mesh.panels)
            
            k = 0        
            for j in range(nx-1):
                for i in range(ny):
                    r = r_list[k]
                    dr = (velocity_list[k] + self.V_fs) * dt
                    r = r + dr
                    node_id = mesh.nodes_ids["wake lines"][i][j+1]
                    mesh.nodes[node_id] = (r.x, r.y, r.z)
                    
                    k = k + 1
            
            mesh.jit_update_wake_panel_vertices()
            
            
            CL = self.LiftCoeff(mesh.panels, RefArea)
            CD = self.inducedDragCoeff(mesh.panels, RefArea)
            
            print("CL = " + str(CL) + ",  CD = " + str(CD) + "\n")
            
            dCL_dt = (CL-CL_prev)/dt
            dCD_dt = (CD-CD_prev)/dt
            
            print("dCL/dt = " + str(dCL_dt) +
                  ",  dCD/dt = " + str(dCD_dt) + "\n")
                      
            
            # reset panel strengths
            for panel in mesh.panels:
                panel.sigma, panel.mu, panel.Cp = 0.0, 0.0, 0.0
                panel.Velocity = Vector((0.0, 0.0, 0.0))
                        
            if ( abs(dCL_dt) <= convergence_value
                and abs(dCD_dt) <= convergence_value ):
                
                print("solution converged \n")
                
                break
            
            CL_prev, CD_prev = CL, CD
            
            # mesh.plot_mesh_bodyfixed_frame(
            #     elevation=-150, azimuth=-120, plot_wake=True
            # )
        
        print("final iteration: ", it+1)
        self.solve(mesh)
        
        mesh.plot_mesh_bodyfixed_frame(
            elevation=-150, azimuth=-120, plot_wake=True
        )
        

class UnSteady_PanelMethod(PanelMethod):
    
    def __init__(self, V_wind=Vector((0, 0, 0))):
        # V_wind: wind velocity observed from inertial frame of reference F
        
        super().__init__(V_freestream=Vector((0, 0, 0)))
        self.V_wind = V_wind
        self.set_V_fs(Vo=Vector((0, 0, 0)), V_wind=self.V_wind)
        self.wake_shed_factor = 0.3
        self.triangular_wakePanels = False
        self.dt = 0.1
    
    def set_wakePanelType(self, type:str):
        if type == "triangular":
            self.triangular_wakePanels = True
        elif type == "quadrilateral":
            self.triangular_wakePanels = False
        else:
            self.triangular_wakePanels = False            
    
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
    
    def LiftCoeff(self, mesh, ReferenceArea):
        body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
        C_force = AerodynamicForce(body_panels, ReferenceArea)
        V_fs = self.V_fs.transformation(mesh.R.T)
        CL_vec = LiftCoefficient(C_force, V_fs)
        if CL_vec.z <=0:
            CL = CL_vec.norm()
        else:
            CL = -CL_vec.norm()
        return CL
    
    def inducedDragCoeff(self, mesh, ReferenceArea):
        body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
        C_force = AerodynamicForce(body_panels, ReferenceArea)
        V_fs = self.V_fs.transformation(mesh.R.T)
        CD_vec = inducedDragCoefficient(C_force, V_fs)
        if CD_vec.x >= 0:
            CD = CD_vec.norm()
        else:
            CD = -CD_vec.norm() 
        return CD
               
    def set_WakeShedFactor(self, wake_shed_factor):
        self.wake_shed_factor = wake_shed_factor
    
    def advance_solution(self, mesh:PanelAeroMesh):
                    
        body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]
        wake_panels = [mesh.panels[id] for id in mesh.panels_ids["wake"]]
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
                    
        for id_i in mesh.TrailingEdge["suction side"]:
            id_j = mesh.wake_sheddingPanels[id_i][-1]
            panel_j = mesh.panels[id_j]
            panel_j.mu = panel_j.mu + doublet_strengths[id_i]
            if self.triangular_wakePanels:
                id_j = mesh.wake_sheddingPanels[id_i][-2]
                panel_j = mesh.panels[id_j]
                panel_j.mu = panel_j.mu + doublet_strengths[id_i]
                
        for id_i in mesh.TrailingEdge["pressure side"]:
            id_j = mesh.wake_sheddingPanels[id_i][-1]
            panel_j = mesh.panels[id_j]
            panel_j.mu = panel_j.mu - doublet_strengths[id_i]
            if self.triangular_wakePanels:
                id_j = mesh.wake_sheddingPanels[id_i][-2]
                panel_j = mesh.panels[id_j]
                panel_j.mu = panel_j.mu - doublet_strengths[id_i]       

        # compute Velocity and pressure coefficient at panels' control points
               
        # for panel in body_panels:
            
        #     v_rel = self.V_wind  # velocity relative to inertial frame (V_wind)
        #     r_cp = panel.r_cp
        #     # v = v_rel - (mesh.Vo + Vector.cross_product(mesh.omega, r_cp))
        #     v = v_rel - (mesh.Vo + mesh.omega.cross(r_cp))
        #     # V_wind - Vo = V_fs
        #     # v = self.V_fs - Vector.cross_product(mesh.omega, r_cp)
        #     v = v.transformation(mesh.R.T)
            
        #     # Velocity caclulation with least squares approach (faster)
            
        #     panel_neighbours = mesh.give_neighbours(panel)
        #     panel.Velocity = panel_velocity(panel, panel_neighbours, v)
            
          
        #     # Velocity calculation with disturbance velocity functions
            
        #     # Δεν δουλεύει αυτή η μέθοδος. Δεν μπορώ να καταλάβω γιατ΄ί
        #     # Είναι πιο straight forward (σε αντίθεση με την παραπάνω μέθοδο
        #     # που απαιτεί προσεγγιστική επίλυση των gradients της έντασης μ)
        #     # παρ' ότι πολύ πιο αργή
            
        #     # panel.Velocity = Velocity(v, panel.r_cp, body_panels,
        #     #                           wake_panels)
            
        #     # pressure coefficient calculation
        #     # Katz & Plotkin eq. 13.168
            
        #     panel.Cp = 1 - (panel.Velocity.norm()/v.norm())**2
            
        #     # dφ/dt = dμ/dt
        #     phi_dot = (panel.mu - doublet_strength_old[panel.id])/self.dt
            
        #     panel.Cp = panel.Cp - 2 * phi_dot

        VSAERO_usteady_onbody_analysis(
            self.V_wind, mesh, doublet_strength_old, self.dt
        )
        
        
    def solve(self, mesh:PanelAeroMesh, dt, iters):
        if self.triangular_wakePanels:
            type = "triangular"
        else:
            type = "quadrilateral"
            
        self.dt = dt
        self.set_V_fs(mesh.Vo, self.V_wind)
        
        for i in range(iters):
            print(i)
            mesh.move_body(self.dt)
            
            mesh.shed_wake(self.V_wind, self.dt, self.wake_shed_factor, type)
            # mesh.ravel_wake(self.V_wind, self.dt, type)  # if shed factor = 1
            
            self.advance_solution(mesh)
            
            # mesh.convect_wake(induced_velocity, dt)  # not parallel
            mesh.jit_convect_wake(jit_induced_velocity_function, dt) # parallel
            
            
            # mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120,
            #                                plot_wake=True)
            # mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120,
            #                               plot_wake=True)
            
        
        # mesh.plot_mesh_bodyfixed_frame(elevation=-150, azimuth=-120,
        #                                plot_wake=True)
        # mesh.plot_mesh_inertial_frame(elevation=-150, azimuth=-120,
        #                               plot_wake=True)
        pass
    
    def solve_steady(self, mesh:PanelAeroMesh, RefArea, dt, max_iters,
                     convergence_value = 10**(-3)):
        
        if self.triangular_wakePanels:
            type = "triangular"
        else:
            type = "quadrilateral"
            
        self.dt = dt
        self.set_V_fs(mesh.Vo, self.V_wind)
        
        CL_prev, CD_prev = 0, 0
        
        for i in range(max_iters):
            
            print("iteration: " + str(i) + "\n")
            
            mesh.move_body(self.dt)
            
            mesh.shed_wake(self.V_wind, self.dt, self.wake_shed_factor, type)
            # mesh.ravel_wake(self.V_wind, self.dt, type)  # if shed factor = 1
            
            self.advance_solution(mesh)
            
            # mesh.convect_wake(induced_velocity, dt)  # not parallel
            mesh.jit_convect_wake(jit_induced_velocity_function, dt) # parallel
            
            CL = self.LiftCoeff(mesh, RefArea)
            CD = self.inducedDragCoeff(mesh, RefArea)
            
            print("CL = " + str(CL) + ",  CD = " + str(CD) + "\n")
            
            dCL_dt = (CL-CL_prev)/dt
            dCD_dt = (CD-CD_prev)/dt
            
            print("dCL/dt = " + str(dCL_dt) +
                  ",  dCD/dt = " + str(dCD_dt) + "\n")
                        
            if ( abs(dCL_dt) <= convergence_value
                and abs(dCD_dt) <= convergence_value ):
                
                print("solution converged \n")
                
                break
            
            CL_prev, CD_prev = CL, CD 

    def influence_coeff_matrices(self, mesh:PanelAeroMesh):
        
        # Compute Influence coefficient matrices
        # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)
        
        Nb = len(mesh.panels_ids["body"])
        Nw = len(mesh.panels_ids["wake"])
        Nte = len(mesh.TrailingEdge["suction side"])
        B = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb + Nw))
        A = np.zeros_like(B)
        
        # loop all over panels' control points
        for id_i in mesh.panels_ids["body"]:
            panel_i = mesh.panels[id_i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for id_j in mesh.panels_ids["body"]:
                
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
                        elif (id_k == mesh.wake_sheddingPanels[id_j][-2]
                              and self.triangular_wakePanels):
                            A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
                        
                elif id_j in mesh.TrailingEdge["pressure side"]:
                    for id_k in mesh.wake_sheddingPanels[id_j]:
                        panel_k = mesh.panels[id_k]
                        C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                        if id_k == mesh.wake_sheddingPanels[id_j][-1]:
                            A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                        elif (id_k == mesh.wake_sheddingPanels[id_j][-2]
                              and self.triangular_wakePanels):
                            A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                
        return A, B, C

    def influence_coeff_matrices(self, mesh:PanelAeroMesh):
        
        # Compute Influence coefficient matrices
        # Katz & Plotkin eq(9.24, 9.25) or eq(12.34, 12.35)      
        
        panels = typed.List(mesh.panels)
        panels_ids = pyObjToNumbaObj(mesh.panels_ids)
        TrailingEdge = pyObjToNumbaObj(mesh.TrailingEdge)
        wake_sheddingPanels = pyObjToNumbaObj(mesh.wake_sheddingPanels)
        
        if self.triangular_wakePanels:
            wake_type = "triangular"
        else:
            wake_type = "quadrilateral"
        
        A, B, C = self.jit_influence_coeff_matrices(panels, panels_ids,
                                                    TrailingEdge, wake_sheddingPanels,
                                                    wake_type)
                
        return A, B, C
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def jit_influence_coeff_matrices(panels:list, panels_ids:dict,
                                TrailingEdge:dict, wake_sheddingPanels:dict,
                                wake_type:str):
        
        Nb = len(panels_ids["body"])
        Nw = len(panels_ids["wake"])
        B = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb + Nw))
        A = np.zeros_like(B)
        
        # loop all over panels' control points
        for i in prange(Nb):
            id_i = panels_ids["body"][i]
            panel_i = panels[id_i]
            r_cp = panel_i.r_cp
            
            # loop all over panels
            for j in prange(Nb):
                id_j = panels_ids["body"][j]
                panel_j = panels[id_j]
                
                # B[id_i][id_j] = Src_influence_coeff(r_cp, panel_j)
                # C[id_i][id_j] = Dblt_influence_coeff(r_cp, panel_j)
                B[id_i][id_j], C[id_i][id_j] = influence_coeff(r_cp, panel_j)
                A[id_i][id_j] = C[id_i][id_j]
                                            
            for j in prange(len(TrailingEdge["suction side"])):
                id_j = TrailingEdge["suction side"][j]
                for id_k in wake_sheddingPanels[id_j]:
                    panel_k = panels[id_k]
                    C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                    if id_k == wake_sheddingPanels[id_j][-1]:
                        A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
                    elif (id_k == wake_sheddingPanels[id_j][-2]
                            and wake_type=="triangular"):
                        A[id_i][id_j] = A[id_i][id_j] + C[id_i][id_k]
                        
            for j in prange(len(TrailingEdge["pressure side"])):
                id_j = TrailingEdge["pressure side"][j]
                for id_k in wake_sheddingPanels[id_j]:
                    panel_k = panels[id_k]
                    C[id_i][id_k] = Dblt_influence_coeff(r_cp, panel_k)
                    if id_k == wake_sheddingPanels[id_j][-1]:
                        A[id_i][id_j] = A[id_i][id_j] - C[id_i][id_k]
                    elif (id_k == wake_sheddingPanels[id_j][-2]
                        and wake_type=="triangular"):
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

def additional_right_hand_side(body_panels, wake_panels, C):
    
    Nb = len(body_panels)
    RHS = np.zeros(Nb)
    
    # loop all over panels' control points
    for panel_i in body_panels:
        id_i = panel_i.id
        
        # loop all over wake panels
        """
        Κανονικά πρέπει η <<λούπα>> πρέπει να περιλαμβάνει μόνο τα πάνελ του απόρρου της προγούμενη επανάλληψης καθώς μόνο γι αυτά έχει υπολογιστεί η τιμή της έντασης τους. Παρ' όλα αυτά αφού στα νέα πάνελ ορίζεται μηδενική τιμή έντασης κατά την ορισμό τους, δεν θα συμβάλουν στο δεξί μέλος
        """
        for panel_j in wake_panels:
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

def VSAERO_panel_velocity(V_fs, panel, panel_neighbours, is_neighbour_1=True,
                          is_neighbour_2=True, is_neighbour_3=True, is_neighbour_4=True):

    """
    this function computes the surface velocity, following the notation of NASA Contractor Report 4023 "Program VSAERO theory Document,
    A Computer Program for Calculating Nonlinear Aerodynamic Characteristics
    of Arbitrary Configurations, Brian Maskew"
    
    check pages 48-50 and 23-25
    """

    # if is_neighbour_1 and is_neighbour_3:
    #     neighbour_1 = panel_neighbours[0]
    #     neighbour_3 = panel_neighbours[2]
    #     SMQ_k, SMQ_n1, SMQ_n3 = panel.SMQ, neighbour_1.SMQ, neighbour_3.SMQ
    #     SA = - (SMQ_k + SMQ_n1)
    #     SB = SMQ_k + SMQ_n3
    #     DA = (neighbour_1.mu - panel.mu)/SA
    #     DB = (neighbour_3.mu - panel.mu)/SB

    #     DELQ = (DA * SB - DB * SA)/(SB - SA)

    # if is_neighbour_2 and is_neighbour_4:
    #     neighbour_2, neighbour_4 = panel_neighbours[1], panel_neighbours[3]
    #     SMP_k, SMP_n2, SMP_n4 = panel.SMP, neighbour_2.SMP, neighbour_4.SMP

    #     SA = - (SMP_k + SMP_n4)
    #     SB = SMP_k + SMP_n2
    #     DA = (neighbour_4.mu - panel.mu)/SA
    #     DB = (neighbour_2.mu - panel.mu)/SB

    #     DELP = (DA * SB - DB * SA)/(SB - SA)


    if is_neighbour_1 and is_neighbour_3:

        neighbour_1 = panel_neighbours[0]
        neighbour_3 = panel_neighbours[2]

        panel_j_minus1 = neighbour_1
        panel_j_plus1 = neighbour_3

        x1 = 0
        x0 = x1 - panel.SMQ - panel_j_minus1.SMQ
        x2 = x1 + panel.SMQ + panel_j_plus1.SMQ
        mu0 = panel_j_minus1.mu
        mu1 = panel.mu
        mu2 = panel_j_plus1.mu

        DELQ = mu0 * (x1 - x2)/(x0 - x1)/(x0 - x2) \
                + mu1 * (2*x1 - x0 - x2)/(x1 - x0)/(x1 - x2) \
                + mu2 * (x1 - x0)/(x2 - x0)/(x2 - x1)

    elif is_neighbour_1:
        neighbour_1 = panel_neighbours[0]
        neighbour_3 = panel_neighbours[2]
        panel_j_minus1 = neighbour_1
        panel_j_minus2 = neighbour_3

        x2 = 0
        x1 = x2 - panel.SMQ - panel_j_minus1.SMQ
        x0 = x1  - panel_j_minus1.SMQ - panel_j_minus2.SMQ

        mu0 = panel_j_minus2.mu
        mu1 = panel_j_minus1.mu
        mu2 = panel.mu

        DELQ = mu0 * (x2 - x1)/(x0 - x1)/(x0 - x2) \
                + mu1 * (x2 - x0)/(x1 - x0)/(x1 - x2) \
                + mu2 * (2*x2 - x0 - x1)/(x2 - x0)/(x2 - x1)

    elif is_neighbour_3:
        neighbour_1 = panel_neighbours[0]
        neighbour_3 = panel_neighbours[2]
        panel_j_plus1 = neighbour_3
        panel_j_plus2 = neighbour_1

        x0 = 0
        x1 = x0 + panel.SMQ + panel_j_plus1.SMQ
        x2 = x1 + panel_j_plus1.SMQ + panel_j_plus2.SMQ

        mu0 = panel.mu
        mu1 = panel_j_plus1.mu
        mu2 = panel_j_plus2.mu

        DELQ = mu0 * (2*x0 - x1 - x2)/(x0 - x1)/(x0 - x2) \
                + mu1 * (x0 - x2)/(x1 - x0)/(x1 - x2) \
                + mu2 * (x0 -x1)/(x2 - x0)/(x2 - x1)


    if is_neighbour_2 and is_neighbour_4:
        neighbour_2, neighbour_4 = panel_neighbours[1], panel_neighbours[3]

        panel_i_minus1 = neighbour_4
        panel_i_plus1 = neighbour_2

        x1 = 0
        x0 = x1 - panel.SMP - panel_i_minus1.SMP
        x2 = x1 + panel.SMP + panel_i_plus1.SMP

        mu0 = panel_i_minus1.mu
        mu1 = panel.mu
        mu2 = panel_i_plus1.mu

        DELP = mu0 * (x1 - x2)/(x0 - x1)/(x0 - x2) \
                + mu1 * (2*x1 - x0 - x2)/(x1 - x0)/(x1 - x2) \
                + mu2 * (x1 - x0)/(x2 - x0)/(x2 - x1)

    elif is_neighbour_2:
        neighbour_2, neighbour_4 = panel_neighbours[1], panel_neighbours[3]

        panel_i_plus1 = neighbour_2
        panel_i_plus2 = neighbour_4

        x0 = 0
        x1 = x0 + panel.SMP + panel_i_plus1.SMP
        x2 = x1 + panel_i_plus1.SMP + panel_i_plus2.SMP

        mu0 = panel.mu
        mu1 = panel_i_plus1.mu
        mu2 = panel_i_plus2.mu

        DELP = mu0 * (2*x0 - x1 - x2)/(x0 - x1)/(x0 - x2) \
                + mu1 * (x0 - x2)/(x1 - x0)/(x1 - x2) \
                + mu2 * (x0 - x1)/(x2 - x0)/(x2 - x1)

    elif is_neighbour_4:
        neighbour_2, neighbour_4 = panel_neighbours[1], panel_neighbours[3]

        panel_i_minus1 = neighbour_4
        panel_i_minus2 = neighbour_2

        x2 = 0
        x1 = x2 - panel.SMP - panel_i_minus1.SMP
        x0 = x1 - panel_i_minus1.SMP - panel_i_minus2.SMP

        mu0 = panel_i_minus2.mu
        mu1 = panel_i_minus1.mu
        mu2 = panel.mu

        DELP = mu0 * (x2 - x1)/(x0 - x1)/(x0 - x2) \
                + mu1 * (x2 - x0)/(x1 - x0)/(x1 - x2) \
                + mu2 * (2*x2 - x0 - x1)/(x2 - x0)/(x2 - x1)



    T = panel.T
    T = T.transformation(panel.R)
    TM = T.y
    TL = T.x

    VL = - (panel.SMP * DELP - TM * DELQ)/TL 
    VM = - DELQ
    VN = panel.sigma

    Vl, Vm, Vn = VL, VM, VN

    V_disturb_local = Vector((Vl, Vm, Vn))
    V_disturb = V_disturb_local.transformation(panel.R.T)

    V = V_fs + V_disturb

    return V

def VSAERO_onbody_analysis(V_fs:Vector, mesh:PanelAeroMesh):

    mesh.locate_VSAERO_adjacency()

    body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]

    V_fs_norm = V_fs.norm()

    for panel in body_panels:

        if len(mesh.shell_neighbours[panel.id])==4:
            # all 4 adjacent panels exist
            panel_neighbours = mesh.give_neighbours(panel)
            panel.Velocity = VSAERO_panel_velocity(
                V_fs, panel, panel_neighbours
            )

        elif len(mesh.shell_neighbours[panel.id])>4:

            neighbours_ids = []
            i = 4

            if mesh.shell_neighbours[panel.id][0] == -1:
                is_neighbour_1 = False
                neighbours_ids.append(mesh.shell_neighbours[panel.id][i])
                i = i + 1
            else:
                is_neighbour_1 = True
                neighbours_ids.append(mesh.shell_neighbours[panel.id][0])

            if mesh.shell_neighbours[panel.id][1] == -1:
                is_neighbour_2 = False
                neighbours_ids.append(mesh.shell_neighbours[panel.id][i])
                i = i + 1
            else:
                is_neighbour_2 = True
                neighbours_ids.append(mesh.shell_neighbours[panel.id][1])

            if mesh.shell_neighbours[panel.id][2] == -1:
                is_neighbour_3 = False
                neighbours_ids.append(mesh.shell_neighbours[panel.id][i])
                i = i + 1
            else:
                is_neighbour_3 = True
                neighbours_ids.append(mesh.shell_neighbours[panel.id][2])

            if mesh.shell_neighbours[panel.id][3] == -1:
                is_neighbour_4 = False
                neighbours_ids.append(mesh.shell_neighbours[panel.id][i])
            else:
                is_neighbour_4 = True
                neighbours_ids.append(mesh.shell_neighbours[panel.id][3])

            mesh.shell_neighbours[panel.id] = neighbours_ids
            panel_neighbours = mesh.give_neighbours(panel)


            panel.Velocity = VSAERO_panel_velocity(
                V_fs, panel, panel_neighbours, is_neighbour_1,
                is_neighbour_2, is_neighbour_3, is_neighbour_4
            )

        else:
            # standard least squares method
            panel_neighbours = mesh.give_neighbours(panel)
            panel.Velocity = panel_velocity(panel, panel_neighbours, V_fs)


        # pressure coefficient calculation
        panel.Cp = 1 - (panel.Velocity.norm()/V_fs_norm)**2

def VSAERO_usteady_onbody_analysis(v_rel:Vector, mesh:PanelAeroMesh,
                                   mu_previous:np.ndarray, dt:float):
    
    mesh.locate_VSAERO_adjacency()

    body_panels = [mesh.panels[id] for id in mesh.panels_ids["body"]]

    for panel in body_panels:
        
        # v_rel velocity relative to inertial frame (V_wind)
        r_cp = panel.r_cp
        # v = v_rel - (mesh.Vo + Vector.cross_product(mesh.omega, r_cp))
        v = v_rel - (mesh.Vo + mesh.omega.cross(r_cp))
        # V_wind - Vo = V_fs
        # v = self.V_fs - Vector.cross_product(mesh.omega, r_cp)
        v = v.transformation(mesh.R.T)
        
        if len(mesh.shell_neighbours[panel.id])==4:
            # all 4 adjacent panels exist
            panel_neighbours = mesh.give_neighbours(panel)
            panel.Velocity = VSAERO_panel_velocity(v, panel, panel_neighbours)

        elif len(mesh.shell_neighbours[panel.id])>4:

            neighbours_ids = []
            i = 4

            if mesh.shell_neighbours[panel.id][0] == -1:
                is_neighbour_1 = False
                neighbours_ids.append(mesh.shell_neighbours[panel.id][i])
                i = i + 1
            else:
                is_neighbour_1 = True
                neighbours_ids.append(mesh.shell_neighbours[panel.id][0])

            if mesh.shell_neighbours[panel.id][1] == -1:
                is_neighbour_2 = False
                neighbours_ids.append(mesh.shell_neighbours[panel.id][i])
                i = i + 1
            else:
                is_neighbour_2 = True
                neighbours_ids.append(mesh.shell_neighbours[panel.id][1])

            if mesh.shell_neighbours[panel.id][2] == -1:
                is_neighbour_3 = False
                neighbours_ids.append(mesh.shell_neighbours[panel.id][i])
                i = i + 1
            else:
                is_neighbour_3 = True
                neighbours_ids.append(mesh.shell_neighbours[panel.id][2])

            if mesh.shell_neighbours[panel.id][3] == -1:
                is_neighbour_4 = False
                neighbours_ids.append(mesh.shell_neighbours[panel.id][i])
            else:
                is_neighbour_4 = True
                neighbours_ids.append(mesh.shell_neighbours[panel.id][3])

            mesh.shell_neighbours[panel.id] = neighbours_ids
            panel_neighbours = mesh.give_neighbours(panel)


            panel.Velocity = VSAERO_panel_velocity(
                v, panel, panel_neighbours, is_neighbour_1,
                is_neighbour_2, is_neighbour_3, is_neighbour_4
            )

        else:
            # standard least squares method
            panel_neighbours = mesh.give_neighbours(panel)
            panel.Velocity = panel_velocity(panel, panel_neighbours, v)


        # pressure coefficient calculation
        # Katz & Plotkin eq. 13.168
        
        panel.Cp = 1 - (panel.Velocity.norm()/v.norm())**2
        
        # dφ/dt = dμ/dt
        phi_dot = (panel.mu - mu_previous[panel.id])/dt
        
        panel.Cp = panel.Cp - 2 * phi_dot
    
def body_induced_velocity(r_p, body_panels):
    # Velocity calculation with disturbance velocity functions
    
    velocity = Vector((0, 0, 0))
    
    for panel in body_panels:
        
        
        Src_disturb_velocity(r_p, panel)
        
        # Doublet panels
        velocity = (velocity
                    + Src_disturb_velocity(r_p, panel)
                    + Dblt_disturb_velocity(r_p, panel))
        
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

def Cm_about_point(r_point, body_panels, ReferenceArea):
    """
    r_point: position vector meassured from body-fixed frame of reference f'
    
    Cm = Σ(r_i X CF_i) = Σ{(r_cp_i - r_p) X CF_i}
    
    Cm: moment coefficient about point p
    C_F = n * (-Cp * A/A_ref)
    Cp: panel's pressure coefficient
    A: panel's area
    A_ref: wing's reference area
    r_cp: panel's control point
    r_p: r_point
    r_cp = r + r_p
    """
    Cm = Vector((0, 0, 0))
    
    for panel in body_panels:
        C_Fi = panel.n * (-panel.Cp * panel.area/ReferenceArea)
        r = panel.r_cp - r_point
        Cm = Cm + r.cross(C_Fi)
    
    return Cm

def Center_of_Pressure(body_panels, ReferenceArea):
    """
    a X b = c => b = (c X a)/(a*a) + ka, k: arbitary scalar constant
    
    r_CoP X F = Σ{r X Fi} => F X r_CoP = - Σ{r X Fi} =>
    => r_CoP = (- Σ{r X Fi} X F)/(F*F) + kF 
    => r_CoP = (F X Σ{r X Fi} )/(F*F) + kF
    
    r_CoP: position vector of Center of Pressure
    F: Aerodynamic force = ΣFi
    r: position vector of the point where the force Fi act meassured from the body-fixed frame of reference f'
    r X Fi : moment of force Fi about body-fixed frame's of refernce origin
    Σ{r X Fi}: resultant Moment about the origin of the body-fixed frame of reference f'
    """
    
    CF = AerodynamicForce(body_panels, ReferenceArea)
    
    # r_o: position vector of point about which Cm is calculated
    # r_o is meassured from body-fixed frame of reference f'
    r_o = Vector((0, 0, 0))  
    Cm = Cm_about_point(r_o, body_panels, ReferenceArea)
    
    k = 0 # arbitary constant
    # r_cop.z = 0 =>  [CF.cross(Cm) / CF.norm()**2].z + k*CF.z = 0 =>
    k =-(CF.cross(Cm) / CF.norm()**2).z/CF.z 
    r_cop = CF.cross(Cm) / CF.norm()**2 + CF*k
    
    return r_cop

def Trefftz_Plane_Analysis(mesh:PanelAeroMesh, V_fs:Vector, RefArea:float):
    
    """
    CL and CD computed in Trefftz plane.
    Calculations are based on xflr5's function PanelAnalysis::panelTrefftz()
    """
    
    CL = 0
    CD = 0
    CF = Vector((0, 0, 0))
    
    ny, nx = mesh.nodes_ids["wake lines"].shape
    
    j_max = nx - 1
    
    r = [
        Vector(mesh.nodes[mesh.nodes_ids["wake lines"][i][j_max]])
        for i in range(ny)
    ]
    
    r_c = [(r[i] + r[i+1])/2 for i in range(len(r)-1)]
    
    v_induced = jit_induced_velocity_function(r_c, mesh.panels)
    
    # panel's bound vortex vector. Check xflr5's Panel class
    vortex = [r[i+1] - r[i] for i in range(len(r)-1)] 
    
    for i in range(len(r_c)):
        
        panel_id = mesh.wake_sheddingPanels[
            mesh.TrailingEdge["suction side"][i]
        ][-1]
        
        panel = mesh.panels[panel_id]
        
        w_i = v_induced[i] + V_fs
        
        CF = CF + w_i.cross(vortex[i]) * panel.mu
                
        # CF = CF + vortex[i].cross(w_i) * panel.mu
        
    CF = CF / (0.5 * V_fs.norm()**2 * RefArea)
    
    CD = CF.dot(V_fs/V_fs.norm())
    
    CL_vec = CF - V_fs/V_fs.norm() * CD
    
    if CL_vec.z <=0:
        CL = CL_vec.norm()
    else:
        CL = -CL_vec.norm()

    return CL, CD
   

# numba modifications with just-in-time decorator

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
        """
        Κανονικά πρέπει η <<λούπα>> πρέπει να περιλαμβάνει μόνο τα πάνελ του απόρρου της προγούμενη επανάλληψης καθώς μόνο γι αυτά έχει υπολογιστεί η τιμή της έντασης τους. Παρ' όλα αυτά αφού στα νέα πάνελ ορίζεται μηδενική τιμή έντασης κατά την ορισμό τους, δεν θα συμβάλουν στο δεξί μέλος
        """ 
        for j in prange(Nw):
            
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

@jit(nopython=True, parallel=True)
def induced_velocity_array(r_p_list, panels):
    
    num_points = len(r_p_list)
    num_panels = len(panels)
    
    disturb_velocity_matrix = np.empty((num_points, num_panels, 3))
    
    for i in prange(num_points):
        
        r_p = r_p_list[i]
        
        for j in prange(num_panels):
        
            panel = panels[j]
            
            # Velocity = Vrtx_ring_disturb_velocity(r_p, panel)
            
            _, Vdblt = Dblt_NASA4023(r_p, panel)
            Velocity = Vdblt
            
            if panel.sigma != 0:
                # Velocity = Velocity + Src_disturb_velocity(r_p, panel)
                
                _, Vsrc = Src_NASA4023(r_p, panel)
                Velocity = Velocity + Vsrc
            
            disturb_velocity_matrix[i][j] = Velocity.x, Velocity.y, Velocity.z
    
    velocity_array = np.empty((num_points, 3))
    
    for i in prange(num_points):
        velocity_array[i] = np.sum(disturb_velocity_matrix[i], axis=0)
       
    return velocity_array

def jit_induced_velocity_function(r_p_list, panels):
       
    velocity_array = induced_velocity_array(
        typed.List(r_p_list), typed.List(panels)
    )
    
    velocity_list = [
        Vector(velocity_array[i]) for i in range(len(r_p_list))
    ]
    
    return velocity_list
  

       
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
        