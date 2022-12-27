import numpy as np
from vector_class import Vector
from panel_class import Panel, quadPanel, triPanel
from numba import jit

@jit(nopython=True)
def Src_NASA4023(r_p:Vector, panel:Panel, alpha=10, core_radius = 10**(-6)):
    
    phi = 0.0
    V = Vector((0.0, 0.0, 0.0))
    
    C = r_p
    CollPt = panel.r_cp
    
    PJK = C - CollPt
    
    PN = PJK.dot(panel.n)
    pjk = PJK.norm()
    
    if pjk > alpha * panel.char_length:
        
        phi = panel.area/pjk
        V = PJK * panel.area /pjk/pjk/pjk * panel.sigma
        
        return phi/(-4*np.pi), V/(4*np.pi)
    
    n = panel.num_vertices
    for i in range(n):
        
        a = C - panel.r_vertex[i%n]
        b = C - panel.r_vertex[(i+1)%n]
        s = panel.r_vertex[(i+1)%n] - panel.r_vertex[i%n]
        
        A = a.norm()
        B = b.norm()
        S = s.norm()
        
        # SM = s * panel.m
        # SL = s * panel.l
        SM = s.dot(panel.m)
        SL = s.dot(panel.l)
        # AM = a * panel.m
        # AL = a * panel.l
        # Al = AM * SL - AL * SM
        # PA = PN * PN * SL + Al * AM
        # Al = panel.n * Vector.cross_product(s, a)
        Al = panel.n.dot(s.cross(a)) 
        # PA = a * ( Vector.cross_product(panel.l, Vector.cross_product(a, s)) )
        PA = a.dot(panel.l.cross(a.cross(s)))
        PB = PA - Al * SM
        
        
        # h = Vector.cross_product(a, s)
        h = a.cross(s)
        
        # first the potential
        
        if (h.norm()/s.norm() <= core_radius and a.dot(s) >= 0 and b.dot(s) <=0
            or A < core_radius or B < core_radius):
            
            # if lying on the panel's side... no contribution
            # CJKi = 0.0
            pass

        else:
            
            # first the potential
            
            if A + B - S > 0 :
                GL = 1/S * np.log((A + B + S)/(A + B - S))
            else:
                GL = 0
            
            RNUM = SM * PN * (B * PA - A * PB)
            DNOM = PA * PB + PN * PN * A * B * SM * SM
            
            if abs(PN) < 10**(-7):
                
                # side is >0 if on the panel's right side
                side = panel.n.dot(h)
                
                sign = 1  if side >= 0.0  else -1
                                
                if DNOM < 0.0:
                    
                    CJKi = np.pi * sign  if PN > 0  else  -np.pi * sign
                                    
                elif DNOM == 0.0:
                    
                    CJKi = np.pi/2 * sign  if PN > 0  else  -np.pi/2 * sign
                                   
                else:
                    
                    CJKi = 0.0
                    
            else:
                
                CJKi = np.arctan2(RNUM, DNOM)
                
            phi = phi + Al * GL - PN * CJKi
            
            # next the induced velocity
            T1 = panel.l * SM * GL
            T2 = panel.m * SL * GL
            
            V = V + panel.n * CJKi + T1 - T2
        
    
    return phi/(-4*np.pi), V/(4*np.pi) * panel.sigma


@jit(nopython=True)
def Dblt_NASA4023(r_p:Vector, panel:Panel, alpha=10, core_radius = 10**(-6)):
    
    phi = 0.0
    V = Vector((0.0, 0.0, 0.0))
    
    C = r_p
    CollPt = panel.r_cp
    
    PJK = C - CollPt
    
    # PN = PJK * panel.n
    PN = PJK.dot(panel.n)
    
    pjk = PJK.norm() 
    
    if pjk > alpha * panel.char_length:
        
        phi = PN * panel.area/pjk/pjk/pjk
        T = PJK * 3 * PN - panel.n * pjk * pjk
        V = T * panel.area /pjk/pjk/pjk/pjk/pjk 
        
        return phi/(-4*np.pi), V/(4*np.pi) * panel.mu
    
    n = panel.num_vertices
    for i in range(n):
        
        a = C - panel.r_vertex[i%n]
        b = C - panel.r_vertex[(i+1)%n]
        s = panel.r_vertex[(i+1)%n] - panel.r_vertex[i%n]
        
        A = a.norm()
        B = b.norm()
        
        # SM = s * panel.m
        SM = s.dot(panel.m)
        # SL = s * panel.l
        # AM = a * panel.m
        # AL = a * panel.l
        # Al = AM * SL - AL * SM
        # PA = PN * PN * SL + Al * AM
        # Al = panel.n * Vector.cross_product(s, a)
        Al = panel.n.dot(s.cross(a))
        # PA = a * ( Vector.cross_product(panel.l, Vector.cross_product(a, s)) )
        PA = a.dot(panel.l.cross(a.cross(s)))
        PB = PA - Al * SM
        
        
        # h = Vector.cross_product(a, s)
        h = a.cross(s)
        
        # first the potential
        
        if (h.norm()/s.norm() <= core_radius and a.dot(s) >= 0 and b.dot(s) <=0
            or A < core_radius or B < core_radius):
            
            # speed is singular at panel edge, the value of the potential is unknown
            CJKi = 0.0 

        else:
            
            RNUM = SM * PN * (B * PA - A * PB)
            DNOM = PA * PB + PN * PN * A * B * SM * SM
            
            if abs(PN) < 10**(-7):
                
                # side is >0 if on the panel's right side
                side = panel.n.dot(h)
                
                sign = 1  if side >= 0.0  else -1
                                
                if DNOM < 0.0:
                    CJKi = np.pi * sign  if PN > 0  else  -np.pi * sign
                                    
                elif DNOM == 0.0:
                    CJKi =  np.pi/2 * sign  if PN > 0  else  -np.pi/2 * sign
                                    
                else:
                    CJKi = 0.0
                    
            else:
                
                CJKi = np.arctan2(RNUM, DNOM)
                
            
            # next induced velocity
            
            # h = Vector.cross_product(a, b)
            h = a.cross(b)
            GL = ( (A+B) /A/B/(A * B + a.dot(b)))                       
            
            V = V + h*GL
        
        phi = phi + CJKi
        
    if (PJK).norm() <= 10**(-10):
        
        phi = -2 * np.pi
        
        
    return phi/(-4*np.pi), V/(4*np.pi) * panel.mu


if __name__=='__main__':
    from matplotlib import pyplot as plt
    from influence_coefficient_functions import Src_influence_coeff, Dblt_influence_coeff
    from disturbance_velocity_functions import Src_disturb_velocity, Vrtx_ring_disturb_velocity, Dblt_disturb_velocity
    
    vertex1 = Vector((0, 0, 1))
    vertex2 = Vector((1, 0, 1))
    vertex3 = Vector((1, 1, 1))
    vertex4 = Vector((0, 1, 1))
    
    # Quadrilateral panel
    panel = quadPanel(vertex4, vertex3, vertex2, vertex1)
    
    # Triangular panel
    # panel = triPanel(vertex1, vertex2, vertex3)
    
    r_p = panel.r_cp  
    # r_p = panel.r_vertex[0] # !
    # r_p = (panel.r_vertex[3] + panel.r_vertex[2])/2 # !
    # r_p = Vector((0, 0.2, 1))
    # r_p = Vector((0.2, 0.2, 1)) 
    # r_p = Vector((0.5, 0.5, 1)) 
    # r_p = Vector((-0.5, 0, 1))
    # r_p = Vector((-2, -2, 2))
    # r_p = Vector((20, 200, 100))
    
    C = Dblt_influence_coeff(r_p + Vector((0, 0, 0)), panel)
    B = Src_influence_coeff(r_p, panel)
    print("C = " + str(C))
    print("B = " + str(B))
    
    C, _ = Dblt_NASA4023(r_p, panel)
    B, _ = Src_NASA4023(r_p, panel)
    print("NASA4023 C = " + str(C))
    print("NASA4023 B = " + str(B))
    
    
    panel.sigma, panel.mu = 1, 1
    
    Vsrc = Src_disturb_velocity(r_p, panel)
    Vdblt = Dblt_disturb_velocity(r_p, panel)
    Vring = Vrtx_ring_disturb_velocity(r_p, panel)
    
    print("V_source = " + str((Vsrc.x, Vsrc.y, Vsrc.z)))
    print("V_doublet = " + str((Vdblt.x, Vdblt.y, Vdblt.z))
        + "\nV_vortex_ring = " + str((Vring.x, Vring.y, Vring.z)))    
    
    _, Vsrc = Src_NASA4023(r_p, panel)
    _, Vdblt = Dblt_NASA4023(r_p, panel)
    
    print("NASA4023 V_source = " + str((Vsrc.x, Vsrc.y, Vsrc.z)))
    print("NASA4023 V_doublet = " + str((Vdblt.x, Vdblt.y, Vdblt.z)))
    
    # plot panel
    
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    # ax.view_init(0, 0)
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
    
    ax.scatter(r_p.x, r_p.y, r_p.z)
    
    ax.legend()
    
    plt.show()