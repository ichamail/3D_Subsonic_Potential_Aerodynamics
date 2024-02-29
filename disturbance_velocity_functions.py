import numpy as np
from vector_class import Vector
from panel_class import Panel, quadPanel, triPanel
from is_inside_polygon import is_inside_polygon


def Src_disturb_velocity(r_p:Vector, panel:Panel, alpha=10):
    
    n = panel.num_vertices
    r_vertex = panel.r_vertex_local
    r_cp = panel.r_cp
    R = panel.R 
    
    r = r_p - r_cp
    r_local = r.transformation(R)
    r = r_local
    
    if r.norm() >= alpha * panel.char_length:
        
        u = panel.sigma/(4*np.pi) * panel.area * r.x/(r.norm()**3)
        v = panel.sigma/(4*np.pi) * panel.area * r.y/(r.norm()**3)
        w = panel.sigma/(4*np.pi) * panel.area * r.z/(r.norm()**3)
    
    elif r.norm() == 0:
        # point p coincide with centroid (control point)
        u = 0
        v = 0
        w = 0.5 * panel.sigma
        
    elif r.z == 0 or abs(r.z) <= 10**(-12):
        point = (r.x, r.y)  
        polygon = [(r_vertex[i].x, r_vertex[i].y) for i in range(n)]
        
        if is_inside_polygon(polygon, point):
            # point p lies on panel's surface as z-->0
            w = 0.5 * panel.sigma 
        else:
            # point p lies outside of panel's surface as z-->0
            w = 0
            
        u = 0
        v = 0
        for i in range(n-1, -1, -1):
            # panel numbering follow counter clock wise direction
            # Hess and Smith integrals are calculated with clock wise ordering 
            a = (i+1)%n  # 0, 3, 2, 1 (cw) (instead 0, 1, 2, 3 (cw))
            b = i # 3, 2, 1, 0, (clock wise) (instead 1, 2, 3, 0 (cw))

            r_ab = r_vertex[b] - r_vertex[a]
            d_ab = r_ab.norm()
            r_a = r - r_vertex[a]
            r_a = r_a.norm()
            r_b = r - r_vertex[b]
            r_b = r_b.norm() 
            
            if (r_a + r_b - d_ab) == 0:
                # point p coincide lies on a panel's edge
                # u, v --> inf
                log_term = 0
            else:
                
                log_term = np.log((r_a + r_b - d_ab)/(r_a + r_b + d_ab))
            
            u = u + (r_vertex[b].y - r_vertex[a].y)/d_ab  * log_term 
            
            v = v + (r_vertex[a].x - r_vertex[b].x)/d_ab * log_term
        
        # Katz & Plotkin    
        u = panel.sigma/(4*np.pi) * u
        v = panel.sigma/(4*np.pi) * v
        
        # paper of Lothar birk 
        # u = - panel.sigma/(4*np.pi) * u
        # v = - panel.sigma/(4*np.pi) * v
        
    else:
        u, v, w = 0, 0, 0
        for i in range(n-1, -1, -1):
            # panel numbering follow counter clock wise direction
            # Hess and Smith integrals are calculated with clock wise ordering 
            a = (i+1)%n  # 0, 3, 2, 1 (cw) (instead 0, 1, 2, 3 (cw))
            b = i # 3, 2, 1, 0, (clock wise) (instead 1, 2, 3, 0 (cw))
            
            r_ab = r_vertex[b] - r_vertex[a]
            d_ab = r_ab.norm()
            r_a = r - r_vertex[a]
            r_a = r_a.norm()
            r_b = r - r_vertex[b]
            r_b = r_b.norm() 
            
            if (r_vertex[b].x - r_vertex[a].x) == 0:
                m_ab = np.inf 
            else:
                m_ab = ( (r_vertex[b].y - r_vertex[a].y)
                    /(r_vertex[b].x - r_vertex[a].x) )
            
            
            e_a = (r.x - r_vertex[a].x)**2 + r.z**2
            e_b = (r.x - r_vertex[b].x)**2 + r.z**2
            h_a = (r.x - r_vertex[a].x)*(r.y - r_vertex[a].y)
            h_b = (r.x - r_vertex[b].x)*(r.y - r_vertex[b].y)
            
            log_term = np.log((r_a + r_b - d_ab)/(r_a + r_b + d_ab))
            
            u = u + (r_vertex[b].y - r_vertex[a].y)/d_ab  * log_term 
            
            v = v + (r_vertex[a].x - r_vertex[b].x)/d_ab * log_term
            
            w = w + (  np.arctan((m_ab * e_a - h_a)/(r.z * r_a))
                        - np.arctan((m_ab * e_b - h_b)/(r.z * r_b))  )
        
        # Katz & Plotkin     
        u = panel.sigma/(4*np.pi) * u
        v = panel.sigma/(4*np.pi) * v           
        w = panel.sigma/(4*np.pi) * w
        
        # paper of Lothar birk 
        # u = - panel.sigma/(4*np.pi) * u
        # v = - panel.sigma/(4*np.pi) * v           
        # w = - panel.sigma/(4*np.pi) * w
                
    disturb_velocity_local = Vector((u, v, w))
    disturb_velocity = disturb_velocity_local.transformation(R.T)
    
    return disturb_velocity

def Dblt_disturb_velocity(r_p:Vector, panel:Panel, alpha=10):
    n = panel.num_vertices
    r_vertex = panel.r_vertex_local
    r_cp = panel.r_cp
    R = panel.R 
    
    r = r_p - r_cp
    r_local = r.transformation(R)
    r = r_local
    
    point = (r.x, r.y)  
    polygon = [(r_vertex[i].x, r_vertex[i].y) for i in range(n)]
    
    if r.norm() >= alpha * panel.char_length:
    
        u =  3/(4*np.pi) * panel.mu * panel.area * (r.x * r.z)/(r.norm()**5)
        v = 3/(4*np.pi) * panel.mu * panel.area * (r.y * r.z)/(r.norm()**5)
        w = ( -1/(4*np.pi) * panel.mu * panel.area 
             * (r.x**2 + r.y**2 - 2 * r.z**2)/(r.norm()**5))
        
    elif r.z==0 or abs(r.z)<=10**(-12) and is_inside_polygon(polygon, point):
        u = 0
        v = 0
        
        w = 0   
        for i in range(n-1, -1, -1):
            # panel numbering follow counter clock wise direction
            # Hess and Smith integrals are calculated with clock wise ordering 
            a = (i+1)%n  # 0, 3, 2, 1 (cw) (instead 0, 1, 2, 3 (cw))
            b = i # 3, 2, 1, 0, (clock wise) (instead 1, 2, 3, 0 (cw))
            
            r_a = r - r_vertex[a]
            r_a = r_a.norm()
            r_b = r - r_vertex[b]
            r_b = r_b.norm() 
            
            denominator = (
                r_a * r_b
                * (r_a*r_b + ( (r.x - r_vertex[a].x) * (r.x - r_vertex[b].x)
                              + (r.y - r_vertex[a].y) * (r.y - r_vertex[b].y)
                              + r.z**2)
                   )
                )
            
            numerator = ( ((r.x - r_vertex[b].x) * (r.y - r_vertex[a].y)
                           - (r.x - r_vertex[a].x) * (r.y - r_vertex[b].y))
                         * (r_a + r_b) )
            
            if denominator == 0:
                pass
            else:
                w = w + numerator/denominator
            
               
        w = 1/(4*np.pi) * panel.mu * w
        
    else:
        u, v, w = 0, 0, 0
        for i in range(n-1, -1, -1):
            # panel numbering follow counter clock wise direction
            # Hess and Smith integrals are calculated with clock wise ordering 
            a = (i+1)%n  # 0, 3, 2, 1 (cw) (instead 0, 1, 2, 3 (cw))
            b = i # 3, 2, 1, 0, (clock wise) (instead 1, 2, 3, 0 (cw))
            
            r_a = r - r_vertex[a]
            r_a = r_a.norm()
            r_b = r - r_vertex[b]
            r_b = r_b.norm() 
            
            # υπάρχει λάθος στον παρονομαστή στην εξίσωση του βιβλίου
            denominator = (
                r_a * r_b
                * (r_a*r_b + ((r.x - r_vertex[a].x) * (r.x - r_vertex[b].x)
                              + (r.y - r_vertex[a].y) * (r.y - r_vertex[b].y)
                              + r.z**2)
                   )
                )
            
            numerator = r.z * (r_vertex[a].y - r_vertex[b].y) * (r_a + r_b)
            u = u + numerator/denominator 
            
            numerator = r.z * (r_vertex[b].x - r_vertex[a].x) * (r_a + r_b)
            v = v + numerator/denominator 
            
            numerator = ( ((r.x - r_vertex[b].x) * (r.y - r_vertex[a].y)
                           - (r.x - r_vertex[a].x) * (r.y - r_vertex[b].y))
                         * (r_a + r_b) )
            w = w + numerator/denominator
        
        u = 1/(4*np.pi) * panel.mu * u
        v = 1/(4*np.pi) * panel.mu * v
        w = 1/(4*np.pi) * panel.mu * w
       
    disturb_velocity_local = Vector((u, v, w))
    disturb_velocity = disturb_velocity_local.transformation(R.T)

    return disturb_velocity

# cut-off vortex core model
def Vrtx_ring_disturb_velocity(r_p:Vector, panel:Panel, epsilon = 10**(-6)):
    
    # with a vortex ring we work with global coordinates
    
    r_vertex = panel.r_vertex
    n = panel.num_vertices
    disturb_velocity = Vector((0, 0, 0))
    for i in range(n):
        
        next = (i+1)%n
        
        # Katz & Plotking figure 2.16 and 10.23
        r_1 = r_p - r_vertex[i]
        r_2 = r_p - r_vertex[next]
        r_0 = r_1 - r_2
        # or
        # r_0 = r_vertex[next] - r_vertex[i]
        
        # Katz & Plotking (eq 10.115)
        term1 = panel.mu/(4 * np.pi)
        
        norm1 = r_1.norm()
        norm2 = r_2.norm()
        
        vec12 = Vector.cross_product(r_1, r_2)
        norm12 = vec12.norm()
        
        if norm1 < epsilon or norm2 < epsilon or norm12**2 < epsilon:
            # epsilon : vortex core radius
            u, v, w = 0, 0, 0
            velocity_term = Vector((u, v, w))
            
        else:
                      
            vec12 = vec12/(norm12**2)
            vec1 = r_1/norm1
            vec2 = r_2/norm2
            vec3 = vec1 - vec2
            # term2 = Vector.dot_product(r_0, vec3)
            term2 = r_0 * vec3
            # velocity_term = vec12.scalar_product(term1 * term2)
            velocity_term = term1 * term2 * vec12
            
            
        disturb_velocity = disturb_velocity + velocity_term
    
    
    return disturb_velocity

# Lamb–Osseen finite vortex core model
def Vrtx_ring_induced_veloctiy(r_p:Vector, panel:Panel, core_size=10**(-5)):
    r_vertex = panel.r_vertex
    n = panel.num_vertices
    V_induced = Vector((0, 0, 0))

    for i in range(n):

        next = (i+1)%n

        # Katz & Plotking figure 2.16 and 10.23
        r_1 = r_p - r_vertex[i]
        r_2 = r_p - r_vertex[next]
        r_0 = r_1 - r_2
        # or
        # r_0 = r_vertex[next] - r_vertex[i]

        # Katz & Plotking (eq 10.115)
        term1 = panel.mu/(4 * np.pi)

        norm1 = r_1.norm()
        norm2 = r_2.norm()

        # vec12 = Vector.cross_product(r_1, r_2)
        vec12 = r_1.cross(r_2)
        norm12 = vec12.norm()

        if norm1 ==0 or norm2 ==0 or norm12**2 ==0:
            u, v, w = 0, 0, 0
            velocity_term = Vector((u, v, w))

        else:

            vec12 = vec12/(norm12**2)
            vec1 = r_1/norm1
            vec2 = r_2/norm2
            vec3 = vec1 - vec2
            # term2 = r_0 * vec3
            term2 = r_0.dot(vec3)
            # velocity_term = term1 * term2 * vec12
            velocity_term = vec12 * term1 * term2

            """
            Lamb-Osseen finite vortex core model
            ref: "Computational Methods With Vortices—The 1988 Freeman Scholar Lecture", Turgut Sarpkaya
            ref: "Convergence of different wake alignment methods in a panel code
            for steady-state flows", Youjiang Wang, Moustafa Abdel-Maksoud, Baowei Song
            """

            # distance from panel's i-th edge
            h = r_1.cross(r_0).norm()/r_0.norm()

            # radius of vortex core model
            r_c = core_size

            velocity_term = velocity_term * (
                1 - np.exp(- (h**2)/(r_c**2) )
            ) 

        V_induced = V_induced + velocity_term

    return V_induced

if __name__=='__main__':    
    pass