import numpy as np
from vector_class import Vector
from panel_class import Panel


def Src_influence_coeff(r_p:Vector, panel:Panel, alpha=10):
    n = panel.num_vertices
    r_vertex = panel.r_vertex_local
    r_cp = panel.r_cp
    R = panel.R 
    
    r = Vector.addition(r_p, r_cp.scalar_product(-1))
    r_local = r.transformation(R)
    r = r_local 
    
    if r.norm() >= alpha * panel.char_length:
        
        B = panel.area/r.norm()
    
    
    elif r.z == 0:
        
        B = 0
        for i in range(n-1, -1, -1):
            # panel numbering follow counter clock wise direction
            # Hess and Smith integrals are calculated with clock wise ordering 
            a = (i+1)%n  # 0, 3, 2, 1 (cw) (instead 0, 1, 2, 3 (cw))
            b = i # 3, 2, 1, 0, (clock wise) (instead 1, 2, 3, 0 (cw))
            
            r_ab = Vector.addition(r_vertex[b], r_vertex[a].scalar_product(-1))
            d_ab = r_ab.norm()
            r_a = Vector.addition(r, r_vertex[a].scalar_product(-1))
            r_a = r_a.norm()
            r_b = Vector.addition(r, r_vertex[b].scalar_product(-1))
            r_b = r_b.norm()     
            
            first_term = (
                (r.x - r_vertex[a].x) * (r_vertex[b].y - r_vertex[a].y)
                - (r.y - r_vertex[a].y) * (r_vertex[b].x - r_vertex[a].x)
                )
            
            first_term = first_term/d_ab
               
        if (r_a + r_b - d_ab) == 0:
            # point p coincide lies on a panel's edge
            return 0
        
        else:  
            log_term = np.log((r_a + r_b + d_ab)/(r_a + r_b - d_ab))
                
            B = B + first_term * log_term  
              
    else:
        
        B = 0
        for i in range(n-1, -1, -1):
            # panel numbering follow counter clock wise direction
            # Hess and Smith integrals are calculated with clock wise ordering 
            a = (i+1)%n  # 0, 3, 2, 1 (cw) (instead 0, 1, 2, 3 (cw))
            b = i # 3, 2, 1, 0, (clock wise) (instead 1, 2, 3, 0 (cw))
            
            r_ab = Vector.addition(r_vertex[b], r_vertex[a].scalar_product(-1))
            d_ab = r_ab.norm()
            r_a = Vector.addition(r, r_vertex[a].scalar_product(-1))
            r_a = r_a.norm()
            r_b = Vector.addition(r, r_vertex[b].scalar_product(-1))
            r_b = r_b.norm()     
            
            first_term = (
                (r.x - r_vertex[a].x) * (r_vertex[b].y - r_vertex[a].y)
                - (r.y - r_vertex[a].y) * (r_vertex[b].x - r_vertex[a].x)
                )
            
            first_term = first_term/d_ab
            
            log_term = np.log((r_a + r_b + d_ab)/(r_a + r_b - d_ab))
                
            if (r_vertex[b].x - r_vertex[a].x) == 0:
                m_ab = np.inf 
            else:
                m_ab = ( (r_vertex[b].y - r_vertex[a].y)
                        /(r_vertex[b].x - r_vertex[a].x) )
            
            e_a = (r.x - r_vertex[a].x)**2 + r.z**2
            e_b = (r.x - r_vertex[b].x)**2 + r.z**2
            h_a = (r.x - r_vertex[a].x)*(r.y - r_vertex[a].y)
            h_b = (r.x - r_vertex[b].x)*(r.y - r_vertex[b].y)
            
            arctan_term = ( np.abs(r.z)
                            * ( np.arctan((m_ab*e_a - h_a)/(r.z*r_a))
                                - np.arctan((m_ab*e_b - h_b)/(r.z*r_b)))
                            )    
                        
            B = B + first_term * log_term - arctan_term
            
    B = - 1/(4 * np.pi) * B
    
    return B

def Dblt_influence_coeff(r_p:Vector, panel:Panel, alpha=10):
    n = panel.num_vertices
    r_vertex = panel.r_vertex_local
    r_cp = panel.r_cp
    R = panel.R 
    
    r = Vector.addition(r_p, r_cp.scalar_product(-1))
    r_local = r.transformation(R)
    r = r_local
    
    if r.norm() >= alpha * panel.char_length:
        C = - 1/(4*np.pi) * ( panel.area * r.z)/(r.norm()**3)
        
    elif r.z == 0:
        
        C = - 2 * np.pi
        
    else:
        C = 0
        for i in range(n-1, -1, -1):
            # panel numbering follow counter clock wise direction
            # Hess and Smith integrals are calculated with clock wise ordering 
            a = (i+1)%n  # 0, 3, 2, 1 (cw) (instead 0, 1, 2, 3 (cw))
            b = i # 3, 2, 1, 0, (clock wise) (instead 1, 2, 3, 0 (cw))
            
            r_a = Vector.addition(r, r_vertex[a].scalar_product(-1))
            r_a = r_a.norm()
            r_b = Vector.addition(r, r_vertex[b].scalar_product(-1))
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
            
                
            C = C + ( np.arctan((m_ab*e_a - h_a)/(r.z*r_a)) 
                    - np.arctan((m_ab*e_b - h_b)/(r.z*r_b)) )
            
        C = - 1/(4*np.pi) * C
        
    return C


if __name__=='__main__':
    pass