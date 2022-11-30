import numpy as np
from vector_class import Vector
from panel_class import Panel, quadPanel, triPanel
from is_inside_polygon import is_inside_polygon


def Src_influence_coeff(r_p:Vector, panel:Panel, alpha=10):
    n = panel.num_vertices
    r_vertex = panel.r_vertex_local
    r_cp = panel.r_cp
    R = panel.R 
    
    r = r_p - r_cp
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
            
            r_ab = r_vertex[b] - r_vertex[a]
            d_ab = r_ab.norm()
            r_a = r - r_vertex[a]
            r_a = r_a.norm()
            r_b = r - r_vertex[b]
            r_b = r_b.norm()     
            
            first_term = (
                (r.x - r_vertex[a].x) * (r_vertex[b].y - r_vertex[a].y)
                - (r.y - r_vertex[a].y) * (r_vertex[b].x - r_vertex[a].x)
                )
            
            first_term = first_term/d_ab
               
            if (r_a + r_b - d_ab) == 0:
                # point p coincide lies on a panel's edge
                # log term ---> inf => B ---> inf
                log_term = 0
        
            else:
                # katz & Plotkin
                log_term = np.log((r_a + r_b + d_ab)/(r_a + r_b - d_ab))
                
                # paper of Lothar birk  
                # log_term = np.log((r_a + r_b - d_ab)/(r_a + r_b + d_ab))
                
            B = B + first_term * log_term  
              
    else:
        
        B = 0
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
            
            first_term = (
                (r.x - r_vertex[a].x) * (r_vertex[b].y - r_vertex[a].y)
                - (r.y - r_vertex[a].y) * (r_vertex[b].x - r_vertex[a].x)
                )
            
            first_term = first_term/d_ab
            
            # katz & Plotkin
            log_term = np.log((r_a + r_b + d_ab)/(r_a + r_b - d_ab))
            
            # paper of Lothar birk  
            # log_term = np.log((r_a + r_b - d_ab)/(r_a + r_b + d_ab))
                
            if (r_vertex[b].x - r_vertex[a].x) == 0:
                m_ab = np.inf 
            else:
                m_ab = ( (r_vertex[b].y - r_vertex[a].y)
                        /(r_vertex[b].x - r_vertex[a].x) )
            
            e_a = (r.x - r_vertex[a].x)**2 + r.z**2
            e_b = (r.x - r_vertex[b].x)**2 + r.z**2
            h_a = (r.x - r_vertex[a].x)*(r.y - r_vertex[a].y)
            h_b = (r.x - r_vertex[b].x)*(r.y - r_vertex[b].y)
            
            arctan_term = ( np.arctan((m_ab*e_a - h_a)/(r.z*r_a))
                           - np.arctan((m_ab*e_b - h_b)/(r.z*r_b)) )
            
            # Katz & Plotkin            
            # B = B + first_term * log_term - np.abs(r.z) * arctan_term
            
            # paper of Lothar birk            
            B = B + first_term * log_term - r.z * arctan_term
                  
    B = - 1/(4 * np.pi) * B
    
    return B

def Dblt_influence_coeff(r_p:Vector, panel:Panel, alpha=10):
    n = panel.num_vertices
    r_vertex = panel.r_vertex_local
    r_cp = panel.r_cp
    R = panel.R 
    
    r = r_p - r_cp
    r_local = r.transformation(R)
    r = r_local
    
    if r.norm() >= alpha * panel.char_length:
        C = - 1/(4*np.pi) * ( panel.area * r.z)/(r.norm()**3)
    
    elif r.z == 0:
        point = (r.x, r.y)  
        polygon = [(r_vertex[i].x, r_vertex[i].y) for i in range(n)]
        
        if is_inside_polygon(polygon, point):
            # point p lies on panel's surface as z-->0
            # C = - 0.5  # if z--> +0
            C = 0.5  # if z--> -0
        else:
            # point p lies outside of panel's surface as z-->0
            C = 0
        
    else:
        C = 0
        for i in range(n-1, -1, -1):
            # panel numbering follow counter clock wise direction
            # Hess and Smith integrals are calculated with clock wise ordering 
            a = (i+1)%n  # 0, 3, 2, 1 (cw) (instead 0, 1, 2, 3 (cw))
            b = i # 3, 2, 1, 0, (clock wise) (instead 1, 2, 3, 0 (cw))
            
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
            
            arctan_term = ( np.arctan((m_ab*e_a - h_a)/(r.z*r_a))
                           - np.arctan((m_ab*e_b - h_b)/(r.z*r_b)) )
                
            C = C + arctan_term
            
        C = - 1/(4*np.pi) * C
        
    return C

def influence_coeff(r_p:Vector, panel:Panel, alpha=10):
    n = panel.num_vertices
    r_vertex = panel.r_vertex_local
    r_cp = panel.r_cp
    R = panel.R 
    
    r = r_p - r_cp
    r_local = r.transformation(R)
    r = r_local
    
    if r.norm() >= alpha * panel.char_length:
        B = panel.area/r.norm()
        C = ( panel.area * r.z)/(r.norm()**3)
    
    elif r.z == 0:
        
        point = (r.x, r.y)  
        polygon = [(r_vertex[i].x, r_vertex[i].y) for i in range(n)]
    
        if is_inside_polygon(polygon, point):
            # point p lies on panel's surface as z-->0
            # C =  2 * np.pi  # if z--> +0
            C = -2 * np.pi  # if z--> -0
            
        else:
            # point p lies outside of panel's surface as z-->0
            C = 0
        
        B = 0
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
            
            first_term = (
                (r.x - r_vertex[a].x) * (r_vertex[b].y - r_vertex[a].y)
                - (r.y - r_vertex[a].y) * (r_vertex[b].x - r_vertex[a].x)
                )
            
            first_term = first_term/d_ab
            
            if (r_a + r_b - d_ab) == 0:
                # point p coincide lies on a panel's edge
                # log term ---> inf => B ---> inf
                log_term = 0
        
            else:
                # katz & Plotkin
                log_term = np.log((r_a + r_b + d_ab)/(r_a + r_b - d_ab))
                
                # paper of Lothar birk  
                # log_term = np.log((r_a + r_b - d_ab)/(r_a + r_b + d_ab))
                
            B = B + first_term * log_term  
                           
    else:
        B = 0
        C = 0
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
            
            first_term = (
                (r.x - r_vertex[a].x) * (r_vertex[b].y - r_vertex[a].y)
                - (r.y - r_vertex[a].y) * (r_vertex[b].x - r_vertex[a].x)
                )
            
            first_term = first_term/d_ab
            
            # katz & Plotkin
            log_term = np.log((r_a + r_b + d_ab)/(r_a + r_b - d_ab))
            
            # paper of Lothar birk  
            # log_term = np.log((r_a + r_b - d_ab)/(r_a + r_b + d_ab))  
            
            if (r_vertex[b].x - r_vertex[a].x) == 0:
                m_ab = np.inf 
            else:
                m_ab = ( (r_vertex[b].y - r_vertex[a].y)
                        /(r_vertex[b].x - r_vertex[a].x) )
            
            e_a = (r.x - r_vertex[a].x)**2 + r.z**2
            e_b = (r.x - r_vertex[b].x)**2 + r.z**2
            h_a = (r.x - r_vertex[a].x)*(r.y - r_vertex[a].y)
            h_b = (r.x - r_vertex[b].x)*(r.y - r_vertex[b].y)
            
             
            arctan_term = ( np.arctan((m_ab*e_a - h_a)/(r.z*r_a))
                           - np.arctan((m_ab*e_b - h_b)/(r.z*r_b)) )
     
            
            
            # Katz & Plotkin            
            # B = B + first_term * log_term - np.abs(r.z) * arctan_term
            
            # paper of Lothar birk            
            B = B + first_term * log_term - r.z * arctan_term
            
            # Katz & Plotkin  
            C = C + arctan_term 
            
    B = - 1/(4 * np.pi) * B
    C = - 1/(4 * np.pi) * C
        
    return B, C

if __name__=='__main__':
    from matplotlib import pyplot as plt
    vertex1 = Vector((-1, -1, 1))
    vertex2 = Vector((1, -1, 1))
    vertex3 = Vector((1, 1, 1))
    vertex4 = Vector((-1, 1, 1))
    
    # Quadrilateral panel
    panel = quadPanel(vertex1, vertex2, vertex3, vertex4)
    
    # Triangular panel
    # panel = triPanel(vertex1, vertex2, vertex3)
    
    r_p = panel.r_cp

    C = Dblt_influence_coeff(r_p, panel)
    B = Src_influence_coeff(r_p, panel)
    print("C = " + str(C) + "\nB = " + str(B))
    
    B, C = influence_coeff(r_p, panel)
    print("C = " + str(C) + "\nB = " + str(B))
    
    
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
    
    ax.legend()
    
    plt.show()
  