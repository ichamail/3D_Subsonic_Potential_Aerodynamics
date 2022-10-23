import numpy as np
from vector_class import Vector
        
class Panel:
    def __init__(self, num_vertices):
        self.id = None  # panel's identity
        self.sigma = 0  # constant source stregth per area
        self.mu = 0  # constant doublet strength per area
        self.num_vertices = num_vertices # panel's number of vertices
        
        # r_vertex[i] : position vector of panel's i-th vertex
        self.r_vertex = np.empty(self.num_vertices, dtype=Vector)
        
        self.r_cp = None # position vector of panel's control point(centroid) 
        self.n = None  # position vector of panel's normal unit vector
        self.l = None # position vector of panel's longitudinal unit vector
        self.m = None  # position vector of panel's transverse unit vector
    
        # all vector are defined in the global coordinate system
        
        # R : Rotation matrix
        # r_p_local = R*(r_cpp) = = R*(r_p - r_cp)
        # r_local = R*r_global or r' = R*r  (x' = Ax συμβολισμός Νατσίαβα)
        self.R = None
        
        self.r_vertex_local = np.empty_like(self.r_vertex)
        
        self.char_length = 0  # maximum diagonal length or maximum edge length
        self.area = 0 # area of the panel
        
        self.Velocity = Vector((0, 0, 0))  # Velocity Vector at control point
        self.Cp = 0  # Pressure coefficient at control point
              
    def set_centroid(self):
        r_cp = Vector((0, 0, 0))
        for i in range(self.num_vertices):
            r_cp = r_cp + self.r_vertex[i]
        
        self.r_cp = r_cp/self.num_vertices
            
    def set_n(self):
        r_1 = self.r_vertex[0]
        r_2 = self.r_vertex[1]
        r_3 = self.r_vertex[2]
        r_31 = r_1 - r_3
        
        if self.num_vertices == 4:
            r_4 = self.r_vertex[3]
            r_24 = r_4 - r_2
            
            n = Vector.cross_product(r_24, r_31)
            n = n/n.norm()
            
        elif self.num_vertices == 3:
            r_21 = r_1 - r_2
            
            n = Vector.cross_product(r_21, r_31)
            n = n/n.norm()
        
        self.n = n
    
    def set_l(self):
        r_1 = self.r_vertex[0]
        r_2 = self.r_vertex[1]
        r_21 = r_1 - r_2
        self.l = r_21/r_21.norm()
    
    def set_m(self):
        self.m = Vector.cross_product(self.n, self.l)
    
    def set_R(self):
        l, m, n = self.l, self.m, self.n
        self.R = np.array([[l.x, l.y, l.z],
                           [m.x, m.y, m.z],
                           [n.x, n.y, n.z]])

    def set_r_vertex_local(self):
        n = self.num_vertices
        r_cp = self.r_cp
        r_vertex = self.r_vertex
        r_vertex_local = self.r_vertex_local
        R = self.R
        
        for i in range(n):
            r_vertex_local[i] = r_vertex[i] - r_cp
            r_vertex_local[i] = r_vertex_local[i].transformation(R)

    def set_char_length(self):
        r_1 = self.r_vertex[0]
        r_2 = self.r_vertex[1]
        r_3 = self.r_vertex[2]
        r_31 = r_1 - r_3
        
        if self.num_vertices == 4:
            r_4 = self.r_vertex[3]
            r_24 = r_4 - r_2
            
            self.char_length = np.max([r_24.norm(), r_31.norm()])
            
        elif self.num_vertices == 3:
            r_21 = r_1 - r_2
            r_32 = r_2 - r_3
            
            self.char_length = np.max([r_21.norm(), r_32.norm(),
                                       r_31.norm()])        
    
    def set_area(self):
        r_1 = self.r_vertex[0]
        r_2 = self.r_vertex[1]
        r_3 = self.r_vertex[2]
        r_31 = r_1 - r_3
        
        if self.num_vertices == 3:
            r_21 = r_1 - r_2
            cross_prod = Vector.cross_product(r_31, r_21)
            self.area = 0.5 * cross_prod.norm()
            
        elif self.num_vertices == 4:
            r_4 = self.r_vertex[3]
            r_24 = r_4 - r_2
            cross_prod = Vector.cross_product(r_31, r_24)
            self.area = 0.5 * cross_prod.norm()               

    # unsteady features
    
    def move(self, v_rel, Vo, omega, R, dt):
        
        """
        v_rel = velocity relative to the inertial frame of reference. (e.g. v_rel = wind speed => v_rel + v_body = v_freestream)    
                
        Ιδιότητα Πρόσθεσης:
        ω_f"/f = ω_f"/f' + ω_f'/f
        
        1)
        f"=f  =>  ω_f"/f' = ω_f/f' και ω_f"/f = 0 
        => 0 = ω_f/f' + ω_f'/f
        => ω_f/f' = - ω_f'/f
        
        2)
        F:αδρανειακό συστημα αναφοράς, F:OXYZ
        f:μεταφερόμενο σύστημα αναφοράς, f:O'xyz
        f:προσδεδεμένο σύστημα αναφοράς στο στερεό, f':O'x'y'z'
        
        Από ιδιότητα πρόσθεσης: 
        ω_f'/F = ω_f'/f + ω_f/F
        (ω_f/F=0) => ω_f'/F = ω_f'/f
        
        από 1) ω_f/f' = - ω_f'/f => ω_f'/F = - ω_F/f'
        (ω_F/F = ω_F/f' + ω_f'/F => 0 = ω_F/f' + ω_f'/F => ω_f'/F = - ω_F/f')
        
        3)
        Μεταφορά παραγώγου διανύσματος από το σύστημα αναφοράς f' στο σύστημα ανφοράς f:
        (r_dot)f = (r_dot)f' + (ω)f'/f X r
        
        
        r = ro' + r' => r_dot = ro'_dot + r'_dot =>
        (r_dot)F = (ro'_dot)F + (r'_dot)F => ...
        (r'_dot)f' = (r_dot)F - [ (ro'_dot)F + (ω)f'/F X r']
        r':position vector of node measured from bodyfixed frame of reference f'
        r: position vector of node measured from inertial frame of reference F
        ω: angular velocity of body-fixed frame observed from inetial frame of reference F (or from frame of refernce f)
         
        (Δες και χειρόγραφες σημειώσεις για περισσότερες λεπτομέρειες)
        """
        
        # Vo: velocity vector of body-fixed frame's origin
        # omega: body-fixed frame's angular velocity
        # v_rel: vector of velocity relative to inertial frame (e.g. wind speed)
        # R: Rotation matrix of body-fixed frame of reference
        
        
        for i in range(self.num_vertices):
            # r_vertex : position vector of panel's vertex meassured from body-fixed frame of reference f'
             
            v = v_rel - (Vo + Vector.cross_product(omega, self.r_vertex[i]))
            
            dr = v*dt
            dr = dr.transformation(R.T)
                        
            self.r_vertex[i] = self.r_vertex[i] + dr
            
                
        self.set_centroid()
        self.set_n() 
        self.set_l() 
        self.set_m()
        self.set_R()
        
        # λογικά δεν χρειάζονται ανανέωση
        self.set_r_vertex_local()
        self.set_char_length()
        self.set_area()
        
   
class quadPanel(Panel):
    def __init__(self, vertex0:Vector, vertex1:Vector,
                 vertex2:Vector, vertex3:Vector):
        super().__init__(4)
        self.r_vertex[0] = vertex0
        self.r_vertex[1] = vertex1
        self.r_vertex[2] = vertex2
        self.r_vertex[3] = vertex3
        self.set_centroid()
        self.set_n() 
        self.set_l() 
        self.set_m()
        self.set_R()
        self.set_r_vertex_local()
        self.set_char_length()
        self.set_area()
        
class triPanel(Panel):
    def __init__(self, vertex0:Vector, vertex1:Vector,
                 vertex2:Vector):
        super().__init__(3)
        self.r_vertex[0] = vertex0
        self.r_vertex[1] = vertex1
        self.r_vertex[2] = vertex2
        self.set_centroid()
        self.set_n() 
        self.set_l() 
        self.set_m()
        self.set_R()
        self.set_r_vertex_local()
        self.set_char_length()
        self.set_area()
              

if __name__=='__main__':
    from matplotlib import pyplot as plt
    vertex1 = Vector((-1, -1, 1))
    vertex2 = Vector((1, -1, 1))
    vertex3 = Vector((1, 1, 1))
    vertex4 = Vector((-1, 1, 1))
    
    # Quadrilateral panel
    # panel = quadPanel(vertex1, vertex2, vertex3, vertex4)
    
    # Triangular panel
    panel = triPanel(vertex1, vertex2, vertex3)
    
    # print(panel.num_vertices)
    # print(panel.n)
    # print(panel.l)
    # print(panel.m)
    
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