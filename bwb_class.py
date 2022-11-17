from airfoil_class import Airfoil
from vector_class import Vector


class WingCrossSection:
    
    def __init__(self, r_leadingEdge:Vector,
                 chord:float, twist:float, airfoil:Airfoil):
        
        self.r_leadingEdge = r_leadingEdge
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil
    
    def translate(self, dr:Vector):
        r_leadingEdge = self.r_leadingEdge + dr
        wing_cross_section = WingCrossSection(r_leadingEdge, self.chord,
                                              self.twist, self.airfoil)
        return wing_cross_section
    
    
    

class WingSpanwiseSection:
    
    def __init__(self, nea):
        pass
    

class BWB:
    
    def __init__(self):
        pass