from matplotlib import pyplot as plt
import numpy as np
from Algorithms import cosspace
from scipy import interpolate


class Airfoil:
    
    filePath="coord_seligFmt/"
    
    def __init__(self, name:str, chord_length:float=1,
                 x_coords = None, y_coords = None, CCW_order=True):
        
        self.name = name
        self.chord = chord_length
        if x_coords is None or y_coords is None:
            self.get_from_data_base()
            self.x_coords = self.chord * self.x_coords
            self.y_coords = self.chord * self.y_coords 
        else:   
            self.x_coords =  np.asarray(x_coords)
            self.y_coords = np.asarray(y_coords)
        
        if not CCW_order:
            self.invert_coords_order()
    
    def get_from_data_base(self):
        
        fileName = self.name + ".dat"
        self.x_coords, self.y_coords = self.load_airfoil(self.filePath,
                                                         fileName)
    
    def give_suctionSide(self):
        index = np.where(self.x_coords==self.x_coords.min())[0][0]
        return self.x_coords[0:index+1], self.y_coords[0:index+1]
    
    def give_pressureSide(self):
        index = np.where(self.x_coords==self.x_coords.min())[0][0]
        return self.x_coords[index:], self.y_coords[index:]
    
    def invert_coords_order(self):
        
        self.x_coords = np.array(
            [self.x_coords[-i] for i in range(len(self.x_coords))]
        )
        
        self.y_coords = np.array(
            [self.y_coords[-i] for i in range(len(self.y_coords))]
        )
    
    def close_trailing_edge(self):
        if (self.x_coords[0] != self.x_coords[-1] 
            and self.y_coords[0] != self.y_coords[-1]):
            
            self.x_coords = np.append(self.x_coords, self.x_coords[0])
            self.y_coords = np.append(self.y_coords, self.y_coords[0])
    
    @staticmethod
    def load_airfoil(filePath, fileName, header_lines=1):
        
        # Load the data from the text file
        fileName = filePath + fileName
        dataBuffer = np.loadtxt(fileName, delimiter=' ', skiprows=header_lines)
        
        # Extract data from the loaded dataBuffer array
        dataX = np.asarray(dataBuffer[:,0])
        dataY = np.asarray(dataBuffer[:,1])
        
        return dataX, dataY
    
    def plot(self):
        plt.plot(self.x_coords, self.y_coords)
        plt.plot(self.x_coords, self.y_coords, 'ko', markerfacecolor='r')
        plt.axis('scaled')
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(self.name)
        plt.show()
    
    def new_x_spacing(self, num_x_points):
        
        x, y, = self.x_coords, self.y_coords
          
        # Circle creation with diameter equal to airfoil chord
        x_max, x_min = max(x), min(x)
        R = (x_max - x_min)/2
        x_center = (x_max + x_min)/2
        theta = np.linspace(0, 2*np.pi, num_x_points+1)
        x_circle = x_center + R * np.cos(theta)
        
        # project circle points on x-axis
        x_project = np.copy(x_circle) # projections of x-cordiantes on airfoil
        y_project = np.empty_like(x_project)
        
        # compute y_project with interpolation
        j=0
        for i in range(num_x_points):
            while j < len(x)-1:
                if (x[j]<=x_project[i]<=x[j+1] or x[j+1]<=x_project[i]<=x[j]):
                    break
                else:
                    j = j+1
                
            # when break interpolate
            a = (y[j+1]-y[j])/(x[j+1]-x[j])
            b = y[j+1] - a * x[j+1]
            y_project[i] = a * x_project[i] + b
                
        y_project[num_x_points] = y_project[0]
        
        X, Y = x_project, y_project
        
        self.x_coords , self.y_coords = X, Y
        
    def new_suctionSide_x_spacing(self, num_x_points):
        
        x, y = self.give_suctionSide()
          
        # Circle creation with diameter equal to airfoil chord
        x_max, x_min = max(x), min(x)
        R = (x_max - x_min)/2
        x_center = (x_max + x_min)/2
        theta = np.linspace(0, np.pi, num_x_points+1)
        x_circle = x_center + R * np.cos(theta)
        
        # project circle points on x-axis
        x_project = np.copy(x_circle) # projections of x-cordiantes on airfoil
        y_project = np.empty_like(x_project)
        
        # compute y_project with interpolation
        j=0
        for i in range(num_x_points):
            while j < len(x)-1:
                if (x[j]<=x_project[i]<=x[j+1] or x[j+1]<=x_project[i]<=x[j]):
                    break
                else:
                    j = j+1
                
            # when break interpolate
            a = (y[j+1]-y[j])/(x[j+1]-x[j])
            b = y[j+1] - a * x[j+1]
            y_project[i] = a * x_project[i] + b
                
        y_project[num_x_points] = y_project[0]
        
        X, Y = x_project, y_project
        
        return X, Y
    
    def new_pressureSide_x_spacing(self, num_x_points):
        
        x, y = self.give_pressureSide()
          
        # Circle creation with diameter equal to airfoil chord
        x_max, x_min = max(x), min(x)
        R = (x_max - x_min)/2
        x_center = (x_max + x_min)/2
        theta = np.linspace(np.pi, 2*np.pi, num_x_points+1)
        x_circle = x_center + R * np.cos(theta)
        
        # project circle points on x-axis
        x_project = np.copy(x_circle) # projections of x-cordiantes on airfoil
        y_project = np.empty_like(x_project)
        
        # compute y_project with interpolation
        j=0
        for i in range(num_x_points):
            while j < len(x)-1:
                if (x[j]<=x_project[i]<=x[j+1] or x[j+1]<=x_project[i]<=x[j]):
                    break
                else:
                    j = j+1
                
            # when break interpolate
            a = (y[j+1]-y[j])/(x[j+1]-x[j])
            b = y[j+1] - a * x[j+1]
            y_project[i] = a * x_project[i] + b
                
        y_project[num_x_points] = y_project[0]
        
        X, Y = x_project, y_project
        
        return X, Y
        
    def plot(self):
        plt.plot(self.x_coords, self.y_coords)
        plt.plot(self.x_coords, self.y_coords, 'ko', markerfacecolor='r')
        plt.axis('scaled')
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(self.name)
        plt.show()
    
    def new_x_spacing2(self, num_x_points, UpperLowerSpacing_equal=True):
        if UpperLowerSpacing_equal:
            SS_x, SS_y = self.new_suctionSide_x_spacing(num_x_points)
            PS_x, PS_y = self.new_pressureSide_x_spacing(num_x_points)
        else:
            quotient, remainder = np.divmod(num_x_points, 2)
            SS_x, SS_y = self.new_suctionSide_x_spacing(quotient)
            PS_x, PS_y = self.new_pressureSide_x_spacing(quotient+remainder)
        
        self.x_coords = np.append(SS_x, PS_x[1:])
        self.y_coords = np.append(SS_y, PS_y[1:])   

    def repanel(self, n_points_per_side:int):
        """
        improved version of new_x_spacing() and new_x_spacing2()
        """
        
        # upper and lower side coordinates
        x_u, y_u = self.give_suctionSide()
        x_l, y_l = self.give_pressureSide()
        
        # distances between coordinates
        dr_u = np.sqrt( (x_u[:-1] - x_u[1:])**2 + (y_u[:-1] - y_u[1:])**2 )
        dr_l = np.sqrt( (x_l[:-1] - x_l[1:])**2 + (y_l[:-1] - y_l[1:])**2 )
        
        # distances from trailing edge
        dr_u = np.hstack((0, np.cumsum(dr_u)))
        dr_l = np.hstack((0, np.cumsum(dr_l)))
        
        # normalize
        dr_u = dr_u/dr_u[-1]
        dr_l = dr_l/dr_l[-1]
        
        dr = np.hstack((dr_u, 1 + dr_l[1:]))
        
        # cosine-spaced list of points from 0 to 1
        x = cosspace(0, 1, n_points_per_side)
        s = np.hstack((x, 1 + x[1:]))
        
        # Check that there are no duplicate points in the airfoil.
        if np.any(np.diff(dr))==0:
            raise ValueError(
                "This airfoil has a duplicated point (i.e. two adjacent points with the same (x, y) coordinates), so you can't repanel it!"
            )
        
        self.x_coords = interpolate.PchipInterpolator(dr, self.x_coords)(s)
        self.y_coords = interpolate.PchipInterpolator(dr, self.y_coords)(s)
        

if __name__=="__main__":
    name = "naca0012_new"
    chord = 2
    airfoil = Airfoil(name, chord)
    # airfoil.new_x_spacing(10)
    # airfoil.new_x_spacing2(5)
    # airfoil.new_x_spacing2(11, UpperLowerSpacing_equal=False)
    # print(airfoil.x_coords)
    # print(airfoil.y_coords)
    airfoil.repanel(6)
    airfoil.plot()