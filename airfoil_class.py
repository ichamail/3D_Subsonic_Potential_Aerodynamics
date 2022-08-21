from matplotlib import pyplot as plt
import numpy as np


class Airfoil:
    
    filePath="C:/Users/winuser/PythonProjects/Panel Methods 3D/coord_seligFmt/"
    
    def __init__(self, name:str, chord_length:float,
                 x_coords = None, y_coords = None):
        
        self.name = name
        self.chord = chord_length
        if x_coords is None or y_coords is None:
            self.get_from_data_base()
            self.x_coords = self.chord * self.x_coords
            self.y_coords = self.chord * self.y_coords 
        else:   
            self.x_coords =  np.asarray(x_coords)
            self.y_coords = np.asarray(y_coords)
        
        self.z_coords = np.zeros_like(self.x_coords)
    
    def get_from_data_base(self):
        
        fileName = self.name + ".dat"
        self.x_coords, self.y_coords = self.load_airfoil(self.filePath,
                                                         fileName)
    
    def set_z_coords(self, z_coords):
        self.z_coords = np.asarray(z_coords)
    
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
    
    def new_x_spacing(self, num_x_points, indexing='xy'):
        if indexing == 'xy':
            x, y, = self.x_coords, self.y_coords
        
        elif indexing == 'xz':
            x, y, = self.x_coords, -self.z_coords
            
            
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
        
        if indexing == 'xy':
            self.x_coords , self.y_coords = X, Y
            self.z_coords = self.z_coords[0] * np.ones_like(X)
            
        elif indexing == 'xz':
            self.x_coords , self.z_coords = X, -Y
            self.y_coords = self.y_coords[0] * np.ones_like(X)
            
           

if __name__=="__main__":
    name = "naca0012"
    chord = 2
    airfoil = Airfoil(name, chord)
    airfoil.new_x_spacing(10)
    airfoil.plot()