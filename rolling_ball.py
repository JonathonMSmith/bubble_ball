import pysat
import numpy as np
from scipy.spatial import Delaunay
from scipy.signal import medfilt
import matplotlib.pyplot as plt

clean_level='none'

class OrbitalBallRoller(points):
    '''class that takes time series data of ion density data and performs a 
       rolling ball algorithm to detect depletions or enhancements in the
       density.
       Parameters:
       points: array-like containing X and Y data, where Y is the ion density
       alpha: size factor for the determination of the Alpha shape of the data
    '''

    def treat_points(points):
        '''prepare data for triangulation if it is not already done
           it is recommended that the density be interpreted on a log scale
           it is also recommended that the density data be smoothed somewhat
           nan values cannot be included
           for the triangulation the axes must be scaled for geometric reasons
           Parameters:
           points: array-like containing X and Y data, where Y is ion density
        '''
        slt = ivm['slt']
        n_i = medfilt([np.log10(x) for x in ivm['ionDensity']], 7)
        points = zip(slt, n_i)
        points = np.array([x for x in points if not np.isnan(x).any()])
        
        scale_factor = get_scale_factor(points[:, 0], points[:, 1])
        points[:,0] /= np.sqrt(scale_factor)
        points[:,1] *= np.sqrt(scale_factor)


    def get_scale_factor(x, y):
        '''returns scaling factor for ion density in delaunay triangulation
        the ion density must be scaled so that the delaunay triangulation produces 
        meaningful geometry for the detection of bubbles and background density.
    
        More specifically: If the delta X (or time) between two distant 
        points is smaller than the delta Y for two points closer together, 
        a tringle edge will be placed between the two points closer together. This 
        creates a triangle that goes 'through' the ion density curve, corrupting 
        the alpha shape desired for the detection of bubbles. 
    
        So we make the largest delta y the same as the smallest delta x
    
        parameters:
        x : array-like, typically solar or magnetic local time
        y : array-like, ion density
        '''
        xdiff = np.diff(x)
        ydiff = np.diff(y)
        
        xdiff = np.sort(xdiff)
        minxdiff = next((x for x in xdiff if x != 0), None)
        maxydiff = np.max(np.abs(ydiff))
    
        return minxdiff / maxydiff
    
    def sq_norm(v): #squared norm 
        return np.linalg.norm(v)**2
    
    def tri_area(points, simplex):
        #points is a 2d array of points in a plane, simplexes is an nx3 array with indices for each point
        #get area and circumradius of circle abc
        #coordinates of each vertex
        A = points[simplex[0]]
        B = points[simplex[1]]
        C = points[simplex[2]]
        #vector of each edge in each triangle
        AC = C - A
        AB = B - A
        #lengths of each edge in each triangle
        area = 0.5 * np.cross(AB, AC)
        #radius of circumcircle of each triangle
        return area
    
    def circumcircle(points, simplex):
        A = [points[simplex[k]] for k in range(3)]
        M=[[1.0]*4]
        M+=[[sq_norm(A[k]), A[k][0], A[k][1], 1.0 ] for k in range(3)]
        M=np.asarray(M, dtype=np.float32)
        S=np.array([0.5*np.linalg.det(M[1:,[0,2,3]]), -0.5*np.linalg.det(M[1:,[0,1,3]])])
        a=np.linalg.det(M[1:, 1:])
        b=np.linalg.det(M[1:, [0,1,2]])
        return S/a,  np.sqrt(b/a+sq_norm(S)/a**2) #center=S/a, radius=np.sqrt(b/a+sq_norm(S)/a**2)
    
    def get_alpha_complex(alpha, points, simplexes):
        '''gets the alpha shape of triangulation
        parameters:
        alpha: alpha factor
        simplices: simplices from triangulation
        '''
        return filter(lambda simplex: circumcircle(points, simplex)[1]>alpha and tri_area(points, simplex)<0, simplexes)
    
    def plot_delaunay(points, tri):
        plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
        plt.plot(points[:,0], points[:,1])
        plt.show()

#may need to take log10 of density first, will test
info = {'index':'slt', 'kind':'local time'}
ivm = pysat.Instrument(platform='cnofs', name='ivm', orbit_info=info, clean_level=clean_level)
start = pysat.datetime(2009,7,8)
stop = pysat.datetime(2009,7,9)
ivm.bounds = (start, stop)
ivm.load(date=start)
ivm.orbits[8]
ivm.data = ivm.data.resample('1S', label='left').ffill(limit=7)


tri = Delaunay(points)
simplexes=np.asarray(np.sort(tri.simplices)) #may need this for upper/lower envelopes
alpha_complex = get_alpha_complex(400, points, simplexes)
#X, Y = Plotly_data(points, alpha_complex)
alpha_arr = np.stack(alpha_complex)
plt.plot(points[:,0], points[:,1])
plt.triplot(points[:,0], points[:,1], alpha_arr)
#plt.plot(X,Y)
plt.show()
