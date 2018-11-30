'''
class and methods for detecting depletions/enhancements in plasma density
using a rolling ball technique
'''
import numpy as np
from scipy.spatial import Delaunay
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import pysat

clean_level = 'none'


def sq_norm(vector):
    '''squared norm '''
    return np.linalg.norm(vector)**2


class OrbitalBallRoller():
    '''class that takes time series data of ion density data and performs a
       rolling ball algorithm to detect depletions or enhancements in the
       density.
       Parameters:
       points: array-like containing X and Y data, where Y is the ion density
       alpha: size factor for the determination of the Alpha shape of the data
    '''
    def __init__(self, points):
        self.in_points = points
        self.treat_points()
        self.tri = Delaunay(self.points)
        self.simplexes = np.asarray(np.sort(self.tri.simplices))
        self.alpha_complex = None

    def treat_points(self):
        '''prepare data for triangulation if it is not already done
           it is recommended that the density be interpreted on a log scale
           it is also recommended that the density data be smoothed somewhat
           nan values cannot be included
           for the triangulation the axes must be scaled for geometric reasons
           Parameters:
           points: array-like containing X and Y data, where Y is ion density
        '''
        time = self.in_points[:, 0]
        density = medfilt([np.log10(x) for x in self.in_points[:, 1]], 7)
        points = zip(time, density)
        self.points = np.array([x for x in points if not np.isnan(x).any()])
        scale_factor = self.get_scale_factor()
        self.points[:, 0] /= np.sqrt(scale_factor)
        self.points[:, 1] *= np.sqrt(scale_factor)

    def get_scale_factor(self):
        '''returns scaling factor for ion density in delaunay triangulation
        the ion density must be scaled so that the delaunay triangulation
        produces meaningful geometry for the detection of bubbles and
        background density.
        More specifically: If the delta X (or time) between two distant
        points is smaller than the delta Y for two points closer together,
        a tringle edge will be placed between the two points closer together.
        This creates a triangle that goes 'through' the ion density curve,
        corrupting the alpha shape desired for the detection of bubbles.
        So we make the largest delta y the same as the smallest delta x
        parameters:
        x : array-like, typically solar or magnetic local time
        y : array-like, ion density
        '''
        xdiff = np.diff(self.points[:, 0])
        ydiff = np.diff(self.points[:, 1])
        xdiff = np.sort(xdiff)
        minxdiff = next((x for x in xdiff if x != 0), None)
        maxydiff = np.max(np.abs(ydiff))
        return minxdiff / maxydiff

    def tri_area(self, simplex):
        '''
        points is a 2d array of points in a plane, simplexes is an nx3 array
        with indices for each point get area and circumradius of circle abc
        '''
        # coordinates of each vertex
        A = self.points[simplex[0]]
        B = self.points[simplex[1]]
        C = self.points[simplex[2]]
        # vector of each edge in each triangle
        AC = C - A
        AB = B - A
        # return area of triangle
        return 0.5 * np.cross(AB, AC)

    def circumcircle(self, points, simplex):
        '''method to find the circumcircle of triangle specified by simplex'''
        A = [points[simplex[k]] for k in range(3)]
        M = [[1.0]*4]
        M += [[sq_norm(A[k]), A[k][0], A[k][1], 1.0] for k in range(3)]
        M = np.asarray(M, dtype=np.float32)
        S = np.array([0.5 * np.linalg.det(M[1:, [0, 2, 3]]),
                     -0.5 * np.linalg.det(M[1:, [0, 1, 3]])])
        a = np.linalg.det(M[1:, 1:])
        b = np.linalg.det(M[1:, [0, 1, 2]])
        # center=S/a, radius=np.sqrt(b/a+sq_norm(S)/a**2)
        return S / a, np.sqrt(b / a + sq_norm(S) / a**2)

    def get_alpha_complex(self, alpha, c=1):
        '''gets the alpha shape of triangulation
        parameters:
        a: alpha factor
        simplices: simplices from triangulation
        c: constant to determine if the upper (1) or lower (-1) envelope
        is desired. Upper envelope for depletions, lower for enhancements
        '''
        result = filter(lambda simplex:
                        self.circumcircle(self.points, simplex)[1] > alpha
                        and c * self.tri_area(simplex) > 0,
                        self.simplexes)
        self.alpha_complex = np.stack(result)

    def plot_delaunay(self, tri):
        '''plots the curve and the triangulation'''
        plt.triplot(self.points[:, 0], self.points[:, 1], tri.simplices.copy())
        plt.plot(self.points[:, 0], self.points[:, 1])
        plt.show()

    def get_background(self):
        '''gets the background density from alpha shell'''
        print(type(self.alpha_complex))
        return np.unique(self.alpha_complex.flatten())

    def locate_depletions(self):
        '''using the upper alpha shape of the density curve this method locates
           depletions in density from the background density.
           the current behavior is adapted from Smith et. al. 2017/18
        '''
        depletions = []
        upper_envelope = self.get_background()
        delta_t = np.diff(self.points[upper_envelope, 0])
        ind, = np.where(delta_t > 1)
        for i in ind:
            lead = upper_envelope[i]
            trail = upper_envelope[i+1]
            d_t = delta_t[i]
            dens = self.points[lead:trail, 1]
            min_edge = np.min([dens[0], dens[-1]])
            min_dens = np.min(dens)
            d_n = (min_edge - min_dens) / min_edge
            if d_n > .25 and d_t/d_n < .002:
                depletions.append((lead, trail))

        return depletions


info = {'index': 'slt', 'kind': 'local time'}
ivm = pysat.Instrument(platform='cnofs', name='ivm',
                       orbit_info=info, clean_level=clean_level)
start = pysat.datetime(2009, 7, 8)
stop = pysat.datetime(2009, 7, 9)
ivm.bounds = (start, stop)
ivm.load(date=start)
ivm.orbits[8]
ivm.data = ivm.data.resample('1S', label='left').ffill(limit=7)

orbit = OrbitalBallRoller(np.column_stack([ivm['slt'], ivm['ionDensity']]))
orbit.get_alpha_complex(400)
alpha_arr = orbit.alpha_complex
bkg = orbit.get_background()
plt.plot(orbit.points[:, 0], orbit.points[:, 1])
plt.triplot(orbit.points[:, 0], orbit.points[:, 1], alpha_arr)
plt.plot(orbit.points[bkg, 0], orbit.points[bkg, 1])
plt.show()
