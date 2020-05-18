'''
   class and methods for detecting depletions/enhancements in plasma density
   using a rolling ball technique
'''
import numpy as np
from scipy.spatial import Delaunay
from scipy.signal import medfilt
import xarray as xr
from datetime import datetime
import pysat
import logging


logging.basicConfig(level=logging.DEBUG, filename='bubble.log')

def sq_norm(vector):
    '''squared norm '''
    return np.linalg.norm(vector)**2

def filter_inst(inst):
    '''prepare data for triangulation if it is not already done
       it is recommended that the density be interpreted on a log scale
       it is also recommended that the density data be smoothed somewhat
       nan values cannot be included
       for the triangulation the axes must be scaled for geometric reasons
       I'm not entirely sure why, perhaps a precision issue, but the
       triangulation is much better if both axes are scaled similarly
       Parameters:
       points: array-like containing X and Y data, where Y is ion density
    '''
    inst['filt_density'] = medfilt([np.log10(x) for x in inst['ionDensity']], 7)
    idx, = np.where((~np.isnan(inst['slt'])) & (~np.isnan(inst['filt_density'])))
    inst.data = inst.data.iloc[idx]
    return inst

class OrbitalBallRoller():
    '''class that takes time series data of ion density data and performs a
       rolling ball algorithm to detect depletions or enhancements in the
       density.
       Parameters:
       points: array-like containing X and Y data, where Y is the ion density
       alpha: size factor for the determination of the Alpha shape of the data
    '''
    def __init__(self, inst):
        self.inst = filter_inst(inst)
        self.points = np.column_stack([self.inst['slt'], self.inst['filt_density']]) 
        if self.points.shape[0] < 7:
            raise ValueError("input array must have at least seven rows")
        self.scale_factor = None
        self._scale_points()
        self.tri = Delaunay(self.points)
        self.simplexes = np.asarray(np.sort(self.tri.simplices))
        self.alpha_complex = None

    def _scale_points(self):
        '''prepare data for triangulation if it is not already done
           it is recommended that the density be interpreted on a log scale
           it is also recommended that the density data be smoothed somewhat
           nan values cannot be included
           for the triangulation the axes must be scaled for geometric reasons
           I'm not entirely sure why, perhaps a precision issue, but the
           triangulation is much better if both axes are scaled similarly
           Parameters:
           points: array-like containing X and Y data, where Y is ion density
        '''
        self._get_scale_factor()
        self.points[:, 0] /= np.sqrt(self.scale_factor)
        self.points[:, 1] *= np.sqrt(self.scale_factor)

    def _get_scale_factor(self):
        '''returns scaling factor for ion density in delaunay triangulation
           the ion density must be scaled so that the delaunay triangulation
           produces meaningful geometry for the detection of bubbles and
           background density.
           More specifically: If the delta X (or time) between two distant
           points is smaller than the delta Y for two points closer together,
           a tringle edge will be placed between the two points closer together
           This creates a triangle that goes 'through' the ion density curve,
           corrupting the alpha shape desired for the detection of bubbles.
           So we make the largest delta y the same as the smallest delta x
           parameters:
           x : array-like, typically solar or magnetic local time
           y : array-like, ion density
        '''
        xdiff = np.diff(self.points[:, 0])
        ydiff = np.diff(self.points[:, 1])
        xdiff = np.sort(np.abs(xdiff))
        minxdiff = next((x for x in xdiff if x > 0), None)
        maxydiff = np.max(np.abs(ydiff))
        self.scale_factor = minxdiff / maxydiff

    def _tri_area(self, simplex):
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

    def _circumcircle(self, points, simplex):
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
        result = list(filter(lambda simplex:
                             self._circumcircle(self.points, simplex)[1] > alpha
                             and c * self._tri_area(simplex) > 0,
                             self.simplexes))
        if result:
            self.alpha_complex = np.stack(result)
        else:
            self.alpha_complex = np.array([])

    def plot_delaunay(self):
        '''plots the curve and the triangulation'''
        import matplotlib.pyplot as plt

        plt.triplot(self.points[:, 0], self.points[:, 1],
                    self.tri.simplices.copy())
        plt.plot(self.points[:, 0], self.points[:, 1])
        plt.show()

    def get_background(self):
        '''gets the background density from alpha shell'''
        return np.unique(self.alpha_complex.flatten())

    def locate_depletions(self):
        '''using the upper alpha shape of the density curve this method locates
           depletions in density from the background density.
           the current behavior is adapted from Smith et. al. 2017/18
           For consistency with the source the density data is reverted to its
           original (non log, and non linearly scaled) for the calculation
           of the discrete depth (deltaN / N: d_n) and a shape value that
           ensures the depletion is 'deeper' than it is 'wide'
        '''
        depletions = []
        upper_envelope = self.get_background()
        if upper_envelope.size <= 0:
            self.depletions = depletions
            return

        delta_t = np.diff(self.points[upper_envelope, 0])
        ind, = np.where(delta_t > 0)
        for i in ind:
            lead = upper_envelope[i]
            trail = upper_envelope[i+1]
            d_t = delta_t[i]
            dens = self.points[lead:trail, 1]
            sqsf = np.sqrt(self.scale_factor)
            min_edge = (np.min([dens[0], dens[-1]])) / sqsf
            min_dens = (np.min(dens)) / sqsf
            d_n = (10**min_edge - 10**min_dens) / 10**min_edge
            if d_n > .1 and d_t/d_n < .9/sqsf:
                depletions.append([lead, trail])
        self.depletions = depletions

    def collate_bubble_data(self):
        '''
        at each edge get:
        apex altitude
        altitude
        glon
        glat
        mlon
        mlat
        density
        vz
        slt
        ut

        within edges get:
        min density
        min vz
        max vz
        '''
        if not self.depletions:
            print('No Depletions')
            return
        edge_measures = ['time', 'slt', 'glon', 'glat', 'mlt', 'mlat', 'altitude', 'apex_altitude',
                       'ionDensity', 'ionVelmeridional']
        properties = ['lead', 'trail', 'ut_l', 'ut_t', 'slt_l', 'slt_t', 'glon_l', 'glon_t', 'glat_l', 'glat_t',
                     'mlt_l', 'mlt_t', 'mlat_l', 'mlat_t', 'alt_l', 'alt_t', 'apex_alt_l', 'apex_alt_t',
                     'ion_dens_l', 'ion_dens_t', 'vel_mer_l', 'vel_mer_t', 'min_dens', 'min_vel', 
                     'max_vel', 'rpa_flag', 'dm_flag', 'apex_width', 'depth', 'norm_depth']

        depletion_properties = []
        times = []
        for edges in self.depletions:
            lead = edges[0]
            trail = edges[1]
            dep_props = [lead] # lead
            dep_props.append(trail) # trail
            # add all of the measurements from the edges in order
            for name in edge_measures:
                dep_props.append(self.inst[lead, name]) # lead_measurment
                dep_props.append(self.inst[trail, name]) # trail_measurement

            # add the internal depletion measurements
            dep_props.append(np.min(self.inst[lead:trail, 'ionDensity'])) # min_dens
            dep_props.append(np.min(self.inst[lead:trail, 'ionVelmeridional'])) # min_vel
            dep_props.append(np.max(self.inst[lead:trail, 'ionVelmeridional'])) # max_vel
            dep_props.append(np.max(self.inst[lead:trail, 'RPAflag'])) # rpa_flag
            dep_props.append(np.max(self.inst[lead:trail, 'driftMeterflag'])) # dm_flag
            dep_props.append(np.abs(self.inst[lead, 'apex_altitude'] \
                             - self.inst[trail, 'apex_altitude'])) # apex_width
            depth = np.max([self.inst[lead, 'ionDensity'], self.inst[trail, 'ionDensity']]) \
                    - np.min(self.inst[lead:trail, 'ionDensity']) # depth
            dep_props.append(depth)
            dep_props.append(depth / np.max([self.inst[lead, 'ionDensity'], self.inst[trail, 'ionDensity']])) # norm_depth

            times.append(self.inst.index[lead])
            depletion_properties.append(dep_props)
        return xr.DataArray(depletion_properties, coords=[times, properties], dims=['time', 'properties'])


def climate_survey(start=None, stop=None, save=True):
    if start is None or stop is None:
        print('must include start and stop datetimes')
        return
    clean_level = 'none'
    info = {'index': 'slt', 'kind': 'local time'}
    Ivm = pysat.Instrument(platform='cnofs', name='ivm',
                           orbit_info=info, clean_level=clean_level)
    Ivm.bounds = (start, stop)
    bit = 0
#    Ivm.download(start, stop)
    for orbit_count, ivm in enumerate(Ivm.orbits):
        ivm.data = ivm.data.resample('1S', label='left').ffill(limit=7)
        try:
            orbit = OrbitalBallRoller(ivm)
        except ValueError as err:
            print(err)
            continue
        orbit.get_alpha_complex(400)
        orbit.locate_depletions()
        out = orbit.collate_bubble_data()
        if bit == 1 and out is not None:
            bubble_array = xr.concat([bubble_array, out], dim='time')
        elif bit == 0 and out is not None:
            bubble_array = out
            bit += 1
            continue
        elif bit == 0 and out is None:
            continue
    if save:
        bubble_array.to_netcdf('bubble_properties.nc')
    return bubble_array


def run_survey():
    start = datetime(2008, 8, 1)
    stop = datetime(2015, 7, 1)
    try:
        climate_survey(start=start, stop=stop)
    except:
        logging.exception('uh oh:')
