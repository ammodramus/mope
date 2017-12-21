from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object
import h5py
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np
import numpy.random as npr
#import lru_cache as lru
from . import _interp
from lru import LRU
from .make_transition_matrices import bin_matrix

def isclose(a,b, rtol = 1e-5):
    return np.abs(a-b) / np.abs(a) <= rtol

class TransitionData(object):
    
    def __init__(self, filename, master_filename = None, check = True,
            memory = False):
        assert 3/2 == 1.5
        self._links = {}
        self._filename = filename
        self._N = None
        self._shape = None
        self._frequencies = None
        self._us = set()
        self._gens = set()
        self._sorted_us = None
        self._sorted_gens = None
        self._hdf5_file = None
        self._check_integrity = check
        self._memory = memory
        self._cache = LRU(500)

        self._add_master_file()
            
    def _add_dataset(self, dataset, dataset_name):
        '''
        Adds a particular dataset to the dict of dataset names, keys being
        (gen, u). Also adds gen and u to set of observed values and checks
        that N is correct.
        '''
        attr = dataset.attrs
        N = attr['N']
        gen = attr['gen']
        u = attr['u']
        key = (gen, u)
        if self._check_integrity:
            if self._N is None:
                self._N = N
            elif N != self._N:
                raise Exception('Found multiple population sizes in master \
                        file\'s datasets')
            if self._shape is None:
                self._shape = dataset.shape
            elif dataset.shape != self._shape:
                raise Exception('Found multiple distinct shapes of matrices \
                        in file')
            if key in self._links:
                print(key)
                err_msg = 'Found duplicate generation / mutation rate \
                        combination in matrices'
                raise Exception(err_msg)
        if not self._memory:
            self._links[key] = dataset_name
        else:
            self._links[key] = np.array(dataset[:,:].copy())
        self._gens.add(gen)
        self._us.add(u)
        
    def _add_master_file(self):
        '''
        adds and initializes the datasets in a master file, making ready
        for obtaining distributions
        
        filename   filename of master file
        '''
        with h5py.File(self._filename, 'r') as mf:
            for dsname in mf:
                ds = mf[dsname]
                self._add_dataset(ds, dsname)
            self._frequencies = mf.attrs['frequencies']
            self._breaks = mf.attrs['breaks']
        self._sorted_us = np.array(sorted(list(self._us)), dtype = np.float64)
        #self._sorted_gens = np.array(sorted(list(self._gens)), dtype = np.int)
        self._sorted_gens = np.array(sorted(list(self._gens)), dtype = np.float)
        # check missing members of grid
        missings = []
        for g in self._sorted_gens:
            for u in self._sorted_us:
                key = (g,u)
                if key not in self._links:
                    missings.append(key)
        if len(missings) != 0:
            import sys
            for m in missings:
                print('missing:', m, file=sys.stderr)
            raise ValueError('incomplete transition probability grid')

        self._min_coal_time = self._sorted_gens.min() / self._N
        self._max_coal_time = self._sorted_gens.max() / self._N
        self._min_mutation_rate = self._sorted_us.min() * 2 * self._N
        self._max_mutation_rate = self._sorted_us.max() * 2 * self._N
        if not self._memory:
            self._hdf5_file = h5py.File(self._filename, 'r')
        
    def get_transition_probabilities_time(self, scaled_time):
        desired_gen_time = self._N * scaled_time
        rounded_gen_time = np.round(desired_gen_time)
        if isclose(desired_gen_time, rounded_gen_time):
            desired_gen_time = rounded_gen_time
        sorted_gens = self._sorted_gens
        gen_idx = np.searchsorted(sorted_gens, desired_gen_time)
        # exact_found == False, here
        if gen_idx == 0 or gen_idx == sorted_gens.shape[0]:
            return None
        t = desired_gen_time
        t0 = self._sorted_gens[gen_idx-1]
        t1 = self._sorted_gens[gen_idx]
        #f0 = self.get_distribution(t1,0)  # bug found 15/08/2016 10:00
        f0 = self.get_distribution(t0,0)
        f1 = self.get_distribution(t1,0)
        frac_t1 = (t-t0)/(t1-t0)
        distn = f0*(1-frac_t1) + f1*(frac_t1)
        return distn

    def get_transition_probabilities_time_mutation(self, scaled_time,
            scaled_mut):
        key = (scaled_time, scaled_mut)
        if key in self._cache:
            return self._cache[key]
        else:
            val = self.get_transition_probabilities_time_mutation_not_cached(
                    scaled_time,
                    scaled_mut)
            self._cache[key] = val
            return val

    def get_transition_probabilities_time_mutation_not_cached(self,
            scaled_time, scaled_mut):
        '''
        Returns the interpolated transition probabilities
        
        scaled_time   time, scaled by N
        scaled_mut    2Nu, where u is the mutation rate and N is the haploid
                      population size
        '''

        if scaled_time < self._min_coal_time:
            raise ValueError('drift time too small: {} < {} (min)'.format(
                scaled_time, self._min_coal_time))
        if scaled_time > self._max_coal_time:
            raise ValueError('drift time too large: {} > {} (max)'.format(
                scaled_time, self._max_coal_time))
        if scaled_mut < self._min_mutation_rate:
            raise ValueError('mutation rate too small: {} < {} (min)'.format(
                scaled_mut, self._min_mutation_rate))
        if scaled_mut > self._max_mutation_rate:
            raise ValueError('mutation rate too large: {} > {} (max)'.format(
                scaled_mut, self._max_mutation_rate))

        '''
        searchsorted behavior:

        np.searchsorted(a, v, side = 'left')

        finds the indices of a where the elements of v can be inserted so as to
        maintain order in a. by default (with side = 'left'), the first
        suitable position is given.
        '''
        desired_gen_time = self._N * scaled_time
        gen_idx = np.searchsorted(self._sorted_gens, desired_gen_time)
        desired_u = scaled_mut / (2.0 * self._N)
        u_idx = np.searchsorted(self._sorted_us, desired_u)

        #if gen_idx > 1:
        #    return self.bilinear_interpolation(desired_gen_time, desired_u,
        #            gen_idx, u_idx)
        #else:
        #    return self.biquadratic_interpolation(desired_gen_time, desired_u,
        #            gen_idx, u_idx)

        return self.bilinear_interpolation(desired_gen_time, desired_u,
                gen_idx, u_idx)

    def bilinear_interpolation(self, desired_gen_time, desired_u, gen_idx,
            u_idx):
        # bilinear interpolation for both gen and mutation rate
        '''
        Now interpolate to get distribution. Use bilinear interpolation,
        which will always produce a valid distribution at every point
        (always just a linear combination of the nearest four points) where
        the weights of the points sum to 1.

        See https://en.wikipedia.org/wiki/Bilinear_interpolation.  Using
        that notation, here we interpolate. x is time, y is selection
        '''
        t = desired_gen_time
        u = desired_u
        t1 = self._sorted_gens[gen_idx-1]
        t2 = self._sorted_gens[gen_idx]
        u1 = self._sorted_us[u_idx-1]
        u2 = self._sorted_us[u_idx]
        fQ11 = self.get_distribution(t1,u1)
        fQ12 = self.get_distribution(t1, u2)
        fQ21 = self.get_distribution(t2, u1)
        fQ22 = self.get_distribution(t2, u2)

        #distn = ((fQ11*((t2-t)*(u2-u)) + fQ21*((t-t1)*(u2-u)) +
        #    fQ12*((t2-t)*(u-u1)) + fQ22*((t-t1)*(u-u1))) / ((t2-t1)*(u2-u1)))
        denom = ((t2-t1)*(u2-u1))
        weight11 = (t2-t)*(u2-u) / denom
        weight21 = (t-t1)*(u2-u) / denom
        weight12 = (t2-t)*(u-u1) / denom
        weight22 = (t-t1)*(u-u1) / denom

        #distn = fQ11*weight11 + fQ21*weight21 + fQ12*weight12 + fQ22*weight22
        distn = _interp.bilinear_interpolate(
                fQ11,
                fQ12,
                fQ21,
                fQ22,
                denom,
                weight11,
                weight12,
                weight21,
                weight22)

        return distn


    def biquadratic_interpolation(self, desired_gen_time, desired_u, gen_idx,
            u_idx):
        '''
        Perform a quadratic interpolation along the generations axis and a
        linear interpolation along the mutation axis
        '''
        assert gen_idx < 2  # only do this for very low drift times

        t = desired_gen_time
        t1 = self._sorted_gens[gen_idx-1]
        t2 = self._sorted_gens[gen_idx]
        t3 = self._sorted_gens[gen_idx+1]

        u = desired_u
        u1 = self._sorted_us[u_idx-1]
        u2 = self._sorted_us[u_idx]

        fQ11 = self.get_distribution(t1, u1)
        fQ12 = self.get_distribution(t1, u2)
        fQ21 = self.get_distribution(t2, u1)
        fQ22 = self.get_distribution(t2, u2)
        fQ31 = self.get_distribution(t3, u1)
        fQ32 = self.get_distribution(t3, u2)

        # get coefficients of Lagrange interpolation polynomial for three terms
        c = np.zeros(3)
        c[0] = (t-t2)*(t-t3)/((t1-t2)*(t1-t3))
        c[1] = (t-t1)*(t-t3)/((t2-t1)*(t2-t3))
        c[2] = (t-t1)*(t-t2)/((t3-t1)*(t3-t2))
        #c[c < 0] = 0.0
        #c /= c.sum()
        c1, c2, c3 = c



        # interpolated distribution for mutation rate 1
        iQu1 = c1*fQ11
        iQu1 += c2*fQ21
        iQu1 += c3*fQ31

        pos = iQu1 < 0
        #print '-----'
        #print fQ11[pos][:10]
        #print
        #print fQ21[pos][:10]
        #print
        #print fQ31[pos][:10]
        #print
        #print iQu1[pos][:10]
        #print t1, t2, t3

        # interpolated distribution for mutation rate 2
        iQu2 = c1*fQ12
        iQu2 += c2*fQ22
        iQu2 += c3*fQ32

        # linearly interpolate between mutation rates 
        # first, get coefficients
        coef1 = (u2-u)/(u2-u1)
        coef2 = (u-u1)/(u2-u1)

        # (linearly) interpolate
        distn = coef1*iQu1 + coef2*iQu2

        return distn

    
    def get_distribution(self, gen, u):
        key = (gen, u)
        if not self._memory:
            dataset_name = self._links[key]
            transition_matrix = self._hdf5_file[dataset_name][:,:].copy()
        else:
            transition_matrix = self._links[key].copy()
        return transition_matrix

    def get_identity_matrix(self, u):
        N = self._N
        breaks = self._breaks
        # N+1 x N+1 identity matrix, plus 
        diag = np.diag(np.repeat(1.0-2*u, N+1))
        above_diag = np.diag(np.repeat(u, N), 1)
        below_diag = np.diag(np.repeat(u, N), -1)
        P = diag + above_diag + below_diag
        P[0,0] += u
        P[-1,-1] += u
        P = bin_matrix(P, breaks)
        return P
    
    def get_bin_frequencies(self):
        return self._frequencies

    def get_breaks(self):
        return self._breaks

    def get_N(self):
        return self._N

    def get_min_coal_time(self):
        return self._min_coal_time

    def get_max_coal_time(self):
        assert 3/2 == 1.5
        return self._max_coal_time

    def get_min_mutation_rate(self):
        '''
        returns min mutation rate, scaled by 2N
        '''
        return self._min_mutation_rate
    
    def get_max_mutation_rate(self):
        '''
        returns max mutation rate, scaled by 2N
        '''
        return self._max_mutation_rate

    def close(self):
        self._hdf5_file.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, x,y,z):
        self.close()

    def clear_cache(self):
        self._cache.clear()
