from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object
import h5py
import os
os.environ['OPL_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
import numpy as np
import numpy.random as npr
#import lru_cache as lru
from . import _interp
from lru import LRU

def isclose(a,b, rtol = 1e-5):
    return np.abs(a-b) / np.abs(a) <= rtol

class TransitionDataBottleneck(object):
    
    def __init__(self, filename, master_filename = None, check = True,
            memory = False):
        assert 3/2 == 1.5
        self._links = {}
        self._filename = filename
        self._N = None
        self._shape = None
        self._frequencies = None
        self._us = set()
        self._nbs = set()
        self._sorted_us = None
        self._sorted_nbs = None
        self._hdf5_file = None
        self._check_integrity = check
        self._memory = memory
        self._add_master_file()
        self._cache = LRU(10000)
            
    def _add_dataset(self, dataset, dataset_name):
        '''
        Adds a particular dataset to the dict of dataset names, keys being
        (bottleneck_size, u). Also adds bottleneck_size and u to set of
        observed values and checks that N is correct.
        '''
        attr = dataset.attrs
        N = attr['N']
        bottleneck_size = attr['Nb']
        u = attr['u']
        key = (bottleneck_size, u)
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
                err_msg = 'Found duplicate bottleneck size / mutation rate \
                        combination in matrices'
                raise Exception(err_msg)
        if not self._memory:
            self._links[key] = dataset_name
        else:
            self._links[key] = np.array(dataset[:,:].copy())
        self._nbs.add(bottleneck_size)
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
        self._sorted_nbs = np.array(sorted(list(self._nbs)), dtype = np.int)
        self._min_bottleneck_size = self._sorted_nbs.min()
        self._max_bottleneck_size = self._sorted_nbs.max()
        self._min_mutation_rate = self._sorted_us.min() * 2 * self._N
        self._max_mutation_rate = self._sorted_us.max() * 2 * self._N
        if not self._memory:
            self._hdf5_file = h5py.File(self._filename, 'r')
        
    def get_transition_probabilities_2d(self, nb,
            scaled_mut):
        key = (nb, scaled_mut)
        if key in self._cache:
            val = self._cache[key]
        else:
            val = self.get_transition_probabilities_2d_not_cached(
                    nb,
                    scaled_mut)
            self._cache[key] = val
        return val

    def get_transition_probabilities_2d_not_cached(self,
            nb, scaled_mut):
        '''
        Returns the interpolated transition probabilities
        
        nb            bottleneck size
        scaled_mut    2Nu, where u is the mutation rate and N is the haploid
                      population size
        '''

        if nb < self._min_bottleneck_size:
            raise ValueError('bottleneck size too small: {} < {} (min)'.format(
                nb, self._min_bottleneck_size))
        if nb > self._max_bottleneck_size:
            raise ValueError('bottleneck size too large: {} > {} (max)'.format(
                nb, self._max_bottleneck_size))
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
        nb_idx = np.searchsorted(self._sorted_nbs, nb)
        desired_u = scaled_mut / (2.0 * self._N)
        u_idx = np.searchsorted(self._sorted_us, desired_u)

        # bilinear interpolation for both Nb and mutation rate
        '''
        Now interpolate to get distribution. Use bilinear interpolation,
        which will always produce a valid distribution at every point
        (always just a linear combination of the nearest four points) where
        the weights of the points sum to 1.

        See https://en.wikipedia.org/wiki/Bilinear_interpolation.  Using
        that notation, here we interpolate. x is time, y is selection
        '''
        t = nb
        u = desired_u
        t1 = self._sorted_nbs[nb_idx-1]
        t2 = self._sorted_nbs[nb_idx]
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
    
    #@lru.lru_cache(maxsize = 10000)
    def get_distribution(self, nb, u):
        key = (nb, u)
        if not self._memory:
            dataset_name = self._links[key]
            transition_matrix = self._hdf5_file[dataset_name][:,:].copy()
        else:
            transition_matrix = self._links[key].copy()
        return transition_matrix
    
    def get_bin_frequencies(self):
        return self._frequencies

    def get_breaks(self):
        return self._breaks

    def get_N(self):
        return self._N

    def get_min_bottleneck_size(self):
        return self._min_bottleneck_size

    def get_max_bottleneck_size(self):
        assert 3/2 == 1.5
        return self._max_bottleneck_size

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
