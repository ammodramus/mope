from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
import argparse
import h5py
import os 
import numpy as np
import sys
import numpy.linalg as npl
import logging
import warnings
import gc
import scipy.stats as st
from numpy.core.numeric import binary_repr, asanyarray
from numpy.core.numerictypes import issubdtype
from .generate_transitions_util import log_matrix_power, logdot

from . import util as ut
from . import _transition as trans

def get_bottleneck_transition_matrix(N, Nb, mu):
    '''
    get transition matrix under a model of instantaneous bottleneck
    followed by doubling to the initial population size, N

    N     original haploid population size
    Nb    haploid bottleneck size
    mu     symmetric per-generation mutation probability
    '''
    assert 3/2 == 1.5  # check for __future__ division
    if Nb >= N:
        raise ValueError('Nb must be < N')

    num_doublings = np.floor(np.log2(N/Nb)).astype(int)
    num_gens = np.ceil(np.log2(N/Nb)).astype(int)

    Ns = [N, Nb] 
    for d in range(1,num_doublings+1):
        Ns.append(Nb*2**d)
    if num_gens > num_doublings:
        Ns.append(N)
    P = trans.get_transition_matrix_var_N(Ns[0], Ns[1], N, mu)
    for N1, N2 in zip(Ns[1:-1], Ns[2:]):
        Pp = trans.get_transition_matrix_var_N(N1, N2, N, mu)
        P = np.dot(P, Pp)
    return P


def get_wright_fisher_transition_matrix(N, s, u, v):
    assert 3 / 2 == 1.5, "check for __future__ division"
    P = np.matrix(np.zeros((N + 1, N + 1), dtype=np.float64))
    js = np.arange(0, N + 1)
    for i in range(N + 1):
        p = i / N
        pmut = (1 - u) * p + v * (1 - p)  # first mutation, then selection
        pstar = pmut * (1 + s) / (pmut * (1 + s) + 1 - pmut)
        P[i, :] = np.exp(st.binom.logpmf(js, N, pstar))
    return P

def get_log_wright_fisher_transition_matrix(N, s, u, v):
    assert 3/2 == 1.5  # check for __future__ division
    lP = np.matrix(np.zeros((N+1, N+1), dtype = np.float64))
    js = np.arange(0,N+1)
    for i in xrange(N+1):
        p = i/N
        pmut = (1-u)*p + v*(1-p) # first mutation, then selection
        pstar = pmut*(1+s) / (pmut*(1+s) + 1-pmut)
        lP[i,:] = st.binom.logpmf(js, N, pstar)
    return lP


def get_breaks(N, uniform_weight, min_bin_size):
    '''
    returns the indices in {0,1,...,N} that define quantiles at least as large
    as target_bin_size of the distribution of X*U + (1-X)*S, where U is
    Uniform over {1,2,...,N-1}, and S has a (discrete) distribution
    proportional to {1/i} for i \in {1,2,...,N-1}, and X is
    Bernoulli(uniform_weight). 0 and N are always included as distinct bins.

    params

    N                  population size and number of bins minus 1

    uniform_weight     how much to weight the uniform distribution rather than
                       the theoretical prediction of \propto 1/i

    min_bin_size       the minimum total probability that a sequence of
                       adjacent bins need to have in order to be considered as
                       a bin. the final bin just before the fixation class may
                       have *more* than target_bin_size probability in this
                       mixture.

    return value

    breaks             numpy 1-d array of ints giving the lower bounds of the
                       breaks. I.e., the i'th bin corresponds to the following
                       zero-indexed entries of {0,1,2,...,N}:
                       {bins[i], bins[i]+1, ..., bins[i+1]-1}. The final bin,
                       which is always included, corresponds to the entry
                       bins[-1] == N.
    '''
    assert 0 <= uniform_weight and uniform_weight <= 1
    assert 0 < min_bin_size and min_bin_size <= 1
    assert 3/2 == 1.5

    w = uniform_weight
    u = np.ones(N-1)
    u /= u.sum()
    s = 1/np.arange(1,N)
    s /= s.sum()
    interior_probs = w*u + (1-w)*s

    breaks = [0,1]
    cur_prob = 0.0
    for i, prob in zip(xrange(1, N), interior_probs):
        cur_prob += prob
        if cur_prob >= min_bin_size:
            breaks.append(i+1)
            cur_prob = 0.0
    breaks.append(N)
    breaks = np.array(breaks, dtype = np.int)
    return breaks

def get_breaks_symmetric(N, uniform_weight, min_bin_size):
    '''
    Just like get_breaks, but makes the breaks symmetric about the middle
    frequency. The middle frequency gets its own bin. Because the
    interpretation of min_bin_size stays the same, this will return more breaks
    than the non-symmetric version. Something to keep in mind.

    params

    N                  population size and number of bins minus 1

    uniform_weight     how much to weight the uniform distribution rather than
                       the theoretical prediction of \propto 1/i

    min_bin_size       the minimum total probability that a sequence of
                       adjacent bins need to have in order to be considered as
                       a bin. the final bin just before the fixation class may
                       have *more* than target_bin_size probability in this
                       mixture.

    return value

    breaks             numpy 1-d array of ints giving the lower bounds of the
                       breaks. I.e., the i'th bin corresponds to the following
                       zero-indexed entries of {0,1,2,...,N}:
                       {bins[i], bins[i]+1, ..., bins[i+1]-1}. The final bin,
                       which is always included, corresponds to the entry
                       bins[-1] == N.
    '''
    assert 0 <= uniform_weight and uniform_weight <= 1
    assert 0 < min_bin_size and min_bin_size <= 1
    assert 3/2 == 1.5

    if N % 2 != 0:
        raise ValueError('population size (N) must be even')

    w = uniform_weight
    u = np.ones(N-1)
    u /= u.sum()
    s = 1/np.arange(1,N)
    s /= s.sum()
    interior_probs = w*u + (1-w)*s

    breaks = [0,1]
    cur_prob = 0.0
    for i, prob in zip(xrange(1, N), interior_probs):
        cur_prob += prob
        if cur_prob >= min_bin_size:
            breaks.append(i+1)
            cur_prob = 0.0
        if i >= N/2-1:
            break 
    if breaks[-1] != N/2:
        breaks.append(N/2)
    breaks.append(N/2+1)
    lesser_breaks = [el for el in breaks[::-1] if el < N/2]
    for br in lesser_breaks[:-1]:
        breaks.append(N-br+1)
    breaks = np.array(breaks, dtype = np.int)
    return breaks

def bin_matrix(P, breaks):
    assert 3/2 == 1.5
    breaks = np.array(breaks)
    P_binned = np.zeros((breaks.shape[0], breaks.shape[0]))
    P_colsummed = np.add.reduceat(P, breaks, axis = 1)
    bin_lengths = np.concatenate((np.diff(breaks), [P.shape[1]-breaks[-1]]))
    break_is_even = (bin_lengths % 2 == 0)
    break_is_odd = np.logical_not(break_is_even)
    middles = ((bin_lengths-1)/2).astype(np.int)
    P_binned[break_is_odd,:] = P_colsummed[(breaks+middles)[break_is_odd],:]
    left_middles = np.floor((bin_lengths-1)/2).astype(np.int)
    right_middles = np.ceil((bin_lengths-1)/2).astype(np.int)
    P_binned[break_is_even,:] = (P_colsummed[breaks+left_middles,:][break_is_even,:] +
                           P_colsummed[breaks+right_middles,:][break_is_even,:]) / 2
    return P_binned
def get_binned_frequencies(N, breaks):
    assert 3/2 == 1.5
    full_val = np.arange(N+1) / N
    bin_lengths = np.concatenate((np.diff(breaks), [N+1-breaks[-1]]))
    vals = np.add.reduceat(full_val, breaks) / bin_lengths
    return vals


def get_next_matrix_with_prev(cur_matrix, cur_power, next_power, P,
                                  log_space=False):
    step_power = next_power - cur_power
    if log_space:
        lP_step = log_matrix_power(P, step_power)
        lnext_P = logdot(cur_matrix, lP_step)
        retP = lnext_P
    else:
        P_step = npl.matrix_power(P, step_power)
        next_P = np.matmul(cur_matrix, P_step)
        retP = next_P
    return retP

def get_identity_matrix(N, u, breaks):
    # N+1 x N+1 identity matrix, plus 
    diag = np.diag(np.repeat(1.0-2*u, N+1))
    above_diag = np.diag(np.repeat(u, N), 1)
    below_diag = np.diag(np.repeat(u, N), -1)
    P = diag + above_diag + below_diag
    P[0,0] += u
    P[-1,-1] += u
    if breaks is not None:
        P = bin_matrix(P, breaks)
    return P

def add_matrix(h5file, P, N, s, u, v, gen, idx, breaks = None):
    '''
    add a transition matrix to an HDF5 file

    h5file   file for outputting matrix (h5py File object)
    P        transition matrix
    N        population size
    s        selection coefficient
    u        mutation probability away from focal allele
    v        mutation probability towards the focal allele
    gen      generation
    idx      index of the dataset in the hdf5 file
    breaks   tuple of uniform_weight and min_bin_size (see get_breaks())
    '''
    if np.any(np.isnan(P)):
        warnings.warn('converting NaNs to zeros in add_matrix')
        P[np.isnan(P)] = 0.0
    if breaks is not None:
        P = bin_matrix(P, breaks)
    print('row sums:', P.sum(1).tolist())
    #assert np.all(np.isfinite(P)), "not all elements of P are finite"
    group_name = "P" + str(idx)
    dset = h5file.create_dataset(group_name,
            data = np.array(P, dtype = np.float64))
    dset.attrs['N'] = N
    dset.attrs['s'] = s
    dset.attrs['u'] = u
    dset.attrs['v'] = v
    dset.attrs['gen'] = gen
    return

def add_matrix_bot(h5file, P, N, Nb, u, idx, breaks = None):
    '''
    add a transition matrix to an HDF5 file

    h5file   file for outputting matrix (h5py File object)
    P        transition matrix
    N        population size
    s        selection coefficient
    u        symmetric per-generation mutation probability
    idx      index of the dataset in the hdf5 file
    breaks   tuple of uniform_weight and min_bin_size (see get_breaks())
    '''
    if np.any(np.isnan(P)):
        warnings.warn('converting NaNs to zeros in add_matrix_bot')
        P[np.isnan(P)] = 0.0
    if breaks is not None:
        P = bin_matrix(P, breaks)
    group_name = "P" + str(idx)
    dset = h5file.create_dataset(group_name,
            data = np.array(P, dtype = np.float64))
    dset.attrs['N'] = N
    dset.attrs['Nb'] = Nb
    dset.attrs['u'] = u
    return

def _run_generate(args):

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)-15s %(name)-5s %(levelname)-8s MEM: %(memusg)-15s %(message)s')
    logger = logging.getLogger('gendrift')
    f = ut.MemoryFilter()
    logger.addFilter(f)

    logger.debug('opening output file {}'.format(args.output))
    with h5py.File(args.output, 'w') as h5file:

        h5file.attrs['N'] = args.N
        if args.breaks is not None:
            uniform_weight = args.breaks[0]
            min_bin_size = args.breaks[1]
            breaks = get_breaks_symmetric(args.N, uniform_weight,
                    min_bin_size)
            h5file.attrs['breaks'] = breaks
            h5file.attrs['min_bin_size'] = min_bin_size
            h5file.attrs['uniform_weight'] = uniform_weight
            frequencies = get_binned_frequencies(args.N, breaks)
        else:
            breaks = None
            frequencies = np.arange(args.N+1)/args.N
            h5file.attrs['breaks'] = 0
        h5file.attrs['frequencies'] = frequencies

        if args.bottlenecks:
            dataset_idx = 0
            Nbs = []
            with open(args.inputfile) as nb_in:
                for line in nb_in:
                    try:
                        Nb = int(line.strip())
                    except ValueError:
                        raise ValueError("invalid integer in bottleneck sizes \
                                file")
                    Nbs.append(Nb)
            for Nb in Nbs:
                logger.debug('getting bottleneck P, Nb = {}, {} of {}'.format(
                    Nb, Nbs.index(Nb)+1, len(Nbs)))
                P = get_bottleneck_transition_matrix(args.N, Nb, args.u)
                logger.debug('obtained P, adding matrix with index {}'.format(
                    dataset_idx))
                add_matrix_bot(h5file, P, args.N, Nb, args.u, dataset_idx,
                        breaks)
                dataset_idx += 1

        else:  # not bottlenecks
            logger.debug('calculating Wright-Fisher matrix P')
            gens = []
            with open(args.inputfile) as gen_in:
                for line in gen_in:
                    try:
                        gen = int(line.strip())
                    except ValueError:
                        raise ValueError("invalid integer in generations file")
                    gens.append(gen)
            if args.log:
                lP = get_log_wright_fisher_transition_matrix(args.N, args.s, args.u, args.v)
                if np.any(np.isnan(lP)):
                    warnings.warn('log WF transition matrix P contains nans')
                    warnings.warn('number of nans: {} of {}'.format(
                        np.isnan(P).sum(),
                        P.shape[0] * P.shape[1]))
                logger.debug('log Wright-Fisher matrix P obtained')
                dataset_idx = 0
                logger.debug('calculating P_prime')
                lP_prime = log_matrix_power(lP, gens[0])
                logger.debug('P_prime obtained, adding matrix')
                add_matrix(h5file, np.exp(lP_prime), args.N, args.s, args.u, args.v,
                        gens[0], dataset_idx, breaks)
                logger.debug('adding matrix')
                dataset_idx += 1
                prev_gen = gens[0]
                for gen in gens[1:]:
                    logger.debug('calculating P prime')
                    lP_prime = get_next_matrix_with_prev(
                            lP_prime, prev_gen, gen, lP, log_space=True)
                    logger.debug('P_prime calculated for gen {}, '
                                 'adding matrix'.format(gen))
                    add_matrix(h5file, np.exp(lP_prime), args.N, args.s, args.u, args.v,
                            gen, dataset_idx, breaks)
                    logger.debug('matrix added')
                    dataset_idx += 1
                    prev_gen = gen
                    gc.collect()   # explicit garbage collection
            else:
                P = get_wright_fisher_transition_matrix(args.N, args.s, args.u,
                                                        args.v)
                if np.any(np.isnan(P)):
                    warnings.warn('WF transition matrix P contains nans')
                    warnings.warn('number of nans: {} of {}'.format(
                        np.isnan(P).sum(),
                        P.shape[0] * P.shape[1]))
                logger.debug('Wright-Fisher matrix P obtained')
                dataset_idx = 0
                logger.debug('calculating P_prime')
                P_prime = npl.matrix_power(P, gens[0])
                logger.debug('P_prime obtained, adding matrix')
                add_matrix(h5file, P_prime, args.N, args.s, args.u, args.v,
                        gens[0], dataset_idx, breaks)
                logger.debug('adding matrix')
                dataset_idx += 1
                prev_gen = gens[0]
                for gen in gens[1:]:
                    logger.debug('calculating P prime')
                    P_prime = get_next_matrix_with_prev(
                            P_prime, prev_gen, gen, P, log_space=False)
                    logger.debug('P_prime calculated for gen {}, '
                                 'adding matrix'.format(gen))
                    add_matrix(h5file, P_prime, args.N, args.s, args.u, args.v,
                            gen, dataset_idx, breaks)
                    logger.debug('matrix added')
                    dataset_idx += 1
                    prev_gen = gen
                    gc.collect()   # explicit garbage collection


def _run_master(args):
    if args.out_file in args.files:
        raise argparse.ArgumentError(
            "--out-file value cannot be a provided file")
    master_filename = args.out_file
    filenames = args.files
    mf = h5py.File(master_filename, 'w')
    linked_dataset_idx = 0
    freqs = None
    breaks = None
    N = None
    for target_fn in filenames:
        f_target = h5py.File(target_fn, 'r')
        for ds in f_target:
            name = "D" + str(linked_dataset_idx)
            mf[name] = h5py.ExternalLink(target_fn, ds)
            linked_dataset_idx += 1
        f_freqs = f_target.attrs['frequencies']
        f_breaks = f_target.attrs['breaks']
        f_N = f_target.attrs['N']
        if freqs is None:
            freqs = f_freqs
        else:
            if not np.array_equal(f_freqs, freqs):
                raise ValueError("found distinct sets of frequencies in "
                        "target datasets")
        if breaks is None:
            breaks = f_breaks
        else:
            if not np.array_equal(breaks, f_breaks):
                raise ValueError("found distinct sets of breaks in \
                        target datasets")
        if N is None:
            N = f_N
        else:
            if N != f_N:
                raise ValueError("found distinct values of N in \
                        target datasets")
        f_target.close()
    mf.attrs['frequencies'] = freqs
    mf.attrs['breaks'] = breaks
    mf.attrs['cmd'] = ' '.join(sys.argv)
    mf.close()


def _run_make_gauss(args):
    with h5py.File(args.outfile, 'w') as h5file:

        h5file.attrs['N'] = args.N
        breaks = get_breaks_symmetric(args.N, args.unif_weight,
                args.min_bin_size)
        h5file.attrs['breaks'] = breaks
        h5file.attrs['min_bin_size'] = args.min_bin_size
        h5file.attrs['uniform_weight'] = args.unif_weight
        frequencies = get_binned_frequencies(args.N, breaks)
        h5file.attrs['frequencies'] = frequencies

        s = 0  # no selection

        dataset_idx = 0
        with open(args.gensfile, 'r') as genin:
            for line in genin:
                t = ut.nonneg_float(line.strip())
                P = get_gauss_matrix_with_breaks(t, args.uv, args.N, breaks)
                P[P<0] = 0  # just in case, for small probabilities
                gen = args.N*t  # floating-point generation
                add_gauss_matrix(h5file, P, args.N, s, args.uv, args.uv,
                        gen, dataset_idx)
                dataset_idx += 1

def _run_gencmd(args):

    if args.big:
        N = 2000
    else:
        N = 1000


    default_gens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24,
            26, 28, 30, 32, 34, 36, 38, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
            100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800,
            900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800,
            3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
            5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200,
            7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000, 9200, 9400,
            9600, 9800, 10000]
    default_bots = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50,
            55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150,
            160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
            425, 450, 475, 500]
    default_mutsels = [1.0e-12, 2.5e-12, 5e-12, 7.5e-12, 1.0e-11, 2.5e-11, 5e-11,
            7.5e-11, 1.0e-10, 2.5e-10, 5e-10, 7.5e-10, 1.0e-9, 2.5e-9, 5e-9,
            7.5e-9, 1.0e-8, 2.5e-8, 5e-8, 7.5e-8, 1.0e-7, 2.5e-7, 5e-7, 7.5e-7,
            1.0e-6, 2.5e-6, 5e-6, 7.5e-6, 1.0e-5, 2.5e-5, 5e-5, 7.5e-5, 1.0e-4,
            2.5e-4, 5e-4, 7.5e-4, 1.0e-3, 2.5e-3, 5e-3, 7.5e-3, 1.0e-2, 2.5e-2,
            5e-2, 7.5e-2]
    default_big_gens = np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52,
        54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 73, 75, 77, 80, 82, 85, 87, 90,
        93, 96, 99, 103, 106, 109, 113, 117, 120, 124, 129, 133, 137, 142, 147,
        152, 157, 162, 168, 173, 179, 185, 192, 198, 205, 212, 220, 227, 235,
        244, 252, 261, 270, 280, 290, 300, 311, 322, 334, 346, 359, 372, 385,
        399, 414, 429, 445, 462, 479, 497, 515, 534, 554, 575, 597, 620, 643,
        668, 693, 720, 747, 776, 806, 837, 869, 903, 939, 975, 1013, 1053,
        1095, 1138, 1183, 1230, 1279, 1330, 1383, 1439, 1497, 1557, 1620, 1686,
        1754, 1826, 1901, 1978, 2060, 2144, 2233, 2325, 2422, 2523, 2628, 2738,
        2852, 2972, 3097, 3228, 3365, 3507, 3656, 3812, 3975, 4146, 4324, 4510,
        4704, 4908, 5121, 5343, 5576, 5820, 6074, 6341, 6620, 6912, 7217, 7537,
        7872, 8222, 8589, 8973, 9375,  9796, 10238, 10700, 11184, 11691, 12222,
        12779, 13363, 13974, 14615, 15287, 15991, 16730, 17504, 18316, 19168,
        20061])

    # write gens
    write_gens = default_gens if not args.big else default_big_gens
    with open('gens.txt', 'w') as fout:
        for gen in write_gens:
            fout.write(str(gen) + '\n')

    if not args.selection:
        # write bottlenecks
        with open('bots.txt', 'w') as fout:
            for bot in default_bots:
                fout.write(str(bot) + '\n')

    # print W-F drift commands
    infile = 'gens.txt'
    prefix = 'mkdir -p transitions/drift_matrices && '
    mutsel_txt = 'sel' if args.selection else 'mut'
    for mutsel in default_mutsels:
        outfile = 'transitions/drift_matrices/drift_matrices_{}_{}.h5'.format(
            mutsel_txt, mutsel)
        if args.selection:
            s = mutsel
            u = 0
        else:
            s = 0
            u = mutsel
        cmd = ('mope generate-transitions {N} {s} {u} {u} '
               '{fin} {output} --breaks 0.5 0.01'.format(
                      N=N, s=s, u=u, output=outfile, fin='gens.txt'))
        print(prefix + cmd)

    # print bottleneck commands
    if not args.selection:
        prefix = 'mkdir -p transitions/bottleneck_matrices && '
        for mutsel in default_mutsels:
            outfile = 'transitions/bottleneck_matrices/bottleneck_matrices_mut_{}.h5'.format(mutsel)
            cmd = ('mope generate-transitions {N} {s} {u} {u} '
                   '{fin} {output} --breaks 0.5 0.01 --bottlenecks'.format(
                       N=N, s=0, u=mutsel, output=outfile, fin='bots.txt'))
            print(prefix + cmd)


    # print master-file generation command
    print("#########################################")
    print('# run these after everything has completed')
    print("#########################################")
    if args.selection:
        outfile = 'selection_transitions.h5'
    else:
        outfile = 'drift_transitions.h5'
    cmd = ('# cd transitions/ && mope make-master-transitions '
           'drift_matrices/*.h5 --out-file {outfile}; '
           'cd ..'.format(outfile=outfile))
    print(cmd)
    if not args.selection:
        cmd = ('# cd transitions/ && mope make-master-transitions '
               'bottleneck_matrices/*.h5 --out-file bottleneck_transitions.h5; '
               'cd ..')
        print(cmd)
