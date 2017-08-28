from __future__ import division
import sys
import argparse
import h5py
import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
import numpy as np

import _transition as trans
import util as ut


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

def check_start_end(start, end):
    if end < start:
        raise argparse.ArgumentTypeError(
                "end generation must be >= start generation")


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


def add_matrix(h5file, P, N, Nb, u, idx, breaks = None):
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
    if breaks is not None:
        P = bin_matrix(P, breaks)
    group_name = "P" + str(idx)
    dset = h5file.create_dataset(group_name,
            data = np.array(P, dtype = np.float64))
    dset.attrs['N'] = N
    dset.attrs['Nb'] = Nb
    dset.attrs['u'] = u
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Calculate Wright-Fisher \
            transition matrices by iterative multiplication")
    parser.add_argument('N', help='haploid population size',
            type = ut.positive_int)
    parser.add_argument('mu', type = ut.probability,
            help='symmetric per-generation mutation probability')
    parser.add_argument('start', help='first bottleneck size to record',
            type = ut.positive_int)
    parser.add_argument('step', help='step size of bottleneck sizes',
            type = ut.positive_int)
    parser.add_argument('end', help='final bottleneck size to record',
            type = ut.positive_int)
    parser.add_argument('output', help='filename for output hdf5 file. \
            overwrites if exists.')
    parser.add_argument('--breaks', help = 'uniform weight and minimum bin \
            size for binning of larger matrix into smaller matrix',
            nargs = 2, metavar = ('uniform_weight', 'min_bin_size'),
            type = float)
    parser.add_argument('--sizes-file', type = str,
            help = 'file containing bottleneck sizes to produce, overriding \
                    start, every, and end')

    args = parser.parse_args()
    check_start_end(args.start, args.end)

    cmd = ' '.join(sys.argv)

    with h5py.File(args.output, 'w') as h5file:
        h5file.attrs['cmd'] = cmd
        h5file.attrs['N'] = args.N
        if args.breaks is not None:
            uniform_weight = args.breaks[0]
            min_bin_size = args.breaks[1]
            breaks = get_breaks_symmetric(args.N, uniform_weight,
                    min_bin_size)
            h5file.attrs['breaks'] = breaks
            frequencies = get_binned_frequencies(args.N, breaks)
        else:
            breaks = None
            frequencies = np.arange(args.N+1)/args.N
            h5file.attrs['breaks'] = 0
        h5file.attrs['frequencies'] = frequencies

        dataset_idx = 0
        if args.sizes_file is None:
            Nbs = range(args.start, args.end+1, args.step)
        else:
            Nbs = []
            with open(args.sizes_file) as nb_in:
                for line in nb_in:
                    try:
                        Nb = int(line.strip())
                    except ValueError:
                        raise ValueError("invalid integer in bottleneck sizes \
                                file")
                    Nbs.append(Nb)
        for Nb in Nbs:
            P = get_bottleneck_transition_matrix(args.N, Nb, args.mu)
            add_matrix(h5file, P, args.N, Nb, args.mu, dataset_idx, breaks)
            dataset_idx += 1
