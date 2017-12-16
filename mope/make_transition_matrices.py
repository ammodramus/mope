from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import zip
from builtins import str
from builtins import range
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np
import numpy.linalg as npl
from scipy.misc import comb
import argparse
import h5py
import util as ut
import scipy.stats as st

def check_start_end(start, end):
    if end < start:
        raise argparse.ArgumentTypeError(
                "end generation must be >= start generation")

def get_wright_fisher_transition_matrix(N, s, u, v):
    assert 3/2 == 1.5  # check for __future__ division
    P = np.matrix(np.zeros((N+1, N+1), dtype = np.float64))
    js = np.arange(0,N+1)
    for i in range(N+1):
        p = i/N
        pmut = (1-u)*p + v*(1-p) # first mutation, then selection
        pstar = pmut*(1+s) / (pmut*(1+s) + 1-pmut)
        P[i,:] = comb(N, js)*(pstar**js)*((1-pstar)**(N-js))
    return P

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
    for i, prob in zip(list(range(1, N)), interior_probs):
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
    for i, prob in zip(list(range(1, N)), interior_probs):
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

def get_next_matrix_with_prev(cur_matrix, cur_power, next_power, P):
    step_power = next_power - cur_power
    P_step = npl.matrix_power(P, step_power)
    next_P = np.matmul(cur_matrix, P_step)
    return next_P

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
    if breaks is not None:
        P = bin_matrix(P, breaks)
    group_name = "P" + str(idx)
    dset = h5file.create_dataset(group_name,
            data = np.array(P, dtype = np.float64))
    dset.attrs['N'] = N
    dset.attrs['s'] = s
    dset.attrs['u'] = u
    dset.attrs['v'] = v
    dset.attrs['gen'] = gen
    return

def _run_make_transition_matrices(args):
    check_start_end(args.start, args.end)


    with h5py.File(args.output, 'w') as h5file:

        h5file.attrs['N'] = args.N
        if args.breaks is not None:
            uniform_weight = args.breaks[0]
            min_bin_size = args.breaks[1]
            if args.asymmetric:
                breaks = get_breaks(args.N, uniform_weight, min_bin_size)
            else: # symmetric
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

        P = get_wright_fisher_transition_matrix(args.N, args.s, args.u, args.v)
        dataset_idx = 0
        if args.gens_file is None:
            step_matrix = npl.matrix_power(P, args.every)

            gen = args.start
            if gen > 0:
                P_prime = npl.matrix_power(P, args.start)
                add_matrix(h5file, P_prime, args.N, args.s, args.u, args.v,
                        gen, dataset_idx, breaks)
            elif gen == 0:
                # for now, not taking into account mutation in the zero-gen
                # matrix
                #P_prime = get_identity_matrix(args.N, args.u, breaks)
                P_prime = np.diag(np.repeat(1.0, args.N+1))
                add_matrix(h5file, P_prime, args.N, args.s, args.u, args.v,
                        gen, dataset_idx, breaks)
                P_prime = np.diag(np.repeat(1.0, args.N+1))
            dataset_idx += 1
            step_matrix = npl.matrix_power(P, args.every)
            for gen in range(args.start+args.every, args.end+1, args.every):
                P_prime = np.dot(P_prime, step_matrix)
                add_matrix(h5file, P_prime, args.N, args.s, args.u, args.v, gen,
                        dataset_idx, breaks)
                dataset_idx += 1
        else:
            gens = []
            with open(args.gens_file) as gen_in:
                for line in gen_in:
                    try:
                        gen = int(line.strip())
                    except ValueError:
                        raise ValueError("invalid integer in generations file")
                    gens.append(gen)
            P_prime = npl.matrix_power(P, gens[0])
            add_matrix(h5file, P_prime, args.N, args.s, args.u, args.v,
                    gens[0], dataset_idx, breaks)
            dataset_idx += 1
            prev_gen = gens[0]
            for gen in gens[1:]:
                P_prime = get_next_matrix_with_prev(
                        P_prime, prev_gen, gen, P)
                add_matrix(h5file, P_prime, args.N, args.s, args.u, args.v,
                        gen, dataset_idx, breaks)
                print(gen)
                dataset_idx += 1
                prev_gen = gen


'''
Gaussian matrices

For small-t approximations better than linearly interpolating between
generations 0 (identity matrix) and 1.
'''
def get_middle_freq(li, ui, N):
    assert 3/2 == 1.5
    if li == N:
        return 1.0
    mf = np.mean(np.arange(li, ui)) / N
    return mf
def get_bounds(li, ui, N):
    delta = 1.0/(2*N)
    if li == N:
        return np.array((1.0-delta, np.inf))
    if li == 0:
        return np.array((-np.inf, delta))
    lower = li/N - delta
    lower = max(0, lower)
    upper = (ui-1)/N + delta
    upper = min(1.0, upper)
    return np.array((lower, upper))

def get_mean_sd(x0, t, theta):
    sd=np.sqrt(t*x0*(1-x0))
    mean = x0 + t*0.5*theta*(1.-2.*x0)
    return mean, sd


def get_gauss_transition(t, u, i, j, N, breaks, debug = False):
    theta = 2*N*u   # TODO check factor of 2
    try:
        br_i0, br_i1 = breaks[i:i+2]
    except:
        assert breaks[i] == N
        br_i0, br_i1 = N, N+1
    f_i = get_middle_freq(br_i0, br_i1, N)
    try:
        br_j0, br_j1 = breaks[j:j+2]
    except:
        assert j == breaks.shape[0]-1, j
        assert breaks[j] == N
        br_j0, br_j1 = breaks[j], breaks[j]+1
    if br_i0 in (0,N):
        # poisson for current allele count = 0
        poisrate = theta / 2
        if br_i0 == 0:
            prob = np.sum(st.poisson.pmf(np.arange(br_j0, br_j1), poisrate))
        else:
            assert br_i0 == N
            prob = np.sum(st.poisson.pmf(N-np.arange(br_j0, br_j1), poisrate))
        assert 0 <= prob <= 1
        return prob
    bound_j = get_bounds(br_j0, br_j1, N)
    mean, sd = get_mean_sd(f_i, t, theta)
    cdf_v = st.norm.cdf(bound_j, loc = mean, scale = sd)
    if debug:
        print("mean =", mean, "sd =", sd)
        print(cdf_v)
    return cdf_v[1] - cdf_v[0]

def get_gauss_matrix_with_breaks(t, u, N, br):
    nbr = br.shape[0]
    P = np.zeros((nbr, nbr))
    for i in range(nbr):
        for j in range(nbr):
            P[i,j] = get_gauss_transition(t, u, i, j, N, br)
    return P

def get_gaussian_matrix(t, u, N, unif_weight, min_size):
    br = mtm.get_breaks_symmetric(N, unif_weight, min_size)
    return get_gauss_matrix_with_breaks(t, u, N, br)

def add_gauss_matrix(h5file, P, N, s, u, v, gen, idx):
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

    already binned, no need to bin it.
    '''
    group_name = "P" + str(idx)
    dset = h5file.create_dataset(group_name,
            data = np.array(P, dtype = np.float64))
    dset.attrs['N'] = N
    dset.attrs['s'] = s
    dset.attrs['u'] = u
    dset.attrs['v'] = v
    dset.attrs['gen'] = gen
    return

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
