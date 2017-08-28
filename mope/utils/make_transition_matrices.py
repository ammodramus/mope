from __future__ import division
import numpy as np
import numpy.linalg as npl
from scipy.misc import comb
import argparse
import h5py

def positive_int(val):
    val = int(val)
    if val <= 0:
        raise argparse.ArgumentTypeError("invalid positive integer: {}".format(
            val))
    return val


def nonneg_int(val):
    val = int(val)
    if val < 0:
        raise argparse.ArgumentTypeError(
                "invalid non-negative integer: {}".format(val))
    return val


def nonneg_float(val):
    val = float(val)
    if val < 0:
        raise argparse.ArgumentTypeError( "invalid positive float: {}".format(
            val))
    return val

def probability(val):
    val = float(val)
    if val < 0 or val > 1:
        raise argparse.ArgumentTypeError( "invalid probability: {}".format(
            val))
    return val

def check_start_end(start, end):
    if end < start:
        raise argparse.ArgumentTypeError(
                "end generation must be >= start generation")

def get_wright_fisher_transition_matrix(N, s, u, v):
    assert 3/2 == 1.5  # check for __future__ division
    P = np.matrix(np.zeros((N+1, N+1), dtype = np.float64))
    js = np.arange(0,N+1)
    for i in xrange(N+1):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Calculate Wright-Fisher \
            transition matrices by iterative multiplication")
    parser.add_argument('N', help='haploid population size',
            type = positive_int)
    parser.add_argument('s', help='selection coefficient', type = probability)
    parser.add_argument('u', help='mutation probability away from the focal \
            allele', type = probability)
    parser.add_argument('v', help='mutation probability towards from the \
            focal allele', type = probability)
    parser.add_argument('start', help='first generation to record (generation \
            1 is first generation after the present generation)',
            type = nonneg_int)
    parser.add_argument('every', help='how often to record a generation',
            type = positive_int)
    parser.add_argument('end', help='final generation to record (generation \
            1 is first generation after the present generation)',
            type = nonneg_int)
    parser.add_argument('output', help='filename for output hdf5 file. \
            overwrites if exists.')
    parser.add_argument('--breaks', help = 'uniform weight and minimum bin \
            size for binning of larger matrix into smaller matrix',
            nargs = 2, metavar = ('uniform_weight', 'min_bin_size'),
            type = float)
    parser.add_argument('--gens-file', '-g', type = str,
            help = 'file containing generations to produce, one per line. \
                    overrides start, every, and end.')
    parser.add_argument('--asymmetric', action = 'store_true',
            help = 'bin the frequencies asymmetrically around the middle \
                    frequency')

    args = parser.parse_args()
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
            for gen in xrange(args.start+args.every, args.end+1, args.every):
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
                print gen
                dataset_idx += 1
                prev_gen = gen
