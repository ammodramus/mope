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
import numpy as np
import sys
import numpy.linalg as npl

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
    try:
        from scipy.misc import comb
    except ImportError:
        raise ImportError('generating transition matrices requires scipy')
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
            if args.input_file is None:
                Nbs = range(args.start, args.end+1, args.every)
            else:
                Nbs = []
                with open(args.input_file) as nb_in:
                    for line in nb_in:
                        try:
                            Nb = int(line.strip())
                        except ValueError:
                            raise ValueError("invalid integer in bottleneck sizes \
                                    file")
                        Nbs.append(Nb)
            for Nb in Nbs:
                P = get_bottleneck_transition_matrix(args.N, Nb, args.u)
                add_matrix_bot(h5file, P, args.N, Nb, args.u, dataset_idx,
                        breaks)
                dataset_idx += 1

        else:  # not bottlenecks
            P = get_wright_fisher_transition_matrix(args.N, args.s, args.u, args.v)
            dataset_idx = 0
            if args.input_file is None:
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
                with open(args.input_file) as gen_in:
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
                    dataset_idx += 1
                    prev_gen = gen

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

def _run_gencmd(args):
    default_gens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24,
            26, 28, 30, 32, 34, 36, 38, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
            100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800,
            900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800,
            3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
            5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200,
            7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000, 9200, 9400,
            9600, 9800, 10000]
    # 19 gens, geometrically spaced between 10^-5 and 10^-3, minus the 10^-3
    default_gauss_gens = [
                1.000000e-05, 1.274275e-05, 1.623777e-05, 2.069138e-05,
                2.636651e-05, 3.359818e-05, 4.281332e-05, 5.455595e-05,
                6.951928e-05, 8.858668e-05, 1.128838e-04, 1.438450e-04,
                1.832981e-04, 2.335721e-04, 2.976351e-04, 3.792690e-04,
                4.832930e-04, 6.158482e-04, 7.847600e-04]
    default_bots = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50,
            55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150,
            160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
            425, 450, 475, 500]
    default_muts = [1.0e-11, 2.5e-11, 5e-11, 7.5e-11, 1.0e-10, 2.5e-10, 5e-10,
            7.5e-10, 1.0e-9, 2.5e-9, 5e-9, 7.5e-9, 1.0e-8, 2.5e-8, 5e-8,
            7.5e-8, 1.0e-7, 2.5e-7, 5e-7, 7.5e-7, 1.0e-6, 2.5e-6, 5e-6, 7.5e-6,
            1.0e-5, 2.5e-5, 5e-5, 7.5e-5, 1.0e-4, 2.5e-4, 5e-4, 7.5e-4, 1.0e-3,
            2.5e-3, 5e-3, 7.5e-3, 1.0e-2, 2.5e-2, 5e-2, 7.5e-2]

    # write to gens and bots files
    with open('gens.txt', 'w') as fout:
        for gen in default_gens:
            fout.write(str(gen) + '\n')
    with open('gens_gauss.txt', 'w') as fout:
        for gen in default_gauss_gens:
            fout.write(str(gen) + '\n')
    with open('bots.txt', 'w') as fout:
        for bot in default_bots:
            fout.write(str(bot) + '\n')

    # print out W-F drift commands
    prefix = 'mkdir -p transitions/drift_matrices && '
    for mut in default_muts:
        outfile = 'transitions/drift_matrices/drift_matrices_mut_{}.h5'.format(mut)
        cmd = ('mope generate-transitions {N} {s} {u} {u} {start} {every} '
               '{end} {output} --breaks 0.5 0.01 --input-file {fin}'.format(
                      N = 1000, s = 0, u = mut, start = 1, every = 1, end = 2,
                      output = outfile, fin = 'gens.txt'))
        print(prefix + cmd)

    # print out gauss commands (same directory)
    for mut in default_muts:
        outfile = 'transitions/drift_matrices/gaussian_drift_matrices_mut_{}.h5'.format(mut)
        cmd = ('mope make-gauss {fin} {N} {uv} {output}'.format(
                      N = 1000, uv = mut, output = outfile, fin = 'gens_gauss.txt'))
        print(prefix + cmd)

    # print out bottleneck commands
    prefix = 'mkdir -p transitions/bottleneck_matrices && '
    for mut in default_muts:
        outfile = 'transitions/bottleneck_matrices/bottleneck_matrices_mut_{}.h5'.format(mut)
        cmd = ('mope generate-transitions {N} {s} {u} {u} {start} {every} '
               '{end} {output} --breaks 0.5 0.01 --input-file {fin} '
               '--bottlenecks'.format(
                      N = 1000, s = 0, u = mut, start = 1, every = 1, end = 2,
                      output = outfile, fin = 'bots.txt'))
        print(prefix + cmd)


    # print out master-file generation command
    print("#########################################")
    print('# run these after everything has completed')
    print("#########################################")
    cmd = ('# cd transitions/ && mope make-master-transitions '
           'drift_matrices/*.h5 --out-file drift_transitions.h5; '
           'cd ..')
    print(cmd)
    cmd = ('# cd transitions/ && mope make-master-transitions '
           'bottleneck_matrices/*.h5 --out-file bottleneck_transitions.h5; '
           'cd ..')
    print(cmd)
