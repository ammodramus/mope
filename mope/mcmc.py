from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
import os.path
import emcee
import argparse
import sys
import numpy as np
import scipy.optimize as opt
import pandas as pd
from . import likelihoods as lis
from . import transition_data_mut as tdm
from . import params as par
from functools import partial
import multiprocessing as mp
import numpy.random as npr

from . import newick
from . import inference as inf
from . import util as ut
from . import data as da
from . import _binom
from .pso import pso
from ._util import print_csv_line, print_csv_lines, print_parallel_csv_lines
from . import ascertainment as asc


def _check_input_fns(datas, trees, ages):
    fns = [datas, trees, ages]
    if len(fns[0]) != len(fns[1]) or len(fns[0]) != len(fns[2]):
        raise ValueError('--data-files, --tree-files, and --age-files must be of the same length')
    for grp in fns:
        for fn in grp:
            if not os.path.exists(fn):
                raise ValueError('could not find file {}'.format(fn))
    return

def get_input_files(args):
    '''
    look at args.input_file and (args.data_files, args.tree_files,
    args.age_files)

    args.input_file takes precedence
    '''


    if args.input_file is not None:
        datas, trees, ages = [], [], []
        inf = open(args.input_file)
        for line in inf:
            spline = line.strip().split()
            if len(spline) != 3:
                raise ValueError('--input-files file must have three columns: data file, tree file, and age file')
            datas.append(spline[0])
            trees.append(spline[1])
            ages.append(spline[2])
    else:
        datas = args.data_files.split(',')
        trees = args.tree_files.split(',')
        ages = args.age_files.split(',')

    _check_input_fns(datas, trees, ages)
    return datas, trees, ages

def run_mcmc(args):
    global inf_data
    '''
    =========================
    Initialization / setup
    =========================
    '''

    start_from_bool = (args.start_from_map, args.start_from_true)
    if sum(start_from_bool) > 1:
        err = ('--start-from-map and --start-from-true are mutually exclusive')
        raise ValueError(err)
    if args.start_from_map and (args.prev_chain is not None):
        raise ValueError('--start-from-map and --prev-chain are mutually '
                         'exclusive')
    if args.debug and args.mpi:
        errmsg='--debug and --mpi cannot be simultaneously specified'
        raise ValueError(errmsg)
    # (valid values are 'true', 'map')
    start_from = 'prior'
    if args.start_from_true:
        start_from = 'true'
    elif args.start_from_map:
        start_from = 'map'

    lower_dr, upper_dr = args.drift_limits

    data_files, tree_files, age_files = get_input_files(args)

    inf_data = inf.Inference(
            data_files = data_files,
            tree_files = tree_files,
            age_files = age_files,
            transitions_file = args.drift,
            true_parameters = args.true_parameters,
            start_from = start_from,
            data_are_freqs = args.data_are_frequencies,
            genome_size = args.genome_size,
            bottleneck_file = args.bottlenecks,
            min_freq = args.min_het_freq,
            poisson_like_penalty = args.asc_prob_penalty,
            print_debug = args.debug,
            log_unif_drift = not args.uniform_drift_priors,
            inverse_bot_priors = args.inverse_bottleneck_priors,
            post_is_prior = args.just_prior_debug,
            lower_drift_limit = lower_dr,
            upper_drift_limit = upper_dr,
            min_phred_score = args.min_phred_score,
            mutation_transitions_file = args.mutations)

    # for parallel tempering
    nt = args.num_temperatures
    ei = args.evidence_integral

    do_parallel = ((nt is not None) and nt > 1) or ei

    if not do_parallel:
        inf_data.run_mcmc(
                args.numiter, args.num_walkers, args.num_processes, args.mpi,
                args.prev_chain, start_from, args.init_norm_sd,
                args.chain_alpha)


    else:  # num_temperatures is specified 
        inf_data.run_parallel_temper_mcmc(args.numiter, args.num_walkers,
                args.prev_chain, start_from, args.init_norm_sd,
                do_evidence = ei,
                num_processes = args.num_processes,
                mpi = args.mpi,
                ntemps = nt,
                parallel_print_all = args.parallel_print_all,
                chain_alpha = args.chain_alpha)

