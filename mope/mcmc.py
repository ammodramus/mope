from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
import os
import os.path
import emcee
import argparse
import sys

import numpy as np
import scipy.optimize as opt
import pandas as pd
from functools import partial
import multiprocessing as mp
import numpy.random as npr

from mope import newick
from mope import inference as inf
from mope import util as ut
from mope import data as da
from mope import _binom
from mope.pso import pso
from mope._util import print_csv_line, print_csv_lines
from mope._util import print_parallel_csv_lines
from mope import ascertainment as asc
from mope import likelihoods as lis
from mope import transition_data_mut as tdm
from mope import params as par


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

    data_files, tree_files, age_files = ut.get_input_files(args)

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
        selection_model=args.selection,
    )

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

