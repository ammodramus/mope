from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
import emcee
import argparse
import sys
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np
import scipy.optimize as opt
import pandas as pd
from . import likelihoods as lis
from . import transition_data_mut as tdm
from . import params as par
from . import initialguess as igs
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

def run_mcmc(args):
    global inf_data
    '''
    =========================
    Initialization / setup
    =========================
    '''

    if args.start_from_map and (args.prev_chain is not None):
        raise ValueError('--start-from-map and --prev-chain are mutually '
                         'exclusive')
    if args.debug and args.mpi:
        errmsg='--debug and --mpi cannot be simultaneously specified'
        raise ValueError(errmsg)

    inf_data = inf.Inference(
            data_file = args.data,
            transitions_file = args.drift,
            tree_file = args.tree,
            true_parameters = args.true_parameters,
            start_from_true = args.start_from_true,
            data_are_freqs = args.data_are_frequencies,
            genome_size = args.genome_size,
            bottleneck_file = args.bottlenecks,
            min_freq = args.min_het_freq,
            ages_data_fn = args.agesdata,
            poisson_like_penalty = args.asc_prob_penalty,
            print_debug = args.debug,
            log_unif_drift = args.log_uniform_drift_priors)

    if (not args.num_temperatures > 1) and (not args.evidence_integral):
        inf_data.run_mcmc(
                args.numiter, args.num_walkers, args.num_processes, args.mpi,
                args.prev_chain, args.start_from_map, args.init_norm_sd,
                args.chain_alpha)


    else:
        inf_data.run_parallel_temper_mcmc(args.numiter, args.num_walkers,
                args.prev_chain, args.start_from_map, args.init_norm_sd,
                do_evidence = args.evidence_integral,
                num_processes = args.num_processes,
                mpi = args.mpi,
                num_temperatures = args.num_temperatures)

