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
from . import ascertainment as asc

def run_posterior_mut_loc(args):
    global inf_data
    '''
    =========================
    Initialization / setup
    =========================
    '''

    if args.debug and args.mpi:
        errmsg='--debug and --mpi cannot be simultaneously specified'
        raise ValueError(errmsg)

    lower_dr, upper_dr = args.drift_limits

    data_files, tree_files, age_files = get_input_files(args)

    inf_data = inf.Inference(
            data_files = data_files,
            tree_files = tree_files,
            age_files = age_files,
            transitions_file = args.drift,
            true_parameters = None,
            start_from = 'prior',
            data_are_freqs = args.data_are_frequencies,
            genome_size = args.genome_size,
            bottleneck_file = args.bottlenecks,
            min_freq = args.min_het_freq,
            poisson_like_penalty = args.asc_prob_penalty,
            print_debug = False,
            log_unif_drift = True,
            inverse_bot_priors = False,
            post_is_prior = False,
            lower_drift_limit = lower_dr,
            upper_drift_limit = upper_dr,
            min_phred_score = args.min_phred_score)

    import pdb; pdb.set_trace()
