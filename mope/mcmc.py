from __future__ import division
import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
import emcee
import argparse
import sys
import numpy as np
import scipy.optimize as opt
import pandas as pd
import likelihoods as lis
import transition_data_mut as tdm
import params as par
import initialguess as igs
from functools import partial
import multiprocessing as mp
import numpy.random as npr

import newick
import inference as inf
import util as ut
import data as da
import _binom
from pso import pso
from _util import print_csv_line, print_csv_lines, print_parallel_csv_lines
import ascertainment as asc

def target(x):
    global inf_data
    val = -1.0*inf_data.penalty_bound_log_posterior(x)
    return val

def inf_bound_target(x):
    global inf_data
    val = -1.0*inf_data.log_posterior(x)
    return val


def optimize_posterior(inf_data, pool):
    '''
    maximizes posterior

    inf_data    Inference object

    returns varparams in normal space
    '''

    lower_bound = inf_data.lower+1e-10
    upper_bound = inf_data.upper-1e-10

    swarmsize = 5000
    minfunc = 1e-5
    swarm_init_weight = 0.1


    x, f = pso(target, lower_bound, upper_bound,
            swarmsize = swarmsize, minfunc = minfunc,
            init_params = inf_data.init_params,
            init_params_weight = swarm_init_weight,
            processes = inf_data.num_processes, pool = pool)

    print '! pso:', x, f

    bounds = [[l,u] for l, u in zip(inf_data.lower, inf_data.upper)]
    epsilon = 1e-10
    x, f, d = opt.lbfgsb.fmin_l_bfgs_b(
            target,
            x,
            approx_grad = True,
            factr = 50,
            bounds = bounds,
            epsilon = epsilon)

    print '! lbfgsb:', x, f

    options = {'maxfev': 1000000}
    res = opt.minimize(inf_bound_target,
            x,
            method = "Nelder-Mead",
            options = options)
    x = res.x
    f = res.fun

    print '! nelder-mead:', x, f


    return x

def logl(x):
    global inf_data
    return -1.0*inf_data.inf_bound_like_obj(x)

def logp(x):
    global inf_data
    return inf_data.logprior(x)


def run_mcmc(args):
    global inf_data
    '''
    =========================
    Initialization / setup
    =========================
    '''
    inf_data = inf.Inference(
            data_file = args.data,
            transitions_file = args.transitions,
            tree_file = args.tree,
            true_parameters = args.true_parameters,
            start_from_true = args.start_from_true,
            data_are_freqs = args.study_frequencies,
            genome_size = args.genome_size,
            bottleneck_file = args.bottlenecks,
            min_freq = args.min_het_freq,
            ages_data_fn = args.agesdata,
            poisson_like_penalty = args.asc_prob_penalty,
            print_debug = args.debug)

    inf_data.run_mcmc(
            args.numiter, args.num_walkers, args.processes, args.mpi,
            args.prev_chain, args.start_from_map, args.init_norm_sd,
            args.parallel_temper, args.evidence_integral, args.chain_alpha)

