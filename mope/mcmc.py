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

def post_clone(x):
    global inf_data
    return inf_data.log_posterior(x)


inf_data = None
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
            num_processes = args.processes,
            bottleneck_file = args.bottlenecks,
            min_freq = args.min_het_freq,
            ages_data_fn = args.agesdata,
            poisson_like_penalty = args.asc_prob_penalty,
            print_debug = args.debug)



    def initializer(args):
        global inf_data

        inf_data = inf.Inference(
                data_file = args.data,
                transitions_file = args.transitions,
                tree_file = args.tree,
                true_parameters = args.true_parameters,
                start_from_true = args.start_from_true,
                data_are_freqs = args.study_frequencies,
                genome_size = args.genome_size,
                num_processes = args.processes,
                bottleneck_file = args.bottlenecks,
                min_freq = args.min_het_freq,
                ages_data_fn = args.agesdata,
                poisson_like_penalty = args.asc_prob_penalty,
                print_debug = args.debug)

    if args.processes > 1:
        pool = mp.Pool(args.processes, initializer = initializer,
                initargs = [args])
    elif args.mpi:
        from emcee.utils import MPIPool
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool = None


    '''
    =========================
    MCMC
    =========================
    '''

    # print calling command
    print "# " + " ".join(sys.argv)

    ndim = 2*len(inf_data.varnames) + 2

    ##########################################################################
    # get initial position: previous chains, MAP, or heuristic guess
    ##########################################################################
    if args.prev_chain is not None:
        # start from previous state
        prev_chains = pd.read_csv(args.prev_chain, sep = '\t', header = None)
        prev_chains = prev_chains.iloc[-args.num_walkers:,1:]
        vnames = inf_data.varnames
        prev_chains.columns = ([el+'_l' for el in vnames] +
                [el+'_m' for el in vnames] + ['root', 'ppoly'])
        prev_chains.loc[:,prev_chains.columns.str.endswith('_l')] = (
                prev_chains.loc[:,prev_chains.columns.str.endswith('_l')].abs())        
        init_pos = prev_chains.values
    elif args.start_from_map:
        # start from MAP
        init_params = optimize_posterior(inf_data, pool)
        rx = (1+args.init_norm_sd*npr.randn(ndim*args.num_walkers))
        rx = rx.reshape((args.num_walkers, ndim))
        proposed_init_pos = rx*init_params
        proposed_init_pos = np.apply_along_axis(
                func1d = lambda x: np.maximum(inf_data.lower, x),
                axis = 1, 
                arr = proposed_init_pos)
        proposed_init_pos = np.apply_along_axis(
                func1d = lambda x: np.minimum(inf_data.upper, x),
                axis = 1, 
                arr = proposed_init_pos)
        init_pos = proposed_init_pos
    else:
        # use initial guess
        rx = (1+args.init_norm_sd*npr.randn(ndim*args.num_walkers))
        rx = rx.reshape((args.num_walkers, ndim))
        proposed_init_pos = rx*inf_data.init_params
        proposed_init_pos = np.apply_along_axis(
                func1d = lambda x: np.maximum(inf_data.lower, x),
                axis = 1, 
                arr = proposed_init_pos)
        proposed_init_pos = np.apply_along_axis(
                func1d = lambda x: np.minimum(inf_data.upper, x),
                axis = 1, 
                arr = proposed_init_pos)
        init_pos = proposed_init_pos

    if not args.parallel_temper and (not args.evidence_integral):
        ##############################################################
        # running normal MCMC   
        ##############################################################
        print inf_data.header
        sampler = emcee.EnsembleSampler(args.num_walkers, ndim, post_clone,
                pool = pool, a = args.chain_alpha)
        for ps, lnprobs, cur_seed in sampler.sample(init_pos,
                iterations = args.numiter, storechain = False):
            print_csv_lines(ps, lnprobs)
            inf_data.transition_data.clear_cache()

    else:
        ##############################################################
        # use parallel-tempering
        ##############################################################
        if args.evidence_integral:
            ntemps = 25
        else:  # regular parallel tempering MCMC
            ntemps = 5

        ndim = 2*len(inf_data.varnames)+2
        sampler = emcee.PTSampler(ntemps, nwalkers = args.num_walkers,
                dim = ndim, logl = logl, logp = logp,
                threads = args.num_threads, pool = pool)
        print '! betas:'
        for beta in sampler.betas:
            print '!', beta
        import cPickle
        newshape = [ntemps] + list(init_pos.shape)
        init_pos_new = np.zeros(newshape)
        for i in range(ntemps):
            init_pos_new[i,:,:] = init_pos.copy()

        for p, lnprob, lnlike in sampler.sample(init_pos_new,
                iterations=args.numiter, storechain = True):
            print_parallel_csv_lines(p, lnprob)

        if args.evidence_integral:
            for fburnin in [0.1, 0.25, 0.4, 0.5, 0.75]:
                evidence = sampler.thermodynamic_integration_log_evidence(
                        fburnin=fburnin)
                print '* evidence (fburnin = {}):'.format(fburnin), evidence
