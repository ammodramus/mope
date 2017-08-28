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

from mcmc import run_mcmc
from simulate import run_simulate

def main():
    np.set_printoptions(precision = 10)

    '''
    =========================
    Argument parsing
    =========================
    '''

    parser = argparse.ArgumentParser(
            description = 'mope: mitochondrial ontogenetic '
                          'phylogenetic inference')

    subparsers = parser.add_subparsers()
    parser_run = subparsers.add_parser('run')

    parser_run.add_argument('data', type = str, help = "data file")
    parser_run.add_argument('tree', type = str, help = "file containing newick \
            tree")
    parser_run.add_argument('transitions', type = str, help = "HDF5 file for \
            pre-calculated transition matrices")
    parser_run.add_argument('bottlenecks', type = str,
            help = 'HDF5 file for pre-calculated bottleneck transition \
                    distributions')
    parser_run.add_argument('agesdata', type = str,
            help = 'tap-separated dataset containing the ages for each family')
    parser_run.add_argument('numiter', metavar='n',
            help = 'number of mcmc iterations to perform',
            type = ut.positive_int)
    parser_run.add_argument('--num-optimizations', type = ut.positive_int,
            default = 1, metavar = 'n',
            help = 'perform n optimizations')
    parser_run.add_argument('--true-parameters',
            help = 'file containing true parameters', metavar = "FILE")
    parser_run.add_argument('--start-from-true', action = 'store_true')
    parser_run.add_argument('--start-from-map', action = 'store_true')
    parser_run.add_argument('--study-frequencies', action = 'store_true',
            help = 'use heteroplasmy frequencies from Li et al supplement')
    parser_run.add_argument('--fst-filter', type = ut.probability, metavar = 'X',
            help = 'remove the top X quantile of FST')
    parser_run.add_argument('--genome-size', type = ut.positive_int,
            default = 20000)
    parser_run.add_argument('--num-walkers', type = ut.positive_int, default = 100,
            help = 'number of walkers (chains) to use')
    parser_run.add_argument('--num-threads', type = ut.positive_int, default = 1)
    parser_run.add_argument('--init-norm-sd', type = float, default = 0.2,
            help = 'initial parameters are multiplied by 1+a*X, \
                    where X~norm(0,1) and a is the specified parameter')
    parser_run.add_argument('--processes', type = ut.positive_int, default = 1)
    parser_run.add_argument('--ascertainment', action = 'store_true')
    parser_run.add_argument('--asc-prob-penalty', type = float, default = 1.0,
            help = 'multiplicative factor penalty for Poisson count of \
                    part of likelihood function')
    parser_run.add_argument('--min-het-freq', type = ut.probability,
            help = 'minimum heteroplasmy frequency considered',
            default = 0.001)
    parser_run.add_argument('--parallel-temper', action = 'store_true')
    parser_run.add_argument('--evidence-integral', action = 'store_true')
    parser_run.add_argument('--prev-chain',
            help = 'tab-separated table of previous chain positions, with \
                    the first column giving the logposterior values')
    parser_run.add_argument('--chain-alpha', '-a', type = float, default = 2.0,
            help = 'scale value for emcee ensemble chain proposals')
    parser_run.add_argument('--mpi', action = 'store_true', 
            help = 'use MPI for distribution of chain posterior calculations')
    parser_run.set_defaults(func = run_mcmc)


    # parser for simulations
    parser_sim = subparsers.add_parser('simulate')

    parser_sim.add_argument('tree',
            help = 'newick-formatted tree for simulation')
    parser_sim.add_argument('--num-families', type = ut.positive_int,
            default = 30)
    parser_sim.add_argument('parameters',
            help = 'tab-delimited file giving length, rate, mut. parameters')
    parser_sim.add_argument('-N', help = 'population size',
            type = ut.positive_int,
            default = 1000)
    parser_sim.add_argument('--root-parameter', '-r', type = ut.positive_float,
            help = 'alpha/beta parameter of equilibrium '
            'beta distribution of frequencies at the root',
            metavar = 'AB')
    parser_sim.add_argument('--mean-coverage', '-b', type = ut.positive_int,
            help = 'coverage is Poisson distributed with mean COV, '
                   'independently at each locus', metavar = 'COV',
                   default = 1000)
    parser_sim.add_argument('--sites-per-family', '-s', type = ut.positive_int,
            help = 'number of heteroplasmies per family', default = 50,
            metavar = 'MUT')
    parser_sim.add_argument('--frequencies', action = 'store_true',
            help = 'simulate frequencies rather than allele counts')
    parser_sim.add_argument('--ages',
            help = 'tab-delimited file containing ages to sample from')
    parser_sim.add_argument('--single-beta', action = 'store_true')
    parser_sim.add_argument('--exact-ages', action = 'store_true')
    parser_sim.add_argument('--heteroplasmic-only', action = 'store_true',
            help = 'only output heteroplasmic sites')
    parser_sim.set_defaults(func = run_simulate)

    args = parser.parse_args()
    args.func(args)
