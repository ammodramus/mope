from __future__ import division
import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import argparse
import util as ut


from mcmc import run_mcmc
from simulate import run_simulate
from make_transition_matrices import run_make_transition_matrices
from make_bottleneck_matrices import run_make_bot
from get_ages_from_sims import run_ages
from count_heteroplasmies import count_hets
from simulate_msprime import run_sim_ms


def main():

    '''
    =========================
    Argument parsing
    =========================
    '''

    parser = argparse.ArgumentParser(
            description = 'mope: mitochondrial ontogenetic '
                          'phylogenetic inference')
    subparsers = parser.add_subparsers()

    #############################
    # run MCMC
    #############################
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
            help = 'perform n optimizations [%(default)s]')
    parser_run.add_argument('--true-parameters',
            help = 'file containing true parameters', metavar = "FILE")
    parser_run.add_argument('--start-from-true', action = 'store_true')
    parser_run.add_argument('--start-from-map', action = 'store_true')
    parser_run.add_argument('--study-frequencies', action = 'store_true',
            help = 'use heteroplasmy frequencies from Li et al supplement')
    parser_run.add_argument('--fst-filter', type = ut.probability, metavar = 'X',
            help = 'remove the top X quantile of FST')
    parser_run.add_argument('--genome-size', type = ut.positive_int,
            default = 16569, help = 'genome size is G bp [%(default)s]',
            metavar = 'G')
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
    parser_run.add_argument('--debug', action = 'store_true',
            help = 'print debug output')
    parser_run.set_defaults(func = run_mcmc)


    #############################
    # simulate data
    #############################
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


    #############################
    # make transition matrices
    #############################
    parser_trans = subparsers.add_parser('make-drift')
    parser_trans.add_argument('N', help='haploid population size',
            type = ut.positive_int)
    parser_trans.add_argument('s', help='selection coefficient',
            type = ut.probability)
    parser_trans.add_argument('u', help='mutation probability away from the '
            'focal allele', type = ut.probability)
    parser_trans.add_argument('v', help='mutation probability towards from '
            'the focal allele', type = ut.probability)
    parser_trans.add_argument('start', help='first generation to record '
            '(generation 1 is first generation after the present '
            'generation)', type = ut.nonneg_int)
    parser_trans.add_argument('every', help='how often to record a generation',
            type = ut.positive_int)
    parser_trans.add_argument('end', help='final generation to record '
            '(generation 1 is first generation after the present generation)',
            type = ut.nonneg_int)
    parser_trans.add_argument('output', help='filename for output hdf5 file. '
            'overwrites if exists.')
    parser_trans.add_argument('--breaks', help = 'uniform weight and '
            'minimum bin size for binning of larger matrix into smaller '
            'matrix', nargs = 2, metavar = ('uniform_weight', 'min_bin_size'),
            type = float)
    parser_trans.add_argument('--gens-file', '-g', type = str,
            help = 'file containing generations to produce, one per line. '
            'overrides start, every, and end.')
    parser_trans.add_argument('--asymmetric', action = 'store_true',
            help = 'bin the frequencies asymmetrically around the middle '
                    'frequency')
    parser_trans.set_defaults(func = run_make_transition_matrices)

    #############################
    # make bottleneck matrices
    #############################
    parser_bot = subparsers.add_parser('make-bot')
    parser_bot.add_argument('N', help='haploid population size',
            type = ut.positive_int)
    parser_bot.add_argument('mu', type = ut.probability,
            help='symmetric per-generation mutation probability')
    parser_bot.add_argument('start', help='first bottleneck size to record',
            type = ut.positive_int)
    parser_bot.add_argument('step', help='step size of bottleneck sizes',
            type = ut.positive_int)
    parser_bot.add_argument('end', help='final bottleneck size to record',
            type = ut.positive_int)
    parser_bot.add_argument('output', help='filename for output hdf5 file. '
            'overwrites if exists.')
    parser_bot.add_argument('--breaks', help = 'uniform weight and minimum '
            'bin size for binning of larger matrix into smaller matrix',
            nargs = 2, metavar = ('uniform_weight', 'min_bin_size'),
            type = float)
    parser_bot.add_argument('--sizes-file', type = str,
            help = 'file containing bottleneck sizes to produce, overriding '
            'start, every, and end')
    parser_bot.set_defaults(func = run_make_bot)

    parser_ages = subparsers.add_parser('get-ages')
    parser_ages.add_argument('input', help = 'input filename', type = str)
    parser_ages.set_defaults(func = run_ages)

    parser_count = subparsers.add_parser('count-hets', 
            description='count heteroplasmies in somatic simulation data')
    parser_count.add_argument('data', help = 'data filename', type = str)
    parser_count.add_argument('--frequencies', action = 'store_true')
    parser_count.set_defaults(func = count_hets)

    parser_ms = subparsers.add_parser('simulate-msprime',
            description = 'simulate heteroplasmies using msprime')
    parser_ms.add_argument('tree',
            help = 'newick-formatted tree for simulation')
    parser_ms.add_argument('parameters',
            help = 'tab-delimited file giving length, rate, mut. parameters')
    parser_ms.add_argument('--num-families', type = int,
            help = 'number of families to simulate', metavar = 'R',
            default = 100)
    parser_ms.add_argument('-N', help = 'population size',
            type = ut.positive_int, default = 10000)
    parser_ms.add_argument('--ages',
            help = 'tab-delimited file containing ages to sample from')
    parser_ms.add_argument('--mean-coverage', type = int,
            help = 'mean coverage for variant read count samples. if '
                   'specified, results are reported as read counts rather than'
                   ' frequencies.')
    parser_ms.add_argument('--genome-size', '-g', type = ut.positive_int,
            default = 16571)
    parser_ms.add_argument('--free-recombination', action='store_true',
            help = 'instead of no recombination, simulate free recombination')
    parser_ms.set_defaults(func = run_sim_ms)

    parser_fig = subparsers.add_parser('make-figures',
            description='make plots for MCMC results')
    parser_fig.add_argument('results', type = str, help = 'file containing results '
            'table from mcmc_somatic.py (can be gzipped)')
    parser_fig.add_argument('tree', help = 'tree file', type = str)
    parser_fig.add_argument('--traces', action = 'store_true',
            help = 'also make traces plot')
    parser_fig.add_argument('--num-walkers', type = ut.positive_int,
            default = 500, help = 'number of chains in MCMC ensemble')
    parser_fig.add_argument('--trace-burnin-steps', type = int, default = 500,
            help = 'number of burnin stems for trace plot')
    parser_fig.add_argument('--posterior-burnin-steps', type = ut.positive_int,
            default = 2500,
            help = 'number of burnin stems for posterior histograms')
    parser_fig.add_argument('--prefix', type = str,
            help = 'prefix for image names, if not provided then taken from '
                   'input file name (defaults to current directory)')
    parser_fig.add_argument('--format', type = str, default = 'png',
            help = 'image format')
    parser_fig.add_argument('--true-parameters', type = str,
            help = 'params file for true (simulated) parameters')
    parser_fig.add_argument('--colorsandtypes', '-c',
            help = 'filename for file specifying colors and parameter types '
                   ' for each variable. columns are variable name, color, '
                   'type, whitespace separated. valid types are "fixed", '
                   '"rate", and "bottleneck".')
    parser_fig.add_argument('--mutation-prior-limits', nargs = 2,
            help = 'lower and upper log10 limits for mutation rate parameters, '
                   'two parameters, in log10 space')
    parser_fig.add_argument('--length-prior-limits', nargs = 3,
            help = 'three arguments: lower and upper length-parameter prior '
                   'limits, followed by the family-ages file for the data. '
                   'prior limits are in log10 space')
    parser_fig.add_argument('--add-title', action = 'store_true')
    parser_fig.add_argument('--dpi', type = ut.positive_int, default = 300)

    ############################################
    # parse and run
    args = parser.parse_args()
    args.func(args)
