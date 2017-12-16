from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import argparse
from . import util as ut


from .mcmc import run_mcmc
from .simulate import run_simulate
from .make_transition_matrices import _run_make_transition_matrices
from .make_transition_matrices import _run_make_gauss
from .make_bottleneck_matrices import run_make_bot
from .get_ages_from_sims import run_ages
from .count_heteroplasmies import count_hets
from .simulate_msprime import run_sim_ms
from .make_figures import _run_make_figures
from .download_transitions import _run_download
from .generate_transitions import _run_generate, _run_master, _run_gencmd
from .acceptance import _run_acceptance


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
    parser_run.add_argument('drift', type = str, help = "HDF5 file for \
            pre-calculated drift transition distributions")
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
    parser_run.add_argument('--start-from-true', action = 'store_true',
            help = 'start MCMC from true parameters, for debugging with '
                   'simulations')
    parser_run.add_argument('--start-from-map', action = 'store_true',
            help = 'start MCMC chain from MAP estimate. this improves '
                   'convergence')
    parser_run.add_argument('--start-from-prior', action = 'store_true',
            help = 'start MCMC chain with random sample from prior.')
    parser_run.add_argument('--data-are-frequencies', action = 'store_true',
            help = 'data are frequencies rather than allele counts')
    parser_run.add_argument('--genome-size', type = ut.positive_int,
            default = 16569, help = 'genome size is G bp [%(default)s]',
            metavar = 'G')
    parser_run.add_argument('--num-walkers', type = ut.positive_int,
            default = 500,
            help = 'number of walkers (chains) to use [%(default)s]')
    parser_run.add_argument('--init-norm-sd', type = float, default = 0.2,
            help = 'initial parameters are multiplied by 1+a*X, '
                    'where X~norm(0,1) and a is the specified parameter '
                    '[%(default)s]')
    parser_run.add_argument('--num-processes',
            type = ut.positive_int, default = 1,
            help = 'number of parallel processes to use [%(default)s]')
    parser_run.add_argument('--asc-prob-penalty', type = float, default = 1.0,
            help = 'multiplicative factor penalty for Poisson count of '
                    'part of likelihood function (1 = no penalty, '
                    '0 = full penalty) [%(default)s]')
    parser_run.add_argument('--min-het-freq', type = ut.probability,
            help = 'minimum heteroplasmy frequency considered [%(default)s]',
            default = 0.001)
    parser_run.add_argument('--num-temperatures',
            type = ut.positive_int,
            help = 'number of temperatures for parallel-tempering MCMC. '
                   'specifying > 1 will enable paralle-tempering MCMC.')
    parser_run.add_argument('--parallel-print-all', action = 'store_true',
            help = 'if specifying --num-temperatures > 1, or '
                   '--evidence-integral, use this option to print states and '
                   'log-posterior values for all temperatures. otherwise just '
                   'the chain with temperature 1 (the original posterior) is '
                   'printed. ignored if not doing parallel-tempering MCMC')
    parser_run.add_argument('--evidence-integral', action = 'store_true')
    parser_run.add_argument('--prev-chain',
            help = 'tab-separated table of previous chain positions, with \
                    the first column giving the logposterior values')
    parser_run.add_argument('--chain-alpha', '-a', type = float, default = 1.4,
            help = 'scale value for emcee ensemble chain proposals')
    parser_run.add_argument('--mpi', action = 'store_true', 
            help = 'use MPI for distribution of chain posterior calculations')
    parser_run.add_argument('--debug', action = 'store_true',
            help = 'print debug output')
    parser_run.add_argument('--log-uniform-drift-priors',
            action = 'store_true',
            help = 'use log-uniform prior distributions for drift parameters,'
                   'rather than the default uniform priors')
    parser_run.add_argument('--inverse-bottleneck-priors',
            action = 'store_true',
            help = 'make the prior for bottlenecks reflect the drift caused '
                   'by the bottleneck, which is inversely proportional to '
                   'the bottleneck size')
    parser_run.add_argument('--just-prior-debug', action = 'store_true',
            help = 'let the posterior be the prior, for debugging')
    parser_run.add_argument('--drift-limits', type = float, nargs = 2,
            default = (1e-3, 3),
            help = 'lower and upper length limits for drift variables, in '
                   'natural scale')
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
    parser_trans.set_defaults(func = _run_make_transition_matrices)

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
            'table from mope (can be gzipped)')
    parser_fig.add_argument('tree', help = 'tree file', type = str)
    parser_fig.add_argument('--plot', default = 'histograms',
            choices = ('histograms', 'traces', 'both', 'corner'),
            help = 'which plots to make (histograms, traces, both)')
    parser_fig.add_argument('--num-walkers', type = ut.positive_int,
            default = 500,
            help = 'number of chains in MCMC ensemble [%(default)s]')
    parser_fig.add_argument('--trace-burnin-steps', type = int, default = 500,
            help = 'number of burnin stems for trace plot [%(default)s]')
    parser_fig.add_argument('--posterior-burnin-frac', type = ut.probability,
            default = 0.3,
            help = 'fraction of data that is burnin, for posterior histograms '
                   '[%(default)s]')
    parser_fig.add_argument('--prefix', type = str,
            help = 'prefix for image names, if not provided then taken from '
                   'input file name (defaults to current directory)')
    parser_fig.add_argument('--format', type = str, default = 'png',
            help = 'image format [%(default)s]')
    parser_fig.add_argument('--true-parameters', type = str,
            help = 'params file for true (simulated) parameters')
    parser_fig.add_argument('--colors-and-types', '-c',
            help = 'filename for file specifying colors and parameter types '
                   ' for each variable. columns are variable name, color, '
                   'type, whitespace separated. valid types are "fixed", '
                   '"rate", and "bottleneck".')
    parser_fig.add_argument('--mutation-prior-limits', nargs = 2,
            help = 'lower and upper log10 limits for mutation rate parameters,'
                   ' two parameters, in log10 space')
    parser_fig.add_argument('--length-prior-limits', nargs = 2,
            help = 'three arguments: lower and upper length-parameter prior '
                   'limits, followed by the family-ages file for the data. '
                   'prior limits are in log10 space')
    parser_fig.add_argument('--ages-file', type = str,
            help = 'individual ages in tab-separated table')
    parser_fig.add_argument('--add-title', action = 'store_true')
    parser_fig.add_argument('--dpi', type = ut.positive_int, default = 300)
    parser_fig.add_argument('--log-uniform-drift-priors',
            action = 'store_true',
            help = 'priors for drift are log-uniform instead of uniform')
    parser_fig.set_defaults(func = _run_make_figures)

    parser_download = subparsers.add_parser('download-transitions',
            description='download allele frequency transitions')
    parser_download.add_argument('--directory', '-d',
            help = 'directory in which to place the downloaded transitions/ '
                   'directory', default = '.')
    parser_download.set_defaults(
            func = _run_download)

    parser_master = subparsers.add_parser('make-master-transitions',
            description = 'make master file named MF from many previously '
                          'generated transition files')
    parser_master.add_argument('files', nargs = '+',
            help = 'previously generated HDF5 files containing transitions '
                   '(use mope generate-transitions to generate these files)')
    parser_master.add_argument('--out-file', '-o', type = str,
            help = "filename for output HDF5 file", default = 'master.h5',
            metavar = 'filename')
    parser_master.set_defaults(
            func = _run_master)

    parser_gendrift = subparsers.add_parser('generate-transitions',
            description = 'generate genetic drift and bottleneck transition '
                          'distribution files')
    parser_gendrift.add_argument('N', help='haploid population size',
            type = ut.positive_int)
    parser_gendrift.add_argument('s',
            help='selection coefficient', type = ut.probability)
    parser_gendrift.add_argument('u', help='mutation probability away from '
            'the focal allele', type = ut.probability)
    parser_gendrift.add_argument('v', help='mutation probability towards from '
            'the focal allele', type = ut.probability)
    parser_gendrift.add_argument('start',
            help='first generation (or minimum bottleneck size) to record '
            '(generation 1 is first generation after the present generation)',
            type = ut.nonneg_int)
    parser_gendrift.add_argument('every',
            help='step size for recording generations or bottleneck sizes',
            type = ut.positive_int)
    parser_gendrift.add_argument('end',
            help='final generation or bottleneck size to record',
            type = ut.nonneg_int)
    parser_gendrift.add_argument('output',
            help='filename for output hdf5 file. overwrites if exists.')
    parser_gendrift.add_argument('--bottlenecks', action = 'store_true',
            help = 'generate bottleneck transition files instead of W-F drift')
    parser_gendrift.add_argument('--breaks',
            help = 'uniform weight and minimum bin size for binning of larger '
                   'matrix into smaller matrix',
            nargs = 2, metavar = ('uniform_weight', 'min_bin_size'),
            type = float, default = (0.5, 0.01))
    parser_gendrift.add_argument('--input-file', '-g', type = str,
            help = 'file containing generations or bottleneck sizes to '
                   'produce, one per line. ' 'overrides start, every, and '
                   'end.')
    parser_gendrift.set_defaults(
            func = _run_generate)

    parser_gencmd = subparsers.add_parser('generate-commands',
            description = 'generate commands for producing default '
                          'drift and bottleneck files. This also '
                          'produces gens.txt and bots.txt, containing the '
                          'default generations and bottleneck sizes. Run '
                          'these commands with GNU Parallel to parallelize '
                          'calculations. Notice that two final commands '
                          'need to be run after completion of all others '
                          'in order to make the master files containing '
                          'all transitions.')
    parser_gencmd.set_defaults(
            func = _run_gencmd)

    parser_accept = subparsers.add_parser('acceptance',
            description = 'get acceptance fractions')
    parser_accept.add_argument('datafile',
            help = 'data file produced with mope run')
    parser_accept.add_argument('numwalkers',
            help = 'number of walkers (chains) in mope run',
            type = int)
    parser_accept.add_argument('--burnin-steps', '-b',
            help = 'number of burnin steps', default = 10000)
    parser_accept.add_argument('--all', '-a',
            help = 'print acceptance for each chain / walker individually')
    parser_accept.set_defaults(
            func = _run_acceptance)

    parser_gauss = subparsers.add_parser('make-gauss',
            description = 'make Gaussian allele frequency transition matrices')
    parser_gauss.add_argument('gensfile',
            help = 'filename containing list of generations')
    parser_gauss.add_argument('N', help = 'population size',
            type = ut.positive_int)
    parser_gauss.add_argument('uv', help = 'symmetric mutation parameter',
            type = ut.probability)
    parser_gauss.add_argument('outfile', help = 'output filename')
    parser_gauss.add_argument('--unif-weight', type = ut.probability,
            default = 0.5)
    parser_gauss.add_argument('--min-bin-size', type = ut.positive_float,
            default = 0.01,
            help = 'unifweight and minbinsize are two parameters to control '
                   'the heuristic used for deciding where to bin matrices. '
                   'defaults are recommended')
    parser_gauss.set_defaults(
            func = _run_make_gauss)


    ############################################
    # parse and run
    args = parser.parse_args()
    args.func(args)
