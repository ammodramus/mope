from __future__ import division
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
import sys

from simulate import get_parameters
import util as ut
import newick
from collections import OrderedDict

def get_hatch(paramtype):
    hatch = None
    if paramtype == 'bottleneck':
        hatch = 'o'
    elif paramtype == 'rate':
        hatch = '\\'
    return hatch

def get_parameter_names(filename):
    with open(filename) as fin:
        tree_str = fin.read().strip()
    branch_names = []
    leaf_names = []
    tree = newick.loads(tree_str, look_for_multiplier = True,
                       length_parser = ut.length_parser_str)[0]
    vset = set([])
    for node in tree.walk('postorder'):
        if node == tree:
            continue
        vset.add(node.varname)
    vnames = sorted(list(vset))
    param_names = ['loglike']
    for vname in vnames:
        param_names.append(vname + '_l')
    for vname in vnames:
        param_names.append(vname + '_m')
    param_names.append('root')
    param_names.append('ppoly')
    return param_names


def get_chains(dat, num_walkers):
    chains = []
    for i in xrange(num_walkers):
        chains.append(dat.iloc[i::num_walkers,:].copy())
    return chains

def get_acceptance(chain):
    assert 3/2 == 1.5
    rows0 = chain.iloc[:-1,:].values
    rows1 = chain.iloc[1:,:].values
    num_changes = np.sum(np.all(~np.equal(rows0, rows1), axis = 1))
    num_comparisons = rows0.shape[0]
    return num_changes / num_comparisons

def get_len_limits(tree, ages_file, lower_drift, upper_drift,
        lower_bottleneck = 2, upper_bottleneck = 500):
    is_bottleneck = {}
    ages_dat = pd.read_csv(ages_file, sep = '\t')
    limits = {}
    for node in tree.walk():
        varname = node.varname
        is_bottleneck[varname] = node.is_bottleneck
        mult = node.multipliername
        if mult is None:
            limits[varname] = [1.0, 1.0]
        else:
            minv, maxv = ages_dat[mult].min(), ages_dat[mult].max()
            if varname in limits:
                limits[varname][0] = min(limits[varname][0], minv)
                limits[varname][1] = max(limits[varname][1], maxv)
            else:
                limits[varname] = [minv, maxv]

    # so far, it has just been the multipliers, now need to make them limits
    for varname in limits.keys():
        mults = limits[varname]
        if is_bottleneck[varname]:
            ld, ud = lower_bottleneck, upper_bottleneck
        else:
            ld, ud = lower_drift, upper_drift
        limits[varname] = [np.log10(ld / mults[0]),
                np.log10(ud / mults[1])]

    return limits


parser = argparse.ArgumentParser(description='make plots for MCMC results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('results', type = str, help = 'file containing results '
        'table from mcmc_somatic.py (can be gzipped)')
parser.add_argument('tree', help = 'tree file', type = str)
parser.add_argument('--traces', action = 'store_true',
        help = 'also make traces plot')
parser.add_argument('--num-walkers', type = int, default = 500,
        help = 'number of chains in MCMC ensemble')
parser.add_argument('--trace-burnin-steps', type = int, default = 500,
        help = 'number of burnin stems for trace plot')
parser.add_argument('--posterior-burnin-steps', type = int, default = 2500,
        help = 'number of burnin stems for posterior histograms')
parser.add_argument('--prefix', type = str,
        help = 'prefix for image names, if not provided then taken from '
               'input file name (defaults to current directory)')
parser.add_argument('--format', type = str, default = 'png',
        help = 'image format')
parser.add_argument('--true-parameters', type = str,
        help = 'params file for true (simulated) parameters')
parser.add_argument('--colorsandtypes', '-c',
        help = 'filename for file specifying colors and parameter types for '
               'each variable. columns are variable name, color, type, '
               'whitespace separated. valid types are "fixed", "rate", and '
               '"bottleneck".')
parser.add_argument('--mutation-prior-limits', nargs = 2,
        help = 'lower and upper log10 limits for mutation rate parameters, '
               'two parameters, in log10 space')
parser.add_argument('--length-prior-limits', nargs = 3,
        help = 'three arguments: lower and upper length-parameter prior '
               'limits, followed by the family-ages file for the data. '
               'prior limits are in log10 space')
parser.add_argument('--add-title', action = 'store_true')
parser.add_argument('--dpi', type = int, default = 300)

args = parser.parse_args()

validparamtypes = ('fixed', 'rate', 'bottleneck')

###################
# output filenames
###################

if args.prefix is None:
    path, filename = osp.split(args.results)
    prefix = filename.split('.')[0]

else:
    prefix = args.prefix
trace_file = 'traces_' + prefix + '.' + args.format
posterior_histograms_file = ('posterior_histograms_' + prefix + '.' +
        args.format)

# loading colors and (drift) parameter types
if args.colorsandtypes is not None:
    colors = OrderedDict()
    types = OrderedDict()
    priortypes = OrderedDict()
    priormins = OrderedDict()
    priormaxes = OrderedDict()
    with open(args.colorsandtypes) as inp:
        for line in inp:
            spline = line.split()
            param, color, paramtype = spline[:3]
            if paramtype not in validparamtypes:
                raise ValueError('invalid parameter type: must be "fixed", '
                                 '"rate", or "bottleneck"')
            colors[param] = color
            types[param] = paramtype
            try:
                priortype, priormin, priormax = spline[3:6]
                priortypes[param] = priortype
                priormins[param] = priormin
                priormaxs[param] = priormax
            except ValueError:
                continue
    priors_found = (len(priortypes) > 0)
    for param, ptype in priortypes.iteritems():
        if ptype not in validpriortypes:
            raise ValueError('Invalid prior type: {}. must be "uniform" or '
                             '"loguniform".'.format(ptype))
        if priormins[param] >= priormaxes[param]:
            raise ValueError('prior minima must be less than maxima')

###################
# loading the data
###################
tree_file = args.tree
with open(tree_file) as fin:
    tree_str = fin.read().strip()
tree = newick.loads(tree_str,
                    look_for_multiplier = True,
                    length_parser = ut.length_parser_str)[0]

cols = get_parameter_names(tree_file)
varnames = [col[:-2] for col in cols if col.endswith('_l')]
dat_m1 = pd.read_csv(args.results,
        sep = '\t', header = None, names = cols, dtype = np.float64)
dat_m1.loc[:,dat_m1.columns.str.contains('_l')] = np.log10(np.abs(
        dat_m1.loc[:,dat_m1.columns.str.contains('_l')]))
dat_m1.loc[:,dat_m1.columns.str.contains('_m')] = -np.abs(
        dat_m1.loc[:,dat_m1.columns.str.contains('_m')])
dat_m1.loc[:,'root'] = -np.abs(dat_m1.loc[:,'root'])
dat_m1.loc[:,'ppoly'] = -np.abs(dat_m1.loc[:,'ppoly'])
num_walkers = args.num_walkers
burnin_steps = args.trace_burnin_steps
chains = get_chains(dat_m1.iloc[burnin_steps*num_walkers:,:].copy(),
        num_walkers)

# getting (optional) length prior limits
if args.length_prior_limits is not None:
    lower_l, upper_l, ages_file = args.length_prior_limits
    lower_l = 10**float(lower_l)
    upper_l = 10**float(upper_l)
    length_limits = get_len_limits(tree, ages_file, lower_l, upper_l)



# getting true parameters
if args.true_parameters is not None:
    true_bls, true_mrs, true_ab, true_ppoly = get_parameters(
            args.true_parameters, tree)
    true_params = [true_bls[v] for v in varnames]
    true_params += [true_mrs[v] for v in varnames]
    true_params += [true_ab, true_ppoly]

# traces
if args.traces:
    f, axes = plt.subplots(
            dat_m1.shape[1], 1, figsize = (10, 2.5*dat_m1.shape[1]))
    for i in range(dat_m1.shape[1]):
        col = list(dat_m1.columns)[i]
        ax = axes[i]
        if not np.any(dat_m1[col] <= 0.0):
            for chain in chains:
                ax.plot(np.arange(chain.shape[0]),np.log10(chain.iloc[:,i]),
                        lw = 0.02)
            plot_log = True
        else:
            for chain in chains:
                ax.plot(np.arange(chain.shape[0]),(chain.iloc[:,i]), lw = 0.02)
            plot_log = False
        if args.true_parameters is not None and i > 0:
            true_p = true_params[i-1]
            if plot_log:
                ax.axhline(np.log10(true_p))
            else:
                # now all the parameters are log10 transformed
                ax.axhline(np.log10(true_p))
        ax.set_title(dat_m1.columns[i])
    f.tight_layout()
    plt.savefig(trace_file)

# posterior histograms
if args.colorsandtypes is None:
    final = False  # not a 'final', publication plot
    num_plots = dat_m1.shape[1]
    nrows = 4
    ncols = int(np.ceil(num_plots / nrows))
    f, axes = plt.subplots(nrows,ncols, figsize = (12,12*nrows/ncols))

else:
    final = True   # a 'final', publication plot
    nrows = 2
    ncols = len(varnames)
    num_plots = (dat_m1.shape[1]-3)*2  # (no loglike, or the two root params)
    f, axes = plt.subplots(nrows,ncols, figsize = (2*ncols,4))
    mpl.rcParams.update({'font.size': 5})
    mpl.rcParams.update({'axes.labelsize': 'small'})

burnin_steps = args.posterior_burnin_steps
burnin = num_walkers * burnin_steps
if burnin > dat_m1.shape[0]:
    burnin = 0
plotted = np.tile([False], nrows*ncols)
plotted.shape = (nrows,ncols)

if burnin > dat_m1.shape[0]:
    raise ValueError('burnin exceeds number of samples')


if final:
    if not args.add_title:
        mpl.rc('text', usetex = False)
    #mpl.rcParams.update({'font.family': 'Arial'})
    colornames = colors.keys()
    for counter, var in enumerate(colornames):
        if args.true_parameters is not None:
            true_l = np.log10(true_bls[var])
            true_m = np.log10(true_mrs[var])
        # length
        if args.colorsandtypes is not None:
            try:
                c = colors[var]
                t = types[var]
            except KeyError:
                c = 'gray'
                t = 'fixed'
        else:
            c = 'gray'
            t = 'fixed'
        hatch = get_hatch(t)

        ax = axes[0][counter]
        col = var + '_l'
        if args.length_prior_limits is not None:
            minv, maxv = length_limits[var]
        else:
            minv, maxv = dat_m1[col][burnin:].min(), dat_m1[col][burnin:].max()

        lower = np.floor(minv)
        upper = np.ceil(maxv)
        bins = np.linspace(lower, upper, 50)
        ticks = np.arange(lower, upper+1).astype(np.int)
        if ticks.shape[0] > 5:
            ticks = np.arange(lower, upper+1, 2).astype(np.int)

        if args.length_prior_limits is not None:
            x = np.linspace(minv, maxv, 1000)
            y = 10**x * np.log(10) / (10**maxv - 10**minv)
            ax.fill_between(x, 0, y, alpha = 0.3, color = 'gray')

        ax.hist(dat_m1[col][burnin:], bins, color = c, hatch = hatch,
                normed = True)
        ax.set_xticks(ticks)
        #ax.set_xscale('log')
        ax.get_yaxis().set_visible(False)
        if args.true_parameters is not None:
            ax.axvline(true_l, ls = '--', c = 'black', lw = 2)
        if args.add_title:
            ax.set_title(col)

        ax = axes[1][counter]
        col = var + '_m'
        if args.mutation_prior_limits is None:
            minv, maxv = dat_m1[col][burnin:].min(), dat_m1[col][burnin:].max()
        else:
            minv, maxv = args.mutation_prior_limits
            minv = float(minv)
            maxv = float(maxv)
        lower = np.floor(minv)
        upper = np.ceil(maxv)
        ticks = np.arange(lower, upper+1).astype(np.int)
        if ticks.shape[0] > 5:
            ticks = np.arange(lower, upper+1, 2).astype(np.int)
        bins = np.linspace(lower, upper, 50)

        if args.mutation_prior_limits is not None:
            x = np.linspace(minv, maxv, 1000)
            y = 1.0/(maxv-minv)
            ax.fill_between(x, 0, y, alpha = 0.3, color = 'gray')

        ax.hist(dat_m1[col][burnin:], bins, color = c, hatch = hatch,
                normed = True)
        ax.set_xticks(ticks)
        ax.get_yaxis().set_visible(False)
        if args.true_parameters is not None:
            ax.axvline(true_m, ls = '--', c = 'black', lw = 2)
        if args.add_title:
            ax.set_title(col)


else:  # not final
    counter = 0
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            ax = axes[row_idx][col_idx]
            try:
                col = cols[counter]
            except IndexError:
                break
            if args.true_parameters is not None and counter > 0:
                true_p = true_params[counter-1]
            minv, maxv = dat_m1[col][burnin:].min(), dat_m1[col][burnin:].max()
            if args.colorsandtypes is not None:
                var = col[:-2]
                try:
                    c = colors[var]
                    t = types[var]
                except KeyError:
                    c = 'gray'
                    t = 'fixed'
            else:
                c = 'gray'
                t = 'fixed'
            hatch = get_hatch(t)
            if minv >= 0:
                lower = np.floor(np.log10(minv))
                upper = np.ceil(np.log10(maxv))
                bins = np.logspace(lower, upper, 50)
                ax.hist(dat_m1[col][burnin:], bins, color = c, hatch = hatch)
                ax.set_xscale('log')
            else:
                if col != 'loglike':
                    lower = np.floor(minv)
                    upper = np.ceil(maxv)
                    bins = np.logspace(lower, upper, 50)
                    datp = 10**dat_m1[col].iloc[burnin:]
                    ax.hist(datp, bins, color = 'gray')
                    ax.set_xscale('log')
                else:
                    lower = np.floor(minv)
                    upper = np.ceil(maxv)
                    bins = np.linspace(lower, upper, 50)
                    ax.hist(dat_m1[col][burnin:], bins, color = 'gray')
            if args.true_parameters is not None and counter > 0:
                ax.axvline(true_p)
            ax.set_title(col)
            ax.get_yaxis().set_visible(False)
            counter += 1
            plotted[row_idx][col_idx] = True

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            if not plotted[row_idx][col_idx]:
                ax = axes[row_idx][col_idx]
                f.delaxes(ax)
        
f.tight_layout()
plt.savefig(posterior_histograms_file, dpi = args.dpi)


