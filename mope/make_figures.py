from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import range
import argparse
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
import sys

from .simulate import get_parameters
from . import util as ut
from . import newick
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
    for i in range(num_walkers):
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
    for varname in list(limits.keys()):
        mults = limits[varname]
        if is_bottleneck[varname]:
            ld, ud = lower_bottleneck, upper_bottleneck
        else:
            ld, ud = lower_drift, upper_drift
        limits[varname] = [np.log10(ld / mults[0]),
                np.log10(ud / mults[1])]

    return limits

def make_figures(
        results, tree, num_walkers, ages_file, prefix = None,
        img_format = 'png', dpi = 300, colors_and_types = None,
        length_prior_limits = (-6,0.477), mutation_prior_limits = (-8,-1),
        true_parameters = None, do_traces = False, trace_burnin_steps = 500,
        posterior_burnin_steps = 2500, add_title = True, log_uniform_drift_priors = False):

    validparamtypes = ('fixed', 'rate', 'bottleneck')

    ###################
    # output filenames
    ###################

    if prefix is None:
        path, filename = osp.split(results)
        prefix = filename.split('.')[0]

    trace_file = 'traces_' + prefix + '.' + img_format
    posterior_histograms_file = ('posterior_histograms_' + prefix + '.' +
            img_format)

    # loading colors and (drift) parameter types
    if colors_and_types is not None:
        colors = OrderedDict()
        types = OrderedDict()
        #priortypes = OrderedDict()
        with open(colors_and_types) as inp:
            for line in inp:
                spline = line.split()
                param, color, paramtype = spline[:3]
                if paramtype not in validparamtypes:
                    raise ValueError('invalid parameter type: must be "fixed", '
                                     '"rate", or "bottleneck"')
                colors[param] = color
                types[param] = paramtype
                #try:
                #    priortype1, priortype2 = spline[3:5]
                #    priortypes[param] = (priortype1, priortype2)
                #except ValueError:
                #    continue
        #priors_found = (len(priortypes) > 0)
        #validpriortypes = ('loguniform', 'uniform')
        #for params in priortypes.keys():
        #    if ptype not in validpriortypes:
        #        raise ValueError('Invalid prior type: {}. must be "uniform" or '
        #                         '"loguniform".'.format(ptype))

    ###################
    # loading the data
    ###################
    tree_file = tree
    with open(tree_file) as fin:
        tree_str = fin.read().strip()
    tree = newick.loads(tree_str,
                        look_for_multiplier = True,
                        length_parser = ut.length_parser_str)[0]

    cols = get_parameter_names(tree_file)
    varnames = [col[:-2] for col in cols if col.endswith('_l')]
    try:
        dat_m1 = pd.read_csv(results,
                sep = '\t', header = None, names = cols, dtype = np.float64)
    except ValueError:
        dat_m1 = pd.read_csv(results,
                sep = '\t', header = 0, names = cols, dtype = np.float64)
    # drift parameters are output in natural scale
    dat_m1.loc[:,dat_m1.columns.str.contains('_l')] = np.log10(np.abs(
            dat_m1.loc[:,dat_m1.columns.str.contains('_l')]))
    # mutation parameters are given in log-10 scale
    dat_m1.loc[:,dat_m1.columns.str.contains('_m')] = -np.abs(
            dat_m1.loc[:,dat_m1.columns.str.contains('_m')])
    dat_m1.loc[:,'root'] = -np.abs(dat_m1.loc[:,'root'])
    dat_m1.loc[:,'ppoly'] = -np.abs(dat_m1.loc[:,'ppoly'])
    num_walkers = num_walkers
    burnin_steps = trace_burnin_steps
    chains = get_chains(dat_m1.iloc[burnin_steps*num_walkers:,:].copy(),
            num_walkers)

    # getting (optional) length prior limits
    if length_prior_limits is not None:
        lower_l, upper_l = length_prior_limits
        lower_l = 10**float(lower_l)
        upper_l = 10**float(upper_l)
        length_limits = get_len_limits(tree, ages_file, lower_l, upper_l)

    # getting true parameters
    if true_parameters is not None:
        true_bls, true_mrs, true_ab, true_ppoly = get_parameters(
                true_parameters, tree)
        true_params = [true_bls[v] for v in varnames]
        true_params += [true_mrs[v] for v in varnames]
        true_params += [true_ab, true_ppoly]

    # do_traces
    if do_traces:
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
            if true_parameters is not None and i > 0:
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
    if colors_and_types is None:
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

    burnin_steps = posterior_burnin_steps
    burnin = num_walkers * burnin_steps
    if burnin > dat_m1.shape[0]:
        burnin = 0
    plotted = np.tile([False], nrows*ncols)
    plotted.shape = (nrows,ncols)

    if burnin > dat_m1.shape[0]:
        raise ValueError('burnin exceeds number of samples')


    if final:
        if not add_title:
            mpl.rc('text', usetex = False)
        #mpl.rcParams.update({'font.family': 'Arial'})
        colornames = list(colors.keys())
        for counter, var in enumerate(colornames):
            if true_parameters is not None:
                true_l = np.log10(true_bls[var])
                true_m = np.log10(true_mrs[var])
            # length
            if colors_and_types is not None:
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
            if length_prior_limits is not None:
                minv, maxv = length_limits[var]
            else:
                minv, maxv = dat_m1[col][burnin:].min(), dat_m1[col][burnin:].max()

            lower = np.floor(minv)
            upper = np.ceil(maxv)
            bins = np.linspace(lower, upper, 50)
            ticks = np.arange(lower, upper+1).astype(np.int)
            if ticks.shape[0] > 5:
                ticks = np.arange(lower, upper+1, 2).astype(np.int)

            if length_prior_limits is not None:
                x = np.linspace(minv, maxv, 1000)
                if log_uniform_drift_priors:
                    y = 1.0/(maxv-minv)
                else:
                    y = 10**x * np.log(10) / (10**maxv - 10**minv)
                ax.fill_between(x, 0, y, alpha = 0.3, color = 'gray')

            ax.hist(dat_m1[col][burnin:], bins, color = c, hatch = hatch,
                    normed = True)
            ax.set_xticks(ticks)
            #ax.set_xscale('log')
            ax.get_yaxis().set_visible(False)
            if true_parameters is not None:
                ax.axvline(true_l, ls = '--', c = 'black', lw = 2)
            if add_title:
                ax.set_title(col)

            ax = axes[1][counter]
            col = var + '_m'
            if mutation_prior_limits is None:
                minv, maxv = dat_m1[col][burnin:].min(), dat_m1[col][burnin:].max()
            else:
                minv, maxv = mutation_prior_limits
                minv = float(minv)
                maxv = float(maxv)
            lower = np.floor(minv)
            upper = np.ceil(maxv)
            ticks = np.arange(lower, upper+1).astype(np.int)
            if ticks.shape[0] > 5:
                ticks = np.arange(lower, upper+1, 2).astype(np.int)
            bins = np.linspace(lower, upper, 50)

            if mutation_prior_limits is not None:
                x = np.linspace(minv, maxv, 1000)
                y = 1.0/(maxv-minv)
                ax.fill_between(x, 0, y, alpha = 0.3, color = 'gray')

            ax.hist(dat_m1[col][burnin:], bins, color = c, hatch = hatch,
                    normed = True)
            ax.set_xticks(ticks)
            ax.get_yaxis().set_visible(False)
            if true_parameters is not None:
                ax.axvline(true_m, ls = '--', c = 'black', lw = 2)
            if add_title:
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
                if true_parameters is not None and counter > 0:
                    true_p = true_params[counter-1]
                minv, maxv = dat_m1[col][burnin:].min(), dat_m1[col][burnin:].max()
                if colors_and_types is not None:
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
                if true_parameters is not None and counter > 0:
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
    plt.savefig(posterior_histograms_file, dpi = dpi)


def _run_make_figures(args):

    make_figures(
            results = args.results,
            tree = args.tree,
            num_walkers = args.num_walkers,
            ages_file = args.ages_file,
            prefix = args.prefix,
            img_format = args.format,
            dpi = args.dpi,
            colors_and_types = args.colors_and_types,
            length_prior_limits = args.length_prior_limits,
            mutation_prior_limits = args.mutation_prior_limits,
            true_parameters = args.true_parameters,
            do_traces = args.plot_traces,
            trace_burnin_steps = args.trace_burnin_steps,
            posterior_burnin_steps = args.posterior_burnin_steps,
            add_title = args.add_title,
            log_uniform_drift_priors = args.log_uniform_drift_priors)

