from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import sys
import argparse
import numpy as np
import numpy.random as npr
import scipy.stats as st
import pandas as pd

from scipy.misc import comb
from collections import OrderedDict

from . import likelihoods as lik

from . import _wf
from . import _util
from . import newick
from . import ages
from .util import *

def get_heteroplasmic_filter(results, is_frequencies, tree):
    leaves = []
    for node in tree.walk('postorder'):
        if node.is_leaf:
            leaves.append(node.name)
    if is_frequencies:
        freqs = results.loc[:,leaves]
    else:
        n_cols = [leaf + '_n' for leaf in leaves]

        freqs = (results.loc[:,leaves] /
                results.loc[:,n_cols].astype(np.float64))
    is_het = ~((freqs.sum(1) == len(leaves)) | (freqs.sum(1) == 0.0))
    return is_het

def get_parameters(
        params_file,
        tree,
        num_families = None,
        sites_per_family = 50,
        ages = None,
        exact = False):
    bls = {}
    mrs = {}
    ab = np.nan
    ages_found = False

    age_names = []

    if ages is not None:
        if not exact:
            age_indices = np.sort(npr.choice(ages.shape[0],
                    size = num_families,
                    replace = True))
            # define family_ages below

    with open(params_file) as fin:
        for line in fin:
            line = line.strip()
            if '---' in line:
                if not num_families:
                    break
                ages_found = True
                mathpat = '([^\s]+)\s+([^\s]+)\s*([\+-])\s*([^\s]+)'
                mathre =  re.compile(mathpat)
                normpat = '([^\s]+)\s+([^\s]+)\s+([^\s]+)'
                normre = re.compile(normpat)
                ret_ages = {}
                continue
            if not ages_found:
                spline = [el for el in line.split() if el != '']
                name = spline[0]
                bl = float(spline[1])
                mr = float(spline[2])
                if name == tree.name:
                    # assume the two parameters are the same at the root
                    ab = bl
                    ppoly = mr
                else:
                    bls[name] = bl
                    mrs[name] = mr

            else:  # (ages found)
                normmat = normre.match(line)  # match for Gaussian pattern
                mathmat = mathre.match(line)  # match for arithmetic pattern
                if normmat:
                    name, mean, sd = normmat.group(1,2,3)
                    if ages is None:
                        mean = float(mean)
                        sd = float(sd)
                        family_ages = np.maximum(
                                npr.normal(mean, sd, size = num_families), 0.4)
                        age_names.append(name)
                        rep_ages = np.repeat(
                                family_ages, sites_per_family)
                        ret_ages[name] = rep_ages
                    else:
                        if not exact:
                            family_ages = ages[name].iloc[age_indices].values
                            rep_ages = np.repeat(
                                    family_ages, sites_per_family)
                            ret_ages[name] = rep_ages
                        else:
                            age_names.append(name)
                    # ages, repeated for output. If the number of replicates is
                    # not a multiple of the number of families, the final
                    # family gets fewer (independent) sites

                elif mathmat:
                    name, arg1, signstr, arg2 = mathmat.group(1,2,3,4)
                    if not exact:
                        if signstr == '-':
                            ret_ages[name] = ret_ages[arg1]-ret_ages[arg2]
                        elif signstr == '+':
                            ret_ages[name] = ret_ages[arg1]+ret_ages[arg2]
                        else:
                            raise ValueError('invalid expression in {}'.format(
                                    params_file))
                    age_names.append(name)
                else:
                    raise ValueError('bad age line: {}'.format(line))

    if np.isnan(ab):
        raise ValueError('root parameter not found in {}'.format(params_file))
    for node in tree.walk('postorder'):
        if node == tree:
            continue
        varname = node.varname
        if varname not in bls:
            raise ValueError('parameter(s) for {} not found in {}'.format(
                varname, params_file))

    if num_families:
        if exact:
            family_ages = ages.loc[:,age_names].copy()
            locus_family_indices = np.repeat(np.arange(family_ages.shape[0]),
                    sites_per_family)
            ret_ages = family_ages.iloc[locus_family_indices,:].copy()
            family_idxs = np.repeat(np.arange(family_ages.shape[0]),
                    sites_per_family)
            ret_ages['family'] = family_idxs
            ret_ages = pd.DataFrame(ret_ages)

        else:
            family_idxs = np.repeat(np.arange(num_families), sites_per_family)
            ret_ages['family'] = family_idxs
            ret_ages = pd.DataFrame(ret_ages)

        return bls, mrs, ab, ppoly, ret_ages
    else:
        return bls, mrs, ab, ppoly


def get_stationary_distn(N, a, b, double_beta = False, ppoly = None):
    assert 3/2 == 1.5
    if double_beta:
        assert ppoly is not None, "ppoly must not be None for double beta"
        p = lik.discretize_double_beta_distn(a, b, N, ppoly)
    else:
        p = lik.discretize_beta_distn(a, b, N)
        p /= p.sum()
    return p

def get_header(leaves, frequencies):
    if not frequencies:
        header = leaves + ['coverage', 'age']
    else:
        header = leaves + ['age']
    return header

def run_simulations(reps, tree, bls, mrs, N, stationary_distn,
        mean_coverage, frequencies, family_indices, exact_ages):
    '''
    reps
    tree
    bls
    mrs
    N
    stationary_distn
    binomial_sample_size
    frequencies            bool, whether to write frequencies
    family_indices         indices for each family
    '''

    leaves = []
    multipliers = {}
    for node in tree.walk('postorder'):
        if node.is_leaf:
            leaves.append(node.name)
        if node.multipliername is not None:
            multipliers[node.multipliername] = node.multipliervalues

    header = leaves + list(multipliers.keys())

    # for each node, counts will hold the allele counts for that node, as a
    # numpy array
    counts = {}

    for node in tree.walk():
        node_name = node.name
        if node == tree:
            # mrca, sample from stationary_distn
            counts[node_name] = npr.choice(
                    np.arange(0,N+1),
                    p = stationary_distn,
                    size = reps,
                    replace = True).astype(np.int32)
        else:
            node_name = node.name
            node_varname = node.varname
            multvalues = node.multipliervalues
            anc_counts = counts[node.ancestor.name]
            if node.multipliervalues is not None:
                if node.is_bottleneck:
                    print(node.multipliervalues)
                    print(node_varname)
                    raise ValueError(
                    'bottleneck specified for a branch with rate')
                rate = bls[node_varname]
                lengths = rate*multvalues
                gens = np.array(np.round(N*lengths).astype(np.int32))
                gen_mut_rate = mrs[node.varname] / (2.0*N)
                counts[node_name] = _wf.simulate_transitions(anc_counts, gens,
                                                             N, gen_mut_rate)
            else:
                length = bls[node_varname]
                gen_mut_rate = mrs[node_varname] / (2.0*N)
                if node.is_bottleneck:
                    Nb = length   # bottleneck size
                    counts[node_name] = _wf.simulate_bottlenecks(anc_counts,
                            N, Nb, gen_mut_rate)
                else:    # not a bottleneck
                    gens = np.array(np.round((N*length)*np.ones(reps)).astype(
                        np.int32))
                    counts[node_name] = _wf.simulate_transitions(
                            anc_counts, gens, N, gen_mut_rate)

    # make a pandas DataFrame
    columns = leaves + list(multipliers.keys())

    # always do binomial sampling at the leaves
    #if binomial_sample_size:
    n = mean_coverage

    if not frequencies:
        for leaf in leaves:
            curcov = npr.poisson(n, size = counts[leaf].shape[0]).astype(
                    np.int32)
            counts[leaf] = _wf.bin_sample_freqs(counts[leaf], N,
                    curcov)
            cov_name = leaf + '_n'
            counts[cov_name] = curcov
        cov_names = [leaf + '_n' for leaf in leaves]
        columns += cov_names
        leaf_cols = leaves + cov_names
    else:
        leaf_cols = leaves[:]

    resultsdict = {col:(counts[col] if col in leaf_cols else multipliers[col])
            for col in columns}
    results = pd.DataFrame(resultsdict, columns = columns)

    results['family'] = family_indices

    if frequencies:
        results[leaves] /= n

    return results


def run_simulate(args):

    with open(args.tree) as fin:
        tree_str = fin.read().strip()
    tree = newick.loads(tree_str, length_parser = length_parser_str,
            look_for_multiplier = True)[0]
    ages_dat = None
    if args.ages:
        ages_dat = pd.read_csv(args.ages, sep = '\t')
    branch_lengths, mut_rates, ab, ppoly, ages = (
            get_parameters(args.parameters, tree, args.num_families,
                args.sites_per_family, ages_dat, args.exact_ages))
    tree.set_multiplier_values(ages)

    leaves = []
    for node in tree.walk('postorder'):
        if node.is_leaf:
            leaves.append(node.name)

    a, b = ab, ab
    if a <= 0:
        raise ValueError('root parameter must be > 0')
    N = args.N
    stationary_distn = get_stationary_distn(N, a, b,
            double_beta = (not args.single_beta), ppoly = ppoly)
    comment_strings = []
    for varname in list(branch_lengths.keys()):
        comment_strings.append("# {} {} {}".format(varname,
            branch_lengths[varname], mut_rates[varname]))
    comment_strings.append('# ab: {}'.format(a))
    comment_strings.append('# data columns: ' + ','.join(leaves))
    comment_strings.append('# ' + ' '.join(sys.argv))
    print('\n'.join(comment_strings))

    if args.exact_ages:
        reps = ages.shape[0]
    else:
        reps = args.num_families * args.sites_per_family

    results = run_simulations(reps, tree,  branch_lengths,
            mut_rates, N, stationary_distn,
            args.mean_coverage, args.frequencies,
            family_indices = ages['family'], exact_ages = args.exact_ages)

    if args.heteroplasmic_only:
        cond = get_heteroplasmic_filter(results, args.frequencies, tree)
        results = results.loc[cond,:]

    results.to_csv(sys.stdout, sep = '\t',
            index = False, float_format = '%.4f')

if __name__ == '__main__':
    run_simulate()
