from __future__ import division
import numpy as np
import sys
import pandas as pd
import scipy.optimize as opt
import numpy.linalg as npl
import data as da
from util import *
from data import *
import likelihoods as lis
from functools import partial
from collections import Counter

import newick
import ascertainment as asc

def estimate_initial_parameters(inf):
    init_branch_params = get_initial_params_fst_newick(inf)
    ess = 100
    heterozygosities = {}
    if inf.data_are_counts:
        coverage = inf.data.loc[:, inf.count_names]
    else:
        coverage = None
    heterozygosity = get_heterozygosity(inf.data,
            inf.leaf_names, coverage, inf.genome_size, ess, inf.num_leaves)
    init_mrs_dict = get_thetas(inf, heterozygosity)
    init_mrs = np.zeros(len(inf.varnames))
    for i, par in enumerate(inf.varnames):
        init_mrs[i] = np.log10(init_mrs_dict[par])
    ab_init = -2
    pp_init = -4

    # initial guess for bottleneck sizes will be 2/b, where b is the initial
    # guess in units of N generations  EDIT: 100 for now, will get maximized
    for i, vn in enumerate(inf.varnames):
        if inf.is_bottleneck[vn]:
            #l = init_branch_params[i]
            #init_branch_params[i] = 2/l
            init_branch_params[i] = 100

    init_params = np.concatenate((init_branch_params,
        init_mrs, (ab_init,pp_init)))
    if inf.lower is not None:
        init_params = np.maximum(init_params, 1.01*inf.lower)
    if inf.upper is not None:
        init_params = np.minimum(init_params, 0.99*inf.upper)
    return init_params

def estimate_fst_som(counts_i, counts_j, n):
    '''calculate distance based on Fst'''
    p_i = counts_i/n
    p_j = counts_j/n
    zeros =  np.logical_and(p_i == 0, p_j==0)
    ones =  np.logical_and(p_i == 1, p_j == 1)
    good = np.logical_not(np.logical_or(zeros,ones))
    p_i = p_i[good]
    p_j = p_j[good]
    p = (p_i + p_j) / 2
    #fst = ((p_i-p)**2 + (p_j-p)**2) / (2*p*(1-p))
    num = ((p_i-p_j)**2).sum()
    den = (2*(p-p_i*p_j)).sum()
    return num/den

def estimate_fst(counts_i, n_i, counts_j, n_j):
    '''calculate distance based on Fst'''
    p_i = counts_i/n_i
    p_j = counts_j/n_j
    zeros =  np.logical_and(p_i == 0, p_j==0)
    ones =  np.logical_and(p_i == 1, p_j == 1)
    bad = np.logical_or(zeros,ones)
    p_i = p_i[np.logical_not(bad)]
    p_j = p_j[np.logical_not(bad)]
    n_i = n_i[np.logical_not(bad)]
    n_j = n_j[np.logical_not(bad)]
    p = (p_i + p_j) / 2
    #fst = ((p_i-p)**2 + (p_j-p)**2) / (2*p*(1-p))
    num = ((p_i-p_j)**2).sum()
    den = (2*(p-p_i*p_j)).sum()
    return num/den

def estimate_fst_freqs_wright(p1, p2):
    '''
    calculate mean fst based on called frequencies
    '''
    p = (p1+p2)/2.0
    is_het = ((p>0) & (p<1))
    vp = ((p1-p)**2 + (p2-p)**2) / 2.0
    fsts = vp[is_het]/((p*(1-p))[is_het])
    return fsts.mean()

def estimate_fst_freqs_hudson(p1, p2):
    hw = (2*p1*(1-p1) + 2*p2*(1-p2)) / 2.0
    hb = 1.0/2 * (2*p1*(1-p2) + 2*p2*(1-p1))
    ok = (hb != 0)
    fsts = (1-hw[ok]/hb[ok])
    return fsts.mean()

def estimate_fst_freqs(p_i, p_j):
    p = (p_i + p_j) / 2
    num = ((p_i-p_j)**2).sum()
    den = (2*(p-p_i*p_j)).sum()
    return num/den

def get_dist_matrix(data, num_leaves, count_data = None):
    dist_matrix = np.zeros((num_leaves,num_leaves))
    if count_data is not None:
        ns = count_data.values
        for i in xrange(num_leaves-1):
            countsi = data.iloc[:,i].values.astype(np.float64)
            missing_i = np.isnan(countsi)
            for j in xrange(i+1, num_leaves):
                countsj = data.iloc[:,j].values.astype(np.float64)
                missing_j = np.isnan(countsj)
                either_missing = (missing_i | missing_j)
                fst = estimate_fst(
                        countsi[~either_missing],
                        ns[~either_missing,i],
                        countsj[~either_missing],
                        ns[~either_missing,j])
                # x 2 because distance is twice the divergence time
                dist_matrix[i,j] = -2*np.log(1-fst)
    else:
        for i in xrange(num_leaves-1):
            freqs_i = data.iloc[:,i].values.astype(np.float64)
            missing_i = np.isnan(freqs_i)
            for j in xrange(i+1,num_leaves):
                freqs_j = data.iloc[:,j].values.astype(np.float64)
                missing_j = np.isnan(freqs_j)
                either_missing = (missing_i | missing_j)
                fst = estimate_fst_freqs(freqs_i[~either_missing],
                        freqs_j[~either_missing])
                dist_matrix[i,j] = -2*np.log(1-fst)
    return dist_matrix

def get_connecting_branch_names(node1, node2):
    node1ancestors = []
    node = node1
    while True:
        node1ancestors.append(node.name)
        anc = node.ancestor
        if not anc:
            break
        node = anc
    node = node2
    node2ancestors = []
    while True:
        node2ancestors.append(node.name)
        anc = node.ancestor
        if not anc:
            break
        node = anc
    mrca = [anc for anc in node2ancestors if anc in node1ancestors][0]
    connecting_branches = []
    for anc in node1ancestors:
        if anc == mrca:
            break
        connecting_branches.append(anc)
    for anc in node2ancestors:
        if anc == mrca:
            break
        connecting_branches.append(anc)
    return connecting_branches

def get_initial_params_fst_newick(inf):

    tree = inf.tree
    data = inf.data
    branch_names = inf.branch_names
    num_leaves = len(inf.leaf_names)
    data_are_counts = inf.data_are_counts
    leaf_names = inf.leaf_names
    branch_indices = inf.branch_indices
    leaves = []
    for node in tree.walk('postorder'):
        if node.is_leaf:
            leaves.append(node)

    # leaf_names must be in post order
    leaf_data = inf.data.loc[:,inf.leaf_names]

    if inf.data_are_counts:
        count_data = inf.data.loc[:, inf.count_names]
    else:
        count_data = None
    dist_matrix = get_dist_matrix(leaf_data, num_leaves, count_data)

    varnames_set = set([])
    mean_ages = {}
    name_to_varname = {}
    for col in branch_names:
        multname = inf.multiplierdict[col]
        varname = inf.varnamedict[col]
        name_to_varname[col] = varname
        varnames_set.add(varname)
        if multname:
            not_missing = ~np.isnan(data[multname])
            mean_age = data[multname][not_missing].mean()
            mean_ages[col] = mean_age
        else:
            mean_ages[col] = 1.0

    num_len_vars = len(inf.varnames)
    var_indices = inf.var_indices

    branch_counts = {}
    pair_indices = {}
    pidx = 0
    for i in xrange(num_leaves-1):
        for j in xrange(i+1,num_leaves):
            pair_indices[(i,j)] = pidx
            pidx += 1
            connecting_branches = get_connecting_branch_names(leaves[i],
                    leaves[j])
            branch_counts[(i,j)] = connecting_branches[:]

    num_comparisons = int(num_leaves*(num_leaves-1)/2)
    A = np.zeros((num_comparisons,num_len_vars))
    b = np.zeros(num_comparisons)
    
    state_indices = {}

    idx = 0
    for i in xrange(num_leaves-1):
        for j in xrange(i+1, num_leaves):
            bcs = branch_counts[(i,j)]
            for bc in bcs:
                vname = name_to_varname[bc]
                vidx = var_indices[vname]
                A[idx, vidx] += mean_ages[bc]
            b[idx] = dist_matrix[i,j]
            state_indices[(i,j)] = idx
            idx += 1

    params, resid, rank, s = npl.lstsq(A,b)
    min_value = 1e-3
    params[params < min_value] = min_value
    # now do constrained least squares.
    fun = lambda x: np.dot(A, x) - b
    bounds = ([1e-3] * num_len_vars, [np.inf] * num_len_vars)
    res = opt.least_squares(fun, params, bounds = bounds)

    var_params = res.x
    
    #params = []
    #for brname in inf.branch_names:
    #    varname = inf.varnamedict[brname]
    #    vidx = var_indices[varname]
    #    parvalue = var_params[vidx]
    #    params.append(parvalue)
    
    #params = np.array(params)
    return var_params

def get_heterozygosity(data, data_columns, coverage,
        genome_size, effective_sample_size, num_indivs):
    '''
    now going to assume that any column is data is either counts or frequencies

    data                     pandas DataFrame of counts or frequencies
    data_columns             columns of data giving the counts or freqs
    locus_counts             
    coverage                 coverage, for counts, otherwise None
    genome_size              number of sites in the genome
    effective_sample_size    number of haploid samples, effectively
    '''
    assert 3/2 == 1.5
    if coverage is not None:
        allele_counts = data.loc[:,data_columns].as_matrix().astype(np.float64)
        freqs = allele_counts / coverage
    else:  # data are freqs
        freqs = data.loc[:,data_columns]
    heterozygous = ((freqs != 0) & (freqs != 1.0))
    het_per_site = heterozygous.sum(axis = 1)
    heterozygous_by_site = het_per_site
    num_hets = heterozygous_by_site.sum()
    het = num_hets / (genome_size * num_indivs)
    coeff = np.sum(1/np.arange(1,effective_sample_size))
    return het/coeff

def get_thetas(inf_data, heterozygosity_theta):
    '''
    for each heteroplasmy, find the mrca of all the leaves that are
    heteroplasmic, assume that the mutation occured on that node's branch.
    '''

    desc_list = {}
    for node in inf_data.tree.walk('postorder'):
        # if node is a leaf, add its name as a descendant of each of its
        # ancestors' nodes
        if node.is_leaf:
            node_p = node
            while node_p.ancestor:
                if node_p.ancestor.name not in desc_list:
                    desc_list[node_p.ancestor.name] = [node.name]
                else:
                    desc_list[node_p.ancestor.name].append(node.name)
                node_p = node_p.ancestor


    freq_dat = inf_data.data.loc[:,inf_data.leaf_names].values
    if inf_data.data_are_counts:
        count_dat = inf_data.data.loc[:,inf_data.count_names].values
        freq_dat = freq_dat / count_dat.astype(np.float64)
    is_het = (freq_dat == 0.0) | (freq_dat == 1.0)
    leaf_names_arr = np.array(inf_data.leaf_names)

    num_mutations = {}
    for node in inf_data.tree.walk('postorder'):
        num_mutations[node.name] = 0
    num_mutations_groups = Counter()
    for row in is_het:
        het_leaves = leaf_names_arr[row]
        if het_leaves.shape[0] == 0:
            continue
        if het_leaves.shape[0] == 1:
            num_mutations[het_leaves[0]] += 1
        else:
            het_leaves = frozenset(list(het_leaves))
            num_mutations_groups[het_leaves] += 1

    for key, val in desc_list.iteritems():
        desc_list[key] = frozenset(val)

    for group, count in num_mutations_groups.items():
        for node in inf_data.tree.walk('postorder'):
            if node.is_leaf:
                continue
            # this breaks on the first success, and since we're traversing
            # postorder, we get the MRCA
            if group.issubset(desc_list[node.name]):
                num_mutations[node.name] += count
                break

    # count the number of mutations per varname
    num_mutations_varnames = {}
    vnames_counts = {}
    for nodename in num_mutations:
        if nodename == inf_data.tree.name:
            continue
        vname = inf_data.varnamedict[nodename]
        if vname not in num_mutations_varnames:
            num_mutations_varnames[vname] = 0
        if vname not in vnames_counts:
            vnames_counts[vname] = 0
        num_mutations_varnames[vname] += num_mutations[nodename]
        vnames_counts[vname] += 1

    # count all the mutations and the number of non-root branches
    root_vname = inf_data.tree.varname
    tot_num_muts = 0
    num_keys = 0
    for vname in num_mutations_varnames:
        tot_num_muts += num_mutations_varnames[vname]
        num_keys += vnames_counts[vname]
    avg_num_muts = tot_num_muts / num_keys

    bmrs = {}
    for vname in num_mutations_varnames.keys():
        if vname == root_vname:
            continue
        bmrs[vname] = max(1e-6,
                (float(num_mutations_varnames[vname]) /
                    (vnames_counts[vname]*avg_num_muts) * 
                    heterozygosity_theta))

    return bmrs

def is_anc_het(data, tree, data_are_counts):
    mrca_descs = tree.descendants
    descs_leaves = []
    for desc in mrca_descs:
        desc_leaves = [] 
        for node in desc.walk('postorder'):
            if node.is_leaf:
                desc_leaves.append(node.name)
        descs_leaves.append(desc_leaves)
    
    are_hets = []
    for desc_leaf_list in descs_leaves:
        desc_is_het = ~is_fixed(data, desc_leaf_list, data_are_counts)
        are_hets.append(desc_is_het)
    num_mrca_descs_het = are_hets[0].astype(np.int)
    for are_het in are_hets[1:]:
        num_mrca_descs_het += are_het.astype(np.int)
    return (num_mrca_descs_het > 1)

def get_frac_stat_not_fixed(ab, freqs, breaks, N, target):
    stat_dist = lis.get_stationary_distribution_beta(freqs, breaks, N,
            ab, ab)
    # don't really need full root distribution
    val = np.abs(1.0-stat_dist[0]-stat_dist[-1]-target)
    return val

def estimate_ab(inf_data):
    anc_het = is_anc_het(inf_data.data, inf_data.tree,
            inf_data.data_are_counts)
    num_anc_hets = anc_het.sum()
    frac_anc_het = num_anc_hets / inf_data.genome_size
    freqs = inf_data.transition_data.get_bin_frequencies()
    breaks = inf_data.breaks
    N = inf_data.transition_N
    obj = partial(get_frac_stat_not_fixed, freqs = freqs, breaks = breaks,
            N = N, target = frac_anc_het)
    res = opt.minimize_scalar(obj)
    return res.x

def estimate_ab_2(inf_data,
        init_bls,
        init_bmrs,
        init_lrs,
        init_lmrs):
    '''
    This works, except that initial estimates of the mutation rates are too
    high.
    '''
    num_not_fixed = (~da.is_fixed(
        inf_data.data, inf_data.leaf_names, inf_data.data_are_counts)).sum()
    frac_not_fixed = num_not_fixed / inf_data.genome_size
    # make objective function that calculates ascertainment probability as a
    # function of ab
    mean_age = inf_data.data['age'].mean()
    asc_ages_obj = np.array([mean_age])
    asc_counts_obj = np.array([1.0])
    tmp_mrca = newick.loads(inf_data.tree_str, num_freqs = inf_data.num_freqs,
            num_loci = 2)[0]

    def ab_obj_func(ab, target, asc_mrca, bls, bmrs, lrs, lmrs, branch_indices,
            leaf_indices, ages, counts, transitions, freqs, breaks, N):
        stat_dist = lis.get_stationary_distribution_beta(freqs, breaks, N,
                10**ab, 10**ab)
        log_asc_prob = asc.get_ascertainment_prob_somatic_newick_2(
                tmp_mrca,
                bls,
                bmrs,
                lrs,
                lmrs,
                branch_indices,
                leaf_indices,
                ages,
                counts,
                transitions,
                stat_dist)
        asc_prob = np.exp(log_asc_prob)
        return (asc_prob-target)**2

    obj = partial(ab_obj_func,
        target = frac_not_fixed,
        asc_mrca = inf_data.asc_tree,
        bls = init_bls,
        bmrs = init_bmrs,
        lrs = init_lrs,
        lmrs = init_lmrs,
        branch_indices = inf_data.branch_indices,
        leaf_indices = inf_data.leaf_indices,
        ages = asc_ages_obj,
        counts = asc_counts_obj,
        transitions = inf_data.transition_data,
        freqs = inf_data.freqs,
        breaks = inf_data.breaks,
        N = inf_data.transition_N)

    res = opt.minimize_scalar(obj,
            method = 'bounded',
            bounds = [-8,2])
    return res.x



def randomize_params(params, min_max_factor = 5, all_same = False):
    if not all_same:
        r_size = params.shape[0]
    else:
        r_size = 1
    mult_factors = np.exp(npr.uniform(np.log(1/min_max_factor), 
        np.log(min_max_factor), size = r_size))
    r_params = params * mult_factors
    return r_params

if __name__ == '__main__':
    import pandas as pd
    import data as da
    import newick
    from simulate import get_parameters
    import transition_data_mut as tdm
    import inference as inf
    inf_data = inf.Inference(
            data_file = 'test_data_m1.txt',
            transitions_file = 'transition_matrices_mutation_gens3_symmetric.h5',
            tree_file = 'm1.newick',
            method = 'lbfgs',
            true_parameters = 'm1.params',
            start_from_true = False,
            data_are_freqs = True,
            fst_filter_frac = None,
            genome_size = 10000,
            num_processes = 1,
            ascertainment = True,
            print_res = True)

    params = inf_data.init_params
    #print params
    nv = len(inf_data.varnames)
    for i in xrange(nv):
        print '{}\t{}\t{}'.format(
                inf_data.varnames[i],
                params[i],
                params[nv+i])
    print 'ab\t{}\t{}'.format(params[2*nv], params[2*nv])

    tparams = inf_data.true_params

    print '-----------'
    for i in xrange(nv):
        print '{}\t{}\t{}'.format(
                inf_data.varnames[i],
                tparams[i],
                tparams[nv+i])
    print 'ab\t{}\t{}'.format(tparams[2*nv], tparams[2*nv])
