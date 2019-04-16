from __future__ import division
from __future__ import unicode_literals
from builtins import str
import os
os.environ['OPL_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
import numpy as np
import numpy.random as npr
import pandas as pd
from collections import Counter

def get_ascertainable_only(all_loci):
    assert 3/2 == 1.5
    left_labels = {
            'A': ['M'],
            'B': ['M'],
            'C': ['M'],
            'D': ['M'],
            'E': ['G'],
            'F': ['G'],
            'G': ['G'],
            'H': ['M'],
            'I': ['GG'],
            'J': ['GG'],
            'K': ['A']
            }
    right_labels = {
            'A': list('12'),
            'B': list('123'),
            'C': list('1234'),
            'D': list('12345'),
            'E': list('M12'),
            'F': list('M123'),
            'G': list('AM12'),
            'H': list('12'),
            'I': list('GM123'),
            'J': ['GA', 'G', 'M', '1', '2'],
            'K': list('M12')
            }
    asc_loci = {}
    for ped in list(all_loci.keys()):
        left_count_vars = [('bld'+el, 'buc'+el) for el in left_labels[ped]]
        left_count_vars = [el for sublist in left_count_vars for el in sublist]
        right_count_vars = [('bld'+el, 'buc'+el) for el in right_labels[ped]]
        right_count_vars = [el for sublist in right_count_vars
                for el in sublist]
        left_size_vars = [('bld'+el+'n', 'buc'+el+'n')
                for el in left_labels[ped]]
        left_size_vars = [el for sublist in left_size_vars for el in sublist]
        right_size_vars = [('bld'+el+'n', 'buc'+el+'n')
                for el in right_labels[ped]]
        right_size_vars = [el for sublist in right_size_vars
                for el in sublist]
        left_count = all_loci[ped].eval('+'.join(left_count_vars))
        left_size = all_loci[ped].eval('+'.join(left_size_vars))
        left_freq = left_count / left_size
        right_count = all_loci[ped].eval('+'.join(right_count_vars))
        right_size = all_loci[ped].eval('+'.join(right_size_vars))
        right_freq = right_count / right_size
        asc_loci[ped] = all_loci[ped].loc[(left_freq > 0) & (left_freq < 1) &
                (right_freq > 0) & (right_freq < 1),:].copy()
        if asc_loci[ped].shape[0] == 0:
            del asc_loci[ped]
    return asc_loci

def get_polymorphic_only(all_loci):
    assert 3/2 == 1.5
    left_labels = {
            'A': ['M'],
            'B': ['M'],
            'C': ['M'],
            'D': ['M'],
            'E': ['G'],
            'F': ['G'],
            'G': ['G'],
            'H': ['M'],
            'I': ['GG'],
            'J': ['GG'],
            'K': ['A']
            }
    right_labels = {
            'A': list('12'),
            'B': list('123'),
            'C': list('1234'),
            'D': list('12345'),
            'E': list('M12'),
            'F': list('M123'),
            'G': list('AM12'),
            'H': list('12'),
            'I': list('GM123'),
            'J': ['GA', 'G', 'M', '1', '2'],
            'K': list('M12')
            }
    polymorphic_loci = {}
    for ped in list(all_loci.keys()):
        left_count_vars = [('bld'+el, 'buc'+el) for el in left_labels[ped]]
        left_count_vars = [el for sublist in left_count_vars for el in sublist]
        right_count_vars = [('bld'+el, 'buc'+el) for el in right_labels[ped]]
        right_count_vars = [el for sublist in right_count_vars
                for el in sublist]
        left_size_vars = [('bld'+el+'n', 'buc'+el+'n')
                for el in left_labels[ped]]
        left_size_vars = [el for sublist in left_size_vars for el in sublist]
        right_size_vars = [('bld'+el+'n', 'buc'+el+'n')
                for el in right_labels[ped]]
        right_size_vars = [el for sublist in right_size_vars
                for el in sublist]
        left_count = all_loci[ped].eval('+'.join(left_count_vars))
        left_size = all_loci[ped].eval('+'.join(left_size_vars))
        left_freq = left_count / left_size
        right_count = all_loci[ped].eval('+'.join(right_count_vars))
        right_size = all_loci[ped].eval('+'.join(right_size_vars))
        right_freq = right_count / right_size
        polymorphic_loci[ped] = all_loci[ped].loc[((left_freq > 0) & (left_freq < 1)) |
                ((right_freq > 0) & (right_freq < 1)),:].copy()
        if polymorphic_loci[ped].shape[0] == 0:
            del polymorphic_loci[ped]
    return polymorphic_loci

def get_avg_min_max_age(data):
    min_age = data['age'].values.min()
    max_age = data['age'].values.max()
    avg_age = np.mean(data['age'].values)
    return avg_age, min_age, max_age

def get_A_data_from_ped_data(data, pedtype):
    '''
    Returns just the frequencies / allele counts for the M, 1, and 2 samples,
    so that FST can be estimated from any pedigree, not just A.

    data            pandas DataFrame representing data from a pedigree, not
                    necessarily type A

    Returns

    data_A_only     pandas DataFrame with just the relevant (M, 1, and 2)
                    columns
    '''
    which_ys = {
            'A': (1,2),
            'B': (1,2),
            'C': (1,2),
            'D': (1,2),
            'E': (2,3),
            'F': (2,3),
            'G': (3,4),
            'H': (1,2),
            'I': (3,4),
            'J': (4,5),
            'K': (3,4)
            }
    y_col_names = ['y'+str(el) for el in which_ys[pedtype]]
    y_col_translation = {y_col_names[0]:'y1', y_col_names[1]:'y2'}
    keeper_columns = ['bldM', 'bldMn', 'bucM', 'bucMn', 'bld1', 'bld1n',
            'buc1', 'buc1n', 'bld2', 'bld2n', 'buc2', 'buc2n']
    keeper_columns = y_col_names + keeper_columns
    data_A_only = data[keeper_columns].copy()
    data_A_only = data_A_only.rename(columns = y_col_translation)
    return data_A_only

def get_max_loo(all_data, transitions):
    max_y = -1.0
    for ped_type in all_data:
        d = all_data[ped_type]
        y_cols = [el for el in d.columns.astype(str) if el.startswith('y')]
        ped_max_y = np.array(d.loc[:,y_cols]).max()
        max_y = max(max_y, ped_max_y)
    max_gentime = transitions.get_max_coal_time()
    max_loo = max_gentime / max_y
    return max_loo

#####################################################
##
## filtering
##
#####################################################

def calculate_global_fst(ps):
    ps = ps[~np.isnan(ps)]
    p = ps.mean()
    q = 1-p
    qs = 1-ps
    subpop_hets = 2*ps*qs
    HT = 2*p*q
    HS = subpop_hets.mean()
    if HT == 0:
        return 0
    fst = (HT - HS)/HT
    return fst

def find_high_fst_indices(data, data_columns, quantile, data_are_counts):
    '''
    filter data by global fst

    data              pandas DataFrame representing counts or frequencies of
                      heteroplasmies
    quantile          top 100*quantile percentile of FST indices will be
                      returned
    data_are_counts   bool, if true, then there must also be a coverage column
                      in data
    '''
    ps = data.loc[:,data_columns]
    if data_are_counts:
        ps = ps.divide(data['coverage'], axis = 1)

    fsts = []
    for row in ps.itertuples():
        row = np.array(row[1:])
        fst = calculate_global_fst(row)
        fsts.append(fst)

    fsts = np.array(fsts)
    cutoff = np.percentile(fsts[~np.isnan(fsts)], 100.0*(1-quantile))
    # np.where returns tuple in same number of dimensions as its input
    is_high_fst = (fsts >= cutoff)
    return is_high_fst

def is_fixed(data, data_columns, data_are_counts):
    ps = data.loc[:,data_columns]
    if data_are_counts:
        n_columns = [el + '_n' for el in data_columns]
        ns = data.loc[:,n_columns]
        ps = ps.values / ns.values
    freqsums = np.nansum(ps, axis = 1).astype(np.float64)
    ncols = ps.shape[1]
    fixed = np.isclose(freqsums, ncols) | (freqsums == 0.0)
    return fixed

def randomize_alleles(data, data_columns, data_are_counts):
    ret_data = data.reset_index(drop = True)
    nrows = data.shape[0]
    nrows_to_switch = npr.binomial(nrows, 0.5, size = 1)
    rows_to_switch = np.sort(
            npr.choice(nrows, nrows_to_switch, replace = False)).astype(np.int)
    count_columns = [el + '_n' for el in data_columns]
    if not data_are_counts:
        ret_data.ix[rows_to_switch,data_columns] = (
                1.0 - ret_data.ix[rows_to_switch,data_columns])
    else:
        coverage = ret_data[count_columns][rows_to_switch].values
        ret_data.ix[rows_to_switch,data_columns] = (
                coverage[:,np.newaxis] - ret_data.ix[rows_to_switch,data_columns])
    
    return ret_data
