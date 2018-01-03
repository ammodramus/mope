from __future__ import print_function, division
import argparse
import pandas as pd
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np
import numpy.random as npr
import sys

def _add_det_noise(args):

    if args.data == '-':
        args.data = sys.stdin

    dat = pd.read_csv(args.data, sep = '\t', comment = '#')

    comment_lines = []
    with open(args.data) as fin:
        for line in fin:
            if line.strip().startswith('#'):
                comment_lines.append(line.strip())
            else:
                break

    count_cols = [col for col in dat.columns if col.endswith('_n')]
    data_cols = [col.replace('_n','') for col in count_cols]
    if len(count_cols) == 0 or len(count_cols) != len(data_cols):
        raise ValueError('this script requires count data')

    fnr = args.false_negative_rate
    fpr = args.false_positive_rate
    min_freq = args.min_freq
    max_freq = args.max_freq

    dat_c = dat.copy()
    dat_c.loc[:,data_cols] /= dat_c.loc[:,count_cols].values

    for col in data_cols:
        col_idx = dat.columns.get_loc(col)
        col_idx_n = dat.columns.get_loc(col + '_n')

        # the false negative noise
        low_freq_indices = np.where((dat_c[col].values <= max_freq) &
                (dat_c[col].values >= min_freq))[0]
        n_eligible_low = low_freq_indices.shape[0]
        nfalseneg_low = npr.binomial(n_eligible_low, fnr)
        false_negs_low = npr.choice(low_freq_indices, size = nfalseneg_low, replace = False)
        dat.iloc[false_negs_low,col_idx] = 0  # replacing here

        high_freq_indices = np.where((dat_c[col].values < 1-min_freq) &
                (dat_c[col].values > 1-max_freq))[0]
        n_eligible_high = high_freq_indices.shape[0]
        nfalseneg_high = npr.binomial(n_eligible_high, fnr)
        false_negs_high = npr.choice(high_freq_indices, size = nfalseneg_high, replace = False)
        dat.iloc[false_negs_high,col_idx] = dat.iloc[false_negs_high,col_idx_n]  # replacing here

        if args.debug:
            print('adding {} false negatives'.format(nfalseneg_low +
                nfalseneg_high), file = sys.stderr)

        # the false positive noise
        low_freq_indices = np.where(dat_c[col].values < min_freq)[0]
        n_eligible_low = low_freq_indices.shape[0]
        nfalsepos_low = npr.binomial(n_eligible_low, fpr)
        false_poss_low = npr.choice(low_freq_indices, size = nfalsepos_low, replace = False)
        false_pos_freqs = npr.uniform(min_freq, max_freq, size = nfalsepos_low) # get sim freqs
        # replace
        dat.iloc[false_poss_low,col_idx] = (false_pos_freqs*dat.iloc[false_poss_low,col_idx_n] + 0.5).astype(np.int)

        high_freq_indices = np.where(dat_c[col].values > 1-min_freq)[0]
        n_eligible_high = high_freq_indices.shape[0]
        nfalsepos_high = npr.binomial(n_eligible_high, fpr)
        false_poss_high = npr.choice(high_freq_indices, size = nfalsepos_high, replace = False)
        false_pos_freqs = npr.uniform(1-max_freq, 1-min_freq, size = nfalsepos_high) # get sim freqs
        # replace
        dat.iloc[false_poss_high,col_idx] = (false_pos_freqs*dat.iloc[false_poss_high,col_idx_n] + 0.5).astype(np.int)

        if args.debug:
            print('adding {} false positives'.format(nfalsepos_low +
                nfalsepos_high), file = sys.stderr)

    for line in comment_lines:
        sys.stdout.write(line + '\n')
    dat.to_csv(sys.stdout, sep = '\t', index = False)
