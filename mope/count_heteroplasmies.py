from __future__ import print_function
from __future__ import unicode_literals
import argparse
import pandas as pd
import os 
import numpy as np
import sys

def count_hets(args):

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

    datacols = None
    for line in comment_lines:
        if 'data columns' in line:
            cols_str = line.split(':')[1].strip()
            datacols = cols_str.split(',')
    if not datacols:
        raise ValueError('data columns not found in comments')

    datp = dat.loc[:,datacols]

    count_cols = [c + '_n' for c in datacols]
    if not all([c in dat.columns for c in count_cols]):
        args.frequencies = True

    if not args.frequencies:
        zeros = (dat[datacols].values == 0)
        fixes = (dat[datacols].values == dat[count_cols].values)

    else:
        zeros = datp.values == 0
        fixes = datp.values == 1.0

    # count zeros

    heteroplasmic = ~(zeros | fixes)

    num_heteroplasmic = heteroplasmic.sum(axis = 1)

    # want...
    #  ... number of samples that are heteroplasmic (i.e., count each sample at
    #      each site independently)
    #  ... number of sites that are heterplasmic
    #  ... number of sites that are heteroplasmic in 1, 2, and 3+ tissues

    template = '{:42}{:5}'
    print(template.format('number of sites', num_heteroplasmic.shape[0]))
    print(template.format('number of samples',
            heteroplasmic.shape[0]*heteroplasmic.shape[1]))
    print(template.format('number of heteroplasmic samples',
            heteroplasmic.sum()))
    print(template.format('number of heterplasmic sites',
            (num_heteroplasmic != 0).sum()))
    print(template.format('number of sites heterplasmic in 1 tissue',
            (num_heteroplasmic == 1).sum()))
    print(template.format('number of sites heterplasmic in 2 tissues',
            (num_heteroplasmic == 2).sum()))
    print(template.format('number of sites heterplasmic in 3+ tissues',
            (num_heteroplasmic >= 3).sum()))
