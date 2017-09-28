from __future__ import division
import argparse
import sys
import pandas as pd
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np


def run_ages(args):

    dcols = None
    with open(args.input) as fin:
        for line in fin:
            if line.strip().startswith('#') and 'data columns' in line:
                dcols = line.split(':')[1].strip().split(',')
    if dcols is None:
        raise ValueError('data columns not found in input')
    count_cols = [d + '_n' for d in dcols]

    dat = pd.read_csv(args.input, sep = '\t', comment = '#')

    age_columns = list(dat.columns[dat.columns.str.contains('age')])

    try:
        families = dat['family']
    except KeyError:
        raise KeyError("family column not found in input")

    new_age_dat = {}
    for ac in age_columns:
        new_age_dat[ac] = []

    new_age_dat['family'] = families.unique()

    for family in families.unique():
        datp = dat.loc[families == family,age_columns].drop_duplicates()
        if datp.shape[0] != 1:
            raise ValueError('found distinct age combinations in same family')
        for ac in age_columns:
            new_age_dat[ac].append(datp[ac].iloc[0])

    new_age_dat = pd.DataFrame(new_age_dat)
    new_age_dat = new_age_dat.loc[:,['family'] + age_columns]

    new_age_dat.to_csv(sys.stdout, index = False, sep = '\t')
