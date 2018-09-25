from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from builtins import range
import argparse
import numpy as np
import pandas as pd
import sys

from collections import OrderedDict
import pandas as pd

def hpd(trace, mass_frac):
    """
    Returns highest probability density region given by
    a set of samples.
    
    From:
    http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/l06_credible_regions.html

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.
        
    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])


def get_summary(d, lls, cifrac = 0.95):
    med = np.median(d)
    mean = np.mean(d)
    max_idx = np.argmax(lls)
    mapest = d[max_idx]
    hpdl, hpdh = hpd(d, cifrac)
    return mapest, mean, med, hpdl, hpdh

parser = argparse.ArgumentParser(
        description='do some analyses of paper example data')
parser.add_argument('datafile', help = 'data filename, from mope run')
parser.add_argument('--frac-burnin', '-b', type = float,
        help = 'fraction of data to consider burnin [0.3]',
        default = 0.3)
parser.add_argument('--old', action ='store_true',
        help = 'specify for old data, drift in natural scale')
parser.add_argument('--m2', action = 'store_true',
        help = 'use summary statistics from newer data')
parser.add_argument('--no-change', action = 'store_true',
        help = 'include pointmass at zero')
parser.add_argument('--lowers', help = 'file with parameter lower limits, one per line')
args = parser.parse_args()

dat = pd.read_csv(args.datafile, sep = '\t', header = 0, comment = '#')

if args.no_change and not args.old:
    # get lowers
    if args.lowers is None:
        lowers = []
        with open(args.datafile) as fin:
            foundlower = False
            for line in fin:
                line = line.strip()
                if 'lower' in line:
                    foundlower = True
                    continue
                if foundlower:
                    if 'upper' in line:
                        break
                    lowers.append(float(line.split(' ')[-1]))
    else:
        with open(args.lowers) as fin:
            lowers = []
            for line in fin:
                lowers.append(float(line.strip()))

    num_length_cols = dat.columns.str.endswith('_l').sum()
    for i in range(num_length_cols):
        low = lowers[i]
        col = dat.columns[i+1]  # i+1 because first column is log-likelihood!
        dat.loc[dat[col] < low+1, col] = -np.inf

# convert drift back to natural units
if not args.old:
    dat.loc[:,dat.columns.str.contains('_l')] = 10**dat.loc[:,dat.columns.str.contains('_l')]
else:
    dat.loc[:,dat.columns.str.contains('_l')] = dat.loc[:,dat.columns.str.contains('_l')].abs()
    dat.loc[:,dat.columns.str.contains('_m')] = 10**(-1.0*dat.loc[:,dat.columns.str.contains('_m')].abs())
burn_idx = int(dat.shape[0] * args.frac_burnin)
dat_burn = dat.iloc[burn_idx:,:]

# calculate EBSs

# update the new statistic when all of the data is included
if args.m2:
    mean_age = 29.192876
else:
    mean_age = 29.558974


if args.m2:
    bots_mean = 2.0/(dat_burn['eoo_pre_l'] + dat_burn['eoo_post_l'] + dat_burn['som_l'] + mean_age*dat_burn['loo_l']).values
    bots_18 = 2.0/(dat_burn['eoo_pre_l'] + dat_burn['eoo_post_l'] + dat_burn['som_l'] + 18*dat_burn['loo_l']).values
    bots_40 = 2.0/(dat_burn['eoo_pre_l'] + dat_burn['eoo_post_l'] + dat_burn['som_l'] + 40*dat_burn['loo_l']).values
else:
    bots_mean = (2.0/(2.0/dat_burn['eoo_l'] + 
        dat_burn['som_l'] + mean_age*dat_burn['loo_l'])).values
    bots_18 = (2.0/(2.0/dat_burn['eoo_l'] + 
        dat_burn['som_l'] + 18*dat_burn['loo_l'])).values
    bots_40 = (2.0/(2.0/dat_burn['eoo_l'] + 
        dat_burn['som_l'] + 40*dat_burn['loo_l'])).values


# calculate loo rates
if args.m2:
    bots_25 = 2.0/(dat_burn['eoo_pre_l'] + dat_burn['eoo_post_l'] + dat_burn['som_l'] + 25*dat_burn['loo_l']).values
    bots_34 = 2.0/(dat_burn['eoo_pre_l'] + dat_burn['eoo_post_l'] + dat_burn['som_l'] + 34*dat_burn['loo_l']).values
else:
    bots_25 = (2.0/(2.0/dat_burn['eoo_l'] + 
        dat_burn['som_l'] + 25*dat_burn['loo_l'])).values
    bots_34 = (2.0/(2.0/dat_burn['eoo_l'] + 
        dat_burn['som_l'] + 34*dat_burn['loo_l'])).values
rates = (bots_34-bots_25)/(34-25)

# calculate bottlenecks for fblo_l and f_buc_l
bots_fblo = 2.0/dat_burn['fblo_l'].values
bots_fbuc = 2.0/dat_burn['fbuc_l'].values

# frac post-fert
fracpf = dat_burn['som_l'].values / (2.0/bots_mean)

columns = ['map', 'mean', 'median', 'ci05', 'ci95']

lls = dat_burn['ll'].values
dat_dict = OrderedDict()

# now get summaries
for col in list(dat.columns):
    if col == 'll':
        continue
    dat_dict[col] = get_summary(dat_burn[col].values, lls)
dat_dict['bots_mean'] = get_summary(bots_mean, lls)
dat_dict['bots_18'] = get_summary(bots_18, lls)
dat_dict['bots_40'] = get_summary(bots_40, lls)
dat_dict['bot_fblo'] = get_summary(bots_fblo, lls)
dat_dict['bot_fbuc'] = get_summary(bots_fbuc, lls)
dat_dict['loo_rate'] = get_summary(rates, lls)
dat_dict['fracpf'] = get_summary(fracpf, lls)

# print output
outdat = pd.DataFrame.from_dict(dat_dict, orient = 'index')
outdat.columns = columns

outdat.to_csv(sys.stdout, index_label = 'var', sep = '\t')
