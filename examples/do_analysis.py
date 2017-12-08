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
    max_idx = np.argmax(lls)
    mapest = d[max_idx]
    hpdl, hpdh = hpd(d, cifrac)
    return mapest, med, hpdl, hpdh

parser = argparse.ArgumentParser(
        description='do some analyses of paper example data')
parser.add_argument('datafile', help = 'data filename, from mope run')
parser.add_argument('--frac-burnin', '-b', type = float,
        help = 'fraction of data to consider burnin [0.3]',
        default = 0.3)
args = parser.parse_args()

dat = pd.read_csv(args.datafile, sep = '\t', header = 0, comment = '#')
burn_idx = int(dat.shape[0] * args.frac_burnin)
dat_burn = dat.iloc[burn_idx:,:]

# calculate EBSs

mean_age = 29.558974
bots_mean = (2.0/(2.0/dat_burn['eoo_l'] + 
    dat_burn['som_l'] + mean_age*dat_burn['loo_l'])).values
bots_18 = (2.0/(2.0/dat_burn['eoo_l'] + 
    dat_burn['som_l'] + 18*dat_burn['loo_l'])).values
bots_40 = (2.0/(2.0/dat_burn['eoo_l'] + 
    dat_burn['som_l'] + 40*dat_burn['loo_l'])).values

# calculate loo rates
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

columns = ['map', 'median', 'ci05', 'ci95']

lls = dat_burn['ll'].values
dat_dict = OrderedDict()

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

outdat = pd.DataFrame.from_dict(dat_dict, orient = 'index')
outdat.columns = columns

outdat.to_csv(sys.stdout, index_label = 'var', sep = '\t')