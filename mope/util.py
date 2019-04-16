from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from builtins import range
from past.utils import old_div
import argparse
import multiprocessing as mp
import os
os.environ['OPL_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
import numpy as np
from scipy.special import logit, expit
import logging
import resource


class MemoryFilter(logging.Filter):
    '''
    for use in debugging memory
    '''
    def filter(self, record):
        record.memusg = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + 
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        return True


def get_debug_func(debug_opt):
    if debug_opt:
        def debug(s):
            print('@ ' + s)
    else:
        def debug(s):
            return
    return debug

def length_parser_str(x):
    if x:
        return str(x)
    else:
        return None

def positive_int(val):
    val = int(val)
    if val <= 0:
        raise argparse.ArgumentError("invalid positive integer")
    return val

def positive_float(val):
    val = float(val)
    if val <= 0:
        raise argparse.ArgumentError("invalid positive float")
    return val

def probability(val):
    val = float(val)
    if val < 0 or val > 1:
        raise argparse.ArgumentError("invalid probability")
    return val

def nonneg_int(val):
    val = int(val)
    if val < 0:
        raise argparse.ArgumentTypeError(
                "invalid non-negative integer: {}".format(val))
    return val

def nonneg_float(val):
    val = float(val)
    if val < 0:
        raise argparse.ArgumentTypeError(
                "invalid non-negative integer: {}".format(val))
    return val

'''
def mp_approx_fprime(x, f, pool, eps = 1e-8, chunksize = 1):
    num_params = x.shape[0]
    xs = []
    for i in range(num_params):
        xp = x.copy()
        xp[i] += eps
        xs.append(xp)
    fx = f(x)
    fxps = pool.map(f, xs, chunksize)
    fprimes = []
    for i in range(num_params):
        fprime = (fxps[i]-fx) / eps
        fprimes.append(fprime)
    return np.array(fprimes)
'''

def mp_approx_fprime(x, inq, outq, eps = 1e-8):
    num_params = x.shape[0]
    xs = []
    for i in range(num_params):
        xp = x.copy()
        xp[i] += eps
        inq.put((i, xp))
    inq.put((-1, x))
    
    outputs = []
    for i in range(num_params+1):
        outputs.append(outq.get())
    outputs.sort()

    fxps = [fxp for idx, fxp in outputs[1:]]
    fx = outputs[0][1]
    
    fprimes = []
    for i in range(num_params):
        fprime = old_div((fxps[i]-fx), eps)
        fprimes.append(fprime)
    return np.array(fprimes)

def _check_input_fns(datas, trees, ages):
    fns = [datas, trees, ages]
    if len(fns[0]) != len(fns[1]) or len(fns[0]) != len(fns[2]):
        raise ValueError('--data-files, --tree-files, and --age-files must be of the same length')
    for grp in fns:
        for fn in grp:
            if not os.path.exists(fn):
                raise ValueError('could not find file {}'.format(fn))
    return



def get_input_files(args):
    '''
    look at args.input_file and (args.data_files, args.tree_files,
    args.age_files)

    args.input_file takes precedence
    '''


    if args.input_file is not None:
        datas, trees, ages = [], [], []
        inf = open(args.input_file)
        for line in inf:
            spline = line.strip().split()
            if len(spline) != 3:
                raise ValueError('--input-files file must have three columns: data file, tree file, and age file')
            datas.append(spline[0])
            trees.append(spline[1])
            ages.append(spline[2])
    else:
        datas = args.data_files.split(',')
        trees = args.tree_files.split(',')
        ages = args.age_files.split(',')

    _check_input_fns(datas, trees, ages)
    return datas, trees, ages
