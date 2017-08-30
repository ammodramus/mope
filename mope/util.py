import argparse
import multiprocessing as mp
import numpy as np
from scipy.special import logit, expit

def get_debug_func(debug_opt):
    if debug_opt:
        def debug(s):
            print '@ ' + s
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
        fprime = (fxps[i]-fx) / eps
        fprimes.append(fprime)
    return np.array(fprimes)
