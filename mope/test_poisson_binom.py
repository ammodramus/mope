from __future__ import division
import numpy as np
import numpy.random as npr

import _poisson_binom as _pb

#npr.seed(100)

cov = 10000

min_alt = 8

num_reps = 10000

with open('test_quals.txt') as fin:
    testquals = fin.read().strip()

qualities = []
for ch in testquals:
    q = ord(ch)-33
    qualities.append(q)
qualities = np.array(qualities, dtype = np.float64)
print 'coverage:', qualities.shape[0]

#freqs = np.arange(101)/100
freqs = np.array([0.24])

min_freq = 0.23

asc_probs = 1.0-_pb.get_non_asc_probs(qualities, freqs, min_freq)

# printing numerical probabilities
for f, p in zip(freqs, asc_probs):
    print '{}\t{}'.format(f,p)

# now, simulations
f = 0.24

error_probs = 10.0**(-qualities/10.0)

ps = f*(1-error_probs) + (1-f)*error_probs

min_num = np.round(ps.shape[0]*min_freq).astype(np.int)

#counts = []
#for rep in xrange(num_reps):
#    count = 0
#    for p in ps:
#        count += npr.binomial(1,p,size=1)[0]
#    counts.append(count)
#counts = np.array(counts)
#print 'num counts:', counts.shape

#emp_asc_prob = (counts/ps.shape[0] >= min_freq).sum() / num_reps
#print emp_asc_prob

emp_non_asc = _pb.empirical_non_asc_probs(
        qualities,
        num_reps,
        f,
        min_freq)

print emp_non_asc
