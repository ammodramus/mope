from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
import re

def _run_acceptance(args):
    datapat = re.compile('[0-9-]')  # line starts with a number

    k = args.numwalkers

    prev = [None] * k
    accept_count = [0] * k
    
    datlinecount = 0
    with open(args.datafile, 'r') as fin:
        for line in fin:
            if re.match(datapat, line):
                chain_idx = datlinecount % k
                datlinecount += 1
                if line != prev[chain_idx]:
                    prev[chain_idx] = line
                    accept_count[chain_idx] += 1

    if args.all:
        for i, count in enumerate(accept_count):
            print(i, count / (datlinecount/k))
    else:
        avg_accept = sum([el/(datlinecount/k) for el in accept_count]) / k
        print(avg_accept)
