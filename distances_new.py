from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from sklearn.metrics.pairwise import pairwise_distances

import argparse
argParser = argparse.ArgumentParser(prog="pairwise_distances.py",
                                    description="sklearn pairwise_distances benchmark",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argParser.add_argument('-i', '--iteration', default='10', help="iteration", type=int)
argParser.add_argument('-p', '--proc', default=-1, help="n_jobs for algorithm", type=int)
args = argParser.parse_args()

REP = args.iteration 
try:
    from daal.services import Environment
    nThreadsInit = Environment.getInstance().getNumberOfThreads()
    Environment.getInstance().setNumberOfThreads(args.proc)
except:
    pass


def st_time(func):
    def st_func(*args, **keyArgs):
        times = []
        for n in range(REP):
            t1 = timeit.default_timer()
            r = func(*args, **keyArgs)
            t2 = timeit.default_timer()
            times.append(t2-t1)
        print (min(times), end='')
        return r
    return st_func

problem_sizes = [
        (500, 10000),
        (500, 50000),
        (500, 100000),
        (500, 150000),
        (500, 200000),
        (1000, 10000),
        (1000, 50000),
        (1000, 100000),
        (1000, 150000),
        (1000, 200000)]

X={}
for p, n in problem_sizes:
    #print('Generating %dx%d'%(p,n))
    X[(p,n)] = rand(p,n)


@st_time
def cosine(X):
    cos_dist = pairwise_distances(X, metric='cosine', n_jobs=args.proc) 
@st_time
def correlation(X):
    cor_dist = pairwise_distances(X, metric='correlation', n_jobs=args.proc) 

for p, n in problem_sizes:
    print (p,n, end=' ')
    X_local = X[(p,n)]
    cosine(X_local)
    print(' ', end='')
    correlation(X_local)
    print('')
