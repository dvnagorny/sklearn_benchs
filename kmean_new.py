from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from sklearn.cluster import KMeans

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
        (10000,   2),
        (10000,   25),
        (10000,   50),
        (50000,   2),
        (50000,   25),
        (50000,   50),
        (100000,  2),
        (100000,  25),
        (100000,  50)]

X={}
for p, n in problem_sizes:
    X[(p,n)] = rand(p,n)


kmeans = KMeans(n_clusters=10, n_jobs=args.proc)
@st_time
def train(X):
    kmeans.fit(X)

for p, n in problem_sizes:
    print (p,n, end=' ')
    X_local = X[(p,n)]
    train(X_local)
    print('')
