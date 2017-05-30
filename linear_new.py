from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from sklearn import linear_model

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
        (1000000,5),
        (1000000,25),
        (1000000,50),
        (5000000,5),
        (5000000,25),
        (5000000,50),
        (10000000,5),
        (10000000,25),
        (10000000,50)]
X={}
Xp={}
y={}
for p, n in problem_sizes:
    X[(p,n)] = rand(p,n)
    Xp[(p,n)] = rand(p,n)
    y[(p,n)] = rand(p,n)

regr = linear_model.LinearRegression(n_jobs=args.proc)

@st_time
def test_fit(X,y):
    regr.fit(X,y)
    
@st_time
def test_predict(X):
    regr.predict(X)

for p, n in problem_sizes:
    print (p,n, end=' ')
    X_local = X[(p,n)]
    Xp_local = X[(p,n)]
    y_local = X[(p,n)]
    test_fit(X_local, y_local)
    print(' ', end='')
    test_predict(Xp_local)
    print('')
