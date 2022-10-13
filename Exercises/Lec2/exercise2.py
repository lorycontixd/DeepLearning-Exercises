import numpy as np
import numba as nb
from functools import wraps
import time
from matplotlib import pyplot as plt

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, len(args), len(kw), te-ts))
        return result
    return wrap


def primitive_dot(A,v):
    result = np.zeros(v.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[i] += A[i][j] * v[j]
    return result

@nb.njit(parallel=True)
def primitive_dot2(A,v):
    result = np.zeros(v.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[i] += A[i][j] * v[j]
    return result

@nb.njit(parallel=True, fastmath=True)
def primitive_dot3(A,v):
    result = np.zeros(v.shape)
    for i in nb.prange(A.shape[0]):
        for j in nb.prange(A.shape[1]):
            result[i] += A[i][j] * v[j]
    return result


if __name__=='__main__':
    N = 10000
    v = np.random.uniform(0,1,N).astype(np.float64)
    A = np.random.uniform(0,1,size=(N,N)).astype(np.float64)

    funcs = [primitive_dot, primitive_dot2, primitive_dot3]
    results = []
    times = []
    for f in funcs:
        a = time.time()
        results.append(f(A,v))
        b = time.time()
        times.append(b-a)
    
    fig, ax = plt.subplots(figsize=(14,7))
    plt.title("Execution time of different dot products")
    plt.hist(times, bins=3, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel("Function")
    plt.ylabel("Time")
    plt.show()
    