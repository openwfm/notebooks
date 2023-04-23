import numpy as np
from functools import singledispatch
import pandas as pd
import numbers

numeric_kinds = {'i', 'u', 'f', 'c'}

def is_numeric_ndarray(array):
    if isinstance(array, np.ndarray):
        return array.dtype.kind in numeric_kinds
    else:
        return False

def vprint(*args):
    import inspect
    
    frame = inspect.currentframe()
    if 'verbose' in frame.f_back.f_locals:
        verbose = frame.f_back.f_locals['verbose']
    else:
        verbose = False
    
    if verbose: 
        for s in args[:(len(args)-1)]:
            print(s, end=' ')
        print(args[-1])
        
        
## Generic function to hash dictionary of various types

@singledispatch
## Top level hash function with built-in hash function for str, float, int, etc
def hash2(x):
    return hash(x)

@hash2.register(np.ndarray)
## Hash numpy array, hash array with pandas and return integer sum
def _(x):
    # return hash(x.tobytes())
    return np.sum(pd.util.hash_array(x))

@hash2.register(list)
## Hash list, convert to tuple
def _(x):
    return hash2(tuple(x))

@hash2.register(tuple)
def _(x):
    r = 0
    for i in range(len(x)):
        r+=hash2(x[i])
    return r

@hash2.register(dict)
## Hash dict, loop through keys and hash each element via dispatch. Return hashed integer sum of hashes
def _(x, keys = None, verbose = False):
    r = 0 # return value integer
    if keys is None: # allow user input of keys to hash, otherwise hash them all
        keys = [*x.keys()]
    keys.sort()
    for key in keys:
        if (verbose): print('Hashing', key)
        r += hash2(x[key])
    return hash(r)

def print_args(func, *args, **kwargs):
# wrapper to trace function call and arguments
    print(f"Called: {func.__name__}")
    print("Arguments:")
    for arg in args:
        print(f"  {arg}")
    for key, value in kwargs.items():
        print(f"  {key}={value}")
    return func(*args, **kwargs)

def print_args_test():
    def my_function(a, b):
        # some code here
        return a + b
    print_args(my_function, a=1, b=2)
