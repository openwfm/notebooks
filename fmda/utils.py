import numpy as np
from functools import singledispatch
import pandas as pd
import numbers
from datetime import datetime

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
    
import inspect
def get_item(dict,var,**kwargs):
    caller_name = inspect.stack()[1][3]
    if var in dict:
        value = dict[var]
    elif 'default' in kwargs:
        value = kwargs['default']
    try:
        verbose
    except NameError:
        verbose = True
    if verbose:
        print(caller_name,':',var,'=',value)
    return value
        
def print_dict_summary(d,indent=0,print_first=[],first_num=3):
    """
    Prints a summary for each array in the dictionary, showing the key and the size of the array.

    Arguments:
     d (dict): The dictionary to summarize.
     first_items (list): Print the first items for any arrays with these names
    
    """
    indent_str = ' ' * indent
    for key, value in d.items():
        # Check if the value is list-like using a simple method check
        if isinstance(value, dict):
            print(f"{indent_str}{key}")
            print_dict_summary(value,indent=indent+5)
        elif isinstance(value,np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                print(f"{indent_str}{key}: NumPy array of shape {value.shape}, min: {value.min()}, max: {value.max()}")
            else:
                # Handle non-numeric arrays differently 
                print(f"{indent_str}{key}: NumPy array of shape {value.shape}, type {value.dtype}")
            if key in print_first:
                print_first(value,first_num)
        elif hasattr(value, "__iter__") and not isinstance(value, str):  # Check for iterable that is not a string
            print(f"{indent_str}{key}: Array of {len(value)} items")
            if key in print_first:
                print_first(value,first_num)

        else:
            print(indent_str,key,":",value)
            
def print_first(item_list,num=3):
    """
    Print the first num items of the list followed by '...' 

    :param item_list: List of items to be printed
    :param num: number of items to list
    """
    if len(item_list) > num:
        print(type(item_list[0]))
    for i in range(min(num,len(item_list))):
        print(item_list[i])
    if len(item_list) > num:
        print('...')
                   
from datetime import datetime

def str2time(input):
    """
    Convert a single string timestamp or a list of string timestamps to corresponding datetime object(s).
    """
    if isinstance(input, str):
        return datetime.strptime(input.replace('Z', '+00:00'), '%Y-%m-%dT%H:%M:%S%z')
    elif isinstance(input, list):
        return [str2time(s) for s in input]
    else:
        raise ValueError("Input must be a string or a list of strings")


# interpolate linearly over nans

def filter_nan_values(t1, v1):
    # Filter out NaN values from v1 and corresponding times in t1
    valid_indices = ~np.isnan(v1)  # Indices where v1 is not NaN
    t1_filtered = np.array(t1)[valid_indices]
    v1_filtered = np.array(v1)[valid_indices]
    return t1_filtered, v1_filtered

def time_intp(t1, v1, t2):
    t1_no_nan, v1_no_nan = filter_nan_values(t1, v1)
    
    # Convert datetime objects to timestamps
    t1_stamps = np.array([t.timestamp() for t in t1_no_nan])
    t2_stamps = np.array([t.timestamp() for t in t2])
    
    # Interpolate using the filtered data
    v2_interpolated = np.interp(t2_stamps, t1_stamps, v1_no_nan)
    
    return v2_interpolated


def str2time(strlist):
    # convert array of strings to array of datetime
    return np.array([np.datetime64(dt_str) for dt_str in strlist])