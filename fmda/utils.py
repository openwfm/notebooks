import numpy as np
from functools import singledispatch
import pandas as pd
import numbers
from datetime import datetime
import logging
import sys
import inspect
import yaml
import hashlib
import pickle
import os.path as osp
from urllib.parse import urlparse
import subprocess

from itertools import islice

def rmse_3d(preds, y_test):
    """
    Calculate RMSE for ndarrays structured as (batch_size, timesteps, features). 
    The first dimension, batch_size, could denote distinct locations. The second, timesteps, is length of sequence
    """
    squared_diff = np.square(preds - y_test)
    
    # Mean squared error along the timesteps and dimensions (axis 1 and 2)
    mse = np.mean(squared_diff, axis=(1, 2))
    
    # Root mean squared error (RMSE) for each timeseries
    rmses = np.sqrt(mse)
    
    return rmses
    

class Dict(dict):
    """
    A dictionary that allows member access to its keys.
    A convenience class.
    """

    def __init__(self, d):
        """
        Updates itself with d.
        """
        self.update(d)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, item, value):
        self[item] = value

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        else:
            for key in self:
                if isinstance(key,(range,tuple)) and item in key:
                    return super().__getitem__(key)
            raise KeyError(item)

    def keys(self):
        if any([isinstance(key,(range,tuple)) for key in self]):
            keys = []
            for key in self:
                if isinstance(key,(range,tuple)):
                    for k in key:
                        keys.append(k)
                else:
                    keys.append(key)
            return keys
        else:
            return super().keys()

def save(obj, file):
    with open(file,'wb') as output:
        dill.dump(obj, output )

def load(file):
    with open(file,'rb') as input:
        returnitem = dill.load(input)
        return returnitem

# Utility to retrieve files from URL
def retrieve_url(url, dest_path, force_download=False):
    """
    Downloads a file from a specified URL to a destination path.

    Parameters:
    -----------
    url : str
        The URL from which to download the file.
    dest_path : str
        The destination path where the file should be saved.
    force_download : bool, optional
        If True, forces the download even if the file already exists at the destination path.
        Default is False.

    Warnings:
    ---------
    Prints a warning if the file extension of the URL does not match the destination file extension.

    Raises:
    -------
    AssertionError:
        If the download fails and the file does not exist at the destination path.

    Notes:
    ------
    This function uses the `wget` command-line tool to download the file. Ensure that `wget` is 
    installed and accessible from the system's PATH.

    Prints:
    -------
    A message indicating whether the file was downloaded or if it already exists at the 
    destination path.
    """    
    if not osp.exists(dest_path) or force_download:
        print(f"Attempting to downloaded {url} to {dest_path}")
        target_extension = osp.splitext(dest_path)[1]
        url_extension = osp.splitext(urlparse(url).path)[1]
        if target_extension != url_extension:
            print("Warning: file extension from url does not match destination file extension")
        subprocess.call(f"wget -O {dest_path}  {url}", shell=True)
        assert osp.exists(dest_path)
        print(f"Successfully downloaded {url} to {dest_path}")
    else:
        print(f"Target data already exists at {dest_path}")

# Function to check if lists are nested, or all elements in given list are in target list
def all_items_exist(source_list, target_list):
    """
    Checks if all items from the source list exist in the target list.

    Parameters:
    -----------
    source_list : list
        The list of items to check for existence in the target list.
    target_list : list
        The list in which to check for the presence of items from the source list.

    Returns:
    --------
    bool
        True if all items in the source list are present in the target list, False otherwise.

    Example:
    --------
    >>> source_list = [1, 2, 3]
    >>> target_list = [1, 2, 3, 4, 5]
    >>> all_items_exist(source_list, target_list)
    True

    >>> source_list = [1, 2, 6]
    >>> all_items_exist(source_list, target_list)
    False
    """    
    return all(item in target_list for item in source_list)

# Generic helper function to read yaml files
def read_yml(yaml_path, subkey=None):
    """
    Reads a YAML file and optionally retrieves a specific subkey.

    Parameters:
    -----------
    yaml_path : str
        The path to the YAML file to be read.
    subkey : str, optional
        A specific key within the YAML file to retrieve. If provided, only the value associated 
        with this key will be returned. If not provided, the entire YAML file is returned as a 
        dictionary. Default is None.

    Returns:
    --------
    dict or any
        The contents of the YAML file as a dictionary, or the value associated with the specified 
        subkey if provided.

    """    
    with open(yaml_path, 'r') as file:
        d = yaml.safe_load(file)
        if subkey is not None:
            d = d[subkey]
    return d

# Use to load nested fmda dictionary of cases
def load_and_fix_data(filename):
    # Given path to FMDA training dictionary, read and return cleaned dictionary
    # Inputs: 
    # filename: (str) path to file with .pickle extension
    # Returns:
    # FMDA dictionary with NA values "fixed"
    print(f"loading file {filename}")
    with open(filename, 'rb') as handle:
        test_dict = pickle.load(handle)
        for case in test_dict:
            test_dict[case]['case'] = case
            test_dict[case]['filename'] = filename
            for key in test_dict[case].keys():
                var = test_dict[case][key]    # pointer to test_dict[case][key]
                if isinstance(var,np.ndarray) and (var.dtype.kind == 'f'):
                    nans = np.sum(np.isnan(var))
                    if nans:
                        print('WARNING: case',case,'variable',key,'shape',var.shape,'has',nans,'nan values, fixing')
                        fixnan(var)
                        nans = np.sum(np.isnan(test_dict[case][key]))
                        print('After fixing, remained',nans,'nan values')
            if not 'title' in test_dict[case].keys():
                test_dict[case]['title']=case
            if not 'descr' in test_dict[case].keys():
                test_dict[case]['descr']=f"{case} FMDA dictionary"
    return test_dict

# Generic helper function to read pickle files
def read_pkl(file_path):
    """
    Reads a pickle file and returns its contents.

    Parameters:
    -----------
    file_path : str
        The path to the pickle file to be read.

    Returns:
    --------
    any
        The object stored in the pickle file.

    Prints:
    -------
    A message indicating the file path being loaded.

    Notes:
    ------
    This function uses Python's `pickle` module to deserialize the contents of the file. Ensure 
    that the pickle file was created in a safe and trusted environment to avoid security risks 
    associated with loading arbitrary code.

    """    
    with open(file_path, 'rb') as file:
        print(f"loading file {file_path}")
        d = pickle.load(file)
    return d

def logging_setup():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
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
        
        
## Function for Hashing numpy arrays 
def hash_ndarray(arr: np.ndarray) -> str:
    """
    Generates a unique hash string for a NumPy ndarray.

    Parameters:
    -----------
    arr : np.ndarray
        The NumPy array to be hashed.

    Returns:
    --------
    str
        A hexadecimal string representing the MD5 hash of the array.

    Notes:
    ------
    This function first converts the NumPy array to a bytes string using the `tobytes()` method, 
    and then computes the MD5 hash of this bytes string. Performance might be bad for very large arrays.
    
    Example:
    --------
    >>> arr = np.array([1, 2, 3])
    >>> hash_value = hash_ndarray(arr)
    >>> print(hash_value)
    '2a1dd1e1e59d0a384c26951e316cd7e6'
    """    
    # If input is list, attempt to concatenate and then hash
    if type(arr) == list:
        arr = np.vstack(arr)
        arr_bytes = arr.tobytes()
    else:
        # Convert the array to a bytes string
        arr_bytes = arr.tobytes()
    # Use hashlib to generate a unique hash
    hash_obj = hashlib.md5(arr_bytes)
    return hash_obj.hexdigest()
    
## Function for Hashing tensorflow models
def hash_weights(model):
    """
    Generates a unique hash string for a the weights of a given Keras model.

    Parameters:
    -----------
    model : A keras model
        The Keras model to be hashed.

    Returns:
    --------
    str
        A hexadecimal string representing the MD5 hash of the model weights.

    """
    # Extract all weights and biases
    weights = model.get_weights()
    
    # Convert each weight array to a string
    weight_str = ''.join([np.array2string(w, separator=',') for w in weights])
    
    # Generate a SHA-256 hash of the combined string
    weight_hash = hashlib.md5(weight_str.encode('utf-8')).hexdigest()
    
    return weight_hash

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
    if var in dict:
        value = dict[var]
    elif 'default' in kwargs:
        value = kwargs['default']
    else:
        logging.error('Variable %s not in the dictionary and no default',var)
        raise NameError()
    logging.info('%s = %s',var,value)
    return value

def print_first(item_list,num=3,indent=0,id=None):
    """
    Print the first num items of the list followed by '...' 

    :param item_list: List of items to be printed
    :param num: number of items to list
    """
    indent_str = ' ' * indent
    if id is not None:
        print(indent_str, id)
    if len(item_list) > 0:
        print(indent_str,type(item_list[0]))
    for i in range(min(num,len(item_list))):
        print(indent_str,item_list[i])
    if len(item_list) > num:
        print(indent_str,'...')
        
def print_dict_summary(d,indent=0,first=[],first_num=3):
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
            print_dict_summary(value,first=first,indent=indent+5,first_num=first_num)
        elif isinstance(value,np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                print(f"{indent_str}{key}: NumPy array of shape {value.shape}, min: {value.min()}, max: {value.max()}")
            else:
                # Handle non-numeric arrays differently 
                print(f"{indent_str}{key}: NumPy array of shape {value.shape}, type {value.dtype}")
        elif hasattr(value, "__iter__") and not isinstance(value, str):  # Check for iterable that is not a string
            print(f"{indent_str}{key}: Array of {len(value)} items")
        else:
            print(indent_str,key,":",value)
        if key in first:
            print_first(value,num=first_num,indent=indent+5)
            
                   
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
    # Check if t1 v1 t2 are 1D arrays
    if t1.ndim != 1:
        logging.error("Error: t1 is not a 1D array. Dimension: %s", t1.ndim)
        return None
    if v1.ndim != 1:
        logging.error("Error: v1 is not a 1D array. Dimension %s:", v1.ndim)
        return None
    if t2.ndim != 1:
        logging.errorr("Error: t2 is not a 1D array. Dimension: %s", t2.ndim)
        return None
    # Check if t1 and v1 have the same length
    if len(t1) != len(v1):
        logging.error("Error: t1 and v1 have different lengths: %s %s",len(t1),len(v1))
        return None
    t1_no_nan, v1_no_nan = filter_nan_values(t1, v1)
    # print('t1_no_nan.dtype=',t1_no_nan.dtype)
    # Convert datetime objects to timestamps
    t1_stamps = np.array([t.timestamp() for t in t1_no_nan])
    t2_stamps = np.array([t.timestamp() for t in t2])
    
    # Interpolate using the filtered data
    v2_interpolated = np.interp(t2_stamps, t1_stamps, v1_no_nan)
    if np.isnan(v2_interpolated).any():
        logging.error('time_intp: interpolated output contains NaN')
    
    return v2_interpolated

def str2time(strlist):
    # Convert array of strings to array of datetime objects
    return np.array([datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%SZ') for dt_str in strlist])

def check_increment(datetime_array,id=''):
    # Calculate time differences between consecutive datetime values
    diffs = [b - a for a, b in zip(datetime_array[:-1], datetime_array[1:])]
    diffs_hours = np.array([diff.total_seconds()/3600 for diff in diffs])
    # Check if all time differences are exactlyu 1 hour
    if all(diffs_hours == diffs_hours[0]):
        logging.info('%s time array increments are %s hours',id,diffs_hours[0])
        if diffs_hours[0] <= 0 : 
            logging.error('%s time array increements are not positive',id)
        return diffs_hours[0]
    else:
        logging.info('%s time array increments are min %s max %s',id,
                        np.min(diffs_hours),np.max(diffs_hours))
        return -1