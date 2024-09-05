# v2 training and prediction class infrastructure

# Environment
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.callbacks import Callback, EarlyStopping, TerminateOnNaN
# from sklearn.metrics import mean_squared_error
import logging
from tensorflow.keras.layers import LSTM, SimpleRNN, Input, Dropout, Dense
# Local modules
import reproducibility
# from utils import print_dict_summary
from abc import ABC, abstractmethod
from utils import hash2, all_items_exist, hash_ndarray, hash_weights
from data_funcs import rmse, plot_data, compare_dicts
import copy
# import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

#*************************************************************************************
# Data Formatting Functions

def staircase(x,y,timesteps,datapoints,return_sequences=False, verbose = False):
    # x [datapoints,features]    all inputs
    # y [datapoints,outputs]
    # timesteps: split x and y into samples length timesteps, shifted by 1
    # datapoints: number of timesteps to use for training, no more than y.shape[0]
    if verbose:
        print('staircase: shape x = ',x.shape)
        print('staircase: shape y = ',y.shape)
        print('staircase: timesteps=',timesteps)
        print('staircase: datapoints=',datapoints)
        print('staircase: return_sequences=',return_sequences)
    outputs = y.shape[1]
    features = x.shape[1]
    samples = datapoints-timesteps+1
    if verbose:
        print('staircase: samples=',samples,'timesteps=',timesteps,'features=',features)
    x_train = np.empty([samples, timesteps, features])
    if return_sequences:
        if verbose:
            print('returning all timesteps in a sample')
        y_train = np.empty([samples, timesteps, outputs])  # all
        for i in range(samples):
            for k in range(timesteps):
                x_train[i,k,:] = x[i+k,:]
                y_train[i,k,:] = y[i+k,:]
    else:
        if verbose:
            print('returning only the last timestep in a sample')
        y_train = np.empty([samples, outputs])
        for i in range(samples):
            for k in range(timesteps):
                x_train[i,k,:] = x[i+k,:]
            y_train[i,:] = y[i+timesteps-1,:]

    return x_train, y_train

def staircase_2(x,y,timesteps,batch_size=None,trainsteps=np.inf,return_sequences=False, verbose = False):
    # create RNN training data in multiple batches
    # input:
    #     x (,features)  
    #     y (,outputs)
    #     timesteps: split x and y into sequences length timesteps
    #                a.k.a. lookback or sequence_length    
    
    # print params if verbose
   
    if batch_size is None:
        raise ValueError('staircase_2 requires batch_size')
    if verbose:
        print('staircase_2: shape x = ',x.shape)
        print('staircase_2: shape y = ',y.shape)
        print('staircase_2: timesteps=',timesteps)
        print('staircase_2: batch_size=',batch_size)
        print('staircase_2: return_sequences=',return_sequences)
    
    nx,features= x.shape
    ny,outputs = y.shape
    datapoints = min(nx,ny,trainsteps)   
    if verbose:
        print('staircase_2: datapoints=',datapoints)
    
    # sequence j in a given batch is assumed to be the continuation of sequence j in the previous batch
    # https://www.tensorflow.org/guide/keras/working_with_rnns Cross-batch statefulness
    
    # example with timesteps=3 batch_size=3 datapoints=15
    #     batch 0: [0 1 2]      [1 2 3]      [2 3 4]  
    #     batch 1: [3 4 5]      [4 5 6]      [5 6 7] 
    #     batch 2: [6 7 8]      [7 8 9]      [8 9 10] 
    #     batch 3: [9 10 11]    [10 11 12]   [11 12 13] 
    #     batch 4: [12 13 14]   [13 14 15]    when runs out this is the last batch, can be shorter
    #
    # TODO: implement for multiple locations, same starting time for each batch
    #              Loc 1         Loc 2       Loc 3
    #     batch 0: [0 1 2]      [0 1 2]      [0 1 2]  
    #     batch 1: [3 4 5]      [3 4 5]      [3 4 5] 
    #     batch 2: [6 7 8]      [6 7 8]      [6 7 8] 
    # TODO: second epoch shift starting time at batch 0 in time
    
    # TODO: implement for multiple locations, different starting times for each batch
    #              Loc 1       Loc 2       Loc 3
    #     batch 0: [0 1 2]   [1 2 3]      [2 3 4]  
    #     batch 1: [3 4 5]   [4 5 6]      [5 6 57 
    #     batch 2: [6 7 8]   [7 8 9]      [8 9 10] 
    
    #
    #     the first sample in batch j starts from timesteps*j and ends with timesteps*(j+1)-1
    #     e.g. the final hidden state of the rnn after the sequence of steps [0 1 2] in batch 0
    #     becomes the starting hidden state of the rnn in the sequence of steps [3 4 5] in batch 1, etc.
    #     
    #     sample [0 1 2] means the rnn is used twice to map state 0 -> 1 -> 2
    #     the state at time 0 is fixed but the state is considered a variable at times 1 and 2 
    #     the loss is computed from the output at time 2 and the gradient of the loss function by chain rule which ends at time 0 because the state there is a constant -> derivative is zero
    #     sample [3 4 5] means the rnn is used twice to map state 3 -> 4 -> 5    #     the state at time 3 is fixed to the output of the first sequence [0 1 2]
    #     the loss is computed from the output at time 5 and the gradient of the loss function by chain rule which ends at time 3 because the state there is considered constant -> derivative is zero
    #     how is the gradient computed? I suppose keras adds gradient wrt the weights at 2 5 8 ... 3 6 9... 4 7 ... and uses that to update the weights
    #     there is only one set of weights   h(2) = f(h(1),w)  h(1) = f(h(0),w)   but w is always the same 
    #     each column is a one successive evaluation of h(n+1) = f(h(n),w)  for n = n_startn n_start+1,... 
    #     the cannot be evaluated efficiently on gpu because gpu is a parallel processor
    #     this of it as each column served by one thread, and the threads are independent because they execute in parallel, there needs to be large number of threads (32 is a good number)\
    #     each batch consists of independent calculations
    #     but it can depend on the result of the previous batch (that's the recurrent parr)
    
    
    
    max_batches = datapoints // timesteps
    max_sequences = max_batches * batch_size

    if verbose:
        print('staircase_2: max_batches=',max_batches)
        print('staircase_2: max_sequences=',max_sequences)
                                      
    x_train = np.zeros((max_sequences, timesteps, features)) 
    if return_sequences:
        y_train = np.empty((max_sequences, timesteps, outputs))
    else:
        y_train = np.empty((max_sequences, outputs ))
        
    # build the sequences    
    k=0
    for i in range(max_batches):
        for j in range(batch_size):
            begin = i*timesteps + j
            next  = begin + timesteps
            if next > datapoints:
                break
            if verbose:
                print('sequence',k,'batch',i,'sample',j,'data',begin,'to',next-1)
            x_train[k,:,:] = x[begin:next,:]
            if return_sequences:
                 y_train[k,:,:] = y[begin:next,:]
            else:
                 y_train[k,:] = y[next-1,:]
            k += 1   
    if verbose:
        print('staircase_2: shape x_train = ',x_train.shape)
        print('staircase_2: shape y_train = ',y_train.shape)
        print('staircase_2: sequences generated',k)
        print('staircase_2: batch_size=',batch_size)
    k = (k // batch_size) * batch_size
    if verbose:
        print('staircase_2: removing partial and empty batches at the end, keeping',k)
    x_train = x_train[:k,:,:]
    if return_sequences:
         y_train = y_train[:k,:,:]
    else:
         y_train = y_train[:k,:]

    if verbose:
        print('staircase_2: shape x_train = ',x_train.shape)
        print('staircase_2: shape y_train = ',y_train.shape)

    return x_train, y_train


# Dictionary of scalers, used to avoid multiple object creation and to avoid multiple if statements
scalers = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler() 
}


## DEPRECATED, use RNNData class instead
def create_rnn_data2(dict1, params, atm_dict="HRRR", verbose=False, train_ind=None, test_ind=None):
    # Given fmda data and hyperparameters, return formatted dictionary to be used in RNN
    # Inputs:
    # d: (dict) fmda dictionary
    # params: (dict) hyperparameters
    # atm_dict: (str) string specifying name of subdictionary for atmospheric vars
    # train_frac: (float) fraction of data to use for training (starting from time 0)
    # val_frac: (float) fraction of data to use for validation data (starting from end of train)
    # Returns: (dict) formatted data used in RNN 
    logging.info('create_rnn_data start')
    # Copy Dictionary to avoid changing the input to this function
    d=copy.deepcopy(dict1)
    scale = params['scale']
    scaler= params['scaler']
    # Features list given by params dict to be used in training
    features_list = params["features_list"]
    # All available features list, corresponds to shape of X
    features_all = d["features_list"]
    # Indices to subset all features with based on params features
    indices = []
    for item in features_list:
        if item in features_all:
            indices.append(features_all.index(item))
        else:
            print(f"Warning: feature name '{item}' not found in list of all features from input data")
        
    # Extract desired features based on params, combine into matrix
    # Extract response vector 
    y = d['y']
    y = np.reshape(y,(-1,1))
    # Extract Features matrix, subset to desired features
    X_raw = d['X'][:, indices].copy() # saw untransformed features matrix 
    X = d['X']
    X = X[:, indices]

    # Check total observed hours
    hours=d['hours']    
    assert hours == y.shape[0] # Check that it matches response
    
    logging.info('create_rnn_data: total_hours=%s',hours)
    logging.info('feature matrix X shape %s',np.shape(X))
    logging.info('target  matrix Y shape %s',np.shape(y))
    logging.info('features_list: %s',features_list)

    logging.info('splitting train/val/test')
    if train_ind is None:
        train_ind = round(hours * params['train_frac']) # index of last training observation
    test_ind= train_ind + round(hours * params['val_frac'])# index of first test observation, if no validation data it is equal to train_ind
    logging.info('Final index of training data=%s',train_ind)
    logging.info('First index of Test data=%s',test_ind)
    # Training data from 0 to train_ind
    X_train = X[:train_ind]
    y_train = y[:train_ind].reshape(-1,1)
    # Validation data from train_ind to test_ind
    X_val = X[train_ind:test_ind]
    y_val = y[train_ind:test_ind].reshape(-1,1)
    # Test data from test_ind to end
    X_test = X[test_ind:]
    y_test = y[test_ind:].reshape(-1,1)

    # Scale Data if required
    # TODO:
        # Remove need for "scale_fm" param
        # Reset reproducibility with this scaling
    if scale:
        logging.info('Scaling feature data with scaler: %s',scaler)
        # scale=1
        if scaler=="reproducibility":
            scale_fm = 17.076346687085564
            scale_rain = 0.01
        else:
            scale_fm=1.0
            scale_rain=1.0
            # Fit scaler to training data
            scalers[scaler].fit(X_train)
            # Apply scaling to all data using in-place operations
            X_train[:] = scalers[scaler].transform(X_train)
            if X_val.shape[0] > 0:
                X_val[:] = scalers[scaler].transform(X_val)
            X_test[:] = scalers[scaler].transform(X_test)
            
            
    else:
        print("Not scaling data")
        scale_fm=1.0
        scale_rain=1.0
        scaler=None
    
    logging.info('x_train shape=%s',X_train.shape)
    logging.info('y_train shape=%s',y_train.shape)
    if test_ind == train_ind:
        logging.info('No validation data')
    elif X_val.shape[0]!= 0:
        logging.info('X_val shape=%s',X_val.shape)
        logging.info('y_val shape=%s',y_val.shape)    
    logging.info('X_test shape=%s',X_test.shape)
    logging.info('y_test shape=%s',y_test.shape)
    
    # Set up return dictionary
    rnn_dat={
        'case':d['case'],
        'hours':hours,
        'features_list':features_list,
        'n_features': len(features_list),
        'scaler':scaler,
        'train_ind':train_ind,
        'test_ind':test_ind,
        'X_raw': X_raw,
        'X':X,
        'y':y,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }

    if X_val.shape[0] > 0:
            rnn_dat.update({
                'X_val': X_val,
                'y_val': y_val
            })

    # Update RNN params using data attributes
    logging.info('Updating model params based on data')
    timesteps = params['timesteps']
    batch_size = params['batch_size']
    logging.info('batch_size=%s',batch_size)
    logging.info('timesteps=%s',timesteps)
    features = len(features_list)
    # params.update({
    #         'n_features': features,
    #         'batch_shape': (params["batch_size"],params["timesteps"],features),
    #         'pred_input_shape': (None, features),
    #         'scaler': scaler,
    #         'scale_fm': scale_fm,
    #         'scale_rain': scale_rain
    #     })
    rnn_dat.update({
        'scaler': scaler, 
        'scale_fm': scale_fm,
        'scale_rain': scale_rain
    })
    
    logging.info('create_rnn_data2 done')
    return rnn_dat

def batch_setup(ids, batch_size):
    """
    Sets up stateful batched training data scheme for RNN training.

    This function takes a list or array of identifiers (`ids`) and divides them into batches of a specified size (`batch_size`). If the last batch does not have enough elements to meet the `batch_size`, the function will loop back to the start of the identifiers and continue filling the batch until it reaches the required size.

    Parameters:
    -----------
    ids : list or numpy array
        A list or numpy array containing the ids to be batched.

    batch_size : int
        The desired size of each batch. 

    Returns:
    --------
    batches : list of lists
        A list where each element is a batch (itself a list) of identifiers. Each batch will contain exactly `batch_size` elements.

    Example:
    --------
    >>> ids = [1, 2, 3, 4, 5]
    >>> batch_size = 3
    >>> batch_setup(ids, batch_size)
    [[1, 2, 3], [4, 5, 1]]

    Notes:
    ------
    - If `ids` is shorter than `batch_size`, the returned list will contain a single batch where identifiers are repeated from the start of `ids` until the batch is filled.
    """   
    # Ensure ids is a numpy array
    x = np.array(ids)
    
    # Initialize the list to hold the batches
    batches = []
    
    # Use a loop to slice the list/array into batches
    for i in range(0, len(x), batch_size):
        batch = list(x[i:i + batch_size])
        
        # If the batch is not full, continue from the start
        while len(batch) < batch_size:
            # Calculate the remaining number of items needed
            remaining = batch_size - len(batch)
            # Append the needed number of items from the start of the array
            batch.extend(x[:remaining])
        
        batches.append(batch)
    
    return batches

def staircase_spatial(X, y, batch_size, timesteps, hours=None, start_times = None, verbose = True):
    """
    Prepares spatially formatted time series data for RNN training by creating batches of sequences across different locations, stacked to be compatible with stateful models.

    This function processes multi-location time series data by slicing it into batches and formatting it to fit into a recurrent neural network (RNN) model. It utilizes a staircase-like approach to prepare sequences for each location and then interlaces them to align with stateful RNN structures.

    Parameters:
    -----------
    X : list of numpy arrays
        A list where each element is a numpy array containing features for a specific location. The shape of each array is `(time_steps, features)`.

    y : list of numpy arrays
        A list where each element is a numpy array containing the target values for a specific location. The shape of each array is `(time_steps,)`.

    batch_size : int
        The number of sequences to include in each batch.

    timesteps : int
        The number of time steps to include in each sequence for the RNN.

    hours : int, optional
        The length of each time series to consider for each location. If `None`, it defaults to the minimum length of `y` across all locations.

    start_times : numpy array, optional
        The initial time step for each location. If `None`, it defaults to an array starting from 0 and incrementing by 1 for each location.

    verbose : bool, optional
        If `True`, prints additional information during processing. Default is `True`.

    Returns:
    --------
    XX : numpy array
        A 3D numpy array with shape `(total_sequences, timesteps, features)` containing the prepared feature sequences for all locations.

    yy : numpy array
        A 2D numpy array with shape `(total_sequences, 1)` containing the corresponding target values for all locations.

    n_seqs : int
        Number of sequences per location. Used to reset states when location changes

    Notes:
    ------
    - The function handles spatially distributed time series data by batching and formatting it for stateful RNNs.
    - `hours` determines how much of the time series is used for each location. If not provided, it defaults to the shortest series in `y`.
    - If `start_times` is not provided, it assumes each location starts its series at progressively later time steps.
    - The `batch_setup` function is used internally to manage the creation of location and time step batches.
    - The returned feature sequences `XX` and target sequences `yy` are interlaced to align with the expected input format of stateful RNNs.
    """
    
    # Generate ids based on number of distinct timeseries provided
    n_loc = len(y) # assuming each list entry for y is a separate location
    loc_ids = np.arange(n_loc)

    # Generate hours and start_times if None
    if hours is None:
        print("Setting total hours to minimum length of y in provided dictionary")
        hours = min(len(yi) for yi in y)
    if start_times is None:
        print("Setting Start times to offset by 1 hour by location")
        start_times = np.arange(n_loc)
    # Set up batches
    loc_batch, t_batch =  batch_setup(loc_ids, batch_size), batch_setup(start_times, batch_size)
    if verbose:
        print(f"Location ID Batches: {loc_batch}")
        print(f"Start Times for Batches: {t_batch}")

    # Loop over batches and construct with staircase_2
    Xs = []
    ys = []
    for i in range(0, len(loc_batch)):
        locs_i = loc_batch[i]
        ts = t_batch[i]
        for j in range(0, len(locs_i)):
            t0 = ts[j]
            tend = t0 + hours
            # Create RNNData Dict
            # Subset data to given location and time from t0 to t0+hours
            X_temp = X[j][t0:tend,:]
            y_temp = y[j][t0:tend].reshape(-1,1)

            # Format sequences
            Xi, yi = staircase_2(
                X_temp, 
                y_temp, 
                timesteps = timesteps, 
                batch_size = 1,  # note: using 1 here to format sequences for a single location, not same as target batch size for training data
                verbose=False)
        
            Xs.append(Xi)
            ys.append(yi)    

    # Drop incomplete batches
    lens = [yi.shape[0] for yi in ys]
    n_seqs = min(lens)
    if verbose:
        print(f"Minimum number of sequences by location: {n_seqs}")
        print(f"Applying minimum length to other arrays.")
    Xs = [Xi[:n_seqs] for Xi in Xs]
    ys = [yi[:n_seqs] for yi in ys]

    # Interlace arrays to match stateful structure
    n_features = Xi.shape[2]
    XXs = []
    yys = []
    for i in range(0, len(loc_batch)):
        locs_i = loc_batch[i]
        XXi = np.empty((Xs[0].shape[0]*batch_size, 5, n_features))
        yyi = np.empty((Xs[0].shape[0]*batch_size, 1))
        for j in range(0, len(locs_i)):
            XXi[j::(batch_size)] =  Xs[locs_i[j]]
            yyi[j::(batch_size)] =  ys[locs_i[j]]
        XXs.append(XXi)
        yys.append(yyi)
    yy = np.concatenate(yys, axis=0)
    XX = np.concatenate(XXs, axis=0)

    if verbose:
        print(f"Spatially Formatted X Shape: {XX.shape}")
        print(f"Spatially Formatted X Shape: {yy.shape}")
    
    
    return XX, yy, n_seqs

#***********************************************************************************************
### RNN Class Functionality

class RNNParams(dict):
    """
    A custom dictionary class for handling RNN parameters. Automatically calculates certain params based on others. Overwrites the update method to protect from incompatible parameter choices. Inherits from dict
    """    
    def __init__(self, input_dict):
        """
        Initializes the RNNParams instance and runs checks and shape calculations.

        Parameters:
        -----------
        input_dict : dict,
            A dictionary containing RNN parameters.
        """
        super().__init__(input_dict)
        # Automatically run checks on initialization
        self.run_checks()           
        # Automatically calculate shapes on initialization
        self.calc_param_shapes()        
    def run_checks(self, verbose=True):
        """
        Validates that required keys exist and are of the correct type.

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """
        print("Checking params...")
        # Keys must exist and be integers
        int_keys = [
            'batch_size', 'timesteps', 'rnn_layers', 
            'rnn_units', 'dense_layers', 'dense_units', 'epochs'
        ]
        
        for key in int_keys:
            assert key in self, f"Missing required key: {key}"
            assert isinstance(self[key], int), f"Key '{key}' must be an integer"      

        # Keys must exist and be lists
        list_keys = ['activation', 'features_list', 'dropout']
        for key in list_keys:
            assert key in self, f"Missing required key: {key}"
            assert isinstance(self[key], list), f"Key '{key}' must be a list" 

        # Keys must exist and be floats
        float_keys = ['learning_rate', 'train_frac', 'val_frac']
        for key in float_keys:
            assert key in self, f"Missing required key: {key}"
            assert isinstance(self[key], float), f"Key '{key}' must be a float"  

        print("Input dictionary passed all checks.")
    def calc_param_shapes(self, verbose=True):
        """
        Calculates and updates the shapes of certain parameters based on input data.

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """
        if verbose:
            print("Calculating shape params based on features list, timesteps, and batch size")
            print(f"Input Feature List: {self['features_list']}")
            print(f"Input Timesteps: {self['timesteps']}")
            print(f"Input Batch Size: {self['batch_size']}")
            
        n_features = len(self['features_list'])
        batch_shape = (self["batch_size"], self["timesteps"], n_features)
        if verbose:
            print("Calculated params:")
            print(f"Number of features: {n_features}")
            print(f"Batch Shape: {batch_shape}")
            
        # Update the dictionary
        super().update({
            'n_features': n_features,
            'batch_shape': batch_shape
        })
        if verbose:
            print(self)   
            
    def update(self, *args, verbose=True, **kwargs):
        """
        Updates the dictionary, with restrictions on certain keys, and recalculates shapes if necessary.

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """
        # Prevent updating n_features and batch_shape
        restricted_keys = {'n_features', 'batch_shape'}
        keys_to_check = {'features_list', 'timesteps', 'batch_size'}
        
        # Check for restricted keys in args
        if args:
            if isinstance(args[0], dict):
                if restricted_keys & args[0].keys():
                    raise KeyError(f"Cannot directly update keys: {restricted_keys & args[0].keys()}, \n Instead update one of: {keys_to_check}")
            elif isinstance(args[0], (tuple, list)) and all(isinstance(i, tuple) and len(i) == 2 for i in args[0]):
                if restricted_keys & {k for k, v in args[0]}:
                    raise KeyError(f"Cannot directly update keys: {restricted_keys & {k for k, v in args[0]}}, \n Instead update one of: {keys_to_check}")

        # Check for restricted keys in kwargs
        if restricted_keys & kwargs.keys():
            raise KeyError(f"Cannot update restricted keys: {restricted_keys & kwargs.keys()}")

        
        # Track if specific keys are updated
        keys_updated = set()

        # Update using the standard dict update method
        if args:
            if isinstance(args[0], dict):
                keys_updated.update(args[0].keys() & keys_to_check)
            elif isinstance(args[0], (tuple, list)) and all(isinstance(i, tuple) and len(i) == 2 for i in args[0]):
                keys_updated.update(k for k, v in args[0] if k in keys_to_check)
        
        if kwargs:
            keys_updated.update(kwargs.keys() & keys_to_check)

        # Call the parent update method
        super().update(*args, **kwargs)

        # Recalculate shapes if necessary
        if keys_updated:
            self.calc_param_shapes(verbose=verbose)


## Class for handling input data
class RNNData(dict):
    """
    A custom dictionary class for managing RNN data, with validation, scaling, and train-test splitting functionality.
    """    
    required_keys = {"loc", "time", "X", "y", "features_list"}  
    def __init__(self, input_dict, scaler=None, features_list=None):
        """
        Initializes the RNNData instance, performs checks, and prepares data.

        Parameters:
        -----------
        input_dict : dict
            A dictionary containing the initial data.
        scaler : str, optional
            The name of the scaler to be used (e.g., 'minmax', 'standard'). Default is None.
        features_list : list, optional
            A subset of features to be used. Default is None which means all features.
        """

        # Copy to avoid changing external input
        input_data = input_dict.copy()
        # Initialize inherited dict class
        super().__init__(input_data)
        
        # Check if input data is one timeseries dataset or multiple
        if type(self.loc['STID']) == str:
            self.spatial = False
            print("Input data is single timeseries.")
        elif type(self.loc['STID']) == list:
            self.spatial = True
            print("Input data from multiple timeseries.")
        else:
            raise KeyError(f"Input locations not list or single string")
        
        # Set up Data Scaling
        self.scaler = None
        if scaler is not None:
            self.set_scaler(scaler)
        
        # Rename and define other stuff.
        if self.spatial:
            self['hours'] = min(arr.shape[0] for arr in self.y)
        else:
            self['hours'] = len(self['y'])
        
        self['all_features_list'] = self.pop('features_list')
        if features_list is None:
            print("Using all input features.")
            self.features_list = self.all_features_list
        else:
            self.features_list = features_list
        # self.run_checks()
        self.__dict__.update(self)

    # TODO: Fix checks for multilocation
    def run_checks(self, verbose=True):
        """
        Validates that required keys are present and checks the integrity of data shapes.

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """        
        missing_keys = self.required_keys - self.keys()
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
        # # Check y 1-d
        # y_shape = np.shape(self.y)
        # if not (len(y_shape) == 1 or (len(y_shape) == 2 and y_shape[1] == 1)):
        #     raise ValueError(f"'y' must be one-dimensional, with shape (N,) or (N, 1). Current shape is {y_shape}.")
        
        # # Check if 'hours' is provided and matches len(y)
        # if 'hours' in self:
        #     if self.hours != len(self.y):
        #         raise ValueError(f"Provided 'hours' value {self.hours} does not match the length of 'y', which is {len(self.y)}.")
        # Check desired subset of features is in all input features
        if not all_items_exist(self.features_list, self.all_features_list):
            raise ValueError(f"Provided 'features_list' {self.features_list} has elements not in input features.")
    def set_scaler(self, scaler):
        """
        Sets the scaler to be used for data normalization.

        Parameters:
        -----------
        scaler : str
            The name of the scaler (e.g., 'minmax', 'standard').
        """        
        recognized_scalers = ['minmax', 'standard']
        if scaler in recognized_scalers:
            print(f"Setting data scaler: {scaler}")
            self.scaler = scalers[scaler]
        else:
            raise ValueError(f"Unrecognized scaler '{scaler}'. Recognized scalers are: {recognized_scalers}.")
    def train_test_split(self, train_frac, val_frac=0.0, subset_features=True, features_list=None, split_time=True, split_space=False, verbose=True):
        """
        Splits the data into training, validation, and test sets.

        Parameters:
        -----------
        train_frac : float
            The fraction of data to be used for training.
        val_frac : float, optional
            The fraction of data to be used for validation. Default is 0.0.
        subset_features : bool, optional
            If True, subsets the data to the specified features list. Default is True.
        features_list : list, optional
            A list of features to use for subsetting. Default is None.
        split_time : bool, optional
            Whether to split the data based on time. Default is True.
        split_space : bool, optional
            Whether to split the data based on space. Default is False.
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """
        # Indicate whether multi timeseries or not
        spatial = self.spatial
        
        # Extract data to desired features
        X = self.X.copy()
        y = self.y.copy()
        if subset_features:
            if verbose and self.features_list != self.all_features_list:
                print(f"Subsetting input data to features_list: {self.features_list}")
            # Indices to subset all features with based on params features
            indices = []
            for item in self.features_list:
                if item in self.all_features_list:
                    indices.append(self.all_features_list.index(item))
                else:
                    print(f"Warning: feature name '{item}' not found in list of all features from input data")
            if spatial:
                X = [Xi[:, indices] for Xi in X]
            else:
                X = X[:, indices]
                
        # Setup train/test in time
        train_ind = int(np.floor(self.hours * train_frac)); self.train_ind = train_ind
        test_ind= int(train_ind + round(self.hours * val_frac)); self.test_ind = test_ind

        # Check for any potential issues with indices
        if test_ind > self.hours:
            print(f"Setting test index to {self.hours}")
            test_ind = self.hours
        if train_ind >= test_ind:
            raise ValueError("Train index must be less than test index.")        
        
        # Training data from 0 to train_ind
        # Validation data from train_ind to test_ind
        # Test data from test_ind to end
        if spatial:
            self.X_train = [Xi[:train_ind] for Xi in X]
            self.y_train = [yi[:train_ind].reshape(-1,1) for yi in y]
            if val_frac >0:
                self.X_val = [Xi[train_ind:test_ind] for Xi in X]
                self.y_val = [yi[train_ind:test_ind].reshape(-1,1) for yi in y]
            self.X_test = [Xi[:train_ind] for Xi in X]
            self.y_test = [yi[:train_ind].reshape(-1,1) for yi in y]
        else:
            self.X_train = X[:train_ind]
            self.y_train = y[:train_ind].reshape(-1,1) # assumes y 1-d, change this if vector output
            if val_frac >0:
                self.X_val = X[train_ind:test_ind]
                self.y_val = y[train_ind:test_ind].reshape(-1,1) # assumes y 1-d, change this if vector output                
            self.X_test = X[test_ind:]
            self.y_test = y[test_ind:].reshape(-1,1) # assumes y 1-d, change this if vector output
        


        # Print statements if verbose
        if verbose:
            print(f"Train index: 0 to {train_ind}")
            print(f"Validation index: {train_ind} to {test_ind}")
            print(f"Test index: {test_ind} to {self.hours}")

            if spatial:
                print(f"X_train[0] shape: {self.X_train[0].shape}, y_train[0] shape: {self.y_train[0].shape}")
                print(f"X_val[0] shape: {self.X_val[0].shape}, y_val[0] shape: {self.y_val[0].shape}")
                print(f"X_test[0] shape: {self.X_test[0].shape}, y_test[0] shape: {self.y_test[0].shape}")
            else:
                print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
                print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
                print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
                
    def scale_data(self, verbose=True):
        """
        Scales the training data using the set scaler.

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """        
        # Indicate whether multi timeseries or not
        spatial = self.spatial
        if self.scaler is None:
            raise ValueError("Scaler is not set. Use 'set_scaler' method to set a scaler before scaling data.")
        if not hasattr(self, "X_train"):
            raise AttributeError("No X_train within object. Run train_test_split first. This is to avoid fitting the scaler with prediction data.")
        if verbose:
            print(f"Scaling training data with scaler {self.scaler}, fitting on X_train")

        if spatial:
            # Fit scaler on row-joined training data
            self.scaler.fit(np.vstack(self.X_train))
            # Transform data using fitted scaler
            self.X_train = [self.scaler.transform(Xi) for Xi in self.X_train]
            if hasattr(self, 'X_val'):
                self.X_val = [self.scaler.transform(Xi) for Xi in self.X_val]
            self.X_test = [self.scaler.transform(Xi) for Xi in self.X_test]
        else:
            # Fit the scaler on the training data
            self.scaler.fit(self.X_train)      
            # Transform the data using the fitted scaler
            self.X_train = self.scaler.transform(self.X_train)
            if hasattr(self, 'X_val'):
                self.X_val = self.scaler.transform(self.X_val)
            self.X_test = self.scaler.transform(self.X_test)

    # NOTE: only works for non spatial
    def scale_all_X(self, verbose=True):
        """
        Scales the all data using the set scaler.

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.
        Returns:
        -------
        ndarray
            Scaled X matrix, subsetted to features_list.
        """      
        if self.spatial:
            raise ValueError("Not implemented for spatial data")
        
        if self.scaler is None:
            raise ValueError("Scaler is not set. Use 'set_scaler' method to set a scaler before scaling data.")
        if verbose:
            print(f"Scaling all X data with scaler {self.scaler}, fitted on X_train")
        # Subset features
        indices = []
        for item in self.features_list:
            if item in self.all_features_list:
                indices.append(self.all_features_list.index(item))
            else:
                print(f"Warning: feature name '{item}' not found in list of all features from input data")
        X = self.X[:, indices]
        X = self.scaler.transform(X)

        return X    

    def inverse_scale(self, return_X = 'all_hours', save_changes=False, verbose=True):
        """
        Inversely scales the data to its original form.

        Parameters:
        -----------
        return_X : str, optional
            Specifies what data to return after inverse scaling. Default is 'all_hours'.
        save_changes : bool, optional
            If True, updates the internal data with the inversely scaled values. Default is False.
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """        
        if verbose:
            print("Inverse scaling data...")
        X_train = self.scaler.inverse_transform(self.X_train)
        X_val = self.scaler.inverse_transform(self.X_val)
        X_test = self.scaler.inverse_transform(self.X_test)

        if save_changes:
            print("Inverse transformed data saved")
            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test
        else:
            if verbose:
                print("Inverse scaled, but internal data not changed.")
        if verbose:
            print(f"Attempting to return {return_X}")
        if return_X == "all_hours":
            return np.concatenate((X_train, X_val, X_test), axis=0)
        else:
            print(f"Unrecognized or unimplemented return value {return_X}")
    def batch_reshape(self, timesteps, batch_size, hours=None, verbose=False):
        """
        Restructures input data to RNN using batches and sequences.

        Parameters:
        ----------
        batch_size : int
            The size of each training batch to reshape the data.
        timesteps : int
            The number of timesteps or sequence length. Consistitutes a single sample
        timesteps : int
            Number of timesteps or sequence length used for a single sequence in RNN training. Constitutes a single sample to the model

        batch_size : int
            Number of sequences used within a batch of training

        Returns:
        -------
        None
            This method reshapes the data in place.
        Raises:
        ------
        AttributeError
            If either 'X_train' or 'y_train' attributes do not exist within the instance.

        Notes:
        ------
        The reshaping method depends on self param "spatial".
        - spatial == False: Reshapes data assuming no spatial dimensions.
        - spatial == True: Reshapes data considering spatial dimensions.
        
        """

        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise AttributeError("Both 'X_train' and 'y_train' must be set before reshaping batches.")

        # Indicator of spatial training scheme or not
        spatial = self.spatial
        
        if spatial:
            print(f"Reshaping spatial training data using batch size: {batch_size} and timesteps: {timesteps}")
            self.X_train, self.y_train, self.n_seqs = staircase_spatial(self.X_train, self.y_train, timesteps = timesteps, batch_size=batch_size, hours=hours, verbose=verbose)
            if hasattr(self, "X_val"):
                print(f"Reshaping validation data using batch size: {batch_size} and timesteps: {timesteps}")
                self.X_val, self.y_val, _ = staircase_spatial(self.X_val, self.y_val, timesteps = timesteps, batch_size=batch_size, hours=None, verbose=verbose)
        else:
            print(f"Reshaping training data using batch size: {batch_size} and timesteps: {timesteps}")
            self.X_train, self.y_train = staircase_2(self.X_train, self.y_train, timesteps = timesteps, batch_size=batch_size, verbose=verbose)
            if hasattr(self, "X_val"):
                print(f"Reshaping validation data using batch size: {batch_size} and timesteps: {timesteps}")
                self.X_val, self.y_val = staircase_2(self.X_val, self.y_val, timesteps = timesteps, batch_size=batch_size, verbose=verbose)
        
    def print_hashes(self, attrs_to_check = ['X', 'y', 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']):
        """
        Prints the hash of specified data attributes.

        Parameters:
        -----------
        attrs_to_check : list, optional
            A list of attribute names to hash and print. Default includes 'X', 'y', and split data.
        """
        for attr in attrs_to_check:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if self.spatial:
                    pass
                else:
                    print(f"Hash of {attr}: {hash_ndarray(value)}")        
    def __getattr__(self, key):
        """
        Allows attribute-style access to dictionary keys, a.k.a. enables the "." operator for get elements
        """        
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'rnn_data' object has no attribute '{key}'")

    def __setitem__(self, key, value):
        """
        Ensures dictionary and attribute updates stay in sync for required keys.
        """        
        super().__setitem__(key, value)  # Update the dictionary
        if key in self.required_keys:
            super().__setattr__(key, value)  # Ensure the attribute is updated as well

    def __setattr__(self, key, value):
        """
        Ensures dictionary keys are updated when setting attributes.
        """
        self[key] = value    


# Function to check reproduciblity hashes, environment info, and model parameters
def check_reproducibility(dict0, params, m_hash, w_hash):
    """
    Performs reproducibility checks on a model by comparing current settings and outputs with stored reproducibility information.

    Parameters:
    -----------
    dict0 : dict
        The data dictionary that should contain reproducibility information under the 'repro_info' attribute.
    params : dict
        The current model parameters to be checked against the reproducibility information.
    m_hash : str
        The hash of the current model predictions.
    w_hash : str
        The hash of the current fitted model weights.

    Returns:
    --------
    None
        The function returns None. It issues warnings if any reproducibility checks fail.

    Notes:
    ------
    - Checks are only performed if the `dict0` contains the 'repro_info' attribute.
    - Issues warnings for mismatches in model weights, predictions, Python version, TensorFlow version, and model parameters.
    - Skips checks if physics-based initialization is used (not implemented).
    """    
    if not hasattr(dict0, "repro_info"):
        warnings.warn("The provided data dictionary does not have the required 'repro_info' attribute. Not running reproduciblity checks.")
        return 
    
    repro_info = dict0.repro_info
    # Check Hashes
    if params['phys_initialize']:
        hashes = repro_info['phys_initialize']
        warnings.warn("Physics Initialization not implemented yet. Not running reproduciblity checks.")
    else:
        hashes = repro_info['rand_initialize']
        print(f"Fitted weights hash: {w_hash} \n Reproducibility weights hash: {hashes['fitted_weights_hash']}")
        print(f"Model predictions hash: {m_hash} \n Reproducibility preds hash: {hashes['preds_hash']}")
        if (w_hash != hashes['fitted_weights_hash']) or (m_hash != hashes['preds_hash']):
            if w_hash != hashes['fitted_weights_hash']:
                warnings.warn("The fitted weights hash does not match the reproducibility weights hash.")        
            if m_hash != hashes['preds_hash']:
                warnings.warn("The predictions hash does not match the reproducibility predictions hash.")
        else:
            print("***Reproducibility Checks passed - model weights and model predictions match expected.***")
            
    # Check Environment
    current_py_version = sys.version[0:6]
    current_tf_version = tf.__version__
    if current_py_version != repro_info['env_info']['py_version']:
        warnings.warn(f"Python version mismatch: Current Python version is {current_py_version}, "
                      f"expected {repro_info['env_info']['py_version']}.")
        
    if current_tf_version != repro_info['env_info']['tf_version']:
        warnings.warn(f"TensorFlow version mismatch: Current TensorFlow version is {current_tf_version}, "
                      f"expected {repro_info['env_info']['tf_version']}.")    
    
    # Check Params
    repro_params = repro_info.get('params', {})
    
    for key, repro_value in repro_params.items():
        if key in params:
            if params[key] != repro_value:
                warnings.warn(f"Parameter mismatch for '{key}': Current value is {params[key]}, "
                              f"repro value is {repro_value}.")
        else:
            warnings.warn(f"Parameter '{key}' is missing in the current params.")

    return 

class RNNModel(ABC):
    """
    Abstract base class for RNN models, providing structure for training, predicting, and running reproducibility checks.
    """
    def __init__(self, params: dict):
        """
        Initializes the RNNModel with the given parameters.

        Parameters:
        -----------
        params : dict
            A dictionary containing model parameters.
        """
        self.params = copy.deepcopy(params)
        self.params['n_features'] = len(self.params['features_list'])
        if type(self) is RNNModel:
            raise TypeError("MLModel is an abstract class and cannot be instantiated directly")
        super().__init__()

    @abstractmethod
    def _build_model_train(self):
        """Abstract method to build the training model."""
        pass

    @abstractmethod
    def _build_model_predict(self, return_sequences=True):
        """Abstract method to build the prediction model. This model copies weights from the train model but with input structure that allows for easier prediction of arbitrary length timeseries. This model is not to be used for training, or don't use with .fit calls"""
        pass

    def is_stateful(self):
        """
        Checks whether any of the layers in the internal model (self.model_train) are stateful.

        Returns:
        bool: True if at least one layer in the model is stateful, False otherwise.
        
        This method iterates over all the layers in the model and checks if any of them
        have the 'stateful' attribute set to True. This is useful for determining if 
        the model is designed to maintain state across batches during training.

        Example:
        --------
        model.is_stateful()
        """          
        for layer in self.model_train.layers:
            if hasattr(layer, 'stateful') and layer.stateful:
                return True
        return False
        
    def fit(self, X_train, y_train, plot=True, plot_title = '', 
            weights=None, callbacks=[], validation_data=None, *args, **kwargs):
        """
        Trains the model on the provided training data.

        Parameters:
        -----------
        X_train : np.ndarray
            The input matrix data for training.
        y_train : np.ndarray
            The target vector data for training.
        plot : bool, optional
            If True, plots the training history. Default is True.
        plot_title : str, optional
            The title for the training plot. Default is an empty string.
        weights : optional
            Initial weights for the model. Default is None.
        callbacks : list, optional
            A list of callback functions to use during training. Default is an empty list.
        validation_data : tuple, optional
            Validation data to use during training, expected format (X_val, y_val). Default is None.
        """        
        # verbose_fit argument is for printing out update after each epoch, which gets very long
        # These print statements at the top could be turned off with a verbose argument, but then
        # there would be a bunch of different verbose params
        verbose_fit = self.params['verbose_fit']
        verbose_weights = self.params['verbose_weights']
        if verbose_weights:
            print(f"Training simple RNN with params: {self.params}")
        # X_train, y_train = self.format_train_data(X_train, y_train)
        if validation_data is not None:
            X_val, y_val =validation_data[0], validation_data[1]
        if verbose_weights:
            print(f"Formatted X_train hash: {hash_ndarray(X_train)}")
            print(f"Formatted y_train hash: {hash_ndarray(y_train)}")
            if validation_data is not None:
                print(f"Formatted X_val hash: {hash_ndarray(X_val)}")
                print(f"Formatted y_val hash: {hash_ndarray(y_val)}")
            print(f"Initial weights before training hash: {hash_weights(self.model_train)}")
        # Setup callbacks
        if self.params["reset_states"]:
            callbacks=callbacks+[ResetStatesCallback(batch_reset = self.params['batch_reset']), TerminateOnNaN()]
        if validation_data is not None:
            print("Using early stopping callback.")
            callbacks=callbacks+[EarlyStoppingCallback(patience = self.params['early_stopping_patience'])]
        
        # Note: we overload the params here so that verbose_fit can be easily turned on/off at the .fit call 

        # Evaluate Model once to set nonzero initial state
        if self.params["batch_size"]>= X_train.shape[0]:
            self.model_train(X_train)
        if validation_data is not None:
            history = self.model_train.fit(
                X_train, y_train+self.params['centering'][1], 
                epochs=self.params['epochs'], 
                batch_size=self.params['batch_size'],
                callbacks = callbacks,
                verbose=verbose_fit,
                validation_data = (X_val, y_val),
                *args, **kwargs
            )
        else:
            history = self.model_train.fit(
                X_train, y_train+self.params['centering'][1], 
                epochs=self.params['epochs'], 
                batch_size=self.params['batch_size'],
                callbacks = callbacks,
                verbose=verbose_fit,
                *args, **kwargs
            )
        
        if plot:
            self.plot_history(history,plot_title)
        if self.params["verbose_weights"]:
            print(f"Fitted Weights Hash: {hash_weights(self.model_train)}")

        # Update Weights for Prediction Model
        w_fitted = self.model_train.get_weights()
        self.model_predict.set_weights(w_fitted)

    def predict(self, X_test):
        """
        Generates predictions on the provided test data using the internal prediction model.

        Parameters:
        -----------
        X_test : np.ndarray
            The input data for generating predictions.

        Returns:
        --------
        np.ndarray
            The predicted values.
        """        
        print("Predicting")
        X_test = self.format_pred_data(X_test)
        preds = self.model_predict.predict(X_test).flatten()
        return preds

    # DEPRECATED, USED WITHIN RNNData object now
    # def format_train_data(self, X, y, verbose=False):
    #     """
    #     Formats the training data for RNN input.

    #     Parameters:
    #     -----------
    #     X : np.ndarray
    #         The input data.
    #     y : np.ndarray
    #         The target data.
    #     verbose : bool, optional
    #         If True, prints status messages. Default is False.

    #     Returns:
    #     --------
    #     tuple
    #         Formatted input and target data.
    #     """        
    #     X, y = staircase_2(X, y, timesteps = self.params["timesteps"], batch_size=self.params["batch_size"], verbose=verbose)
    #     return X, y
    def format_pred_data(self, X):
        """
        Formats the prediction data for RNN input.

        Parameters:
        -----------
        X : np.ndarray
            The input data.

        Returns:
        --------
        np.ndarray
            The formatted input data.
        """        
        return np.reshape(X,(1, X.shape[0], self.params['n_features']))

    def plot_history(self, history, plot_title):
        """
        Plots the training history.

        Parameters:
        -----------
        history : History object
            The training history object from model fitting. Output of keras' .fit command
        plot_title : str
            The title for the plot.
        """
        plt.figure()
        plt.semilogy(history.history['loss'], label='Training loss')
        if 'val_loss' in history.history:
            plt.semilogy(history.history['val_loss'], label='Validation loss')
        plt.title(f'{plot_title} Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()

    def run_model(self, dict0, reproducibility_run=False, plot_period='all'):
        """
        Runs the RNN model, including training, prediction, and reproducibility checks.

        Parameters:
        -----------
        dict0 : dict
            The dictionary containing the input data and configuration.
        reproducibility_run : bool, optional
            If True, performs reproducibility checks after running the model. Default is False.

        Returns:
        --------
        tuple
            Model predictions and a dictionary of RMSE errors.
        """        
        verbose_fit = self.params['verbose_fit']
        verbose_weights = self.params['verbose_weights']
        # Make copy to prevent changing in place
        dict1 = copy.deepcopy(dict0)
        if verbose_weights:
            print("Input data hashes, NOT formatted for rnn sequence/batches yet")
            dict1.print_hashes()
        # Extract Fields
        X_train, y_train, X_test, y_test = dict1['X_train'].copy(), dict1['y_train'].copy(), dict1["X_test"].copy(), dict1['y_test'].copy()
        if 'X_val' in dict1:
            X_val, y_val = dict1['X_val'].copy(), dict1['y_val'].copy()
        else:
            X_val = None
        case_id = dict1['case']
        
        # Fit model
        if X_val is None:
            self.fit(X_train, y_train, plot_title=case_id)
        else:
            self.fit(X_train, y_train, validation_data = (X_val, y_val), plot_title=case_id)
        # Generate Predictions, 
        # run through training to get hidden state set proporly for forecast period
        X = dict1.scale_all_X()
        y = dict1.y.flatten()
        # Predict
        if verbose_weights:
            print(f"Predicting Training through Test")
            print(f"All X hash: {hash_ndarray(X)}")
        
        m = self.predict(X).flatten()
        if verbose_weights:
            print(f"Predictions Hash: {hash_ndarray(m)}")
        dict1['m']=m
        dict0['m']=m # add to outside env dictionary, should be only place this happens
        
        if reproducibility_run:
            print("Checking Reproducibility")
            check_reproducibility(dict0, self.params, hash_ndarray(m), hash_weights(self.model_predict))

        # print(dict1.keys())
        # Plot final fit and data
        dict1['y'] = y
        plot_data(dict1, title="RNN", title2=dict1['case'], plot_period=plot_period)
        
        # Calculate Errors
        err = rmse(m, y)
        train_ind = dict1["train_ind"] # index of final training set value
        test_ind = dict1["test_ind"] # index of first test set value
        
        err_train = rmse(m[:train_ind], y[:train_ind].flatten())
        err_pred = rmse(m[test_ind:], y[test_ind:].flatten())
        rmse_dict = {
            'all': err, 
            'training': err_train, 
            'prediction': err_pred
        }
        return m, rmse_dict



## Callbacks
class ResetStatesCallback(Callback):
    """
    Custom callback to reset the states of RNN layers at the end of each epoch and optionally after a specified number of batches.

    Parameters:
    -----------
    batch_reset : int, optional
        If provided, resets the states of RNN layers after every `batch_reset` batches. Default is None.
    """    
    def __init__(self, batch_reset=None, loc_batch_reset=None):
        """
        Initializes the ResetStatesCallback with an optional batch reset interval.

        Parameters:
        -----------
        batch_reset : int, optional
            The interval of batches after which to reset the states of RNN layers. Default is None.
        loc_batch_reset : int, optional
            The interval of batches after which the location changes for a given batch number, then reset the states of RNN layers. Default is None.
        """        
        super(ResetStatesCallback, self).__init__()
        self.batch_reset = batch_reset 
        self.loc_batch_reset = loc_batch_reset 
    def on_epoch_end(self, epoch, logs=None):
        """
        Resets the states of RNN layers at the end of each epoch.

        Parameters:
        -----------
        epoch : int
            The index of the current epoch.
        logs : dict, optional
            A dictionary containing metrics from the epoch. Default is None.
        """        
        # print(f"Resetting hidden state after epoch: {epoch+1}", flush=True)
        # Iterate over each layer in the model
        for layer in self.model.layers:
            # Check if the layer has a reset_states method
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
    def on_train_batch_end(self, batch, logs=None):
        """
        Resets the states of RNN layers during training after a specified number of batches, if `batch_reset` or `loc_batch_reset` are provided. The `batch_reset` is used for stability and to avoid exploding gradients at the beginning of training when a hidden state is being passed with weights that haven't learned yet. The `loc_batch_reset` is used to reset the states when a particular batch is from a new location and thus the hidden state should be passed.

        Parameters:
        -----------
        batch : int
            The index of the current batch.
        logs : dict, optional
            A dictionary containing metrics from the batch. Default is None.
        """        
        batch_reset = self.batch_reset
        loc_batch_reset = self.loc_batch_reset
        if (batch_reset is not None and batch % batch_reset == 0) or (loc_batch_reset is not None and batch % loc_batch_reset == 0):
            # print(f"Resetting states after batch {batch + 1}")
            # Iterate over each layer in the model
            for layer in self.model.layers:
                # Check if the layer has a reset_states method
                if hasattr(layer, 'reset_states'):
                    layer.reset_states()
    def on_test_batch_end(self, batch, logs=None):
        """
        Resets the states of RNN layers during validation if `loc_batch_reset` is provided to demarcate a new location and thus avoid passing a hidden state to a wrong location.

        Parameters:
        -----------
        batch : int
            The index of the current batch.
        logs : dict, optional
            A dictionary containing metrics from the batch. Default is None.
        """          
        loc_batch_reset = self.loc_batch_reset
        if (loc_batch_reset is not None and batch % loc_batch_reset == 0):
            # print(f"Resetting in test batch states after batch {batch + 1}")
            # Iterate over each layer in the model
            for layer in self.model.layers:
                # Check if the layer has a reset_states method
                if hasattr(layer, 'reset_states'):
                    layer.reset_states()        
                   
# class ResetStatesCallback(Callback):
#     """
#     Custom callback to reset the states of RNN layers at the end of each epoch and optionally after a specified number of batches.

#     Parameters:
#     -----------
#     batch_reset : int, optional
#         If provided, resets the states of RNN layers after every `batch_reset` batches. Default is None.
#     """    
#     def __init__(self, batch_reset=None, loc_batch_reset=None):
#         """
#         Initializes the ResetStatesCallback with an optional batch reset interval.

#         Parameters:
#         -----------
#         batch_reset : int, optional
#             The interval of batches after which to reset the states of RNN layers. Default is None.
#         loc_batch_reset : int, optional
#             The interval of batches after which the location changes for a given batch number, then reset the states of RNN layers. Default is None.            
#         """        
#         super(ResetStatesCallback, self).__init__()
#         self.batch_reset = batch_reset 
#         self.loc_batch_reset = loc_batch_reset 
#     def on_epoch_end(self, epoch, logs=None):
#         """
#         Resets the states of RNN layers at the end of each epoch.

#         Parameters:
#         -----------
#         epoch : int
#             The index of the current epoch.
#         logs : dict, optional
#             A dictionary containing metrics from the epoch. Default is None.
#         """        
#         # Iterate over each layer in the model
#         for layer in self.model.layers:
#             # Check if the layer has a reset_states method
#             if hasattr(layer, 'reset_states'):
#                 layer.reset_states()
#     def on_train_batch_end(self, batch, logs=None):
#         """
#         Resets the states of RNN layers after a specified number of batches, if `batch_reset` or `loc_batch_reset` are provided. The batch_reset parameter resets the state for stability, and the loc_batch_reset parameter resets when the underlying timeseries for the batch changes (typically when location changes). If None provided for either used parameter, set to inf so mod batch is never zero.

#         Parameters:
#         -----------
#         batch : int
#             The index of the current batch.
#         logs : dict, optional
#             A dictionary containing metrics from the batch. Default is None.
#         """        
#         batch_reset = self.batch_reset
#         if batch_reset is None:
#             batch_reset = np.inf
#         loc_batch_reset = self.loc_batch_reset
#         if loc_batch_reset is None:
#             loc_batch_reset = np.inf
#         if (batch % batch_reset == 0) or (batch % loc_batch_reset == 0):
#             # print(f"Resetting states after batch {batch}")
#             # Iterate over each layer in the model
#             for layer in self.model.layers:
#                 # Check if the layer has a reset_states method
#                 if hasattr(layer, 'reset_states'):
#                     layer.reset_states()


## Learning Schedules
## NOT TESTED YET
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    alpha=0.0,
    name='CosineDecay',
    # warmup_target=None,
    # warmup_steps=100
)

def EarlyStoppingCallback(patience=5):
    """
    Creates an EarlyStopping callback with the specified patience.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        EarlyStopping: Configured EarlyStopping callback.
    """
    return EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
# early_stopping = EarlyStopping(
#     monitor='val_loss',  # Metric to monitor, e.g., 'val_loss', 'val_accuracy'
#     patience=5,          # Number of epochs with no improvement after which training will be stopped
#     verbose=1,           # Print information about early stopping
#     mode='min',          # 'min' means training will stop when the quantity monitored has stopped decreasing
#     restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
# )

# with open("params.yaml") as file:
#     phys_params = yaml.safe_load(file)["physics_initializer"]

phys_params = {
    'DeltaE': [0,-1],                    # bias correction
    'T1': 0.1,                           # 1/fuel class (10)
    'fm_raise_vs_rain': 0.2              # fm increase per mm rain 
}



def get_initial_weights(model_fit,params,scale_fm=1):
    # Given a RNN architecture and hyperparameter dictionary, return array of physics-initiated weights
    # Inputs:
    # model_fit: output of create_RNN_2 with no training
    # params: (dict) dictionary of hyperparameters
    # rnn_dat: (dict) data dictionary, output of create_rnn_dat
    # Returns: numpy ndarray of weights that should be a rough solution to the moisture ODE
    DeltaE = phys_params['DeltaE']
    T1 = phys_params['T1']
    fmr = phys_params['fm_raise_vs_rain']
    centering = params['centering']  # shift activation down
    
    w0_initial={'Ed':(1.-np.exp(-T1))/2, 
                'Ew':(1.-np.exp(-T1))/2,
                'rain':fmr * scale_fm}   # wx - input feature
                                 #  wh      wb   wd    bd = bias -1
    
    w_initial=np.array([np.nan, np.exp(-0.1), DeltaE[0]/scale_fm, # layer 0
                        1.0, -centering[0] + DeltaE[1]/scale_fm])                 # layer 1
    if params['verbose_weights']:
        print('Equilibrium moisture correction bias',DeltaE[0],
              'in the hidden layer and',DeltaE[1],' in the output layer')
    
    w_name = ['wx','wh','bh','wd','bd']
                        
    w=model_fit.get_weights()
    for j in range(w[0].shape[0]):
            feature = params['features_list'][j]
            for k in range(w[0].shape[1]):
                    w[0][j][k]=w0_initial[feature]
    for i in range(1,len(w)):            # number of the weight
        for j in range(w[i].shape[0]):   # number of the inputs
            if w[i].ndim==2:
                # initialize all entries of the weight matrix to the same number
                for k in range(w[i].shape[1]):
                    w[i][j][k]=w_initial[i]/w[i].shape[0]
            elif w[i].ndim==1:
                w[i][j]=w_initial[i]
            else:
                print('weight',i,'shape',w[i].shape)
                raise ValueError("Only 1 or 2 dimensions supported")
        if params['verbose_weights']:
            print('weight',i,w_name[i],'shape',w[i].shape,'ndim',w[i].ndim,
                  'initial: sum',np.sum(w[i],axis=0),'\nentries',w[i])
    
    return w, w_name

class RNN(RNNModel):
    """
    A concrete implementation of the RNNModel abstract base class, using simple recurrent cells for hidden recurrent layers.

    Parameters:
    -----------
    params : dict
        A dictionary of model parameters.
    loss : str, optional
        The loss function to use during model training. Default is 'mean_squared_error'.
    """
    def __init__(self, params, loss='mean_squared_error'):
        """
        Initializes the RNN model by building the training and prediction models.

        Parameters:
        -----------
        params : dict or RNNParams
            A dictionary containing the model's parameters. 
        loss : str, optional
            The loss function to use during model training. Default is 'mean_squared_error'.
        """        
        super().__init__(params)
        self.model_train = self._build_model_train()
        self.model_predict = self._build_model_predict()

    def _build_model_train(self):
        """
        Builds and compiles the training model, with batch & sequence shape specifications for input.

        Returns:
        --------
        model : tf.keras.Model
            The compiled Keras model for training.
        """        
        inputs = tf.keras.Input(batch_shape=self.params['batch_shape'])
        x = inputs
        for i in range(self.params['rnn_layers']):
            return_sequences = True if i < self.params['rnn_layers'] - 1 else False
            x = SimpleRNN(
                units=self.params['rnn_units'],
                activation=self.params['activation'][0],
                dropout=self.params["dropout"][0],
                recurrent_dropout = self.params["recurrent_dropout"],
                stateful=self.params['stateful'],
                return_sequences=return_sequences)(x)
        if self.params["dropout"][1] > 0:
            x = Dropout(self.params["dropout"][1])(x)            
        for i in range(self.params['dense_layers']):
            x = Dense(self.params['dense_units'], activation=self.params['activation'][1])(x)
        # Add final output layer, must be 1 dense cell with linear activation if continuous scalar output
        x = Dense(units=1, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        # optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'], clipvalue=self.params['clipvalue'])
        # optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        if self.params["verbose_weights"]:
            print(f"Initial Weights Hash: {hash_weights(model)}")
            # print(model.get_weights())

        if self.params['phys_initialize']:
            assert self.params['scaler'] == 'reproducibility', f"Not implemented yet to do physics initialize with given data scaling {self.params['scaler']}"
            assert self.params['features_list'] == ['Ed', 'Ew', 'rain'], f"Physics initiation can only be done with features ['Ed', 'Ew', 'rain'], but given features {self.params['features_list']}"
            print("Initializing Model with Physics based weights")
            w, w_name=get_initial_weights(model, self.params)
            model.set_weights(w)
            print('initial weights hash =',hash_weights(model))
        return model
    def _build_model_predict(self, return_sequences=True):
        """
        Builds and compiles the prediction model, doesn't use batch shape nor sequence length.

        Parameters:
        -----------
        return_sequences : bool, optional
            Whether to return the full sequence of outputs. Default is True.

        Returns:
        --------
        model : tf.keras.Model
            The compiled Keras model for prediction.
        """        
        inputs = tf.keras.Input(shape=(None,self.params['n_features']))
        x = inputs
        for i in range(self.params['rnn_layers']):
            x = SimpleRNN(self.params['rnn_units'],activation=self.params['activation'][0],
                  stateful=False,return_sequences=return_sequences)(x)
        for i in range(self.params['dense_layers']):
            x = Dense(self.params['dense_units'], activation=self.params['activation'][1])(x)
        # Add final output layer, must be 1 dense cell with linear activation if continuous scalar output
        x = Dense(units=1, activation='linear')(x)        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(loss='mean_squared_error', optimizer=optimizer)  

        # Set Weights to model_train
        w_fitted = self.model_train.get_weights()
        model.set_weights(w_fitted)
        
        return model


class RNN_LSTM(RNNModel):
    """
    A concrete implementation of the RNNModel abstract base class, use LSTM cells for hidden recurrent layers.

    Parameters:
    -----------
    params : dict
        A dictionary of model parameters.
    loss : str, optional
        The loss function to use during model training. Default is 'mean_squared_error'.
    """
    def __init__(self, params, loss='mean_squared_error'):
        """
        Initializes the RNN model by building the training and prediction models.

        Parameters:
        -----------
        params : dict or RNNParams
            A dictionary containing the model's parameters. 
        loss : str, optional
            The loss function to use during model training. Default is 'mean_squared_error'.
        """           
        super().__init__(params)
        self.model_train = self._build_model_train()
        self.model_predict = self._build_model_predict()

    def _build_model_train(self):
        """
        Builds and compiles the training model, with batch & sequence shape specifications for input.

        Returns:
        --------
        model : tf.keras.Model
            The compiled Keras model for training.
        """               
        inputs = tf.keras.Input(batch_shape=self.params['batch_shape'])
        x = inputs
        for i in range(self.params['rnn_layers']):
            return_sequences = True if i < self.params['rnn_layers'] - 1 else False
            x = LSTM(
                units=self.params['rnn_units'],
                activation=self.params['activation'][0],
                dropout=self.params["dropout"][0],
                recurrent_dropout = self.params["recurrent_dropout"],
                recurrent_activation=self.params["recurrent_activation"],
                stateful=self.params['stateful'],
                return_sequences=return_sequences)(x)
        if self.params["dropout"][1] > 0:
            x = Dropout(self.params["dropout"][1])(x)            
        for i in range(self.params['dense_layers']):
            x = Dense(self.params['dense_units'], activation=self.params['activation'][1])(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        # optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'], clipvalue=self.params['clipvalue'])
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        if self.params["verbose_weights"]:
            print(f"Initial Weights Hash: {hash_weights(model)}")
        return model
    def _build_model_predict(self, return_sequences=True):
        """
        Builds and compiles the prediction model, doesn't use batch shape nor sequence length.

        Parameters:
        -----------
        return_sequences : bool, optional
            Whether to return the full sequence of outputs. Default is True.

        Returns:
        --------
        model : tf.keras.Model
            The compiled Keras model for prediction.
        """           
        inputs = tf.keras.Input(shape=(None,self.params['n_features']))
        x = inputs
        for i in range(self.params['rnn_layers']):
            x = LSTM(
                units=self.params['rnn_units'],
                activation=self.params['activation'][0],
                stateful=False,return_sequences=return_sequences)(x)
        for i in range(self.params['dense_layers']):
            x = Dense(self.params['dense_units'], activation=self.params['activation'][1])(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(loss='mean_squared_error', optimizer=optimizer)  

        # Set Weights to model_train
        w_fitted = self.model_train.get_weights()
        model.set_weights(w_fitted)
        
        return model




