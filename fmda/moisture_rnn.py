# v2 training and prediction class infrastructure

# Environment
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import mean_squared_error
import logging
from tensorflow.keras.layers import LSTM, SimpleRNN, Input, Dropout, Dense
# Local modules
import reproducibility
from utils import print_dict_summary
from data_funcs import load_and_fix_data, rmse
from abc import ABC, abstractmethod
from utils import hash2
from data_funcs import rmse, plot_data, compare_dicts
import copy
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

# def scale_transform(X, method='minmax'):
#     # Function to scale data in place
#     # Inputs: 
#     # X: (ndarray) data to be scaled
#     # method: (str) one of keys in scalers dictionary above
#     scaler = scalers[method]
#     scaler.fit(X)
#     # Modify X in-place
#     X[:] = scaler.transform(X)

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

#***********************************************************************************************
### RNN Class Functionality

# Custom class for parameters dictionary. Inherits from dict, but adds checks and safeguards
class RNNParams(dict):
    def __init__(self, input_dict=None):
        super().__init__(input_dict)
        # Automatically run checks on initialization
        self.run_checks()           
        # Automatically calculate shapes on initialization
        self.calc_param_shapes()        
    def run_checks(self, verbose=True):
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
    
repro_hashes = {
    'phys_initialize': {
        'fitted_weight_hash' : 4.2030588308041834e+19,
        'predictions_hash' :3.59976005554199219
    },
    'rand_initialize': {
        'fitted_weight_hash' : 4.4965532557938975e+19,
        'predictions_hash' : 3.71594738960266113
    },
    'params':{'id':0,
        'purpose':'reproducibility',
        'batch_size':32,
        'training':5,
        'cases':['case11'],
        'scale':1,        # every feature in [0, scale]
        'rain_do':True,
        'verbose':False,
        'timesteps':5,
        'activation':['linear','linear'],
        'hidden_units':20,  
        'dense_units':1,    # do not change
        'dense_layers':1,   # do not change
        'centering':[0.0,0.0],  # should be activation at 0
        'DeltaE':[0,-1],    # bias correction
        'synthetic':False,  # run also synthetic cases
        'T1': 0.1,          # 1/fuel class (10)
        'fm_raise_vs_rain': 0.2,         # fm increase per mm rain 
        'train_frac':0.5,  # time fraction to spend on training
        'epochs':200,
        'verbose_fit':0,
        'verbose_weights':False,
        'initialize': True,
        'learning_rate': 0.001 # default learning rate
        }
}


class RNNModel(ABC):
    def __init__(self, params: dict):
        self.params = params
        self.params['n_features'] = len(self.params['features_list'])
        if type(self) is RNNModel:
            raise TypeError("MLModel is an abstract class and cannot be instantiated directly")
        super().__init__()

    @abstractmethod
    def _build_model_train(self):
        pass

    @abstractmethod
    def _build_model_predict(self, return_sequences=True):
        pass
    
    def fit(self, X_train, y_train, plot=True, plot_title = '', 
            weights=None, callbacks=[], verbose_fit=None, validation_data=None, *args, **kwargs):
        # verbose_fit argument is for printing out update after each epoch, which gets very long
        # These print statements at the top could be turned off with a verbose argument, but then
        # there would be a bunch of different verbose params
        print(f"Training simple RNN with params: {self.params}")
        X_train, y_train = self.format_train_data(X_train, y_train)
        print(f"X_train hash: {hash2(X_train)}")
        print(f"y_train hash: {hash2(y_train)}")
        if validation_data is not None:
            X_val, y_val = self.format_train_data(validation_data[0], validation_data[1])
            print(f"X_val hash: {hash2(X_val)}")
            print(f"y_val hash: {hash2(y_val)}")
        
        print(f"Initial weights before training hash: {hash2(self.model_train.get_weights())}")
        # Setup callbacks
        if self.params["reset_states"]:
            callbacks=callbacks+[ResetStatesCallback()]
        # if validation_data is not None:
        #     callbacks=callbacks+[early_stopping]
        
        # Note: we overload the params here so that verbose_fit can be easily turned on/off at the .fit call 
        if verbose_fit is None:
            verbose_fit = self.params['verbose_fit']
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
            print(f"Fitted Weights Hash: {hash2(self.model_train.get_weights())}")

        # Update Weights for Prediction Model
        w_fitted = self.model_train.get_weights()
        self.model_predict.set_weights(w_fitted)

    def predict(self, X_test):
        print("Predicting")
        X_test = self.format_pred_data(X_test)
        preds = self.model_predict.predict(X_test).flatten()
        return preds

    def format_train_data(self, X, y, verbose=False):
        X, y = staircase_2(X, y, timesteps = self.params["timesteps"], batch_size=self.params["batch_size"], verbose=verbose)
        return X, y
    def format_pred_data(self, X):
        return np.reshape(X,(1, X.shape[0], self.params['n_features']))

    def plot_history(self, history, plot_title):
        plt.figure()
        plt.semilogy(history.history['loss'], label='Training loss')
        if 'val_loss' in history.history:
            plt.semilogy(history.history['val_loss'], label='Validation loss')
        plt.title(f'{plot_title} Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()

    def run_model(self, dict0):
        # Make copy to prevent changing in place
        dict1 = copy.deepcopy(dict0)
        # Extract Fields
        scale_fm = dict1['scale_fm']
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
        if X_val is None:
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test)).flatten()
        else:
            X = np.concatenate((X_train, X_val, X_test))
            y = np.concatenate((y_train, y_val, y_test)).flatten()
        # Predict
        print(f"Predicting Training through Test \n features hash: {hash2(X)} \n response hash: {hash2(y)} ")
        
        m = self.predict(X).flatten()
        dict1['m']=m
        dict0['m']=m # add to outside env dictionary, should be only place this happens
        if self.params['scale']:
            print(f"Rescaling data using {self.params['scaler']}")
            if self.params['scaler'] == "reproducibility":
                m  *= scale_fm
                y  *= scale_fm
                y_train *= scale_fm
                y_test *= scale_fm
        # Check Reproducibility, TODO: old dict calls it hidden_units not rnn_units, so this doens't check that
        if (case_id == "reproducibility") and compare_dicts(self.params, repro_hashes['params'], ['epochs', 'batch_size', 'scale', 'activation', 'learning_rate']):
            print("Checking Reproducibility")
            checkm = m[350]
            hv = hash2(self.model_predict.get_weights())
            if self.params['phys_initialize']:
                hv5 = repro_hashes['phys_initialize']['fitted_weight_hash']
                mv = repro_hashes['phys_initialize']['predictions_hash']
            else:
                hv5 = repro_hashes['rand_initialize']['fitted_weight_hash']
                mv = repro_hashes['rand_initialize']['predictions_hash']           
            
            print(f"Fitted weights hash (check 5): {hv} \n Reproducibility weights hash: {hv5} \n Error: {hv5-hv}")
            print(f"Model predictions hash: {checkm} \n Reproducibility preds hash: {mv} \n Error: {mv-checkm}")

        # print(dict1.keys())
        # Plot final fit and data
        dict1['y'] = y
        plot_data(dict1, title="RNN", title2=dict1['case'])
        
        # Calculate Errors
        err = rmse(m, y)
        train_ind = dict1["train_ind"] # index of final training set value
        test_ind = dict1["test_ind"] # index of first test set value
        err_train = rmse(m[:train_ind], y_train.flatten())
        err_pred = rmse(m[test_ind:], y_test.flatten())
        rmse_dict = {
            'all': err, 
            'training': err_train, 
            'prediction': err_pred
        }
        return m, rmse_dict



## Callbacks

class ResetStatesCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Iterate over each layer in the model
        for layer in self.model.layers:
            # Check if the layer has a reset_states method
            if hasattr(layer, 'reset_states'):
                layer.reset_states()

    

## Learning Schedules
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    alpha=0.0,
    name='CosineDecay',
    # warmup_target=None,
    # warmup_steps=100
)


early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor, e.g., 'val_loss', 'val_accuracy'
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # Print information about early stopping
    mode='min',          # 'min' means training will stop when the quantity monitored has stopped decreasing
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# with open("params.yaml") as file:
#     phys_params = yaml.safe_load(file)["physics_initializer"]

phys_params = {
    'DeltaE': [0,-1],                    # bias correction
    'T1': 0.1,                           # 1/fuel class (10)
    'fm_raise_vs_rain': 0.2              # fm increase per mm rain 
}



def get_initial_weights(model_fit,params,scale_fm):
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
    def __init__(self, params, loss='mean_squared_error'):
        super().__init__(params)
        self.model_train = self._build_model_train()
        self.model_predict = self._build_model_predict()

    def _build_model_train(self):
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
        # optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        if self.params["verbose_weights"]:
            print(f"Initial Weights Hash: {hash2(model.get_weights())}")
            # print(model.get_weights())

        if self.params['phys_initialize']:
            assert self.params['scaler'] == 'reproducibility', f"Not implemented yet to do physics initialize with given data scaling {self.params['scaler']}"
            assert self.params['features_list'] == ['Ed', 'Ew', 'rain'], f"Physics initiation can only be done with features ['Ed', 'Ew', 'rain'], but given features {self.params['features_list']}"
            print("Initializing Model with Physics based weights")
            w, w_name=get_initial_weights(model, self.params, scale_fm = scale_fm)
            model.set_weights(w)
            print('initial weights hash =',hash2(model.get_weights()))
        return model
    def _build_model_predict(self, return_sequences=True):
        
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
    def __init__(self, params, loss='mean_squared_error'):
        super().__init__(params)
        self.model_train = self._build_model_train()
        self.model_predict = self._build_model_predict()

    def _build_model_train(self):
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'], clipvalue=self.params['clipvalue'])
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        if self.params["verbose_weights"]:
            print(f"Initial Weights Hash: {hash2(model.get_weights())}")
        return model
    def _build_model_predict(self, return_sequences=True):
        
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




