# Environment
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error
import logging
from tensorflow.keras.layers import Input, SimpleRNN, Dropout, Dense
# Local modules
import reproducibility
from utils import print_dict_summary
from data_funcs import load_and_fix_data, rmse
from abc import ABC, abstractmethod
from utils import hash2
from data_funcs import rmse



def staircase(x,y,timesteps,datapoints,return_sequences=False, verbose = False):
    # x [datapoints,features]    all inputs
    # y [datapoints,outputs]
    # timesteps: split x and y into samples length timesteps, shifted by 1
    # datapoints: number of timesteps to use for training, no more than y.shape[0]
    print('staircase: shape x = ',x.shape)
    print('staircase: shape y = ',y.shape)
    print('staircase: timesteps=',timesteps)
    print('staircase: datapoints=',datapoints)
    print('staircase: return_sequences=',return_sequences)
    outputs = y.shape[1]
    features = x.shape[1]
    samples = datapoints-timesteps+1
    print('staircase: samples=',samples,'timesteps=',timesteps,'features=',features)
    x_train = np.empty([samples, timesteps, features])
    if return_sequences:
        print('returning all timesteps in a sample')
        y_train = np.empty([samples, timesteps, outputs])  # all
        for i in range(samples):
            for k in range(timesteps):
                x_train[i,k,:] = x[i+k,:]
                y_train[i,k,:] = y[i+k,:]
    else:
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
    print('staircase_2: shape x = ',x.shape)
    print('staircase_2: shape y = ',y.shape)
    print('staircase_2: timesteps=',timesteps)
    print('staircase_2: batch_size=',batch_size)
    print('staircase_2: return_sequences=',return_sequences)
    
    nx,features= x.shape
    ny,outputs = y.shape
    datapoints = min(nx,ny,trainsteps)   
    print('staircase_2: datapoints=',datapoints)
    
    # sequence j in a given batch is assumed to be the continuation of sequence j in the previous batch
    # https://www.tensorflow.org/guide/keras/working_with_rnns Cross-batch statefulness
    
    # example with timesteps=3 batch_size=3 datapoints=10
    #     batch 0: [0 1 2]   [1 2 3]   [2 3 4]  
    #     batch 1: [3 4 5]   [4 5 6]   [5 6 7] 
    #     batch 2: [6 7 8]   [7 8 9]    when runs out this is the last batch, can be shorter
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
    
    print('staircase_2: shape x_train = ',x_train.shape)
    print('staircase_2: shape y_train = ',y_train.shape)
    print('staircase_2: sequences generated',k)
    print('staircase_2: batch_size=',batch_size)
    k = (k // batch_size) * batch_size
    print('staircase_2: removing partial and empty batches at the end, keeping',k)
    x_train = x_train[:k,:,:]
    if return_sequences:
         y_train = y_train[:k,:,:]
    else:
         y_train = y_train[:k,:]
            
    print('staircase_2: shape x_train = ',x_train.shape)
    print('staircase_2: shape y_train = ',y_train.shape)

    return x_train, y_train

def create_rnn_data(dict1, params, atm_dict="HRRR", verbose=False, train_ind=None, test_ind=None, scaler=None):
    # Given fmda data and hyperparameters, return formatted dictionary to be used in RNN
    # Inputs:
    # d: (dict) fmda dictionary
    # params: (dict) hyperparameters
    # atm_dict: (str) string specifying name of subdictionary for atmospheric vars
    # train_frac: (float) fraction of data to use for training (starting from time 0)
    # val_frac: (float) fraction of data to use for validation data (starting from end of train)
    # Returns: (dict) formatted data used in RNN 
    logging.info('create_rnn_data start')
    # Copy Dictionary 
    d=dict1.copy()
    scale = params['scale']
    features_list = params["features_list"]

    # Check if reproducibility case
    if dict1['case']=="reproducibility":
        params.update({'scale':1})
        atm_dict="RAWS"
    
    # Scale Data if required
    if scale:
        scale=1
        if dict1['case']=="reproducibility":
            # Note: this was calculated from the max observed fm, Ed, Ew in a whole timeseries originally with using data from test period
            scale_fm = 17.076346687085564
            logging.info("REPRODUCIBILITY scaling moisture features: using %s", scale_fm)
            logging.info('create_rnn_data: scaling to range 0 to 1')
            d[atm_dict]['Ed'] = d[atm_dict]['Ed'] / scale_fm
            d[atm_dict]['Ew'] = d[atm_dict]['Ew'] / scale_fm
            d[atm_dict]['fm'] = d[atm_dict]['fm'] / scale_fm
    else:
        scale_fm=1.0
        scaler=None
    # Extract desired features based on params, combine into matrix
    fm = d[atm_dict]['fm']
    values = [d[atm_dict][key] for key in features_list]
    X = np.vstack(values).T
    # Extract response vector 
    y = np.reshape(fm,[fm.shape[0],1])
    # Calculate total observed hours
    hours = X.shape[0]
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
        'features': len(features_list),
        'scaler':scaler,
        'train_ind':train_ind,
        'test_ind':test_ind,
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
    params.update({
            'features': features,
            'batch_shape': (params["batch_size"],params["timesteps"],features),
            'pred_input_shape': (hours, features)
        })

    
    logging.info('create_rnn_data_2 done')
    return rnn_dat

class ResetStatesCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()
    


class RNNModel(ABC):
    def __init__(self, params: dict):
        self.params = params
        if type(self) is RNNModel:
            raise TypeError("MLModel is an abstract class and cannot be instantiated directly")
        super().__init__()

    @abstractmethod
    def fit(self, X_train, y_train, weights=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def run_model(self, dict1):
        # Extract Fields
        X_train, y_train, X_test, y_test = dict1['X_train'], dict1['y_train'], dict1["X_test"], dict1['y_test']
        case_id = dict1['case']
        # Fit model
        self.fit(X_train, y_train)
        # Generate Predictions, 
        # run through training to get hidden state set proporly for forecast period
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test)).flatten()
        print(f"Predicting Training through Test \n features hash: {hash2(X)} \n response hash: {hash2(y)} ")
        m = self.predict(X).flatten()
        # Calculate Errors
        err = rmse(m, y)
        h2 = X_train.shape[0] # index of final training set value
        err_train = rmse(m[:h2], y_train.flatten())
        err_pred = rmse(m[h2:], y_test.flatten())
        rmse_dict = {
            'all': err, 
            'training': err_train, 
            'prediction': err_pred
        }
        return rmse_dict
        
class ResetStatesCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()

class RNN(RNNModel):
    def __init__(self, params, loss='mean_squared_error'):
        super().__init__(params)
        self.model_fit = self._build_model_fit()
        self.model_predict = self._build_model_predict()

    def _build_model_fit(self, return_sequences=False):
        inputs = tf.keras.Input(batch_shape=self.params['batch_shape'])
        x = inputs
        for i in range(self.params['rnn_layers']):
            x = SimpleRNN(
                self.params['rnn_units'],
                activation=self.params['activation'][0],
                dropout=self.params["dropout"][0],
                stateful=self.params['stateful'],
                return_sequences=return_sequences)(x)
        if self.params["dropout"][1] > 0:
            x = Dropout(self.params["dropout"][1])(x)            
        for i in range(self.params['dense_layers']):
            x = Dense(self.params['dense_units'], activation=self.params['activation'][1])(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        if self.params["verbose_weights"]:
            print(f"Initial Weights Hash: {hash2(model.get_weights())}")
        
        return model
    def _build_model_predict(self, return_sequences=True):
        
        inputs = tf.keras.Input(shape=self.params['pred_input_shape'])
        x = inputs
        for i in range(self.params['rnn_layers']):
            x = SimpleRNN(self.params['rnn_units'],activation=self.params['activation'][0],
                  stateful=False,return_sequences=return_sequences)(x)
        for i in range(self.params['dense_layers']):
            x = Dense(self.params['dense_units'], activation=self.params['activation'][1])(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(loss='mean_squared_error', optimizer=optimizer)  

        # Set Weights to model_fit
        w_fitted = self.model_fit.get_weights()
        model.set_weights(w_fitted)
        
        return model
    def format_train_data(self, X, y, verbose=False):
        X, y = staircase_2(X, y, timesteps = self.params["timesteps"], batch_size=self.params["batch_size"], verbose=verbose)
        return X, y
    def format_pred_data(self, X):
        return np.reshape(X,(1, X.shape[0], self.params['features']))
    def fit(self, X_train, y_train, plot=True, plot_title = '', 
            weights=None, callbacks=[], verbose_fit=None, validation_data=None, *args, **kwargs):
        # verbose_fit argument is for printing out update after each epoch, which gets very long
        # These print statements at the top could be turned off with a verbose argument, but then
        # there would be a bunch of different verbose params
        print(f"Training simple RNN with params: {self.params}")
        X_train, y_train = self.format_train_data(X_train, y_train)
        if validation_data is not None:
            X_val, y_val = self.format_train_data(validation_data[0], validation_data[1])
        print(f"X_train hash: {hash2(X_train)}")
        print(f"y_train hash: {hash2(y_train)}")
        print(f"Initial weights before training hash: {hash2(self.model_fit.get_weights())}")
        # Setup callbacks
        if self.params["reset_states"]:
            callbacks=callbacks+[ResetStatesCallback()]
        
        # Note: we overload the params here so that verbose_fit can be easily turned on/off at the .fit call 
        if verbose_fit is None:
            verbose_fit = self.params['verbose_fit']
        # Evaluate Model once to set nonzero initial state
        if self.params["batch_size"]>= X_train.shape[0]:
            self.model_fit(X_train)
        if validation_data is not None:
            history = self.model_fit.fit(
                X_train, y_train+self.params['centering'][1], 
                epochs=self.params['epochs'], 
                batch_size=self.params['batch_size'],
                callbacks = callbacks,
                verbose=verbose_fit,
                validation_data = (X_val, y_val),
                *args, **kwargs
            )
        else:
            history = self.model_fit.fit(
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
            print(f"Fitted Weights Hash: {hash2(self.model_fit.get_weights())}")

        # Update Weights for Prediction Model
        w_fitted = self.model_fit.get_weights()
        self.model_predict.set_weights(w_fitted)
    def predict(self, X_test):
        print("Predicting with simple RNN")
        X_test = self.format_pred_data(X_test)
        preds = self.model_predict.predict(X_test).flatten()
        return preds


    def plot_history(self, history, plot_title):
        plt.semilogy(history.history['loss'], label='Training loss')
        if 'val_loss' in history.history:
            plt.semilogy(history.history['val_loss'], label='Validation loss')
        plt.title(f'{plot_title} Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()



