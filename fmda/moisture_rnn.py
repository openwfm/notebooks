import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
# from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import tensorflow as tf
import pandas as pd
from utils import vprint, hash2, get_item, print_dict_summary, print_first
import reproducibility
from data_funcs import check_data, rmse_data, plot_data
import moisture_models as mod
import sys
import logging
from utils import print_dict_summary, print_first, str2time, check_increment, time_intp
import pickle
import os.path as osp


def staircase(x,y,timesteps,datapoints,return_sequences=False, verbose = False):
    # x [datapoints,features]    all inputs
    # y [datapoints,outputs]
    # timesteps: split x and y into samples length timesteps, shifted by 1
    # datapoints: number of timesteps to use for training, no more than y.shape[0]
    vprint('staircase: shape x = ',x.shape)
    vprint('staircase: shape y = ',y.shape)
    vprint('staircase: timesteps=',timesteps)
    vprint('staircase: datapoints=',datapoints)
    vprint('staircase: return_sequences=',return_sequences)
    outputs = y.shape[1]
    features = x.shape[1]
    samples = datapoints-timesteps+1
    vprint('staircase: samples=',samples,'timesteps=',timesteps,'features=',features)
    x_train = np.empty([samples, timesteps, features])
    if return_sequences:
        vprint('returning all timesteps in a sample')
        y_train = np.empty([samples, timesteps, outputs])  # all
        for i in range(samples):
            for k in range(timesteps):
                x_train[i,k,:] = x[i+k,:]
                y_train[i,k,:] = y[i+k,:]
    else:
        vprint('returning only the last timestep in a sample')
        y_train = np.empty([samples, outputs])
        for i in range(samples):
            for k in range(timesteps):
                x_train[i,k,:] = x[i+k,:]
            y_train[i,:] = y[i+timesteps-1,:]

    return x_train, y_train

def staircase_2(x,y,timesteps,batch_size=None,trainsteps=np.inf,return_sequences=False, verbose = True):
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
            vprint('sequence',k,'batch',i,'sample',j,'data',begin,'to',next-1)
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

def create_RNN_2(hidden_units, dense_units, activation, stateful=False,
                 batch_shape=None, input_shape=None, dense_layers=1,
                 rnn_layers=1,return_sequences=False,
                 initial_state=None, verbose = True):
    if verbose:
        print("Function: moisture_rnn.create_RNN_2")
        print("Arguments:")
        arg_dict = locals().copy()
        for arg in arg_dict:
            if arg != "self":
                print(f"  {arg} = {arg_dict[arg]}")
    if stateful:
        inputs = tf.keras.Input(batch_shape=batch_shape)
    else:
        inputs = tf.keras.Input(shape=input_shape)
    # https://stackoverflow.com/questions/43448029/how-can-i-print-the-values-of-keras-tensors
    x = inputs
    for i in range(rnn_layers):
        x = tf.keras.layers.SimpleRNN(hidden_units,activation=activation[0],
              stateful=stateful,return_sequences=return_sequences)(x)
    for i in range(dense_layers):
        x = tf.keras.layers.Dense(dense_units, activation=activation[1])(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_rnn_data_1(dat, params, hours=None, h2=None):
    # Given fmda data and hyperparameters, return formatted dictionary to be used in RNN
    # Inputs:
    # dat: (dict) fmda dictionary
    # params: (dict) hyperparameters
    # hours: (int) optional parameter to set total length of train+predict
    # h2: (int) optional parameter to set as length of training period (h2 = total - hours)
    # Returns: (dict) formatted datat used in RNN 
    logging.info('create_rnn_data_1 start')
    
    timesteps = params['timesteps']
    scale = params['scale']
    rain_do = params['rain_do']
    verbose = params['verbose']
    
    logging.info('create_rnn_data_1: hours=%s h2=%s',hours,h2)
    # extract inputs the windown of interest
    Ew = dat['Ew']
    Ed = dat['Ed']
    rain = dat['rain']
    fm = dat['fm']
    # temp = dat['temp']
    
    # Average Equilibrium
    # E = (Ed + Ew)/2         # why?

    # Scale Data if required
    if scale:
        logging.info('create_rnn_data_1: scaling to range 0 to %s',scale)
        scale_fm=max(max(Ew),max(Ed),max(fm))/scale
        scale_rain=max(max(rain),0.01)/scale
        Ed = Ed/scale_fm
        Ew = Ew/scale_fm
        fm = fm/scale_fm
        rain = rain/scale_rain
    else:
        scale_fm=1.0
        scale_rain=1.0
     
    if params['verbose_weights']:
        logging.info('scale_fm=%s scale_rain=%s',scale_fm,scale_rain)
    
    # transform as 2D, (timesteps, features) and (timesteps, outputs)
    
    if rain_do:
        X = np.vstack((Ed, Ew, rain)).T
        features_list = ['Ed', 'Ew', 'rain']
    else:
        X = np.vstack((Ed, Ew)).T        
        features_list = ['Ed', 'Ew']
    Y = np.reshape(fm,[fm.shape[0],1])
    
    logging.info('feature matrix X shape %s',np.shape(X))
    logging.info('target  matrix Y shape %s',np.shape(Y))
    logging.info('features_list: %s',features_list)
    
    if hours is None:
        hours = dat['hours']
    if h2 is None:
        h2 = dat['h2'] 
    
    rnn_dat={
        'case':dat['case'],
        'hours':hours,
        'h2':h2,
        'rain_do':rain_do,
        'features_list':features_list,
        'scale':scale,
        'scale_fm':scale_fm,
        'scale_rain':scale_rain,
        'X':X,
        'Y':Y
    }
    
    return rnn_dat
    
def create_rnn_data_2(rnn_dat, params):
    
    logging.info('create_rnn_data_2 start')
    
    timesteps = params['timesteps']
    scale = params['scale']
    verbose = params['verbose']
    batch_size = params['batch_size']
    X = rnn_dat['X']
    Y = rnn_dat['Y']
    h2= rnn_dat['h2']
           
    logging.info('batch_size=%s',batch_size)
    if batch_size is None or batch_size is np.inf:
        x_train, y_train = staircase(X,Y,timesteps=timesteps,datapoints=h2,
                                 return_sequences=False, verbose = verbose)
        batches = None
    else:
        x_train, y_train = staircase_2(X,Y,timesteps,batch_size,trainsteps=h2,
                                 return_sequences=False, verbose = verbose)
    
    logging.info('x_train shape=%s',x_train.shape)
    logging.info('y_train shape=%s',y_train.shape)
    
    samples, timesteps, features = x_train.shape
    
    h0 = tf.convert_to_tensor(X[:samples],dtype=tf.float32)
    
    # Set up return dictionary
    
    rnn_dat.update({
        'x_train': x_train,
        'y_train': y_train,
        'samples': samples,
        'timesteps': timesteps,
        'features':features,
    })
    
    logging.info('create_rnn_data_2 done')
    
    # return rnn_dat


from tensorflow.keras.callbacks import Callback

class ResetStatesCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()


def train_rnn(rnn_dat, params,hours, fit=True, callbacks=[]):

    # operates on matrix input/output, identity of features not visible here
    
    logging.info("train_rnn start, hours=%s fit=%s",hours,fit)
    
    verbose = params['verbose']
    case = get_item(rnn_dat,'case')
    
    if hours is None:
        hours = get_item(rnn_dat,'hours')
    
    samples = get_item(rnn_dat,'samples')
    features = get_item(rnn_dat,'features')
    timesteps = get_item(rnn_dat,'timesteps')
    centering = get_item(params,'centering')
    training = get_item(params,'training')
    batch_size = get_item(params,'batch_size')
    initialize=get_item(params,'initialize',default=1)

    single_batch = batch_size is None or batch_size is np.inf
    if single_batch:
        batch_size=samples
        logging.info('replacing batch_size by %s',batch_size)

    epochs=get_item(params,'epochs')
    
    model_fit=create_RNN_2(hidden_units=params['hidden_units'], 
                        dense_units=params['dense_units'], 
                        batch_shape=(batch_size,timesteps,features),
                        stateful=True,
                        return_sequences=False,
                        # initial_state=h0,
                        activation=params['activation'],
                        dense_layers=params['dense_layers'],
                        verbose = verbose)
    if verbose: print(model_fit.summary())
        
    # X = rnn_dat['X']
    model_predict=create_RNN_2(hidden_units=params['hidden_units'], 
                        dense_units=params['dense_units'],  
                        input_shape=(hours,features),stateful = False,
                        return_sequences=True,
                        activation=params['activation'],
                        dense_layers=params['dense_layers'],
                        verbose = verbose)
    
    if verbose: print(model_predict.summary())
    
    x_train = rnn_dat['x_train']
    y_train = rnn_dat['y_train']
    
    print('x_train shape =',x_train.shape)
    print('y_train shape =',y_train.shape)
    print('x_train hash =',hash2(x_train))
    print('y_train hash =',hash2(y_train))

    if single_batch:
         model_fit(x_train) ## evalue the model once to set nonzero initial state for compatibility
    
    if initialize:
        print('initializing weights')
        w, w_name=get_initial_weights(model_fit, params, rnn_dat)
        print('initial weights hash =',hash2(w))
        model_fit.set_weights(w)
    else:
        print('NOT initializing weights')
        w = model_fit.get_weights()

    if fit:
        history = None
        # model_fit.set_weights(w)
        if single_batch:
            print("~"*50)
            print(f'single batch train with \n epochs={epochs} \n batch_size={samples} \n callbacks={callbacks} \n batch_shape = {model_fit.input_shape}')
            print(f"x_train hash: {hash2(x_train)}")
            print(f"y_train hash: {hash2(y_train)}")
            print(f"Initial Weights hash: {hash2(model_fit.get_weights())}")
            history = model_fit.fit(x_train, 
                      y_train + centering[1] , 
                      epochs=epochs,
                      batch_size=samples,
                      callbacks = callbacks,
                      verbose=get_item(params,'verbose_fit')
                      )
        else:
            if training == 1:
                print('training by batches')
                batches = samples // batch_size
                for i in range(epochs):
                    for j in range(batches):
                        batch = slice(j*batch_size, (j+1)*batch_size)
                        model_fit.fit(x_train[batch], 
                              y_train[batch] + centering[1] , 
                              epochs=1,
                              callbacks = callbacks,
                              verbose=params['verbose_fit'])  
                        model_fit.reset_states()
                        print('epoch',i,'batch j','done') 
                    print('epoch',i,'done')
            elif training==2:
                print('training by epoch, resetting state after each epoch')
                for i in range(epochs):
                    model_fit.fit(x_train, 
                        y_train + centering[1] , 
                        epochs=1, 
                        batch_size=samples,
                        callbacks = callbacks,
                        verbose=params['verbose_fit'])  
                    model_fit.reset_states()
                    print('epoch',i,'done')
            elif training==3:
                print('training by epoch, not resetting state after each epoch')
                for i in range(epochs):
                    model_fit.fit(x_train, 
                        y_train + centering[1] , 
                        epochs=1, 
                        batch_size=samples,
                        callbacks = callbacks,
                        verbose=params['verbose_fit'])  
                    print('epoch',i,'done')
            elif training==4:
                print('training in one pass, not resetting state after each epoch')
                history = model_fit.fit(x_train, 
                    y_train + centering[1] , 
                    epochs=epochs, 
                    batch_size=batch_size,
                    callbacks = callbacks,
                    verbose=params['verbose_fit'])
                
            elif training==5:
                print('training in one pass, resetting state after each epoch by callback')
                history = model_fit.fit(x_train, 
                    y_train + centering[1] , 
                    epochs=epochs, 
                    batch_size=batch_size,
                    callbacks = callbacks + [ResetStatesCallback()],
                    verbose=params['verbose_fit'])

        if history is not None:
            plt.semilogy(history.history['loss'], label='Training loss')
            if 'val_loss' in history.history:
                plt.semilogy(history.history['val_loss'], label='Validation loss')
            plt.title(case + ' Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper left')
            plt.show()

           
        w_fitted=model_fit.get_weights()
        print('fitted weights hash =',hash2(w_fitted))
        if params['verbose_weights'] and params["initialize"]:
            for i in range(len(w_fitted)):
                print('weight',i,w_name[i],'shape',w[i].shape,'ndim',w[i].ndim,
                  'fitted: sum',np.sum(w_fitted[i],axis=0),'\nentries',w_fitted[i])
    else:
        print('Fitting skipped, using initial weights')
        w_fitted=w
        
    model_predict.set_weights(w_fitted)
    if params['verbose_weights']:
        print(f"model_predict weights hash: {hash2(model_predict.get_weights())}")
    
    return model_predict

def get_initial_weights(model_fit,params,rnn_dat):
    # Given a RNN architecture and hyperparameter dictionary, return array of physics-initiated weights
    # Inputs:
    # model_fit: output of create_RNN_2 with no training
    # params: (dict) dictionary of hyperparameters
    # rnn_dat: (dict) data dictionary, output of create_rnn_dat
    # Returns: numpy ndarray of weights that should be a rough solution to the moisture ODE
    DeltaE = params['DeltaE']
    T1 = params['T1']
    fmr = params['fm_raise_vs_rain']
    centering = params['centering']  # shift activation down
    
    w0_initial={'Ed':(1.-np.exp(-T1))/2, 
                'Ew':(1.-np.exp(-T1))/2,
                'rain':fmr * rnn_dat['scale_fm']/rnn_dat['scale_rain']}   # wx - input feature
                                 #  wh      wb   wd    bd = bias -1
    
    w_initial=np.array([np.nan, np.exp(-0.1), DeltaE[0]/rnn_dat['scale_fm'], # layer 0
                        1.0, -centering[0] + DeltaE[1]/rnn_dat['scale_fm']])                 # layer 1
    if params['verbose_weights']:
        print('Equilibrium moisture correction bias',DeltaE[0],
              'in the hidden layer and',DeltaE[1],' in the output layer')
    
    w_name = ['wx','wh','bh','wd','bd']
                        
    w=model_fit.get_weights()
    for j in range(w[0].shape[0]):
            feature = rnn_dat['features_list'][j]
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

def rnn_predict(model, params, rnn_dat):
    # Given compiled model, hyperparameters, and data dictionary, return array of predicted values.
    # NOTE: this does not distinguish in-sample vs out-of-sample data, it just fits the model to the given data
    # Inputs:
    # model: compiled moisture model
    # params: (dict) dictionary of hyperparameters
    # rnn_dat: (dict) dictionary of data to fit model to, output of create_rnn_dat
    # Returns: numpy array of fitted moisture values
    verbose = params['verbose']
    centering = params['centering']
    features = rnn_dat['features']
    hours = rnn_dat['hours']
    
    # model.set_weights(w_fitted)
    x_input=np.reshape(rnn_dat['X'],(1, hours, features))
    y_output = model.predict(x_input, verbose = verbose) - centering[1]
    
    vprint('x_input.shape=',x_input.shape,'y_output.shape=',y_output.shape)
    
    m = np.reshape(y_output,hours)
    # print('weights=',w)
    if 'scale_fm' in rnn_dat:
        vprint('scaling the output')
        m = m*rnn_dat['scale_fm']
    m = np.reshape(m,hours)
    return m

def run_rnn(case_data,params,fit=True,title2=''):
    # Run RNN on given case dictionary of FMDA data with certain params. Used internally in run_case
    # Inputs:
    # case_data: (dict) collection of cases with FMDA data. Cases are different locations and times, or synthetic data
    # params: (dict) collection of run options for model
    # fit: (bool) whether or not to fit RNN to data
    # title2: (str) title of RNN run to be joined to string with other info for plotting title
    # called from: run_case
    
    logging.info('run_rnn start')
    verbose = params['verbose']
    
    reproducibility.set_seed() # Set seed for reproducibility
    rnn_dat = create_rnn_data_1(case_data,params)
    create_rnn_data_2(rnn_dat,params)
    if params['verbose']:
        check_data(rnn_dat,case=0,name='rnn_dat')
    model_predict = train_rnn(
        rnn_dat,
        params,
        rnn_dat['hours'],
        fit=fit
    )

    m = rnn_predict(model_predict, params, rnn_dat)
    case_data['m'] = m

    # Reproducibility Checks
    hv = hash2(model_predict.get_weights())
    if case_data['case']=='case11' and fit:
        if params['initialize']:
            hv5 = 4.2030588308041834e+19
            mv = 3.59976005554199219
        else:
            hv5 = 4.4965532557938975e+19
            mv = 3.71594738960266113               
        print('check 5:',hv, 'should be',hv5,'error',hv-hv5)
        # assert (hv == hv5)
        checkm = case_data['m'][350]
        print('checkm=',format(checkm, '.17f'),' error',checkm-mv)
        print('params:',params)
    else:
        print('check - hash weights:',hv)

    # Print model output hash
    print(f"Model Output Hash: {hash2(m)}")
    
    plot_data(case_data,title2=title2)
    plt.show()
    logging.info('run_rnn end')
    return m, rmse_data(case_data)
    
    
def run_case(case_data,params, check_data=False):
    # Given a formatted FMDA dictionary of data, run RNN model using given run params
    # Inputs:
    # case_data: (dict) one case
    #      its name is in case_data['case']
    #      check_data
    # params: (dict) collection of run options for model
    # Return: (dict) collection of RMSE error for RNN & EKF, for train and prediction periods
    # called from fmda_rnn_rain
    
    print('\n***** ',case_data['case'],' *****\n')
    case_data['rain'][np.isnan(case_data['rain'])] = 0
    print(params)
    if check_data:
        check_data(case_data)
    hours=case_data['hours']
    if ('train_frac' in params) and ('h2' not in case_data):
        case_data['h2'] = round(hours * params['train_frac'])
    h2=case_data['h2']
    plot_data(case_data,title2='case data on input')
    m,Ec = mod.run_augmented_kf(case_data)  # extract from state
    case_data['m']=m
    case_data['Ec']=Ec
    plot_data(case_data,title2='augmented KF')
    rmse =      {'Augmented KF':rmse_data(case_data)}
    del case_data['Ec']  # cleanup
    rmse.update({'RNN initial':run_rnn(case_data,params,fit=False,title2='with initial weights, no fit')[1]})
    print("~"*50)
    rmse.update({'RNN trained':run_rnn(case_data,params,fit=True,title2='with trained RNN')[1]})
    return rmse




