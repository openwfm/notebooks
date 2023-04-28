import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import tensorflow as tf
from utils import vprint, hash2
import reproducibility
from data_funcs import check_data, mse_data, plot_data
import moisture_models as mod



def staircase(x,y,timesteps,trainsteps,return_sequences=False, verbose = False):
    # x [trainsteps+forecaststeps,features]    all inputs
    # y [trainsteps,outputs]
    # timesteps: split x and y into samples length timesteps, shifted by 1
    # trainsteps: number of timesteps to use for training, no more than y.shape[0]
    vprint('shape x = ',x.shape)
    vprint('shape y = ',y.shape)
    vprint('timesteps=',timesteps)
    vprint('trainsteps=',trainsteps)
    outputs = y.shape[1]
    features = x.shape[1]
    forecaststeps = x.shape[0]-trainsteps
    samples = trainsteps-timesteps+1
    vprint('staircase: samples=',samples,'timesteps=',timesteps,'features=',features)
    x_train = np.empty([samples, timesteps, features])
    vprint('return_sequences=',return_sequences)
    if return_sequences:
        vprint('returning all timesteps in a sample')
        y_train = np.empty([samples, timesteps, outputs])  # all
        for i in range(samples):
            for k in range(timesteps):
                for j in range(features):
                    x_train[i,k,j] = x[i+k,j]
                for j in range(outputs):
                    y_train[i,k,j] = y[i+k,j]
    else:
        vprint('returning only the last timestep in a sample')
    y_train = np.empty([samples, outputs])
    for i in range(samples):
        for j in range(features):
            for k in range(timesteps):
                x_train[i,k,j] = x[i+k,j]
        for j in range(outputs):
            y_train[i,j] = y[i+timesteps-1,j]

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
    # inputs2 = K.print_tensor(inputs, message='inputs = ')  # change allso inputs to inputs2 below, must be used
    x = inputs
    for i in range(rnn_layers):
        x = tf.keras.layers.SimpleRNN(hidden_units,activation=activation[0],
              stateful=stateful,return_sequences=return_sequences)(x
              # ,initial_state=initial_state
              )
    # x = tf.keras.layers.Dense(hidden_units, activation=activation[1])(x)
    for i in range(dense_layers):
        x = tf.keras.layers.Dense(dense_units, activation=activation[1])(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_rnn_data(dat, hours=None, h2=None, scale = 0, verbose = False,
                   timesteps=5):
    if hours is None:
        hours = dat['hours']
    if h2 is None:
        h2 = dat['h2']
    vprint('create_rnn_data: hours=',hours,' h2=',h2)
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
        print('scaling to range 0 to',scale)
        scale_fm=max(max(Ew),max(Ed),max(fm))/scale
        scale_rain=max(max(rain),0.01)/scale
        Ed = Ed/scale_fm
        Ew = Ew/scale_fm
        fm = fm/scale_fm
        rain = rain/scale_rain
    else:
        scale_fm=1.0
        scale_rain=1.0
    
    # transform as 2D, (timesteps, features) and (timesteps, outputs)
    # Et = np.reshape(E,[E.shape[0],1])
    Et = np.vstack((Ed, Ew)).T
    
    datat = np.reshape(fm,[fm.shape[0],1])
    
    # split data
    x_train, y_train = staircase(Et,datat,timesteps=timesteps,trainsteps=h2,
                                 return_sequences=False, verbose = verbose)
    vprint('x_train shape=',x_train.shape)
    samples, timesteps, features = x_train.shape
    vprint('y_train shape=',y_train.shape)

    h0 = tf.convert_to_tensor(datat[:samples],dtype=tf.float32)
    
    # Set up return dictionary
    
    rnn_dat = {
        'hours': hours,
        'x_train': x_train,
        'y_train': y_train,
        'Et': Et,
        'samples': samples,
        'timesteps': timesteps,
        'features': features,
        'h0': h0,
        'hours':hours,
        'h2':h2,
        'scale':scale,
        'scale_fm':scale_fm,
        'scale_rain':scale_rain
    }
    
    return rnn_dat

def train_rnn(rnn_dat, hours, activation, hidden_units, dense_units, dense_layers, 
              verbose = False, fit=True, DeltaE=0.0):
    
    if hours is None:
        hours = rnn_dat['hours']
    
    samples = rnn_dat['samples']
    features = rnn_dat['features']
    timesteps = rnn_dat['timesteps']
    
    model_fit=create_RNN_2(hidden_units=hidden_units, 
                        dense_units=dense_units, 
                        batch_shape=(samples,timesteps,features),
                        stateful=True,
                        return_sequences=False,
                        # initial_state=h0,
                        activation=activation,
                        dense_layers=dense_layers,
                        verbose = verbose)
    
    Et = rnn_dat['Et']
    model_predict=create_RNN_2(hidden_units=hidden_units, dense_units=dense_units,  
                            input_shape=(hours,features),stateful = False,
                            return_sequences=True,
                            activation=activation,dense_layers=dense_layers,
                              verbose = verbose)
    
    if verbose: print(model_predict.summary())
    
    x_train = rnn_dat['x_train']
    y_train = rnn_dat['y_train']

    model_fit(x_train) ## evalue the model once to set nonzero initial state
    
    # -1.0 makes no sense but leaving for check 5 in run run_case. Final RMSE is about the same.   
    w_initial=np.array([1.-np.exp(-0.1), np.exp(-0.1), 0., 1.0, -1.0])
    #w_initial=np.array([1.-np.exp(-0.1), np.exp(-0.1), 0., 1.0, 0.0])
    w_name = ['wx','wh','bh','wd','bd']
                        
    w=model_fit.get_weights()
    for i in range(len(w)):
        for j in range(w[i].shape[0]):
            if w[i].ndim==2:
                # initialize all entries of the weight matrix to the same number
                for k in range(w[i].shape[1]):
                    w[i][j][k]=w_initial[i]/w[i].shape[0]
            elif w[i].ndim==1:
                w[i][j]=w_initial[i]
            else:
                print('weight',i,'shape',w[i].shape)
                raise ValueError("Only 1 or 2 dimensions supported")
        print('weight',i,w_name[i],'shape',w[i].shape,'ndim',w[i].ndim,'initial: sum',np.sum(w[i],axis=0),'\nentries',w[i])
    model_fit.set_weights(w)
    
    if fit:
        model_fit.fit(x_train, y_train, epochs=5000,batch_size=samples, verbose=0)
        w_fitted=model_fit.get_weights()
        for i in range(len(w_fitted)):
            print('weight',i,w_name[i],'shape',w[i].shape,'ndim',w[i].ndim,
                  'fitted: sum',np.sum(w_fitted[i],axis=0),'\nentries',w_fitted[i])
    else:
        print('Fitting skipped, using initial weights')
        w_fitted=w
        
    model_predict.set_weights(w_fitted)
    
    return model_predict


def rnn_predict(model, rnn_dat, hours, verbose = False):
    features = rnn_dat['features']
    # model.set_weights(w_fitted)
    x_input=np.reshape(rnn_dat['Et'],(1, hours, features))
    y_output = model.predict(x_input, verbose = verbose)
    
    vprint('x_input.shape=',x_input.shape,'y_output.shape=',y_output.shape)
    
    m = np.reshape(y_output,hours)
    # print('weights=',w)
    if rnn_dat['scale']:
        vprint('scaling')
        m = m*rnn_dat['scale_fm']
    m = np.reshape(m,hours)
    return m

def run_rnn(case_data,fit=True,verbose=False,title2='',scale=0):
    reproducibility.set_seed() # Set seed for reproducibility
    rnn_dat = create_rnn_data(case_data,scale=scale, verbose=verbose)
    check_data(rnn_dat,case=0,name='rnn_dat')
    model_predict = train_rnn(
        rnn_dat,
        rnn_dat['hours'],
        activation=['linear','linear'],
        hidden_units=6,
        dense_units=1,
        dense_layers=1,
        verbose = verbose,
        fit=fit
    )
    hv = hash2(model_predict.get_weights())
    if case_data['case']=='case11' and fit:
        hv5 = 5.55077327554663e+19
        print('check 5:',hv, 'should be',hv5,'error',hv-hv5)
        # assert (hv == hv5)
    else:
        print('check - hash weights:',hv)
    
    case_data['m'] = rnn_predict(model_predict, rnn_dat,rnn_dat['hours'], verbose = verbose)
    mse_data(case_data)
    plot_data(case_data,title2=title2)
    plt.show()
    
    
def run_case(case_data,verbose=False,title2='',scale=0):
    check_data(case_data)
    hours=case_data['hours']
    h2=case_data['h2']
    plot_data(case_data,title2='case data on input')
    m,Ec = mod.run_augmented_kf(case_data)  # extract from state
    case_data['m']=m
    case_data['Ec']=Ec
    plot_data(case_data,title2='augmented KF')
    mse_data(case_data)
    del case_data['Ec']  # cleanup
    run_rnn(case_data,fit=False,verbose=verbose,title2='with initial weights, no fit',scale=scale)
    run_rnn(case_data,fit=True,title2='with trained RNN',scale=scale)