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
from utils import vprint



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

def create_rnn_data(dat, hours=None, h2=None, scale = False, verbose = False,
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
    
    # transform as 2D, (timesteps, features) and (timesteps, outputs)
    # Et = np.reshape(E,[E.shape[0],1])
    Et = np.vstack((Ed, Ew)).T
    
    datat = np.reshape(fm,[fm.shape[0],1])
    
    # Scale Data if required
    scale=False
    if scale:
        scalerx = MinMaxScaler()
        scalerx.fit(Et)
        Et = scalerx.transform(Et)
        scalery = MinMaxScaler()
        scalery.fit(datat)
        datat = scalery.transform(datat)
        
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
        'h2':h2
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

    ## Note: this line executes an in-place operation that changes object. Keeping comment in for tracking purposes
    # vprint('model_predict input shape',Et.shape,'output shape',model_predict(np.reshape(Et,(1, hours, features))).shape)
    if verbose: print(model_predict.summary())
    
    x_train = rnn_dat['x_train']
    y_train = rnn_dat['y_train']

    # print('model_fit input shape',x_train.shape,'output shape',model_fit(x_train).shape) 
    model_fit(x_train) ## evalue the model once to set nonzero initial state
    
    w_exact=  [np.array([[1.-np.exp(-0.1)]]), np.array([[np.exp(-0.1)]]), np.array([0.]),np.array([[1.0]]),np.array([-1.*DeltaE])]
    
    w_initial=[np.array([[1.-np.exp(-0.1)]]), np.array([[np.exp(-0.1)]]), np.array([0.]),np.array([[1.0]]),np.array([-1.0])]
    w=model_fit.get_weights()
    for i in range(len(w)):
        vprint('weight',i,'shape',w[i].shape,'ndim',w[i].ndim,'given',w_initial[i].shape)
        for j in range(w[i].shape[0]):
            if w[i].ndim==2:
                for k in range(w[i].shape[1]):
                    w[i][j][k]=w_initial[i][0][0]/w[i].shape[0]
            else:
                w[i][j]=w_initial[i][0]
    model_fit.set_weights(w)
    
    if fit:
        model_fit.fit(x_train, y_train, epochs=5000,batch_size=samples, verbose=0)
    else:
        print('Fitting skipped, using initial weights')

    w_fitted=model_fit.get_weights()
    for i in range(len(w)):
        vprint('weight',i,' exact:',w_exact[i],':  initial:',w_initial[i],' fitted:',w_fitted[i])
    
    model_predict.set_weights(w_fitted)
    
    return model_predict


def rnn_predict(model, rnn_dat, hours, scale = False, verbose = False):
    features = rnn_dat['features']
    # model.set_weights(w_fitted)
    x_input=np.reshape(rnn_dat['Et'],(1, hours, features))
    y_output = model.predict(x_input, verbose = verbose)
    
    vprint('x_input.shape=',x_input.shape,'y_output.shape=',y_output.shape)
    
    m = np.reshape(y_output,hours)
    # print('weights=',w)
    if scale:
        vprint('scaling')
        m = scalery.inverse_transform(m)
    m = np.reshape(m,hours)
    
    return m

