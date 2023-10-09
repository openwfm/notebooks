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
from utils import vprint, hash2
import reproducibility
from data_funcs import check_data, rmse_data, plot_data
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

def create_rnn_data(dat, params, hours=None, h2=None):
    timesteps = params['timesteps']
    scale = params['scale']
    rain_do = params['rain_do']
    verbose = params['verbose']
    
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
     
    if params['verbose_weights']:
        print('scale_fm=',scale_fm,'scale_rain=',scale_rain)
    
    # transform as 2D, (timesteps, features) and (timesteps, outputs)
    # Et = np.reshape(E,[E.shape[0],1])
    
    if rain_do:
        Et = np.vstack((Ed, Ew, rain)).T
        features_list = ['Ed', 'Ew', 'rain']
    else:
        Et = np.vstack((Ed, Ew)).T        
        features_list = ['Ed', 'Ew']
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
        'scale_rain':scale_rain,
        'rain_do':rain_do,
        'features_list':features_list
    }
    
    return rnn_dat

def train_rnn(rnn_dat, params,hours, fit=True):

    verbose = params['verbose']
    
    if hours is None:
        hours = rnn_dat['hours']
    
    samples = rnn_dat['samples']
    features = rnn_dat['features']
    timesteps = rnn_dat['timesteps']
    centering = params['centering']
    
    model_fit=create_RNN_2(hidden_units=params['hidden_units'], 
                        dense_units=params['dense_units'], 
                        batch_shape=(samples,timesteps,features),
                        stateful=True,
                        return_sequences=False,
                        # initial_state=h0,
                        activation=params['activation'],
                        dense_layers=params['dense_layers'],
                        verbose = verbose)
    
    Et = rnn_dat['Et']
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

    model_fit(x_train) ## evalue the model once to set nonzero initial state
    
    w, w_name=get_initial_weights(model_fit, params, rnn_dat)
    
    model_fit.set_weights(w)
    
    if fit:
        model_fit.fit(x_train, y_train + centering[1] , epochs=params['epochs'],batch_size=samples, verbose=params['verbose_fit'])
        w_fitted=model_fit.get_weights()
        if params['verbose_weights']:
            for i in range(len(w_fitted)):
                print('weight',i,w_name[i],'shape',w[i].shape,'ndim',w[i].ndim,
                  'fitted: sum',np.sum(w_fitted[i],axis=0),'\nentries',w_fitted[i])
    else:
        print('Fitting skipped, using initial weights')
        w_fitted=w
        
    model_predict.set_weights(w_fitted)
    
    return model_predict

def get_initial_weights(model_fit,params,rnn_dat):
    
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
    verbose = params['verbose']
    centering = params['centering']
    features = rnn_dat['features']
    hours = rnn_dat['hours']
    
    # model.set_weights(w_fitted)
    x_input=np.reshape(rnn_dat['Et'],(1, hours, features))
    y_output = model.predict(x_input, verbose = verbose) - centering[1]
    
    vprint('x_input.shape=',x_input.shape,'y_output.shape=',y_output.shape)
    
    m = np.reshape(y_output,hours)
    # print('weights=',w)
    if rnn_dat['scale']:
        vprint('scaling')
        m = m*rnn_dat['scale_fm']
    m = np.reshape(m,hours)
    return m

def run_rnn(case_data,params,fit=True,title2=''):
    verbose = params['verbose']
    
    reproducibility.set_seed() # Set seed for reproducibility
    rnn_dat = create_rnn_data(case_data,params)
    if params['verbose']:
        check_data(rnn_dat,case=0,name='rnn_dat')
    model_predict = train_rnn(
        rnn_dat,
        params,
        rnn_dat['hours'],
        fit=fit
    )
    
    case_data['m'] = rnn_predict(model_predict, params, rnn_dat)

    hv = hash2(model_predict.get_weights())
    if case_data['case']=='case11' and fit:
        hv5 = 5.55077327554663e+19
        print('check 5:',hv, 'should be',hv5,'error',hv-hv5)
        # assert (hv == hv5)
        checkm = case_data['m'][350]
        mv = 3.77920889854431152
        print('checkm=',format(checkm, '.17f'),' error',checkm-mv)
    else:
        print('check - hash weights:',hv)
    
    plot_data(case_data,title2=title2)
    plt.show()
    return rmse_data(case_data)
    
    
def run_case(case_data,params, check_data=False):
    print('\n***** ',case_data['case'],' *****\n')
    case_data['rain'][np.isnan(case_data['rain'])] = 0
    if check_data:
        check_data(case_data)
    hours=case_data['hours']
    if 'train_frac' in params:
        case_data['h2'] = round(hours * params['train_frac'])
    h2=case_data['h2']
    plot_data(case_data,title2='case data on input')
    m,Ec = mod.run_augmented_kf(case_data)  # extract from state
    case_data['m']=m
    case_data['Ec']=Ec
    plot_data(case_data,title2='augmented KF')
    rmse =      {'Augmented KF':rmse_data(case_data)}
    del case_data['Ec']  # cleanup
    rmse.update({'RNN initial':run_rnn(case_data,params,fit=False,title2='with initial weights, no fit')})
    rmse.update({'RNN trained':run_rnn(case_data,params,fit=True,title2='with trained RNN')})
    return rmse