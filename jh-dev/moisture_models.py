import numpy as np

def ext_kf(u,P,F,Q=0,d=None,H=None,R=None):
  """
  One step of the extended Kalman filter. 
  If there is no data, only advance in time.
  :param u:   the state vector, shape n
  :param P:   the state covariance, shape (n,n)
  :param F:   the model function, args vector u, returns F(u) and Jacobian J(u)
  :param Q:   the process model noise covariance, shape (n,n)
  :param d:   data vector, shape (m). If none, only advance in time
  :param H:   observation matrix, shape (m,n)
  :param R:   data error covariance, shape (n,n)
  :return ua: the analysis state vector, shape (n)
  :return Pa: the analysis covariance matrix, shape (n,n)
  """
  def d2(a):
    return np.atleast_2d(a) # convert to at least 2d array

  def d1(a):
    return np.atleast_1d(a) # convert to at least 1d array

  # forecast
  uf, J  = F(u)          # advance the model state in time and get the Jacobian
  uf = d1(uf)            # if scalar, make state a 1D array
  J = d2(J)              # if scalar, make jacobian a 2D array
  P = d2(P)              # if scalar, make Jacobian as 2D array
  Pf  = d2(J.T @ P) @ J + Q  # advance the state covariance Pf = J' * P * J + Q
  # analysis
  if d is None or not d.size :  # no data, no analysis
    return uf, Pf
  # K = P H' * inverse(H * P * H' + R) = (inverse(H * P * H' + R)*(H P))'
  H = d2(H)
  HP  = d2(H @ P)            # precompute a part used twice  
  K   = d2(np.linalg.solve( d2(HP @ H.T) + R, HP)).T  # Kalman gain
  # print('H',H)
  # print('K',K)
  res = d1(H @ d1(uf) - d)          # res = H*uf - d
  ua = uf - K @ res # analysis mean uf - K*res
  Pa = Pf - K @ d2(H @ P)        # analysis covariance
  return ua, d2(Pa)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_augmented_kf(d,Ed,Ew,rain,h2,hours):
  u = np.zeros((2,hours))
  u[:,0]=[0.1,0.0]       # initialize,background state  
  P = np.zeros((2,2,hours))
  P[:,:,0] = np.array([[1e-3, 0.],
                      [0.,  1e-3]]) # background state covariance
  Q = np.array([[1e-3, 0.],
                [0,  1e-3]]) # process noise covariance
  H = np.array([[1., 0.]])  # first component observed
  R = np.array([1e-3]) # data variance

  # ext_kf(u,P,F,Q=0,d=None,H=None,R=None) returns ua, Pa

  # print('initial u=',u,'P=',P)
  # print('Q=',Q,'H=',H,'R=',R)

  for t in range(1,h2):
      # use lambda construction to pass additional arguments to the model 
      u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                  lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                  Q,d[t],H=H,R=R)
      # print('time',t,'data',d[t],'filtered',u[0,t],'Ec',u[1,t])
  for t in range(h2,hours):
      u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                  lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                  Q*0.0)
      # print('time',t,'data',d[t],'forecast',u[0,t],'Ec',u[1,t])
  return u

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_augmented_kf(d,Ed,Ew,rain,h2,hours):
  u = np.zeros((2,hours))
  u[:,0]=[0.1,0.0]       # initialize,background state  
  P = np.zeros((2,2,hours))
  P[:,:,0] = np.array([[1e-3, 0.],
                      [0.,  1e-3]]) # background state covariance
  Q = np.array([[1e-3, 0.],
                [0,  1e-3]]) # process noise covariance
  H = np.array([[1., 0.]])  # first component observed
  R = np.array([1e-3]) # data variance

  # ext_kf(u,P,F,Q=0,d=None,H=None,R=None) returns ua, Pa

  # print('initial u=',u,'P=',P)
  # print('Q=',Q,'H=',H,'R=',R)

  for t in range(1,h2):
      # use lambda construction to pass additional arguments to the model 
      u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                  lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                  Q,d[t],H=H,R=R)
      # print('time',t,'data',d[t],'filtered',u[0,t],'Ec',u[1,t])
  for t in range(h2,hours):
      u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                  lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                  Q*0.0)
      # print('time',t,'data',d[t],'forecast',u[0,t],'Ec',u[1,t])
  return u

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def staircase(x,y,timesteps,trainsteps,return_sequences=False):
  # x [trainsteps+forecaststeps,features]    all inputs
  # y [trainsteps,outputs]
  # timesteps: split x and y into samples length timesteps, shifted by 1
  # trainsteps: number of timesteps to use for training, no more than y.shape[0]
  print('shape x = ',x.shape)
  print('shape y = ',y.shape)
  print('timesteps=',timesteps)
  print('trainsteps=',trainsteps)
  outputs = y.shape[1]
  features = x.shape[1]
  forecaststeps = x.shape[0]-trainsteps
  samples = trainsteps-timesteps+1
  print('staircase: samples=',samples,'timesteps=',timesteps,'features=',features)
  x_train = np.empty([samples, timesteps, features])
  print('return_sequences=',return_sequences)
  if return_sequences:
    print('returning all timesteps in a sample')
    y_train = np.empty([samples, timesteps, outputs])  # all
    for i in range(samples):
      for k in range(timesteps):
        for j in range(features):
          x_train[i,k,j] = x[i+k,j]
        for j in range(outputs):
          y_train[i,k,j] = y[i+k,j]
  else:
    print('returning only the last timestep in a sample')
    y_train = np.empty([samples, outputs])
    for i in range(samples):
      for j in range(features):
        for k in range(timesteps):
          x_train[i,k,j] = x[i+k,j]
      for j in range(outputs):
        y_train[i,j] = y[i+timesteps-1,j]

  return x_train, y_train

def seq2batches(x,y,timesteps,trainsteps):
  # x [trainsteps+forecaststeps,features]    all inputs
  # y [trainsteps,outputs]
  # timesteps: split x and y into samples length timesteps, shifted by 1
  # trainsteps: number of timesteps to use for training, no more than y.shape[0]
  print('shape x = ',x.shape)
  print('shape y = ',y.shape)
  print('timesteps=',timesteps)
  print('trainsteps=',trainsteps)
  outputs = y.shape[1]
  features = x.shape[1]
  samples= trainsteps - timesteps + 1
  print('samples=',samples)
  x_train = np.empty([samples, timesteps, features])
  y_train = np.empty([samples, timesteps, outputs])  # only the last
  print('samples=',samples,' timesteps=',timesteps,
        ' features=',features,' outputs=',outputs)
  for i in range(samples):
    for k in range(timesteps):
      for j in range(features):
        x_train[i,k,j] = x[i+k,j]
      for j in range(outputs):
        y_train[i,k,j] = y[i+k,j]  # return sequences
  return x_train, y_train


def create_RNN_2(hidden_units, dense_units, activation, stateful=False, 
                 batch_shape=None, input_shape=None, dense_layers=1,
                 rnn_layers=1,return_sequences=False,
                 initial_state=None):
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