import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from abc import ABC, abstractmethod
# import xgboost as xg
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils import Dict

# ODE + Augmented Kalman Filter Code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def model_decay(m0,E,partials=0,T1=0.1,tlen=1):  
    # Arguments: 
    #   m0          fuel moisture content at start dimensionless, unit (1)
    #   E           fuel moisture eqilibrium (1)
    #   partials=0: return m1 = fuel moisture contents after time tlen (1)
    #           =1: return m1, dm0/dm0 
    #           =2: return m1, dm1/dm0, dm1/dE
    #           =3: return m1, dm1/dm0, dm1/dE dm1/dT1   
    #   T1          1/T, where T is the time constant approaching the equilibrium
    #               default 0.1/hour
    #   tlen        the time interval length, default 1 hour

    exp_t = np.exp(-tlen*T1)                  # compute this subexpression only once
    m1 = E + (m0 - E)*exp_t                   # the solution at end
    if partials==0:
        return m1
    dm1_dm0 = exp_t
    if partials==1:
        return m1, dm1_dm0          # return value and Jacobian
    dm1_dE = 1 - exp_t      
    if partials==2:
        return m1, dm1_dm0, dm1_dE 
    dm1_dT1 = -(m0 - E)*tlen*exp_t            # partial derivative dm1 / dT1
    if partials==3:
        return m1, dm1_dm0, dm1_dE, dm1_dT1       # return value and all partial derivatives wrt m1 and parameters
    raise('Bad arg partials')


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

### Define model function with drying, wetting, and rain equilibria

# Parameters
r0 = 0.05                                   # threshold rainfall [mm/h]
rs = 8.0                                    # saturation rain intensity [mm/h]
Tr = 14.0                                   # time constant for rain wetting model [h]
S = 250                                     # saturation intensity [dimensionless]
T = 10.0                                    # time constant for wetting/drying

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def model_moisture(m0,Eqd,Eqw,r,t=None,partials=0,T=10.0,tlen=1.0):
    # arguments:
    # m0         starting fuel moistureb (%s
    # Eqd        drying equilibrium      (%) 
    # Eqw        wetting equilibrium     (%)
    # r          rain intensity          (mm/h)
    # t          time
    # partials = 0, 1, 2
    # returns: same as model_decay
    #   if partials==0: m1 = fuel moisture contents after time 1 hour
    #              ==1: m1, dm1/dm0 
    #              ==2: m1, dm1/dm0, dm1/dE  
    
    if r > r0:
        # print('raining')
        E = S
        T1 =  (1.0 - np.exp(- (r - r0) / rs)) / Tr
    elif m0 <= Eqw: 
        # print('wetting')
        E=Eqw
        T1 = 1.0/T
    elif m0 >= Eqd:
        # print('drying')
        E=Eqd
        T1 = 1.0/T
    else: # no change'
        E = m0
        T1=0.0
    exp_t = np.exp(-tlen*T1)
    m1 = E + (m0 - E)*exp_t  
    dm1_dm0 = exp_t
    dm1_dE = 1 - exp_t
    #if t>=933 and t < 940:
    #  print('t,Eqw,Eqd,r,T1,E,m0,m1,dm1_dm0,dm1_dE',
    #        t,Eqw,Eqd,r,T1,E,m0,m1,dm1_dm0,dm1_dE)   
    if partials==0: 
        return m1
    if partials==1:
        return m1, dm1_dm0
    if partials==2:
        return m1, dm1_dm0, dm1_dE
    raise('bad partials')

def model_augmented(u0,Ed,Ew,r,t):
    # state u is the vector [m,dE] with dE correction to equilibria Ed and Ew at t
    # 
    m0, Ec = u0  # decompose state u0
    # reuse model_moisture(m0,Eqd,Eqw,r,partials=0):
    # arguments:
    # m0         starting fuel moistureb (1)
    # Ed         drying equilibrium      (1) 
    # Ew         wetting equilibrium     (1)
    # r          rain intensity          (mm/h)
    # partials = 0, 1, 2
    # returns: same as model_decay
    #   if partials==0: m1 = fuel moisture contents after time 1 hour
    #              ==1: m1, dm0/dm0 
    #              ==2: m1, dm1/dm0, dm1/dE 
    m1, dm1_dm0, dm1_dE  = model_moisture(m0,Ed + Ec, Ew + Ec, r, t, partials=2)
    u1 = np.array([m1,Ec])   # dE is just copied
    J =  np.array([[dm1_dm0, dm1_dE],
                   [0.     ,     1.]])
    return u1, J


### Default Uncertainty Matrices
Q = np.array([[1e-3, 0.],
            [0,  1e-3]]) # process noise covariance
H = np.array([[1., 0.]])  # first component observed
R = np.array([1e-3]) # data variance

def run_augmented_kf(dat0,h2=None,hours=None, H=H, Q=Q, R=R):
    dat = copy.deepcopy(dat0)
    
    if h2 is None:
        h2 = int(dat['h2'])
    if hours is None:
        hours = int(dat['hours'])
    
    d = dat['y']
    feats = dat['features_list']
    Ed = dat['X'][:,feats.index('Ed')]
    Ew = dat['X'][:,feats.index('Ew')]
    rain = dat['X'][:,feats.index('rain')]
    
    u = np.zeros((2,hours))
    u[:,0]=[0.1,0.0]       # initialize,background state  
    P = np.zeros((2,2,hours))
    P[:,:,0] = np.array([[1e-3, 0.],
                      [0.,  1e-3]]) # background state covariance
    # Q = np.array([[1e-3, 0.],
    #             [0,  1e-3]]) # process noise covariance
    # H = np.array([[1., 0.]])  # first component observed
    # R = np.array([1e-3]) # data variance

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

# General Machine Learning Models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MLModel(ABC):
    def __init__(self, params: dict):
        self.params = Dict(params)
        if type(self) is MLModel:
            raise TypeError("MLModel is an abstract class and cannot be instantiated directly")
        super().__init__()

    def filter_params(self, model_cls):
        """Filters out parameters that are not part of the model constructor."""
        model_params = self.params.copy()
        valid_keys = model_cls.__init__.__code__.co_varnames
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}
        return filtered_params
        
    
    def fit(self, X_train, y_train, weights=None):
        print(f"Fitting {self.params.mod_type} with params {self.params}")
        self.model.fit(X_train, y_train, sample_weight=weights)  

    def predict(self, X):
        print(f"Predicting with {self.params.mod_type}")
        preds = self.model.predict(X)
        return preds
        
    def eval(self, X_test, y_test):
        preds = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        # rmse_ros = np.sqrt(mean_squared_error(ros_3wind(y_test), ros_3wind(preds)))
        print(f"Test RMSE: {rmse}")
        # print(f"Test RMSE (ROS): {rmse_ros}")
        return rmse

class XGB(MLModel):
    def __init__(self, params: dict):
        super().__init__(params)
        model_params = self.filter_params(XGBRegressor) 
        self.model = XGBRegressor(**model_params)
        self.params['mod_type'] = "XGBoost"

    def predict(self, X):
        print("Predicting with XGB")
        preds = self.model.predict(X)
        return preds

class RF(MLModel):
    def __init__(self, params: dict):
        super().__init__(params)
        model_params = self.filter_params(RandomForestRegressor)
        self.model = RandomForestRegressor(**model_params)
        self.params['mod_type'] = "RandomForest"

class LM(MLModel):
    def __init__(self, params: dict):
        super().__init__(params)
        model_params = self.filter_params(LinearRegression)
        self.model = LinearRegression(**model_params)
        self.params['mod_type'] = "LinearRegression"



















