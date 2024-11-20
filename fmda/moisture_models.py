import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from abc import ABC, abstractmethod
import xgboost as xg
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils import Dict, all_items_exist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random

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



# Dictionary of scalers, used to avoid multiple object creation and to avoid multiple if statements
scalers = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler() 
}

## Class for handling input data
class MLData(dict):
    """
    A custom dictionary class for managing data, with validation, scaling, and train-test splitting functionality. Simplified as opposed to RNNData custom class since static models have simpler data shaping. 

    Assumes input dict is the result of combine_nested(...), so it is a spatial combination
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
        
        
        # Set up Data Scaling
        self.scaler = None
        if scaler is not None:
            self.set_scaler(scaler)
        
        # Rename and define other stuff.
        self['hours'] = min(arr.shape[0] for arr in self.y)
        
        self['all_features_list'] = self.pop('features_list')
        if features_list is None:
            print("Using all input features.")
            self.features_list = self.all_features_list
        else:
            self.features_list = features_list

        print(f"Setting features_list to {features_list}. \n  NOTE: not subsetting features yet. That happens in train_test_split.")
        
        self._run_checks()
        self.__dict__.update(self)
        
        
    # TODO: Fix checks for multilocation
    def _run_checks(self, verbose=True):
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
    def train_test_split(self, time_fracs=[1.,0.,0.], space_fracs=[1.,0.,0.], subset_features=True, features_list=None, verbose=True):
        # Set up 
        assert np.sum(time_fracs) == np.sum(space_fracs) == 1., f"Provided cross validation params don't sum to 1"
        if (len(time_fracs) != 3) or (len(space_fracs) != 3):
            raise ValueError("Cross-validation params `time_fracs` and `space_fracs` must be lists of length 3, representing (train/validation/test)")
        train_frac = time_fracs[0]
        val_frac = time_fracs[1]
        test_frac = time_fracs[2]        
        
        # Setup train/val/test in time
        train_ind = int(np.floor(self.hours * train_frac)); self.train_ind = train_ind
        test_ind= int(train_ind + round(self.hours * val_frac)); self.test_ind = test_ind
        # Check for any potential issues with indices
        if test_ind > self.hours:
            print(f"Setting test index to {self.hours}")
            test_ind = self.hours
        if train_ind > test_ind:
            raise ValueError("Train index must be less than test index.")        

        # Setup train/val/test in space
        train_frac_sp = space_fracs[0]
        val_frac_sp = space_fracs[1]
        locs = np.arange(len(self.loc['STID'])) # indices of locations
        train_size = int(len(locs) * train_frac_sp)
        val_size = int(len(locs) * val_frac_sp)
        random.shuffle(locs)
        train_locs = locs[:train_size]
        val_locs = locs[train_size:train_size + val_size]
        test_locs = locs[train_size + val_size:]
        # Store Lists of IDs in loc subdirectory
        self.loc['train_locs'] = [self.case[i] for i in train_locs]
        self.loc['val_locs'] = [self.case[i] for i in val_locs]
        self.loc['test_locs'] = [self.case[i] for i in test_locs]        

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
                    print(f"Warning: feature name '{item}' not found in list of all features from input data. Removing from internal features list")
                    # self.features_list.remove(item)
            
            X = [Xi[:, indices] for Xi in X]

            # Split by space
            X_train = [X[i] for i in train_locs]
            X_val = [X[i] for i in val_locs]
            X_test = [X[i] for i in test_locs]
            y_train = [y[i] for i in train_locs]
            y_val = [y[i] for i in val_locs]
            y_test = [y[i] for i in test_locs]

        # Split by time
        self.X_train = [Xi[:train_ind] for Xi in X_train]
        self.y_train = [yi[:train_ind].reshape(-1,1) for yi in y_train]
        if (val_frac >0) and (val_frac_sp)>0:
            self.X_val = [Xi[train_ind:test_ind] for Xi in X_val]
            self.y_val = [yi[train_ind:test_ind].reshape(-1,1) for yi in y_val]
        self.X_test = [Xi[test_ind:] for Xi in X_test]
        self.y_test = [yi[test_ind:].reshape(-1,1) for yi in y_test]


        # Combine List
        print("Combining locations")
        self.X_train = np.vstack(self.X_train)
        self.y_train = np.vstack(self.y_train)
        self.X_test = np.vstack(self.X_test)
        self.y_test = np.vstack(self.y_test)
        if hasattr(self, "X_val"):
            self.X_val = np.vstack(self.X_val)
            self.y_val = np.vstack(self.y_val)
            
        # Print statements if verbose
        if verbose:
            print(f"Train index: 0 to {train_ind}")
            print(f"Validation index: {train_ind} to {test_ind}")
            print(f"Test index: {test_ind} to {self.hours}")

            print("Subsetting locations into train/val/test")
            print(f"Total Locations: {len(locs)}")
            print(f"Train Locations: {len(train_locs)}")
            print(f"Val. Locations: {len(val_locs)}")
            print(f"Test Locations: {len(test_locs)}")
            print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            if hasattr(self, "X_val"):
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

        if self.scaler is None:
            raise ValueError("Scaler is not set. Use 'set_scaler' method to set a scaler before scaling data.")
        # if hasattr(self.scaler, 'n_features_in_'):
        #     warnings.warn("Scale_data has already been called. Exiting to prevent issues.")
        #     return            
        if not hasattr(self, "X_train"):
            raise AttributeError("No X_train within object. Run train_test_split first. This is to avoid fitting the scaler with prediction data.")
        if verbose:
            print(f"Scaling training data with scaler {self.scaler}, fitting on X_train")

        # Fit scaler on row-joined training data
        self.scaler.fit(self.X_train)
        # Transform data using fitted scaler
        self.X_train = self.scaler.transform(self.X_train)
        if hasattr(self, 'X_val'):
            if self.X_val is not None:
                self.X_val = self.scaler.transform(self.X_val)
        if self.X_test is not None:
            self.X_test = self.scaler.transform(self.X_test)

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
                print(f"Hash of {attr}: {hash_ndarray(value)}")        
    def __getattr__(self, key):
        """
        Allows attribute-style access to dictionary keys, a.k.a. enables the "." operator for get elements
        """        
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'MLData' object has no attribute '{key}'")

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














