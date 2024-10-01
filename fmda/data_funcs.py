## Set of Functions to process and format fuel moisture model inputs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np, random
from numpy.random import rand
import tensorflow as tf
import pickle, os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from moisture_models import model_decay, model_moisture
from datetime import datetime, timedelta
from utils import  is_numeric_ndarray, hash2
import json
import copy
import subprocess
import os.path as osp
from utils import Dict, str2time, check_increment, time_intp
import warnings

# Wrapper Functions to Put it all together
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO: ENGINEERED TIME FEATURES:
# hod = rnn_dat.time.astype('datetime64[h]').astype(int) % 24
# doy = np.array([dt.timetuple().tm_yday - 1 for dt in rnn_dat.time])

def create_spatial_train(input_file_paths, params_data, atm_dict = "HRRR", verbose=False):
    train = process_train_dict(file_paths, params_data = params_data, verbose=verbose)
    train_sp = Dict(combine_nested(train))
    return train_sp

def process_train_dict(input_file_paths, params_data, atm_dict = "HRRR", spatial=False, verbose=False):
    if type(input_file_paths) is not list:
        raise ValueError(f"Argument `input_file_paths` must be list, received {type(input_file_paths)}")
    train = {}
    for file_path in input_file_paths:
        # Extract target and features
        di = build_train_dict(file_path, atm=atm_dict, features_all=params_data['features_all'], verbose=verbose)
        # Subset timeseries into shorter stretches
        di = split_timeseries(di, hours=params_data['hours'], verbose=verbose)
        di = discard_keys_with_short_y(di, hours=params_data['hours'], verbose=False)
        # Check for suspect data
        flags = flag_dict_keys(di, params_data['zero_lag_threshold'], params_data['max_intp_time'], max_y = params_data['max_fm'], min_y = params_data['min_fm'], verbose=verbose)
        # Remove flagged cases
        cases = list([*di.keys()])
        flagged_cases = [element for element, flag in zip(cases, flags) if flag == 1]
        remove_key_list(di, flagged_cases, verbose=verbose)
        train.update(di)
    if spatial:
        train = combine_nested(train)
    
    return Dict(train)


def subset_by_features(nested_dict, input_features, verbose=True):
    """
    Subsets a nested dictionary to only include keys where all strings in the input_features
    are present in the dictionary's 'features_list' subkey. Primarily used for RAWS dictionaries where desired features might not be present at all ground stations.

    Parameters:
    nested_dict (dict): The nested dictionary with a 'features_list' subkey.
    input_features (list): The list of features to be checked.

    Returns:
    dict: A subset of the input dictionary with only the matching keys.
    """
    if verbose:
        print(f"Subsetting to cases with features: {input_features}")
    
    # Create a new dictionary to store the result
    result = {}
    
    # Iterate through the keys in the nested dictionary
    for key, value in nested_dict.items():
        # Check if 'features_list' key exists and all input_features are in the list
        if 'features_list' in value and all(feature in value['features_list'] for feature in input_features):
            # Add to the result if all features are present
            result[key] = value
    
    return result

feature_types = {
    # Static features are based on physical location, e.g. location of RAWS site
    'static': ['elev', 'lon', 'lat'],
    # Atmospheric weather features come from either RAWS subdict or HRRR
    'atm': ['temp', 'rh', 'wind', 'solar', 'soilm', 'canopyw', 'groundflux', 'Ed', 'Ew']
}

def build_train_dict(input_file_path,
              forecast_step=1, atm="HRRR",features_all=['Ed', 'Ew', 'solar', 'wind', 'elev', 'lon', 'lat', 'doy', 'hod', 'rain'], verbose=False):
    # in:
    #   file_path       list of strings - files as in read_test_pkl
    #   forecast_step   int - which forecast step to take atmospheric data from (maybe 03, must be >0). 
    #   atm        str - name of subdict where atmospheric vars are located
    #   features_list   list of strings - names of keys in subdicts to collect into features matrix. Default is everything collected
    # return:
    #   train          dictionary with structure
    #                  {key : {'key' : key,    # copied subdict key
    #                          'loc' : {...},  # copied from in dict = {key : {'loc': ... }...}
    #                         'time' : time,   # datetime vector, spacing tres
    #                            'X' : fm      # target fuel moisture from the RAWS, interpolated to time
    #                            'Y' : feat    # features from atmosphere and location
    #                            
    #

    
    # TODO: fix this
    if 'rain' in features_all and (not features_all[-1]=='rain'):
        raise ValueError(f"Make rain in features list last element since (working on fix as of 24-6-24), given features list: {features_list}")
    
    if forecast_step > 0 and forecast_step < 100 and forecast_step == int(forecast_step):
        fstep='f'+str(forecast_step).zfill(2)
        fprev='f'+str(forecast_step-1).zfill(2)
        # logging.info('Using data from step %s',fstep)
        # logging.info('Using rain as the difference of accumulated precipitation between %s and %s',fstep,fprev)
    else:
        # logging.critical('forecast_step must be integer between 1 and 99')
        raise ValueError('bad forecast_step')
        
    train = {}
    with open(input_file_path, 'rb') as file:
        # logging.info("loading file %s", file_path)
        d = pickle.load(file)
    for key in d:
        atm_dict = atm
        features_list = features_all
        # logging.info('Processing subdictionary %s',key)
        if key in train:
            pass
            # logging.warning('skipping duplicate key %s',key)
        else:
            subdict=d[key]    # subdictionary for this case
            loc=subdict['loc']
            train[key] = {
            'id': key,  # store the key inside the dictionary, subdictionary will be used separatedly
            'case':key,
            'filename': input_file_path,
            'loc': loc
            }
            desc='descr'
            if desc in subdict:
                train[desc]=subdict[desc]
            time_hrrr=str2time(subdict[atm_dict]['time'])
            # timekeeping
            hours=len(d[key][atm_dict]['time'])
            train[key]['hours']=hours
            # train[key]['h2']   =hours     # not doing prediction yet    
            hrrr_increment = check_increment(time_hrrr,id=key+f' {atm_dict}.time')
            # logging.info(f'{atm_dict} increment is %s h',hrrr_increment)
            if  hrrr_increment < 1:
                # logging.critical('HRRR increment is %s h must be at least 1 h',hrrr_increment)
                raise(ValueError)
            
            # build matrix of features - assuming all the same length, if not column_stack will fail
            train[key]['time']=time_hrrr
            # logging.info(f"Created feature matrix train[{key}]['X'] shape {train[key]['X'].shape}")
            time_raws=str2time(subdict['RAWS']['time_raws']) # may not be the same as HRRR
            # logging.info('%s RAWS.time_raws length is %s',key,len(time_raws))
            check_increment(time_raws,id=key+' RAWS.time_raws')
            # print_first(time_raws,num=5,id='RAWS.time_raws')
            
            # Set up static vars
            columns=[]
            missing_features = []
            for feat in features_list:
                # For atmospheric features,
                if feat in feature_types['atm']:
                    if atm_dict == "HRRR":
                        vec = subdict['HRRR'][fstep][feat]
                        columns.append(vec)
                    elif atm_dict == "RAWS":
                        if feat in subdict['RAWS'].keys():
                            vec = time_intp(time_raws, subdict['RAWS'][feat], time_hrrr)
                            columns.append(vec)
                        else:
                            missing_features.append(feat)
                
                # For static features, repeat to fit number of time observations
                elif feat in feature_types['static']:
                    columns.append(np.full(hours,loc[feat]))
            # Add Engineered Time features, doy and hod
            hod = time_hrrr.astype('datetime64[h]').astype(int) % 24
            doy = np.array([dt.timetuple().tm_yday - 1 for dt in time_hrrr])
            columns.extend([doy, hod])
            
            # compute rain as difference of accumulated precipitation
            if 'rain' in features_list:
                if atm_dict == "HRRR":
                    rain = subdict[atm_dict][fstep]['precip_accum']- subdict[atm_dict][fprev]['precip_accum']
                    # logging.info('%s rain as difference %s minus %s: min %s max %s',
                             # key,fstep,fprev,np.min(rain),np.max(rain))
                elif atm_dict == "RAWS":
                    if 'rain' in subdict[atm_dict]:
                        rain = time_intp(time_raws,subdict[atm_dict]['rain'],time_hrrr)
                    else:
                        pass
                        # logging.info('No rain data found in RAWS subdictionary %s', key)
                columns.append( rain ) # add rain feature         
            else:
                missing_features.append('rain')

            train[key]['X'] = np.column_stack(columns)
            train[key]['features_list'] = [item for item in features_list if item not in missing_features]
            
            fm=subdict['RAWS']['fm']
            # logging.info('%s RAWS.fm length is %s',key,len(fm))
            # interpolate RAWS sensors to HRRR time and over NaNs
            train[key]['y'] = time_intp(time_raws,fm,time_hrrr)
            # TODO: check endpoint interpolation when RAWS data sparse, and bail out if not enough data
            
            if  train[key]['y'] is None:
                pass
                # logging.error('Cannot create target matrix for %s, using None',key)
            else:
                pass
                # logging.info(f"Created target matrix train[{key}]['y'] shape {train[key]['y'].shape}")

    # logging.info('Created a "train" dictionary with %s items',len(train))
 
    # clean up
    
    keys_to_delete = []
    for key in train:
        if train[key]['X'] is None or train[key]['y'] is None:
            # logging.warning('Deleting training item %s because features X or target Y are None', key)
            keys_to_delete.append(key)

    # Delete the items from the dictionary
    if len(keys_to_delete)>0:
        for key in keys_to_delete:
            del train[key]       
        # logging.warning('Deleted %s items with None for data. %s items remain in the training dictionary.',
                        # len(keys_to_delete),len(train))
        
    # output

    # if output_file_path is not None:
    #     with open(output_file_path, 'wb') as file:
    #         logging.info('Writing pickle dump of the dictionary train into file %s',output_file_path)
    #         pickle.dump(train, file)
    
    # logging.info('pkl2train done')
    
    return train


def remove_key_list(d, ls, verbose=False):
    for key in ls:
        if key in d:
            if verbose:
                print(f"Removing key {key} due to data flags")
            del d[key]

def split_timeseries(dict0, hours, naming_convention = "_set_", verbose=False):
    """
    Given number of hours, splits nested fmda dictionary into smaller stretches. This is used primarily to aid in filtering out and removing stretches of missing data.
    """
    cases = list([*dict0.keys()])
    dict1={}
    for key, data in dict0.items():
        if verbose:
            print(f"Processing case: {key}")
            print(f"Length of y vector: {len(data['y'])}")
        if type(data['time'][0]) == str:
            time=str2time(data['time'])
        else:
            time=data['time']
        X_array = data['X']
        y_array = data['y']
        
        # Determine the start and end time for the 720-hour portions
        start_time = time[0]
        end_time = time[-1]
        current_time = start_time
        portion_index = 1
        while current_time < end_time:
            next_time = current_time + timedelta(hours=hours)
            
            # Create a mask for the time range
            mask = (time >= current_time) & (time < next_time)
            
            # Apply the mask to extract the portions
            new_time = time[mask]
            new_X = X_array[mask]
            new_y = y_array[mask]
            
            # Save the portions in the new dictionary with naming convention if second or more portion
            if portion_index > 1:
                new_key = f"{key}{naming_convention}{portion_index}"
            else:
                new_key = key
            dict1[new_key] = {'time': new_time, 'X': new_X, 'y': new_y}
            
            # Add other keys that aren't subset
            for key2 in dict0[key].keys():
                if key2 not in ['time', 'X', 'y']:
                    dict1[new_key][key2] = dict0[key][key2]
            # Update Case name and Id (same for now, overloaded terminology)
            dict1[new_key]['case'] = new_key
            dict1[new_key]['id'] = new_key
            dict1[new_key]['hours'] = len(dict1[new_key]['y'])
            
            
            # Move to the next portion
            current_time = next_time
            portion_index += 1    
        if verbose:
            print(f"Partitions of length {hours} from case {key}: {portion_index-1}")
    return dict1

def flag_lag_stretches(x, threshold, lag = 1):
    """
    Used to itentify stretches of data that have been interpolated a length greater than or equal to given threshold. Used to identify stretches of data that are not trustworthy due to extensive interpolation and thus should be removed from a ML training set.
    """
    lags = np.round(np.diff(x, n=lag), 8)
    zero_lag_indices = np.where(lags == 0)[0]
    current_run_length = 1
    for i in range(1, len(zero_lag_indices)):
        if zero_lag_indices[i] == zero_lag_indices[i-1] + 1:
            current_run_length += 1
            if current_run_length > threshold:
                return True
        else:
            current_run_length = 1
    else:
        return False   


def flag_dict_keys(dict0, lag_1_threshold, lag_2_threshold, max_y, min_y, verbose=False):
    """
    Loop through dictionary and generate list of flags for if the `y` variable within the dictionary has target lag patterns. The lag_1_threshold parameter sets upper limit for the number of constant, zero-lag stretches in y. The lag_2_threshold parameter sets upper limit for the number of constant, linear stretches of data. Used to identify cases of data that have been interpolated excessively long and thus not trustworthy for inclusion in a ML framework.
    """
    cases = list([*dict0.keys()])
    flags = np.zeros(len(cases))
    for i, case in enumerate(cases):
        if verbose:
            print("~"*50)
            print(f"Case: {case}")
        y = dict0[case]['y']
        if flag_lag_stretches(y, threshold=lag_1_threshold, lag=1):
            if verbose:
                print(f"Flagging case {case} for zero lag stretches greater than param {lag_1_threshold}")
            flags[i]=1
        if flag_lag_stretches(y, threshold=lag_2_threshold, lag=2):
            if verbose:
                print(f"Flagging case {case} for constant linear stretches greater than param {lag_2_threshold}")
            flags[i]=1
        if np.any(y>=max_y) or np.any(y<=min_y):
            if verbose:
                print(f"Flagging case {case} for FMC outside param range {min_y,max_y}. FMC range for {case}: {y.min(),y.max()}")
            flags[i]=1    

    return flags

def discard_keys_with_short_y(input_dict, hours, verbose=False):
    """
    Remove keys from a dictionary where the subkey `y` is less than given hours. Used to remove partial sequences at the end of timeseries after the longer timeseries has been subdivided.
    """
    discarded_keys = [key for key, value in input_dict.items() if len(value['y']) < hours]
    
    if verbose:
        print(f"Discarded keys due to y length less than {hours}: {discarded_keys}")
    
    filtered_dict = {key: value for key, value in input_dict.items() if key not in discarded_keys}
    
    return filtered_dict



# Utility to combine nested fmda dictionaries
def combine_nested(nested_input_dict, verbose=True):
    """
    Combines input data dictionaries.

    Parameters:
    -----------
    verbose : bool, optional
        If True, prints status messages. Default is True.
    """   
    # Setup return dictionary
    d = {}
    # Use the helper function to populate the keys
    d['id'] = _combine_key(nested_input_dict, 'id')
    d['case'] = _combine_key(nested_input_dict, 'case')
    d['filename'] = _combine_key(nested_input_dict, 'filename')
    d['time'] = _combine_key(nested_input_dict, 'time')
    d['X'] = _combine_key(nested_input_dict, 'X')
    d['y'] = _combine_key(nested_input_dict, 'y')

    # Build the loc subdictionary using _combine_key for each loc key
    d['loc'] = {
        'STID': _combine_key(nested_input_dict, 'loc', 'STID'),
        'lat': _combine_key(nested_input_dict, 'loc', 'lat'),
        'lon': _combine_key(nested_input_dict, 'loc', 'lon'),
        'elev': _combine_key(nested_input_dict, 'loc', 'elev'),
        'pixel_x': _combine_key(nested_input_dict, 'loc', 'pixel_x'),
        'pixel_y': _combine_key(nested_input_dict, 'loc', 'pixel_y')
    }
    
    # Handle features_list separately with validation
    features_list = _combine_key(nested_input_dict, 'features_list')
    if features_list:
        first_features_list = features_list[0]
        for fl in features_list:
            if fl != first_features_list:
                warnings.warn("Different features_list found in the nested input dictionaries.")
        d['features_list'] = first_features_list

    return d
        
def _combine_key(nested_input_dict, key, subkey=None):
    combined_list = []
    for input_dict in nested_input_dict.values():
        if isinstance(input_dict, dict):
            try:
                if subkey:
                    combined_list.append(input_dict[key][subkey])
                else:
                    combined_list.append(input_dict[key])
            except KeyError:
                warning_message = f"Missing expected key: '{key}'{f' or subkey: {subkey}' if subkey else ''} in one of the input dictionaries. Setting value to None."
                warnings.warn(warning_message)
                combined_list.append(None)
        else:
            raise ValueError(f"Expected a dictionary, but got {type(input_dict)}")
    return combined_list 


def compare_dicts(dict1, dict2, keys):
    for key in keys:
        if dict1.get(key) != dict2.get(key):
            return False
    return True

items = '_items_'     # dictionary key to keep list of items in
def check_data_array(dat,hours,a,s):
    if a in dat[items]:
         dat[items].remove(a)
    if a in dat:
        ar = dat[a]
        print("array %s %s length %i min %s max %s hash %s %s" %
              (a,s,len(ar),min(ar),max(ar),hash2(ar),type(ar)))
        if hours is not None:
            if len(ar) < hours:
                print('len(%a) = %i does not equal to hours = %i' % (a,len(ar),hours))
                exit(1)
    else:
        print(a + ' not present')
        
def check_data_scalar(dat,a):
    if a in dat[items]:
         dat[items].remove(a)
    if a in dat:
        print('%s = %s' % (a,dat[a]),' ',type(dat[a]))
    else:
        print(a + ' not present' )

def check_data(dat,case=True,name=None):
    dat[items] = list(dat.keys())   # add list of items to the dictionary
    if name is not None:
        print(name)
    if case:
        check_data_scalar(dat,'filename')
        check_data_scalar(dat,'title')
        check_data_scalar(dat,'note')
        check_data_scalar(dat,'hours')
        check_data_scalar(dat,'h2')
        check_data_scalar(dat,'case')
        if 'hours' in dat:
            hours = dat['hours']
        else:
            hours = None
        check_data_array(dat,hours,'E','drying equilibrium (%)')
        check_data_array(dat,hours,'Ed','drying equilibrium (%)')
        check_data_array(dat,hours,'Ew','wetting equilibrium (%)')
        check_data_array(dat,hours,'Ec','equilibrium equilibrium (%)')
        check_data_array(dat,hours,'rain','rain intensity (mm/h)')
        check_data_array(dat,hours,'fm','RAWS fuel moisture data (%)')
        check_data_array(dat,hours,'m','fuel moisture estimate (%)')
    if dat[items]:
        print('items:',dat[items])
        for a in dat[items].copy():
            ar=dat[a]
            if dat[a] is None or np.isscalar(dat[a]):
                check_data_scalar(dat,a)
            elif is_numeric_ndarray(ar):
                print(type(ar))
                print("array", a, "shape",ar.shape,"min",np.min(ar),
                       "max",np.max(ar),"hash",hash2(ar),"type",type(ar))
            elif isinstance(ar, tf.Tensor):
                print("array", a, "shape",ar.shape,"min",np.min(ar),
                       "max",np.max(ar),"type",type(ar))
            else:
                print('%s = %s' % (a,dat[a]),' ',type(dat[a]))
        del dat[items] # clean up

# Note: the project structure has moved towards pickle files, so these json funcs might not be needed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def to_json(dic,filename):
    # Write given dictionary as json file. 
    # This utility is used because the typical method fails on numpy.ndarray 
    # Inputs:
    # dic: dictionary
    # filename: (str) output json filename, expect a ".json" file extension
    # Return: none
    
    print('writing ',filename)
    # check_data(dic)
    new={}
    for i in dic:
        if type(dic[i]) is np.ndarray:
            new[i]=dic[i].tolist()  # because numpy.ndarray is not serializable
        else:
            new[i]=dic[i]
        # print('i',type(new[i]))
    new['filename']=filename
    print('Hash: ', hash2(new))
    json.dump(new,open(filename,'w'),indent=4)

def from_json(filename):
    # Read json file given a filename
    # Inputs: filename (str) expect a ".json" string
    
    print('reading ',filename)
    dic=json.load(open(filename,'r'))
    new={}
    for i in dic:
        if type(dic[i]) is list:
            new[i]=np.array(dic[i])  # because ndarray is not serializable
        else:
            new[i]=dic[i]
    check_data(new)
    print('Hash: ', hash2(new))
    return new

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to simulate moisture data and equilibrium for model testing
def create_synthetic_data(days=20,power=4,data_noise=0.02,process_noise=0.0,DeltaE=0.0):
    hours = days*24
    h2 = int(hours/2)
    hour = np.array(range(hours))
    day = np.array(range(hours))/24.

    # artificial equilibrium data
    E = 100.0*np.power(np.sin(np.pi*day),4) # diurnal curve
    E = 0.05+0.25*E
    # FMC free run
    m_f = np.zeros(hours)
    m_f[0] = 0.1         # initial FMC
    process_noise=0.
    for t in range(hours-1):
        m_f[t+1] = max(0.,model_decay(m_f[t],E[t])  + random.gauss(0,process_noise) )
    data = m_f + np.random.normal(loc=0,scale=data_noise,size=hours)
    E = E + DeltaE    
    return E,m_f,data,hour,h2,DeltaE
    
# the following input or output dictionary with all model data and variables
       
def synthetic_data(days=20,power=4,data_noise=0.02,process_noise=0.0,
    DeltaE=0.0,Emin=5,Emax=30,p_rain=0.01,max_rain=10.0):
    hours = days*24
    h2 = int(hours/2)
    hour = np.array(range(hours))
    day = np.array(range(hours))/24.
    # artificial equilibrium data
    E = np.power(np.sin(np.pi*day),power) # diurnal curve betwen 0 and 1
    E = Emin+(Emax - Emin)*E
    E = E + DeltaE
    Ed=E+0.5
    Ew=np.maximum(E-0.5,0)
    rain = np.multiply(rand(hours) < p_rain, rand(hours)*max_rain)
    # FMC free run
    fm = np.zeros(hours)
    fm[0] = 0.1         # initial FMC
    # process_noise=0.
    for t in range(hours-1):
        fm[t+1] = max(0.,model_moisture(fm[t],Ed[t-1],Ew[t-1],rain[t-1])  + random.gauss(0,process_noise))
    fm = fm + np.random.normal(loc=0,scale=data_noise,size=hours)
    dat = {'E':E,'Ew':Ew,'Ed':Ed,'fm':fm,'hours':hours,'h2':h2,'DeltaE':DeltaE,'rain':rain,'title':'Synthetic data'}
    
    return dat

def plot_one(hmin,hmax,dat,name,linestyle,c,label, alpha=1,type='plot'):
    # helper for plot_data
    if name in dat:
        h = len(dat[name])
        if hmin is None:
            hmin=0
        if hmax is None:
            hmax=len(dat[name])
        hour = np.array(range(hmin,hmax))
        if type=='plot':
            plt.plot(hour,dat[name][hmin:hmax],linestyle=linestyle,c=c,label=label, alpha=alpha)
        elif type=='scatter':
            plt.scatter(hour,dat[name][hmin:hmax],linestyle=linestyle,c=c,label=label, alpha=alpha)

# Lookup table for plotting features
plot_styles = {
    'Ed': {'color': '#EF847C', 'linestyle': '--', 'alpha':.8, 'label': 'drying EQ'},
    'Ew': {'color': '#7CCCEF', 'linestyle': '--', 'alpha':.8, 'label': 'wetting EQ'},
    'rain': {'color': 'b', 'linestyle': '-', 'alpha':.9, 'label': 'Rain'}
}
def plot_feature(x, y, feature_name):
    style = plot_styles.get(feature_name, {})
    plt.plot(x, y, **style)
    
def plot_features(hmin,hmax,dat,linestyle,c,label,alpha=1):
    hour = np.array(range(hmin,hmax))
    for feat in dat.features_list:
        i = dat.all_features_list.index(feat) # index of main data
        if feat in plot_styles.keys():
            plot_feature(x=hour, y=dat['X'][:,i][hmin:hmax], feature_name=feat)
        
def plot_data(dat, plot_period='all', create_figure=False,title=None,title2=None,hmin=0,hmax=None,xlabel=None,ylabel=None):
    # Plot fmda dictionary of data and model if present
    # Inputs:
    # dat: FMDA dictionary
    # inverse_scale: logical, whether to inverse scale data
    # Returns: none

    # dat = copy.deepcopy(dat0)
    
    if 'hours' in dat:
        if hmax is None:
            hmax = dat['hours']
        else:
            hmax = min(hmax, dat['hours'])        
        if plot_period == "all":
            pass
        elif plot_period == "predict":
            assert "test_ind" in dat.keys()
            hmin = dat['test_ind']

        else: 
            raise ValueError(f"unrecognized time period for plotting plot_period: {plot_period}")
    
    
    if create_figure:
        plt.figure(figsize=(16,4))

    plot_one(hmin,hmax,dat,'y',linestyle='-',c='#468a29',label='FM Observed')
    plot_one(hmin,hmax,dat,'m',linestyle='-',c='k',label='FM Model')
    plot_features(hmin,hmax,dat,linestyle='-',c='k',label='FM Model')
    

    if 'test_ind' in dat.keys():
        test_ind = dat["test_ind"]
    else:
        test_ind = None
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Note: the code within the tildes here makes a more complex, annotated plot
    if (test_ind is not None) and ('m' in dat.keys()):
        plt.axvline(test_ind, linestyle=':', c='k', alpha=.8)
        yy = plt.ylim() # used to format annotations
        plot_y0 = np.max([hmin, test_ind]) # Used to format annotations
        plot_y1 = np.min([hmin, test_ind])
        plt.annotate('', xy=(hmin, yy[0]),xytext=(plot_y0,yy[0]),  
                arrowprops=dict(arrowstyle='<-', linewidth=2),
                annotation_clip=False)
        plt.annotate('(Training)',xy=((hmin+plot_y0)/2,yy[1]),xytext=((hmin+plot_y0)/2,yy[1]+1), ha = 'right',
                annotation_clip=False, alpha=.8)
        plt.annotate('', xy=(plot_y0, yy[0]),xytext=(hmax,yy[0]),                  
                arrowprops=dict(arrowstyle='<-', linewidth=2),
                annotation_clip=False)
        plt.annotate('(Forecast)',xy=(hmax-(hmax-test_ind)/2,yy[1]),
                     xytext=(hmax-(hmax-test_ind)/2,yy[1]+1),
                annotation_clip=False, alpha=.8)
     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    if title is not None:
        t = title
    elif 'title' in dat:
        t=dat['title']
        # print('title',type(t),t)
    else:
        t=''
    if title2 is not None:
        t = t + ' ' + title2 
    t = t + ' (' + rmse_data_str(dat)+')'
    if plot_period == "predict":
        t = t + " - Forecast Period"
    plt.title(t, y=1.1)
    
    if xlabel is None:
        plt.xlabel('Time (hours)')
    else:
        plt.xlabel(xlabel)
    if 'rain' in dat:
        plt.ylabel('FM (%) / Rain (mm/h)')
    elif ylabel is None:
        plt.ylabel('Fuel moisture content (%)')
    else:
        plt.ylabel(ylabel)
    plt.legend(loc="upper left")
    
def rmse(a, b):
    return np.sqrt(mean_squared_error(a.flatten(), b.flatten()))

def rmse_skip_nan(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.count_nonzero(mask):
        return np.sqrt(np.mean((x[mask] - y[mask]) ** 2))
    else:
        return np.nan
    
def rmse_str(a,b):
    rmse = rmse_skip_nan(a,b)
    return "RMSE " + "{:.3f}".format(rmse)

def rmse_data_str(dat, predict=True, hours = None, test_ind = None):
    # Return RMSE for model object in formatted string. Used within plotting
    # Inputs:
    # dat: (dict) fmda dictionary 
    # predict: (bool) Whether to return prediction period RMSE. Default True 
    # hours: (int) total number of modeled time periods
    # test_ind: (int) start of test period
    # Return: (str) RMSE value
    
    if hours is None:
        if 'hours' in dat:
            hours = dat['hours']               
    if test_ind is None:
        if 'test_ind' in dat:
            test_ind = dat['test_ind']
    
    if 'm' in dat and 'y' in dat:
        if predict and hours is not None and test_ind is not None:
            return rmse_str(dat['m'][test_ind:hours],dat['y'].flatten()[test_ind:hours])
        else: 
            return rmse_str(dat['m'],dat['y'].flatten())
    else:
        return ''
                    
    
# Calculate mean absolute error
def mape(a, b):
    return ((a - b).__abs__()).mean()
    
def rmse_data(dat, hours = None, h2 = None, simulation='m', measurements='fm'):
    if hours is None:
        hours = dat['hours']
    if h2 is None:
        h2 = dat['h2']
    
    m = dat[simulation]
    fm = dat[measurements]
    case = dat['case']
    
    train =rmse(m[:h2], fm[:h2])
    predict = rmse(m[h2:hours], fm[h2:hours])
    all = rmse(m[:hours], fm[:hours])
    print(case,'Training 1 to',h2,'hours RMSE:   ' + str(np.round(train, 4)))
    print(case,'Prediction',h2+1,'to',hours,'hours RMSE: ' + str(np.round(predict, 4)))
    print(f"All predictions hash: {hash2(m)}")
    
    return {'train':train, 'predict':predict, 'all':all}


    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def get_file(filename, data_dir='data'):
    # Check for file locally, retrieve with wget if not
    if osp.exists(osp.join(data_dir, filename)):
        print(f"File {osp.join(data_dir, filename)} exists locally")        
    elif not osp.exists(filename):
        import subprocess
        base_url = "https://demo.openwfm.org/web/data/fmda/dicts/"
        print(f"Retrieving data {osp.join(base_url, filename)}")
        subprocess.call(f"wget -P {data_dir} {osp.join(base_url, filename)}", shell=True)

