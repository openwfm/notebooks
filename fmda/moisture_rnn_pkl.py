import sys
import logging
from utils import print_dict_summary, print_first, str2time, check_increment, time_intp, hash2
import pickle
import os.path as osp
import pandas as pd
import numpy as np
import reproducibility
# from moisture_rnn import create_rnn_data_2, train_rnn, rnn_predict
from data_funcs import plot_data,rmse_data
import matplotlib.pyplot as plt
import sys
import yaml
import os

feature_types = {
    # Static features are based on physical location, e.g. location of RAWS site
    'static': ['elev', 'lon', 'lat'],
    # Atmospheric weather features come from either RAWS subdict or HRRR
    'atm': ['temp', 'rh', 'wind', 'solar', 'soilm', 'canopyw', 'groundflux', 'Ed', 'Ew']
}


def pkl2train(input_file_paths,
              forecast_step=1, atm="HRRR",features_all=['Ed', 'Ew', 'solar', 'wind', 'elev', 'lon', 'lat', 'rain']):
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
        logging.info('Using data from step %s',fstep)
        logging.info('Using rain as the difference of accumulated precipitation between %s and %s',fstep,fprev)
    else:
        logging.critical('forecast_step must be integer between 1 and 99')
        raise ValueError('bad forecast_step')
        
    train = {}
    for file_path in input_file_paths:
        with open(file_path, 'rb') as file:
            logging.info("loading file %s", file_path)
            d = pickle.load(file)
        for key in d:
            atm_dict = atm
            features_list = features_all
            logging.info('Processing subdictionary %s',key)
            if key in train:
                logging.warning('skipping duplicate key %s',key)
            else:
                subdict=d[key]    # subdictionary for this case
                loc=subdict['loc']
                train[key] = {
                'id': key,  # store the key inside the dictionary, subdictionary will be used separatedly
                'case':key,
                'filename': file_path,
                'loc': loc
                }
                desc='descr'
                if desc in subdict:
                    train[desc]=subdict[desc]
                time_hrrr=str2time(subdict[atm_dict]['time'])
                # timekeeping
                timesteps=len(d[key][atm_dict]['time'])
                hours=timesteps
                train[key]['hours']=hours
                train[key]['h2']   =hours     # not doing prediction yet    
                hrrr_increment = check_increment(time_hrrr,id=key+f' {atm_dict}.time')
                logging.info(f'{atm_dict} increment is %s h',hrrr_increment)
                if  hrrr_increment < 1:
                    logging.critical('HRRR increment is %s h must be at least 1 h',hrrr_increment)
                    raise(ValueError)
                
                # build matrix of features - assuming all the same length, if not column_stack will fail
                train[key]['time']=time_hrrr

                # TODO: REMOVE THIS
                scale_fm = 1
                train[key]["scale_fm"] = scale_fm
                # Set up static vars, but not for repro case
                columns=[]

                for feat in features_list:
                    # For atmospheric features,
                    if feat in feature_types['atm']:
                        if atm_dict == "HRRR":
                            vec = subdict[atm_dict][fstep][feat]
                        elif atm_dict == "RAWS":
                            vec = subdict[atm_dict][feat]
                        if feat in ['Ed', 'Ew']:
                            vec = vec / scale_fm
                        columns.append(vec)
                    
                    # For static features, repeat to fit number of time observations
                    elif feat in feature_types['static']:
                        columns.append(np.full(timesteps,loc[feat]))
                
                # compute rain as difference of accumulated precipitation
                if 'rain' in features_list:
                    if atm_dict == "HRRR":
                        rain = subdict[atm_dict][fstep]['precip_accum']- subdict[atm_dict][fprev]['precip_accum']
                        logging.info('%s rain as difference %s minus %s: min %s max %s',
                                 key,fstep,fprev,np.min(rain),np.max(rain))
                    elif atm_dict == "RAWS":
                        if 'rain' in subdict[atm_dict]:
                            rain = subdict[atm_dict]['rain']
                        else:
                            logging.info('No rain data found in RAWS subdictionary %s', key)
                    columns.append( rain ) # add rain feature             
                train[key]['X'] = np.column_stack(columns)
                train[key]['features_list'] = features_list
                
                logging.info(f"Created feature matrix train[{key}]['X'] shape {train[key]['X'].shape}")
                time_raws=str2time(subdict['RAWS']['time_raws']) # may not be the same as HRRR
                logging.info('%s RAWS.time_raws length is %s',key,len(time_raws))
                check_increment(time_raws,id=key+' RAWS.time_raws')
                # print_first(time_raws,num=5,id='RAWS.time_raws')
                fm=subdict['RAWS']['fm']
                logging.info('%s RAWS.fm length is %s',key,len(fm))
                # interpolate RAWS sensors to HRRR time and over NaNs
                train[key]['y'] = time_intp(time_raws,fm,time_hrrr) / scale_fm
                # TODO: check endpoint interpolation when RAWS data sparse, and bail out if not enough data
                
                if  train[key]['y'] is None:
                    logging.error('Cannot create target matrix for %s, using None',key)
                else:
                    logging.info(f"Created target matrix train[{key}]['y'] shape {train[key]['y'].shape}")
    
    logging.info('Created a "train" dictionary with %s items',len(train))
 
    # clean up
    
    keys_to_delete = []
    for key in train:
        if train[key]['X'] is None or train[key]['y'] is None:
            logging.warning('Deleting training item %s because features X or target Y are None', key)
            keys_to_delete.append(key)

    # Delete the items from the dictionary
    if len(keys_to_delete)>0:
        for key in keys_to_delete:
            del train[key]       
        logging.warning('Deleted %s items with None for data. %s items remain in the training dictionary.',
                        len(keys_to_delete),len(train))
        
    # output

    # if output_file_path is not None:
    #     with open(output_file_path, 'wb') as file:
    #         logging.info('Writing pickle dump of the dictionary train into file %s',output_file_path)
    #         pickle.dump(train, file)
    
    logging.info('pkl2train done')
    
    return train

def run_rnn_pkl(case_data,params, title2=None):
    # analogous to run_rnn after the create_rnn_data_1 stage
    # instead, after pkl2train
    # Inputs:
    # case_data: (dict) one case train[case] after pkl2train()
    #    also plays the role of rnn_dat after create_rnn_data_1
    # title2: (str) string to add to plot titles
    # called from: top level
    
    logging.info('run_rnn start')
    verbose = params['verbose']
    
    if title2 is None:
        title2=case_data['id']
    
    reproducibility.set_seed() # Set seed for reproducibility
    
    print('case_data at entry to run_rnn_pkl')
    print_dict_summary(case_data)
    
    # add batched x_train, y_train
    create_rnn_data_2(case_data,params)  
  
    # train the rnn over period  create prediction model with optimized weights
    model_predict = train_rnn(
        case_data,
        params,
        case_data['hours']
    )

    m = rnn_predict(model_predict, params, case_data)
    case_data['m'] = m
    print(f"Model outputs hash: {hash2(m)}")

    # Plot data needs certain names
    # TODO: make plot_data specific to this context
    case_data.update({"fm": case_data["Y"]*case_data['scale_fm']})
    plot_data(case_data,title2=title2)
    plt.show()

    logging.info('run_rnn_pkl end')
    # Print and return Errors
    # return m, rmse_data(case_data)  # do not have a "measurements" field 
    return m, rmse_data(case_data)
