import sys
import logging
from utils import print_dict_summary, print_first, str2time, check_increment, time_intp
import pickle
import os.path as osp
import pandas as pd
import numpy as np

def pkl2train(input_file_paths,output_file_path='train.pkl',forecast_step=1):
    # in:
    #   file_path       list of strings - files as in read_test_pkl
    #   forecast_step   string - which forecast step to take atmospheric data from (maybe 03, must be >0). 
    #   fprev   strinh - the previous (
    # return:
    #   train          dictionary with structure
    #                  {key : {'key' : key,    # copied subdict key
    #                          'loc' : {...},  # copied from in dict = {key : {'loc': ... }...}
    #                         'time' : time,   # datetime vector, spacing tres
    #                            'X' : fm      # target fuel moisture from the RAWS, interpolated to time
    #                            'Y' : feat    # features from atmosphere and location
    #                            
    #

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
                time_hrrr=str2time(subdict['HRRR']['time'])
                timesteps=len(time_hrrr)
                if check_increment(time_hrrr,id=key+' HRRR.time') < 1:
                    raise(ValueError)
                # build matrix of features - assuming all the same length, if not column_stack will fail
                train[key]['time']=time_hrrr
                columns=[]
                # location as features constant in time come first
                columns.append(np.full(timesteps,loc['elev']))  
                columns.append(np.full(timesteps,loc['lon']))
                columns.append(np.full(timesteps,loc['lat']))
                for i in ["rh","wind","solar","soilm","groundflux","Ed","Ew"]:
                    columns.append(subdict['HRRR'][fstep][i])   # add variables from HRRR forecast steps 
                # compute rain as difference of accumulated precipitation
                rain = subdict['HRRR'][fstep]['precip_accum']- subdict['HRRR'][fprev]['precip_accum']
                logging.info('%s rain as difference %s minus %s: min %s max %s',key,fstep,fprev,np.min(rain),np.max(rain))
                columns.append( rain ) # add rain feature
                train[key]['X'] = np.column_stack(columns)
                logging.info(f"Created feature matrix train[{key}]['X'] shape {train[key]['X'].shape}")
                time_raws=str2time(subdict['RAWS']['time_raws']) # may not be the same as HRRR
                logging.info('%s RAWS.time_raws length is %s',key,len(time_raws))
                check_increment(time_raws,id=key+' RAWS.time_raws')
                # print_first(time_raws,num=5,id='RAWS.time_raws')
                fm=subdict['RAWS']['fm']
                logging.info('%s RAWS.fm length is %s',key,len(fm))
                # interpolate RAWS sensors to HRRR time and over NaNs
                train[key]['Y'] = time_intp(time_raws,fm,time_hrrr)
                if  train[key]['Y'] is None:
                    logging.error('Cannot create target matrix for %s, using None',key)
                else:
                    logging.info(f"Created target matrix train[{key}]['Y'] shape {train[key]['Y'].shape}")
    
    logging.info('Created a "train" dictionary with %s items',len(train))
 
    
    # clean up
    
    keys_to_delete = []
    for key in train:
        if train[key]['X'] is None or train[key]['Y'] is None:
            logging.warning('Deleting training item %s because features X or target Y are None', key)
            keys_to_delete.append(key)

    # Delete the items from the dictionary
    if len(keys_to_delete)>0:
        for key in keys_to_delete:
            del train[key]       
        logging.warning('Deleted %s items with None for data. %s items remain in the training dictionary.',
                        len(keys_to_delete),len(train))
        
    # output

    if output_file_path is not None:
        with open(output_file_path, 'wb') as file:
            logging.info('Writing pickle dump of the dictionary train into file %s',output_file_path)
            pickle.dump(train, file)
    
    logging.info('pkl2train done')
    
    return train