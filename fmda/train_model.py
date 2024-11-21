## Module mean to initiate model training and save a trained prediction model, given user input config file

import numpy as np
from utils import print_dict_summary, print_first, str2time, logging_setup
import pickle
import logging
import os.path as osp
from moisture_rnn_pkl import pkl2train
from moisture_rnn import RNNParams, RNNData, RNN, rnn_data_wrap
from utils import hash2, read_yml, read_pkl, retrieve_url, Dict
from moisture_rnn import RNN
import reproducibility
from data_funcs import rmse, to_json, combine_nested, build_train_dict
from moisture_models import run_augmented_kf, XGB
import copy
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
import json


# Executed Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    print("Training RNN model using config file rnn_model_config.json")

    # Start timer for code 
    code_time_start = time.time()
    
    # Read the JSON config file
    filename = "rnn_model_config.json"
    with open(filename, "r") as json_file:
        config = json.load(json_file)    
    print(json.dumps(config, indent=4))

    # Read Params
    params = read_yml(config["model_params_path"], subkey="rnn")
    params = RNNParams(params) # creates custom params class that runs checks, auto generates some fields, and constrains certain output
    params.update({'plot_history': False}) # Turn off plotting since this is intended to be called from command line
    params_data = read_yml(config["data_params_path"])

    # Process training dictionary
    file_paths = [config['fmda_dict_path']]
    train = build_train_dict(file_paths, atm_source="HRRR", params_data = params_data, spatial=False, verbose=True,forecast_step = 3)

    # Create RNNData and Train Model
    reproducibility.set_seed()
    rnn_dat = rnn_data_wrap(combine_nested(train), params) # wrapper for custom class that runs data scaling and batch reshaping 

    rnn = RNN(params)
    m, errs = rnn.run_model(rnn_dat)
    print(f"Mean Prediction RMSE: {errs.mean()}")

    # Save Model and training data object
    model_filename = config["model_output_filename"]
    data_filename = config["data_output_filename"]
    outpath = config["model_output_path"]
    print(f"Saving trained prediction model to {osp.join(outpath, model_filename)}")
    rnn.model_predict.save(osp.join(outpath, model_filename)) # save prediction model only
    print(f"Saving model training data to {osp.join(outpath, data_filename)}")
    with open(f"{osp.join(outpath, data_filename)}", 'wb') as file:
        pickle.dump(rnn_dat, file)


    