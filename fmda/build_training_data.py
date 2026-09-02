# Module used to initiate model training and save a trained prediction model, given user input config file


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
from data_funcs import rmse, to_json, combine_nested, read_and_clean
from moisture_models import run_augmented_kf, XGB
import copy
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
import json


# Executed Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    print("Building training dataset using config file rnn_model_config.json")

    # Start timer for code 
    code_start_time = time.time()
    
    # Read the JSON config file
    filename = "rnn_model_config.json"
    with open(filename, "r") as json_file:
        config = json.load(json_file)    
    print(json.dumps(config, indent=4))

    # Read Params
    params = read_yml(config["model_params_path"], subkey="rnn")
    params = RNNParams(params) # creates custom params class that runs checks, auto generates some fields, and constrains certain output
    params_data = read_yml(config["data_params_path"])

    # Process training dictionary
    file_paths = [config['input_dict_path']]
    train = read_and_clean(file_paths, atm_source="HRRR", params_data = params_data, verbose=True,forecast_step = 3)

    outpath = config.get("model_output_path")
    data_filename = config.get('training_data_filename')
    print(f"Saving model training data to {osp.join(outpath, data_filename)}")
    with open(f"{osp.join(outpath, data_filename)}", 'wb') as file:
        pickle.dump(train, file)

    # End Timer
    code_end_time = time.time()
    code_elapsed_time = code_end_time - code_start_time
        
    if config.get("time_code"):
        print(f"Data Processing Elapsed time: {code_elapsed_time:.4f} seconds")
    