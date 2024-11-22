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
from data_funcs import rmse, to_json, combine_nested
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
    code_start_time = time.time()
    
    # Read the JSON config file
    filename = "rnn_model_config.json"
    with open(filename, "r") as json_file:
        config = json.load(json_file)    
    print(json.dumps(config, indent=4))

    # Read Params
    params = read_yml(config["model_params_path"], subkey="rnn")
    params = RNNParams(params) # creates custom params class that runs checks, auto generates some fields, and constrains certain output
    params.update({'plot_history': False}) # Turn off plotting since this is intended to be called from command line


    # Read RNNData and Train Model
    train_path = osp.join(config.get("model_output_path"), config.get("training_data_filename"))
    rnn_dat = read_pkl(train_path)
    reproducibility.set_seed()
    rnn = RNN(params)
    m, errs = rnn.run_model(rnn_dat)
    print(f"Mean Prediction RMSE: {errs.mean()}")

    # Save Model and training data object
    model_filename = config.get("model_output_filename")
    outpath = config.get("model_output_path")
    print(f"Saving trained prediction model to {osp.join(outpath, model_filename)}")
    rnn.model_predict.save(osp.join(outpath, model_filename)) # save prediction model only


    # End Timer
    code_end_time = time.time()
    code_elapsed_time = code_end_time - code_start_time
        
    if config.get("time_code"):
        print(f"Model Training Elapsed time: {code_elapsed_time:.4f} seconds")
    