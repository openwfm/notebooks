## FMDA Readme

Code repository for project to build RNN models to predict fuel moisture content

### Setup

Clone repository:
	git clone https://github.com/openwfm/notebooks

Build and activate environment
	fmda/install/....yml
	conda activate tf-211


### Data

The data structure used for this project is nested dictionaries. A "case" of data consists of FMC observations and atmospheric data at a particular location. The FMC data always comes from RAWS ground-level observations, and the atmospheric data can be from a variety of sources. The data acquisition for this project is built off a branch of `wrfxpy`, branch "develop-72-jh", which merges RAWS and HRRR data

- Saved within the repository is a dataset used to ensure reproducibility. As of 20-6-2024, that file is: `data\reproducibility_dict2.pickle`
- Other formatted FMDA dictionaries are staged at the OpenWFM demo page and retrieved via wget:
https://demo.openwfm.org/web/data/fmda/dicts/

### RNN Code

The source code lives in the module "moisture_rnn.py". The module has:

* Superclass RNNModel 
	- Child class for SimpleRNN
	- Child class for LSTM
* Helper code to format data for RNN
	- staircase_2 used to batch/sequence timeseries data

### Notebooks

* version_control/rnn_train_versions.ipynb
	- Shows reproducibility changes and major code restructuring
* rnn_class.ipynb
	- Demonstrates RNN class functionality, different params and configurations
* fmda_rnn_serial.ipynb
	- Trains and predicts the model at multiple locations
