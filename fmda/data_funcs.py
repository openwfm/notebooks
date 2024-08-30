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
                raise ValueError(f"Missing expected key: '{key}'{f' or subkey: {subkey}' if subkey else ''} in one of the input dictionaries")
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

