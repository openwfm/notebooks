## Set of Functions to process and format fuel moisture model inputs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np, random
from numpy.random import rand
from MesoPy import Meso
# import tensorflow as tf
import pickle, os

import matplotlib.pyplot as plt
from moisture_models import model_decay, model_moisture
from datetime import datetime, timedelta
from utils import  is_numeric_ndarray, hash2

import json

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
 
def to_json(dic,filename):
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
            
def plot_data(dat,title=None,title2=None,hmin=None,hmax=None):
    if 'hours' in dat:
        if hmax is None:
            hmax = dat['hours']
        else:
            hmax = min(hmax, dat['hours'])
    plt.figure(figsize=(16,4))
    plot_one(hmin,hmax,dat,'E',linestyle='--',c='r',label='EQ')
    plot_one(hmin,hmax,dat,'Ed',linestyle='--',c='#EF847C',label='drying EQ', alpha=.8)
    plot_one(hmin,hmax,dat,'Ew',linestyle='--',c='#7CCCEF',label='wetting EQ', alpha=.8)
    plot_one(hmin,hmax,dat,'fm',linestyle='-',c='#8BC084',label='FM Observed')
    plot_one(hmin,hmax,dat,'m',linestyle='-',c='k',label='FM Model')
    plot_one(hmin,hmax,dat,'Ec',linestyle='-',c='#8BC084',label='equilibrium correction')
    plot_one(hmin,hmax,dat,'rain',linestyle='-',c='b',label='Rain', alpha=.4)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.axvline(dat['h2'], linestyle=':', c='k', alpha=.8)
    
    plt.annotate('', xy=(0, -6),xytext=(dat['h2'],-6),                  
            arrowprops=dict(arrowstyle='<->'),
            annotation_clip=False)
    plt.annotate('Training',xy=(np.ceil(dat['h2']/2),-7),xytext=(np.ceil(dat['h2']/2),-7),
            annotation_clip=False)
    plt.annotate('', xy=(dat['h2'], -6),xytext=(dat['hours'],-6),                  
            arrowprops=dict(arrowstyle='<->'),
            annotation_clip=False)
    plt.annotate('Forecast',xy=(dat['h2']+np.ceil(dat['h2']/2),-7),xytext=(dat['h2']+np.ceil(dat['h2']/2),-7),
            annotation_clip=False)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    if title is not None:
        t = title
    else:
        t=dat['title']
        # print('title',type(t),t)
    if title2 is not None:
        t = t + ' ' + title2 
    t = t + ' (' + rmse_data_str(dat)+')'
    plt.title(t)
    plt.xlabel('Time (hours)')
    if 'rain' in dat:
        plt.ylabel('FM (%) / Rain (mm/h)')
    else:
        plt.ylabel('Fuel moisture content (%)')
    plt.legend()
    
# Calculate mean squared error
def rmse(a, b):
    return np.sqrt(((a - b)**2).mean())

def rmse_skip_nan(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.count_nonzero(mask):
        return np.sqrt(np.mean((x[mask] - y[mask]) ** 2))
    else:
        return np.nan
    
def rmse_str(a,b):
    rmse = rmse_skip_nan(a,b)
    return "RMSE " + "{:.2f}".format(rmse)

def rmse_data_str(data):
    if 'm' in data and 'fm' in data:
        return rmse_str(data['m'],data['fm'])
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
    
    return {'train':train, 'predict':predict, 'all':all}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## RAWS Data Functions

def format_raws(stn, fixnames = True):
    raws_dat = stn['OBSERVATIONS'].copy() # bug fix for in-place changing of dictionary outside of func call
    
    # Convert to Numpy arrays, check data type for floats
    for key in [*stn['OBSERVATIONS'].keys()]:
        if type(stn['OBSERVATIONS'][key][0]) is float:
            raws_dat[key] = np.array(stn['OBSERVATIONS'][key], dtype = 'float64')
        else:
            raws_dat[key] = np.array(stn['OBSERVATIONS'][key])
    
    # Transform Data
    raws_dat['air_temp_set_1'] = raws_dat['air_temp_set_1'] + 273.15 ## convert C to K
    if 'precip_accum_set_1' in raws_dat.keys():
        raws_dat['precip_accum_set_1'] = format_precip(raws_dat['precip_accum_set_1']) ## format precip data, accumulated to hourly
    
    
    # Calculate Equilibrium Temps
    raws_dat['Ed'] = 0.924*raws_dat['relative_humidity_set_1']**0.679 + 0.000499*np.exp(0.1*raws_dat['relative_humidity_set_1']) + 0.18*(21.1 + 273.15 - raws_dat['air_temp_set_1'])*(1 - np.exp(-0.115*raws_dat['relative_humidity_set_1']))
    raws_dat['Ew'] = 0.618*raws_dat['relative_humidity_set_1']**0.753 + 0.000454*np.exp(0.1*raws_dat['relative_humidity_set_1']) + 0.18*(21.1 + 273.15 - raws_dat['air_temp_set_1'])*(1 - np.exp(-0.115*raws_dat['relative_humidity_set_1']))
    
    # Fix nan values
    for key in [*raws_dat.keys()]:
        if type(raws_dat[key][0]) is float:
            raws_dat[key] = fixnan(raws_dat[key], 2)
    
    # Add station id
    raws_dat['STID'] = stn['STID']
    
    # Add lat/lon
    raws_dat['LATITUDE'] = stn['LATITUDE']
    raws_dat['LONGITUDE'] = stn['LONGITUDE']
    
    # Simplify names 
    if fixnames:
        var_mapping = {
            'date_time': 'time', 'precip_accum': 'rain', 'solar_radiation': 'solar',
            'fuel_moisture': 'fm', 'relative_humidity': 'rh',
            'air_temp': 'temp', 'Ed': 'Ed', 'Ew': 'Ew', 'STID': 'STID',
            'LONGITUDE': 'lon', 'LATITUDE': 'lat'
            }
        old_keys = [*raws_dat.keys()]
        old_keys = [k.replace("_set_1", "") for k in old_keys]
        new_keys = []
        for key in old_keys:
            new_keys.append(var_mapping.get(key, key))
        old_keys = [*raws_dat.keys()]
        old_keys = [k.replace("_set_1", "") for k in old_keys]
        new_keys = []
        for key in old_keys:
            new_keys.append(var_mapping.get(key, key))
        raws_dat2 = dict(zip(new_keys, list(raws_dat.values())))
        return raws_dat2
    
    else: return raws_dat

def format_rtma(rtma):
    td = np.array(rtma['td'])
    t2 = np.array(rtma['temp'])
    rain=np.array(rtma['precipa'])
    # compute relative humidity
    rh = 100*np.exp(17.625*243.04*(td - t2) / (243.04 + t2 - 273.15) / (243.0 + td - 273.15))
    Ed = 0.924*rh**0.679 + 0.000499*np.exp(0.1*rh) + 0.18*(21.1 + 273.15 - t2)*(1 - np.exp(-0.115*rh))
    Ew = 0.618*rh**0.753 + 0.000454*np.exp(0.1*rh) + 0.18*(21.1 + 273.15 - t2)*(1 - np.exp(-0.115*rh))

    rtma_dict = {
        'time': rtma['time_str'],
        'rain': format_precip(rtma['precipa']),
        'rh' : rh,
        'temp' : t2,
        'rh' : rh,
        'Ed' : Ed,
        'Ew' : Ew,
        'lat' : rtma['obs_lat'], 
        'lon' : rtma['obs_lon']
    }
    
    return rtma_dict

def format_precip(precipa):
    rain=np.array(precipa, dtype = 'float64')
    rain = np.diff(rain) # first difference to convert accumulated to hourly
    rain = np.insert(rain, 0, [np.NaN]) # add NaN entry to account for diff
    rain[rain > 1000] = np.NaN # filter out erroneously high
    rain[rain < 0] = np.NaN # filter out negative, results from diff function after precipa goes to zero
    return rain

def fixnan(a,n=99999): # size of gaps we can fill
    if a.ndim > 1:
        print('fixnan: input has',a.ndim,'dimensions, only one supported')
        raise ValueError
    for c in range(n):   
        # try fixing isolated nans first, replace by average
        for i in np.where(np.isnan(a))[0]:
            if i==0:  
                a[i] = a[i+1]
            elif i==len(a)-1:
                a[i] = a[i-1]
            elif not np.isnan(a[i-1]) and not np.isnan(a[i+1]):
                a[i] = 0.5*(a[i-1]+a[i+1])
            elif not np.isnan(a[i-1]):
                a[i] = a[i-1]
            elif not np.isnan(a[i+1]):
                a[i] = a[i+1]
        if not any(np.isnan(a)):
            break
    if any(np.isnan(a)):
        a = np.nan_to_num(a, nan=0.0)

def retrieve_raws(mes, stid, raws_vars, time1, time2):
    meso_ts = mes.timeseries(time1, time2, 
                       stid=stid, vars=raws_vars)
    station = meso_ts['STATION'][0]
    
    raws_dat = format_raws(station)
    
    return station, raws_dat
    
def raws_data(start=None, hours=None, h2=None, stid=None,meso_token=None):
    # input:
    #   start YYYYMMDDhhmm
    #   hours legth of the period
    #   h2 (optional) length of the training period
    #   stid  the station id
    time_start=start
    time_end = datetime.strptime(start, "%Y%m%d%H%M") + timedelta(hours = hours+1) # end time, plus a buffer to control for time shift
    time_end = str(int(time_end.strftime("%Y%m%d%H%M")))
    print('data_raws: Time Parameters:')
    print('-'*50)
    print('Time Start:', datetime.strptime(time_start, "%Y%m%d%H%M").strftime("%Y/%M/%d %H:%M"))
    print('Time End:', datetime.strptime(time_end, "%Y%m%d%H%M").strftime("%Y/%M/%d %H:%M"))
    print('Total Runtime:', hours, 'hours')
    print('Training Time:', h2, 'hours')
    print('-'*50)
    raws_vars='air_temp,relative_humidity,precip_accum,fuel_moisture'
    m=Meso(meso_token)
    station, raws_dat = retrieve_raws(m, stid, raws_vars, time_start, time_end)
    raws_dat['title']='RAWS data station ' + stid
    raws_dat.update({'hours':hours,'h2':h2,'station':station})
    print('Data Read:')
    print('-'*50)
    print('Station ID:', station['STID'])
    print('Lat / Lon:', station['LATITUDE'],', ',station['LONGITUDE'])
    if(station['QC_FLAGGED']): print('WARNING: station flagged for QC')
    print('-'*50)
    raws_dat.update({'hours':hours,'h2':h2})
    return raws_dat
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_and_fix_data(filename):
    with open(filename, 'rb') as handle:
        test_dict = pickle.load(handle)
        for case in test_dict:
            test_dict[case]['case'] = case
            test_dict[case]['filename'] = filename
            for key in test_dict[case].keys():
                var = test_dict[case][key]    # pointer to test_dict[case][key]
                if isinstance(var,np.ndarray) and (var.dtype.kind == 'f'):
                    nans = np.sum(np.isnan(var))
                    if nans:
                        print('WARNING: case',case,'variable',key,'shape',var.shape,'has',nans,'nan values, fixing')
                        fixnan(var)
                        nans = np.sum(np.isnan(test_dict[case][key]))
                        print('After fixing, remained',nans,'nan values')
    return test_dict