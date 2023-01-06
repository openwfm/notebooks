## Set of Functions to process and format fuel moisture model inputs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np, random
from moisture_models import model_decay

# Helper Functions
verbose = False ## Must be declared in environment
def vprint(*args):
    if verbose: 
        for s in args[:(len(args)-1)]:
            print(s, end=' ')
        print(args[-1])

# Function to simulate moisture data and equilibrium for model testing
def create_synthetic_data(days=20,power=4,data_noise=0.02,process_noise=0.0,DeltaE=0.0):
    hours = days*24
    h2 = int(hours/2)
    hour = np.array(range(hours))
    day = np.array(range(hours))/24.

    # artificial equilibrium data
    E = np.power(np.sin(np.pi*day),4) # diurnal curve
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

def synthetic_data(days=20,power=4,data_noise=0.02,process_noise=0.0,DeltaE=0.0,Emin=0.5,Emax=0.3):
    hours = days*24
    h2 = int(hours/2)
    hour = np.array(range(hours))
    day = np.array(range(hours))/24.
    # artificial equilibrium data
    E = np.power(np.sin(np.pi*day),power) # diurnal curve
    E = Emin+(Emax - Emin)*E
    # FMC free run
    m_f = np.zeros(hours)
    m_f[0] = 0.1         # initial FMC
    # process_noise=0.
    for t in range(hours-1):
        m_f[t+1] = max(0.,model_decay(m_f[t],E[t])  + random.gauss(0,process_noise) )
    data = m_f + np.random.normal(loc=0,scale=data_noise,size=hours)
    E = E + DeltaE    
    Ed=E+1.0
    Ew=np.maximum(E-1.0,0)
    return {'E':E,'Ew':Ew,'Ed':Ed,'m_f':m_f,'hour':hour,'h2':h2,'DeltaE':DeltaE}



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## RAWS Data Functions

def format_raws(stn, fixnames = True):
    raws_dat = stn['OBSERVATIONS']
    
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
    
    # Simplify names 
    if fixnames:
        var_mapping = {
            'date_time': 'time', 'precip_accum': 'rain', 
            'fuel_moisture': 'fm', 'relative_humidity': 'rh',
            'air_temp': 'temp', 'Ed': 'Ed', 'Ew': 'Ew'
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

def format_precip(precipa):
    rain=np.array(precipa, dtype = 'float64')
    rain = np.diff(rain) # first difference to convert accumulated to hourly
    rain = np.insert(rain, 0, [np.NaN]) # add NaN entry to account for diff
    rain[rain > 1000] = np.NaN # filter out erroneously high
    rain[rain < 0] = np.NaN # filter out negative, results from diff function after precipa goes to zero
    return rain

# fix isolated nans
def fixnan(a,n):
    for c in range(n):
        for i in np.where(np.isnan(a)):
            a[i]=0.5*(a[i-1]+a[i+1])
        if not any(np.isnan(a)):
            break
    return a

def retrieve_raws(mes, stid, raws_vars, time1, time2):
    meso_ts = mes.timeseries(time1, time2, 
                       stid=stid, vars=raws_vars)
    station = meso_ts['STATION'][0]
    
    raws_dat = format_raws(station)
    
    return station, raws_dat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def check_data_array(dat,h,a,s):
    try:
        ar = dat[a]
    except:
        print('cannot find array ' + a)
        exit(1)
    print("array %s %s length %i min %s max %s\n" % (a,s,len(ar),min(ar),max(ar)))
    if len(ar) < h:
        print('Error: array length less than %i' % hours)
        exit(1)

def check_data(dat,h2,hours):
    check_data_array(dat,hours,'Ed','drying equilibrium (%)')
    check_data_array(dat,hours,'Ew','wetting equilibrium (%)')
    check_data_array(dat,hours,'rain','rain intensity (mm/h)')
    check_data_array(dat,hours,'fm','RAWS fuel moisture (%)')
