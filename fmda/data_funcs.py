## Set of Functions to process and format fuel moisture model inputs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Dependencies
import numpy as np

## RAWS Data Functions

def format_raws(stn):
    # Extract Data
    raws_dat = {
        'time' : np.array(stn['OBSERVATIONS']["date_time"]),
        'temp' : np.array(stn['OBSERVATIONS']["air_temp_set_1"], dtype = 'float64')+273.15,
        'rh' : np.array(stn['OBSERVATIONS']["relative_humidity_set_1"], dtype = 'float64'),
        'fm' : np.array(stn['OBSERVATIONS']["fuel_moisture_set_1"], dtype = 'float64'),
        'rain' : format_precip(stn['OBSERVATIONS']["precip_accum_set_1"])
    }
    
    # Calculate Equilibrium Temps
    raws_dat['Ed'] = 0.924*raws_dat['rh']**0.679 + 0.000499*np.exp(0.1*raws_dat['rh']) + 0.18*(21.1 + 273.15 - raws_dat['temp'])*(1 - np.exp(-0.115*raws_dat['rh']))
    raws_dat['Ew'] = 0.618*raws_dat['rh']**0.753 + 0.000454*np.exp(0.1*raws_dat['rh']) + 0.18*(21.1 + 273.15 - raws_dat['temp'])*(1 - np.exp(-0.115*raws_dat['rh']))
    
    # Fix nan values
    raws_dat['rain']=fixnan(raws_dat['rain'],2)
    raws_dat['temp']=fixnan(raws_dat['temp'],2)
    raws_dat['rh']=fixnan(raws_dat['rh'],2)
    raws_dat['fm']=fixnan(raws_dat['fm'],2)
    raws_dat['Ed']=fixnan(raws_dat['Ed'],2)
    raws_dat['Ew']=fixnan(raws_dat['Ew'],2)
    
    return raws_dat

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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
