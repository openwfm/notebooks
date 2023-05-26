## Set of functions to build a time series of FMC observations at list of lat/lons
## HRRR data source from AWS
## Functions assume hourly data

import os
import subprocess
import pandas as pd
import numpy as np
import xarray as xr
from datetime import date, timedelta, datetime

def download_grib(time,model,source_url="https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.",dest_dir="",fmt = "%Y-%m-%d %H:%M",sector="conus",forecast_hour = 0  ):
    # source_url: starting url string that specifies grib data source
    # time: time of time slice to gather data
    # model: particular grib file to get from time slice
    # dest_dir: destination subdirectory, send grib file to this location
    # fmt: date format string
    # sector: either continental us (conus) or alaska
    # forecast_hour: offset from cycle time
    
    ## Utility function, lightweight so defined within this func
    ## source: chat-gpt4
    def check_file_exists(file_path):
        if os.path.exists(file_path):
            print(f"The file '{file_path}' was successfully downloaded.")
        else:
            raise AssertionError(f"The file '{file_path}' does not exist.")
            
    source_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
    # sector = "conus"
    # start_time = "2022-06-01 00:00"
    fmt = "%Y-%m-%d %H:%M"
    time1 = datetime.strptime(time, fmt)
    day_date=time1.strftime("%Y%m%d")
    cycle = time1.hour          # hour
    product = "wrfsfcf" # 2D surface levels

    # Put it all together
    file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
    grib_url=source_url+"/hrrr."+day_date+"/conus/"+file_path
    
    
    ## Construct full destination file with path
    # time_date = datetime.strptime(time, fmt)
    # day_date = time_date.strftime("%Y%m%d")
    # hour=time_date.strftime("%H")
    # mins=time_date.strftime("%M")
    dest_file=dest_dir+"/hrrr_"+day_date+str(cycle)+".grib2"

    # grib_url=source_url+day_date+"/conus/hrrr.t"+hour+"z."+model+hour+".grib2"
    
    command = " ".join(["wget",grib_url, "-O",dest_file])
    
    print(command)
    
    subprocess.call(command,shell=True)
    
    # Assert T that file exists
    check_file_exists(dest_file)
    
    return dest_file, grib_url
    



def extract_2m_vars(ds, coord, convert_EW=True):
    ## Get variables from the 2m above ground HRRR layer
    ## Vars include: temp (k), RH
    # ds: xarray object from HRRR
    # coord: tuple of the form (lon, lat)
    lon = coord[0]
    if ds.longitude.attrs['units']=="degrees_east" and convert_EW:
        lon = 360 + lon
        # print('Converting target longitude to Deg. E')
    
    lat = coord[1]
    
    abslat = np.abs(ds.latitude-lat)
    abslon = np.abs(ds.longitude-lon)
    c = np.maximum(abslon, abslat)

    ([xloc], [yloc]) = np.where(c == np.min(c))

    # use that index location to get the values, 
    # NOTE: HRRR requires reorder (y,x)
    point_ds = ds.sel(x=yloc, y=xloc)
    
    return point_ds
    
    
    
    
def gather_hrrr_time_range(start, end, pts,
                           source_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.",
                           model = "t18z.wrfsubhf0015",
                           dest_dir = "data",
                           fmt = "%Y-%m-%d %H:%M"):
    # ----------------------------------------------------------
    ## start: starting time
    ## end: ending time
    ## pts: list of tuples of points to built time-series at
    ## source_url: starting url string that specifies grib data source
    ## time: time slice to gather data
    ## model: particular grib file to get from time slice
    ## dest_dir: destination directory, send grib file to this location
    ## fmt: date format string
    # -------------------------------------------------------

    # Handle Dates
    time1 = datetime.strptime(start, fmt)
    time2 = datetime.strptime(end, fmt)
    dates = pd.date_range(start=time1,end=time2, freq="1H") # Series of dates in 1 hour increments
    
    # Initialize matrix of time series obs
    # (x, y, z) dims == (ntime, ncoords, nvars)
    # vars=["rh", "temp", "solar1", "solar2"]
    # nvars=len(vars)
    ## TEMPORARILY just collecting 2 fields: temp and RH
    hrrr_dat = np.zeros((dates.shape[0], len(pts), 2))
    
    
    for t in range(0, dates.shape[0]):
        # Format time
        time=dates[t].strftime("%Y-%m-%d %H:%M")
        print('Time '+str(t)+', '+str(time))
    
        # Temporarily download grib file at given time
        tempfile, url = download_grib(
            source_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
            time = time,
            model = "wrfsfcf",
            dest_dir =  "data" # destination subdirectory for url content
        )

        # Read grib file
        ds=xr.open_dataset(
            tempfile,
            filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2}
        )

        # Extract Data from grib file for each point
        for i in range(0, len(pts)):
            point_ds = extract_2m_vars(ds, pts[i])
            hrrr_dat[t][i][0] = point_ds.t2m.values
            hrrr_dat[t][i][1] = point_ds.r2.values

        os.remove(tempfile)
    
    
    return hrrr_dat