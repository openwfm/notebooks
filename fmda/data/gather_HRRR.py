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
    



def extract_hrrr(ds, coord, convert_EW=True):
    ## Get variables from the given HRRR layer
    ## Vars include: temp (k), RH
    # ds: xarray object from HRRR, layer could be 2m, 10m, surface
    # coord: tuple of the form (lon, lat)
    # convert_EW: whether or not to convert longitude from E to W

    if ds.longitude.attrs['units']=="degrees_east" and convert_EW:
        coord = west_to_east(coord)
        # print('Converting target longitude to Deg. E')
    
    lon = coord[0][0]
    lat = coord[0][1]
    
    abslat = np.abs(ds.latitude-lat)
    abslon = np.abs(ds.longitude-lon)
    c = np.maximum(abslon, abslat)

    ([xloc], [yloc]) = np.where(c == np.min(c))

    # use that index location to get the values, 
    # NOTE: HRRR requires reorder (y,x)
    point_ds = ds.sel(x=yloc, y=xloc)
    
    return point_ds
    
def west_to_east(pts):
    # Convert longitude in list of tuples 
    # from deg W to deg E
    # pts: list of tuples of form (lon, lat)

    # If pts is just one lon/lat pair it is tuple type, treat differently
    if type(pts) is tuple:
        lon = pts[0]+360
        lat = pts[1]
        coords = [(lon, lat)]
    else:
        ## Extract list of lons and lats
        lons = list(map(lambda pt: pt[0], pts))
        lats = list(map(lambda pt: pt[1], pts))
        ## Convert deg west to deg east
        lons = list(map(lambda l: l+360,lons))

        ## Combine back into list of tuples
        coords = [(lons[i], lats[i]) for i in range(0, len(lons))]
    
    return coords    
    

    
def slice_hrrr(tempfile, vs):
    # Given Grib file location and df of variables,
    # slice layers 2m, 10m, and surface
    vars_2m=list(vs['HRRR Name'][vs['Layer'] == '2m'])
    vars_10m=list(vs['HRRR Name'][vs['Layer'] == '10m'])
    vars_surf=list(vs['HRRR Name'][vs['Layer'] == 'surface'])

    # Get 2m vars
    if len(vars_2m)>0:
        # Read grib file
        ds1=xr.open_dataset(
            tempfile,
            filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2}
        )

    # Get surface vars
    if len(vars_surf)>0:
        # Read grib file
        ds2=xr.open_dataset(
            tempfile,
            filter_by_keys={'typeOfLevel': 'surface', 'stepType': 'instant'}
        )
        # Add height above ground field
        ds2=ds2.assign_coords({'heightAboveGround': np.float64(0)})

    # Get 10m vars
    if len(vars_10m)>0:
        # Read grib file
        ds3=xr.open_dataset(
            tempfile,
            filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 10}
        )

    return ds1, ds2, ds3
    
    
    
    
    
    
    
    
    
    
def gather_hrrr_time_range(start, end, pts, vs,
                           source_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.",
                           model = "t18z.wrfsubhf0015",
                           dest_dir = "data",
                           fmt = "%Y-%m-%d %H:%M"):
    # ----------------------------------------------------------
    ## start: starting time
    ## end: ending time
    ## pts: list of tuples of points to built time-series at
    ## vs: pandas dataframe of variables to extract
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
    # ncoords=number of lon/lab coordinates supplied
    # nvars=vs.shape
    hrrr_dat = np.zeros((dates.shape[0], len(pts), vs.shape[0]))
    
    vars_2m=list(vs['HRRR Name'][vs['Layer'] == '2m'])
    vars_10m=list(vs['HRRR Name'][vs['Layer'] == '10m'])
    vars_surf=list(vs['HRRR Name'][vs['Layer'] == 'surface'])
    
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

        # Get 2m vars
        if len(vars_2m)>0:
            # Read grib file
            ds=xr.open_dataset(
                tempfile,
                filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2}
            )
            # Extract Data from grib file for each point
            for i in range(0, len(pts)):
                point_ds = extract_hrrr(ds, pts[i])
                for k in range(0, len(vars_2m)):
                    hrrr_dat[t][i][k] = point_ds.get(vars_2m[k]).values
            
            
        # Get surface vars
        if len(vars_surf)>0:
            # Read grib file
            ds=xr.open_dataset(
                tempfile,
                filter_by_keys={'typeOfLevel': 'surface', 'stepType': 'instant'}
            )
            # Extract Data from grib file for each point
            for i in range(0, len(pts)):
                point_ds = extract_hrrr(ds, pts[i])
                for k in range(0, len(vars_surf)):
                    hrrr_dat[t][i][k+len(vars_2m)] = point_ds.get(vars_surf[k]).values
                
        # Get 10m vars
        if len(vars_10m)>0:
            # Read grib file
            ds=xr.open_dataset(
                tempfile,
                filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 10}
            )
            # Extract Data from grib file for each point
            for i in range(0, len(pts)):
                point_ds = extract_hrrr(ds, pts[i])
                for k in range(0, len(vars_10m)):
                    hrrr_dat[t][i][k+len(vars_2m)+len(vars_surf)] = point_ds.get(vars_10m[k]).values


        os.remove(tempfile)
    
    
    return hrrr_dat



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    print('executing main code')
    
    #-------------------------------------------------------------
    ## Make into json config file
    hrrr_config = {
        'start_time': "2022-06-01 00:00",
        'end_time': "2022-06-01 02:00",
        'dest_dir': "" # optional subdir string, needs more tests
    }
    #-------------------------------------------------------------
    

    # Handle Dates
    fmt = "%Y-%m-%d %H:%M"
    time1 = datetime.strptime(hrrr_config["start_time"], fmt) # use Dict util to get . operator to match wrfxpy
    time2 = datetime.strptime(hrrr_config["end_time"], fmt) # use Dict util to get . operator to match wrfxpy
    dates = pd.date_range(start=time1,end=time2, freq="1H") # Series of dates in 1 hour increments
    
    for t in range(0, dates.shape[0]):
        # Format time
        time=dates[t].strftime("%Y-%m-%d %H:%M")
        print('Time '+str(t)+', '+str(time))

        # Format output subdirectory
        dest_dir = ''
        time_str = datetime.strptime(time,'%Y-%m-%d %H:%M').strftime("%Y-%m-%d_%H")
        os.makedirs(osp.join(dest_dir, time_str), exist_ok=True)

        # Temporarily download grib file at given time
        tempfile, url = download_grib(
            source_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
            time = time,
            model = "wrfsfcf",
            dest_dir = osp.join(dest_dir, time_str)
        )

        # Slice HRRR grib file
        ds1, ds2, ds3 = slice_hrrr(tempfile,vs)

        # Save Slices
        ds1.to_netcdf(osp.join(dest_dir, time_str, time_str +'-hrrr-2m.nc'))
        ds2.to_netcdf(osp.join(dest_dir, time_str, time_str +'-hrrr-surf.nc'))
        ds3.to_netcdf(osp.join(dest_dir, time_str, time_str +'-hrrr-10m.nc'))

        os.remove(tempfile)
    
    
    
    
    
    
    
    
    
    