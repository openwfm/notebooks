## Set of functions to build a time series of FMC observations at list of lat/lons
## HRRR data source from AWS
## Functions assume hourly data

import os
import subprocess

def download_grib(source_url,time,model,dest_dir="",fmt = "%Y-%m-%d %H:%M",sector="conus",forecast_hour = 0  ):
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
    
def gather_hrrr_time_range(start, end, 
                           source_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.",
                           model = "t18z.wrfsubhf0015",
                           dest_dir = "data",
                           fmt = "%Y-%m-%d %H:%M"):
    # ----------------------------------------------------------
    ## start: starting time
    ## end: ending time
    ## source_url: starting url string that specifies grib data source
    ## time: time slice to gather data
    ## model: particular grib file to get from time slice
    ## dest_dir: destination directory, send grib file to this location
    ## fmt: date format string
    # -------------------------------------------------------
    
    
    # Create a range of dates given start and end
    # Calculate time diff in hours
    time1 = datetime.strptime(end_time, fmt)
    time2 = datetime.strptime(start_time, fmt)
    days_diff = time1-time2
    times = (days_diff.days+1)*24
    dates = pd.date_range(start=time1,periods=times, freq="1H") # Series of dates
    
    dates[0].hour
    
    download_grib(
        source_url = source_url,
        date = date,
        model = model,
        dest_dir =  dest_dir # destination subdirectory for url content
    )
    