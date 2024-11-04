# Set of functions to handle spatial data read with xarray and deploy trained model to create predictions

import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
import re
from pyproj import Transformer
from datetime import datetime

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataframe used to organize HRRR metadata
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

band_df_hrrr = pd.DataFrame({
    'Band': [616, 620, 624, 628, 629, 661, 561, 612, 643],
    'hrrr_name': ['TMP', 'RH', "WIND", 'PRATE', 'APCP',
                  'DSWRF', 'SOILW', 'CNWAT', 'GFLUX'],
    'dict_name': ["temp", "rh", "wind", "rain", "precip_accum",
                 "solar", "soilm", "canopyw", "groundflux"],
    'descr': ['2m Temperature [K]', 
              '2m Relative Humidity [%]', 
              '10m Wind Speed [m/s]'
              'surface Precip. Rate [kg/m^2/s]',
              'surface Total Precipitation [kg/m^2]',
              'surface Downward Short-Wave Radiation Flux [W/m^2]',
              'surface Total Precipitation [kg/m^2]',
              '0.0m below ground Volumetric Soil Moisture Content [Fraction]',
              'Plant Canopy Surface Water [kg/m^2]',
              'surface Ground Heat Flux [W/m^2]']
})

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OS-type functions for handling paths and such
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def bands_to_names(bands):
    # Get the bands from band list, assumes band_df_hrrr exists in memory
    # Find matching dict_name values in the dataframe
    dict_names = [
        band_df_hrrr.loc[band_df_hrrr['Band'] == band, 'dict_name'].iloc[0]
            for band in bands if band in band_df_hrrr['Band'].values
    ]
    return dict_names

def names_to_bands(names):
    # Get the names from file names, assumes band_df_hrrr exists in memory
    # Find matching band values in the dataframe
    bands = band_df_hrrr.loc[band_df_hrrr['dict_name'].isin(names), 'Band'].tolist()

    # Check if equilibria were part of names and get rh and temp
    if any(name in names for name in ["Ed", "Ew"]):
        bands.extend([616, 620])
    
    return bands  

def extract_timestamp(file_path):
    # Extract date (parent directory) and hour from the file path
    date_str = re.search(r'(\d{8})', file_path).group(1)  # Matches YYYYMMDD
    hour_str = re.search(r't(\d{2})z', file_path).group(1)  # Matches tHHz
    
    # Combine into a datetime object
    timestamp = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")
    return timestamp

def get_file_list(start_time, end_time, fstep, bands_list, base_path = "."):
    # Set up time
    t0 = datetime.strptime(str(start_time), "%Y%m%d%H")
    t1 = datetime.strptime(str(end_time), "%Y%m%d%H")  
    assert t1 > t0, "end_time must be after start_time"
    times = pd.date_range(start=t0,end=t1, freq="1H")

    # Generate File list based on saved HRRR band format
    file_list = []
    for time in times:
        doy_str = time.strftime("%Y%m%d")
        hr = time.strftime("%H")
        files = [f"{doy_str}/hrrr.t{hr}z.wrfprs{fstep}.{band}.tif" for band in bands_list]
        file_list.append(files) # NOTE: appending to make list nested for data reading with xarray
    
    return file_list

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computational Functions used to transform xarray objects
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Preprocess function to extract time information from filename and set it as a coordinate
def preprocess(ds, add_xy = True):
    # Extract time and assign as coord
    time = extract_timestamp(ds.encoding['source'])
    ds = ds.assign_coords(time=time)  # Add time coordinate
    # Extract band name and assign as coord
    band_number = int(re.search(r'\.(\d{3})\.', ds.encoding['source']).group(1))
    band_name = bands_to_names([band_number])
    ds = ds.assign_coords(band = ("band", band_name))
    
    return ds

def calc_eqs(ds):

    # Calculate Ed based on temp and rh
    temp = ds.sel(band="temp")
    rh = ds.sel(band="rh")

    # Convert temp from C to K if detected
    if (temp.band_data < 150).any().values:
        print('Converting temp from C to F')
        temp += 273.15
    
    Ed = 0.924 * rh**0.679 + 0.000499 * np.exp(0.1 * rh) + 0.18 * (21.1 + 273.15 - temp) * (1 - np.exp(-0.115 * rh))
    Ew = 0.618 * rh**0.753 + 0.000454 * np.exp(0.1 * rh) + 0.18 * (21.1 + 273.15 - temp) * (1 - np.exp(-0.115 * rh))
    
    # Expand dims and assign new band names for Ed and Ew
    Ed = Ed.expand_dims(dim="band").assign_coords(band=["Ed"])
    Ew = Ew.expand_dims(dim="band").assign_coords(band=["Ew"])

    ds = xr.concat([ds, Ed, Ew], dim="band")
    
    return ds


def calc_rain(ds, ds_prev):
    # Check times are the same
    assert np.all(data.time.values == data_prev.time.values), "Time dimension not the same between input xarrays"
    
    rain = ds.sel(band="precip_accum") - ds_prev.sel(band="precip_accum")
    rain = rain.expand_dims(dim="band").assign_coords(band=["rain"])

    ds = xr.concat([ds, rain], dim="band")
    return ds

def bbox_to_xy(bbox, crs, epsg = 4326):
    transformer = Transformer.from_crs(f"EPSG:{epsg}", crs, always_xy=True)
    # Transform the lat/lon bounding box to x/y
    minx, miny = transformer.transform(bbox[1], bbox[0])  # (min_lon, min_lat)
    maxx, maxy = transformer.transform(bbox[3], bbox[2])  # (max_lon, max_lat)

    return minx, miny, maxx, maxy




