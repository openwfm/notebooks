# Set of functions to handle spatial data read with xarray and deploy trained model to create predictions

import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
import re
from pyproj import Transformer
from datetime import datetime
import warnings
import pandas as pd


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
    # Get the variable string names from band number list, assumes band_df_hrrr exists in memory
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

def features_to_bands(feat_list):
    # Given list of features used in a model, return band numbers and variable names needed from HRRR

    bands = names_to_bands(feat_list)
    bnames = bands_to_names(bands)
    
    # Variable "rain" is engineered from precip_accum
    if 'rain' in bnames:
        bands.remove(628)
        bnames.remove("rain")
        bands.append(629)
        bnames.append("precip_accum")

    return bands, bnames

def extract_timestamp(file_path):
    # Extract date (parent directory) and hour from the file path
    date_str = re.search(r'(\d{8})', file_path).group(1)  # Matches YYYYMMDD
    hour_str = re.search(r't(\d{2})z', file_path).group(1)  # Matches tHHz
    
    # Combine into a datetime object
    timestamp = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")
    return timestamp

def get_file_list(start_time, end_time, fstep, bands_list, base_path = "."):
    # Given start time, end time, and list of HRRR bands, return a nested list of files to be read with xr open_mfdataset
    
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

def preprocess(ds, add_xy = True):
    # Preprocess function to extract time information from filename and set it as a coordinate. Used within xr open_mfdataset
    # Extract time and assign as coord
    time = extract_timestamp(ds.encoding['source'])
    ds = ds.assign_coords(time=time)  # Add time coordinate
    # Extract band name and assign as coord
    band_number = int(re.search(r'\.(\d{3})\.', ds.encoding['source']).group(1))
    band_name = bands_to_names([band_number])
    ds = ds.assign_coords(band = ("band", band_name))
    
    return ds

def calc_eqs(ds):

    # Check whether Eqs exist and exit if so
    if any(name in ds.band for name in ["Ed", "Ew"]):
        warnings.warn("Equilibria data already detected in xarray, exiting function")
        return ds 
    
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
    
    # Check whether rain exist and exit if so
    if any(name in ds.band for name in ["rain"]):
        warnings.warn("Rain data already detected in xarray, exiting function")
        return ds 
    
    # Check times are the same
    assert np.all(ds.time.values == ds_prev.time.values), "Time dimension not the same between input xarrays"
    
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computational to Perform transformations between coordinate systems
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def xy_to_grid(x, y, xarray_obj):
    """
    Converts projected x and y coordinates to grid coordinates (fractional indices) 
    based on the affine transform of an xarray object.

    Parameters:
        x (array-like): Array of x coordinates in the projection units.
        y (array-like): Array of y coordinates in the projection units.
        xarray_obj (xarray.Dataset or xarray.DataArray): The xarray object with raster data.

    Returns:
        tuple: A tuple (grid_x, grid_y) where grid_x and grid_y are arrays of fractional indices.
    """
    # Extract the affine transformation from the xarray object
    transform = xarray_obj.rio.transform()

    # Compute grid indices from projected coordinates
    inv_transform = ~transform  # Inverse the affine transform
    grid_x, grid_y = inv_transform * (x, y)  # Apply inverse transform
    
    return grid_x, grid_y


def lonlat_to_xy(longitudes, latitudes, crs):
    """
    Converts latitude and longitude to x and y coordinates based on the given CRS.
    
    Parameters:
        latitudes (array-like): Array or list of latitude values.
        longitudes (array-like): Array or list of longitude values.
        crs: Target CRS (Coordinate Reference System) in pyproj or rasterio format.
    
    Returns:
        tuple: A tuple (x, y) where x and y are arrays of projected coordinates.
    """
    # Define the transformer for converting lat/lon to x/y
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    # Transform lat/lon to x/y
    x, y = transformer.transform(longitudes, latitudes)
    
    return x, y

def xr_to_lonlat(xarray_obj):
    """
    Converts the x and y coordinates of an xarray object to longitude and latitude.
    
    Parameters:
        xarray_obj (xarray.Dataset or xarray.DataArray): The xarray object with 'x' and 'y' dimensions.
        
    Returns:
        numpy.ndarray: A 2D array of shape (2, n_y, n_x), where the first layer is longitude 
                       and the second layer is latitude.
    """
    # Extract the CRS from the xarray object
    crs_xarray = xarray_obj.rio.crs
    
    # Define the transformer for converting x/y to lat/lon
    transformer = Transformer.from_crs(crs_xarray, "EPSG:4326", always_xy=True)
    
    # Extract x and y coordinate arrays
    x_coords = xarray_obj['x'].values
    y_coords = xarray_obj['y'].values
    
    # Create a meshgrid of x and y coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Transform x/y to lon/lat
    lon, lat = transformer.transform(xx, yy)
    
    # Stack lon and lat into a single array
    lonlat_array = np.stack([lon, lat], axis=0)  # Shape (2, n_y, n_x)
    
    return lonlat_array




