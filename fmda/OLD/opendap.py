# Required Libraries
import xarray as xr

# Define the URL for the OpenDAP endpoint
# Replace with the specific URL for the dataset you're interested in
url = 'https://nomads.ncep.noaa.gov:8080/dods/hrrr/hrrr20200910/hrrr_sfc_t18z'

# Load the data using xarray
print('xr.open_dataset',url)
ds = xr.open_dataset(url)

# Specify the point of interest
lon_point = -105.0  # longitude of the point of interest
lat_point = 40.0  # latitude of the point of interest

# Select variable and point of interest
# Temperature is usually "tmp2m" for 2 meter temperature
# Make sure to check the exact variable name in your dataset
print('getting tmp2m at nearest to',lon_point,lat_point)
temp_point = ds['tmp2m'].sel(lon=lon_point, lat=lat_point, method='nearest')

# Print the time series for the selected point
print(temp_point)

