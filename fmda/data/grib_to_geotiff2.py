import argparse
from osgeo import gdal
import numpy as np

gdal.DontUseExceptions()

def grib_to_geotiff2(input_filename, output_filename_base, band_numbers):
    """
    Convert existing grib file to geotiff. 
    :param input_filename:   str, file name w relative path
    :param output_filename_base:  str, output file name that is later appended
    :param band_number:   list of integer
    :return: None
    """
    
    # Open the GRIB file
    ds = gdal.Open(input_filename)

    # Create the output filename
    output_filename = f"{output_filename_base}.fmda_bands.tif"

    # Get first band
    band_number = band_numbers[0]
    band = ds.GetRasterBand(band_number)
    metadata = band.GetMetadata()

    # Print the metadata
    print(f"Metadata for band {band_number}:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
        
    # Check if the band data is 2D
    arr = band.ReadAsArray()
    if len(arr.shape) != 2:
        print(f"Skipping band {band_number} because it is not 2D")
        return
    
    # Create a new data source in memory
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_filename, band.XSize, band.YSize, len(band_numbers), band.DataType)

    # Set the geotransform and projection
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    # Write the data to the new data source
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(arr)

    # Loop Through other bands
    if len(band_numbers)>1:
        for i, band_number in enumerate(band_numbers):
            if i == 0: continue
                
            print("~"*30)
            
            # Get the specified band
            band = ds.GetRasterBand(band_number)
            metadata = band.GetMetadata()
    
            # Print the metadata
            print(f"Metadata for band {band_number}:")
            for key, value in metadata.items():
                print(f"{key}: {value}")
    
            # Check if the band data is 2D
            arr = band.ReadAsArray()

            if len(arr.shape) != 2:
                print(f"Skipping band {band_number} because it is not 2D")
                return

            # Write the data to the new data source
            out_band = out_ds.GetRasterBand(i+1)
            out_band.WriteArray(arr)

    # Close the data sources
    ds = None
    out_ds = None

    print(f"Bands {band_numbers} from {input_filename} saved to {output_filename}")
   

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract a band from a GRIB file and save as a GeoTIFF if it is 2D.')
    parser.add_argument('input_filename', help='Path to the input GRIB file')
    parser.add_argument('output_filename_base', help='Base of the output GeoTIFF filename (band number will be appended)')
    parser.add_argument('band_number', type=int, help='Number of the band to extract (1-based)')

    args = parser.parse_args()

    # Call the function with the arguments
    grib_to_geotiff(args.input_filename, args.output_filename_base, args.band_number)

