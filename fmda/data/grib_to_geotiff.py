import argparse
from osgeo import gdal
import numpy as np

def grib_to_geotiff(input_filename, output_filename, variable_name):
    # Open the GRIB file
    ds = gdal.Open(input_filename)
    
    # Loop through all bands in the dataset and check their descriptions
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i+1)  # 1-based index
        metadata = band.GetMetadata()

        # Check if the description matches the variable we're looking for
        if metadata.get('GRIB_COMMENT') == variable_name:
            # Read the band data
            arr = band.ReadAsArray()

            # Create a new data source in memory
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(output_filename, band.XSize, band.YSize, 1, band.DataType)

            # Set the geotransform and projection
            out_ds.SetGeoTransform(ds.GetGeoTransform())
            out_ds.SetProjection(ds.GetProjection())

            # Write the data to the new data source
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(arr)

            # Close the data sources
            ds = None
            out_ds = None

            print(f"Variable {variable_name} from {input_filename} saved to {output_filename}")
            return  # Exit the function

    print(f"Variable {variable_name} not found in {input_filename}")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract variables from a GRIB file and save as GeoTIFFs.')
    parser.add_argument('input_filename', help='Path to the input GRIB file')
    parser.add_argument('output_filename', help='Path to the output GeoTIFF file')
    parser.add_argument('variable_names', nargs='+', help='Names of the variables to extract')

    args = parser.parse_args()

    # Call the function for each variable
    for variable_name in args.variable_names:
        output_filename = f"{args.output_filename}_{variable_name}.tif"  # Append the variable name to the filename
        grib_to_geotiff(args.input_filename, output_filename, variable_name)

