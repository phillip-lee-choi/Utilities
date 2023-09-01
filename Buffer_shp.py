import geopandas as gpd
import argparse
import os

def buffer_gdf(gdf,buffer):
    '''
    Convert GeoDataFrame to a CRS in meters, buffer, reconvert back to original CRS.
    '''
    epsg_orig = gdf.crs.to_epsg()
    gdf_buffered = gdf.to_crs('EPSG:3857').buffer(buffer)
    gdf_unary_union = gdf_buffered.unary_union
    gdf_buffered = gpd.GeoDataFrame(geometry=[g for g in gdf_unary_union.geoms],crs='EPSG:3857').to_crs(f'EPSG:{epsg_orig}')
    return gdf_buffered

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Input file.',default=None)
    parser.add_argument('--buffer',help='Buffer distance (meters).',default=50,type=float)
    args = parser.parse_args()
    input_file = args.input_file
    buffer = args.buffer
    buffer_str = f'{buffer:.1f}'.replace('.','p').replace('p0','')
    output_file = f'{os.path.splitext(input_file)[0]}_buffered_{buffer_str}m{os.path.splitext(input_file)[1]}'

    gdf_input = gpd.read_file(input_file)
    gdf_buffered = buffer_gdf(gdf_input,buffer)
    gdf_buffered.to_file(output_file)

if __name__ == '__main__':
    main()