import argparse
import numpy as np
import os
import sys
import geopandas as gpd
import shapely
import warnings
import glob
from osgeo import gdal,gdalconst,osr
import pandas as pd

def get_polygon_from_src(src):
    geotransform = src.GetGeoTransform()
    x = np.asarray([geotransform[0],
                    geotransform[0] + geotransform[1]*src.RasterXSize,
                    geotransform[0] + geotransform[1]*src.RasterXSize,
                    geotransform[0],
                    geotransform[0]])
    y = np.asarray([geotransform[3],
                    geotransform[3],
                    geotransform[3] + geotransform[5]*src.RasterYSize,
                    geotransform[3] + geotransform[5]*src.RasterYSize,
                    geotransform[3]])
    p = shapely.geometry.Polygon(zip(x,y))
    return p

def main():
    warnings.simplefilter(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',help='Directory containing files',default=None)
    parser.add_argument('--list',help='List of files',default=None)
    parser.add_argument('--output_file',help='Output file',default='output.shp')
    parser.add_argument('--smooth',help='Replace *dem.tif with *dem_smooth.tif',default=False,action='store_true')
    args = parser.parse_args()
    input_dir = args.dir
    input_list = args.list
    output_file = args.output_file
    smooth_flag = args.smooth


    if input_dir is None and input_list is None:
        print('Either --dir or --list must be specified.')
        sys.exit()
    elif input_dir is not None and input_list is not None:
        print('Only one of --dir or --list can be specified.')
        sys.exit()
    
    if input_dir is not None:
        file_list = sorted([f for f in glob.iglob(f'{input_dir}**/*dem.tif',recursive=True)])
    elif input_list is not None:
        if os.path.isfile(input_list) == False:
            print(f'File {input_list} does not exist.')
            sys.exit()
        file_list = pd.read_csv(input_list,header=None,names=['files']).files.values.tolist()

    if smooth_flag == True:
        file_list = [f.replace('dem.tif','dem_smooth.tif') for f in file_list]

    file_list = np.asarray(file_list)
    epsg_list = np.asarray([osr.SpatialReference(wkt=gdal.Open(s,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1) for s in file_list])
    unique_epsg = np.unique(epsg_list)
    if len(unique_epsg) > 1:
        print('Multiple EPSG codes found. Splitting and appending EPSG code to output file name.')
    
    for epsg_code in unique_epsg:
        idx_epsg = np.argwhere(epsg_list == epsg_code).squeeze()
        file_list_epsg = file_list[idx_epsg]
        filename_list_epsg = [os.path.basename(f) for f in file_list_epsg]
        polygon_list = []
        for f in file_list_epsg:
            src = gdal.Open(f,gdalconst.GA_ReadOnly)
            p = get_polygon_from_src(src)
            polygon_list.append(p)
        gdf = gpd.GeoDataFrame(geometry=polygon_list,data={'file':filename_list_epsg},crs=f'EPSG:{epsg_code}')
        if len(unique_epsg) == 1:
            gdf.to_file(output_file)
        else:
            gdf.to_file(f'{os.path.splitext(output_file)[0]}_{epsg_code}{os.path.splitext(output_file)[1]}')

if __name__ == '__main__':
    main()