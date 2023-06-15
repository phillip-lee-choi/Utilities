import pystac_client
import stackstac
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import sys
import warnings
import os


def get_ndvi(lon,lat,items,buffer=0.0003,method='interp'):
    sentinel_stack = stackstac.stack(items, assets=["red", "nir", "scl"],
                            bounds=[lon-buffer, lat-buffer, lon+buffer, lat+buffer],
                            gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
                                {'GDAL_HTTP_MAX_RETRY': 3,
                                    'GDAL_HTTP_RETRY_DELAY': 5,
                                }),
                            epsg=4326, chunksize=(1, 1, 50, 50)).rename(
                            {'x': 'lon', 'y': 'lat'}).to_dataset(dim='band')
    sentinel_stack['ndvi'] = (sentinel_stack['nir'] - sentinel_stack['red'])/\
                                (sentinel_stack['nir'] + sentinel_stack['red'])
    sentinel_stack = sentinel_stack[['ndvi', 'scl']]
    sentinel_stack = sentinel_stack.drop([c for c in sentinel_stack.coords if not (c in ['time', 'lat', 'lon'])])
    if method == 'interp':
        sentinel_point = sentinel_stack.interp(lon=lon,lat=lat,method="nearest")
    elif method == 'mean':
        sentinel_point = sentinel_stack.mean(dim=['lon','lat'])
    print('Loading data...')
    sentinel_point.load()
    print('Loading complete.')
    sentinel_table = sentinel_point.to_dataframe()
    sentinel_table['time'] = sentinel_table.index
    sentinel_table = sentinel_table.reset_index(drop=True)
    return sentinel_table

def main():
    warnings.simplefilter(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--coords',nargs='*',help='Coordinates (lon,lat)',type=float)
    parser.add_argument('--output_file',nargs='*',help='Path to output csv file(s).')
    parser.add_argument('--plot',help='Plot?',action='store_true',default=False)
    parser.add_argument('--filter',help='Filter for bad acquisitions',action='store_true',default=False)
    parser.add_argument('--method',help='Method for computing NDVI (interp or mean).',default='interp',choices=['interp','mean'])
    args = parser.parse_args()
    coords = args.coords
    output_file_array = np.atleast_1d(args.output_file)
    plot_flag = args.plot
    filter_flag = args.filter
    method = args.method

    if np.mod(len(coords),2) != 0:
        print('Number of coordinates must be even.')
        sys.exit()
    
    lon_array = np.atleast_1d(coords[0::2])
    lat_array = np.atleast_1d(coords[1::2])

    if len(output_file_array) != len(lon_array):
        print('Number of output files must match number of coordinates.')
        sys.exit()

    sentinel_search_url = "https://earth-search.aws.element84.com/v1"
    sentinel_stac_client = pystac_client.Client.open(sentinel_search_url)
    if len(lon_array) > 1:
        coord_dict = dict(type="MultiPoint",coordinates=[[ln,lt] for ln,lt in zip(lon_array,lat_array)])
    elif len(lon_array) == 1:
        coord_dict = dict(type="Point",coordinates=(lon_array[0],lat_array[0]))
    else:
        print('No coordinates provided.')
        sys.exit()
    items = sentinel_stac_client.search(intersects=coord_dict,collections=["sentinel-2-l2a"]).get_all_items()

    for lon,lat,output_file in zip(lon_array,lat_array,output_file_array):
        print(f'Working on {os.path.splitext(os.path.basename(output_file))[0]}.')
        sentinel_table = get_ndvi(lon,lat,items,method=method)
        if filter_flag == True:
            sentinel_table = sentinel_table[np.logical_or(sentinel_table['scl'] == 4,sentinel_table['scl'] == 5)]
        sentinel_table.to_csv(output_file,index=False)
    
    if plot_flag == True:
        if len(output_file_array) <= 3:
            fig,ax = plt.subplots(len(output_file_array),1,sharex=True,sharey=True,figsize=(12,12))
            for i,output_file in enumerate(output_file_array):
                df_table = pd.read_csv(output_file)
                ax[i].plot(pd.to_datetime(df_table['time']),df_table['ndvi'],marker='.',linestyle='none',markersize=5)
                ax[i].set_title(os.path.splitext(os.path.basename(output_file))[0],fontsize=14)
                ax[i].set_ylabel('NDVI',fontsize=14)
                ax[i].set_ylim([0,1])
                ax[i].grid(alpha=0.3)
            ax[i].set_xlabel('Time',fontsize=14)
            plt.show()
        else:
            fig,ax = plt.subplots(1,1,figsize=(14,12))
            for i,output_file in enumerate(output_file_array):
                df_table = pd.read_csv(output_file)
                ax.plot(pd.to_datetime(df_table['time']),df_table['ndvi'],marker='.',linestyle='none',markersize=5,label=os.path.splitext(os.path.basename(output_file))[0])
            ax.set_ylabel('NDVI',fontsize=14)
            ax.set_xlabel('Time',fontsize=14)
            ax.set_ylim([0,1])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=14)
            plt.show()




if __name__ == '__main__':
    main()