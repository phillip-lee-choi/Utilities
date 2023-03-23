import numpy as np
import pandas as pd
import subprocess
import argparse
import os
import sys
from osgeo import gdal,gdalconst,osr



def sample_raster(raster_path, csv_path, output_file,nodata='-9999'):
    cat_command = f"cat {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -wgs84 {raster_path} > tmp.txt"
    subprocess.run(cat_command,shell=True)
    fill_nan_command = f"awk '!NF{{$0=\"NaN\"}}1' tmp.txt > tmp2.txt"
    subprocess.run(fill_nan_command,shell=True)
    paste_command = f"paste -d , {csv_path} tmp2.txt > {output_file}"
    subprocess.run(paste_command,shell=True)
    subprocess.run(f"sed -i '/{nodata}/d' {output_file}",shell=True)
    subprocess.run(f"sed -i '/NaN/d' {output_file}",shell=True)
    subprocess.run(f"sed -i '/nan/d' {output_file}",shell=True)
    subprocess.run(f"rm tmp.txt tmp2.txt",shell=True)
    return None

def filter_outliers(dh,mean_median_mode='mean',n_sigma_filter=2):
    dh_mean = np.nanmean(dh)
    dh_std = np.nanstd(dh)
    dh_median = np.nanmedian(dh)
    if mean_median_mode == 'mean':
        dh_mean_filter = dh_mean
    elif mean_median_mode == 'median':
        dh_mean_filter = dh_median
    dh_filter = np.abs(dh-dh_mean_filter) < n_sigma_filter*dh_std
    return dh_filter

def calculate_shift(df_sampled,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.05):
    count = 0
    cumulative_shift = 0
    original_len = len(df_sampled)
    height_icesat2_original = np.asarray(df_sampled.height_icesat2)
    height_dem_original = np.asarray(df_sampled.height_dem)
    dh_original = height_icesat2_original - height_dem_original
    rmse_original = np.sqrt(np.sum(dh_original**2)/len(dh_original))
    while True:
        count = count + 1
        height_icesat2 = np.asarray(df_sampled.height_icesat2)
        height_dem = np.asarray(df_sampled.height_dem)
        dh = height_icesat2 - height_dem
        dh_filter = filter_outliers(dh,mean_median_mode,n_sigma_filter)
        if mean_median_mode == 'mean':
            incremental_shift = np.mean(dh[dh_filter])
        elif mean_median_mode == 'median':
            incremental_shift = np.median(dh[dh_filter])
        df_sampled = df_sampled[dh_filter].reset_index(drop=True)
        df_sampled.height_dem = df_sampled.height_dem + incremental_shift
        cumulative_shift = cumulative_shift + incremental_shift
        print(f'Iteration        : {count}')
        print(f'Incremental shift: {incremental_shift:.2f} m\n')
        if np.abs(incremental_shift) <= vertical_shift_iterative_threshold:
            break
        if count == 15:
            break
    height_icesat2_filtered = np.asarray(df_sampled.height_icesat2)
    height_dem_filtered = np.asarray(df_sampled.height_dem)
    dh_filtered = height_icesat2_filtered - height_dem_filtered
    rmse_filtered = np.sqrt(np.sum(dh_filtered**2)/len(dh_filtered))
    print(f'Number of iterations: {count}')
    print(f'Number of points before filtering: {original_len}')
    print(f'Number of points after filtering: {len(df_sampled)}')
    print(f'Retained {len(df_sampled)/original_len*100:.1f}% of points.')
    print(f'Cumulative shift: {cumulative_shift:.2f} m')
    print(f'RMSE before filtering: {rmse_original:.2f} m')
    print(f'RMSE after filtering: {rmse_filtered:.2f} m')
    return df_sampled,cumulative_shift


def vertical_shift_raster(raster_path,df_sampled,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.05):
    src = gdal.Open(raster_path,gdalconst.GA_ReadOnly)
    raster_nodata = src.GetRasterBand(1).GetNoDataValue()
    df_sampled_filtered,vertical_shift = calculate_shift(df_sampled,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold)
    raster_base,raster_ext = os.path.splitext(raster_path)
    raster_shifted = f'{raster_base}_shifted_{"{:.2f}".format(vertical_shift).replace(".","p").replace("-","neg")}m{raster_ext}'
    shift_command = f'gdal_calc.py --quiet -A {raster_path} --outfile={raster_shifted} --calc="A+{vertical_shift:.2f}" --NoDataValue={raster_nodata} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    subprocess.run(shift_command,shell=True)
    return df_sampled_filtered,raster_shifted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster', help="Path to DEM file")
    parser.add_argument('--csv', help="Path to txt/csv file")
    parser.add_argument('--mean',default=False,action='store_true')
    parser.add_argument('--median',default=False,action='store_true')
    parser.add_argument('--sigma', nargs='?', type=int, default=2)
    parser.add_argument('--threshold', nargs='?', type=float, default=0.05)
    parser.add_argument('--resample',default=False,action='store_true')
    parser.add_argument('--keep_original_sample',default=False,action='store_true')
    parser.add_argument('--no_writing',default=False,action='store_true')
    parser.add_argument('--nodata', nargs='?', type=str)

    args = parser.parse_args()
    raster_path = args.raster
    csv_path = args.csv
    mean_mode = args.mean
    median_mode = args.median
    n_sigma_filter = args.sigma
    vertical_shift_iterative_threshold = args.threshold
    resample_flag = args.resample
    keep_original_sample_flag = args.keep_original_sample
    no_writing_flag = args.no_writing
    nodata_value = args.nodata
    if np.logical_xor(mean_mode,median_mode) == True:
        if mean_mode == True:
            mean_median_mode = 'mean'
        elif median_mode == True:
            mean_median_mode = 'median'
    else:
        print('Please choose exactly one mode: mean or median.')
        sys.exit()
    

    sampled_file = f'{os.path.splitext(csv_path)[0]}_Sampled_{os.path.splitext(os.path.basename(raster_path))[0]}{os.path.splitext(csv_path)[1]}'
    sample_code = sample_raster(raster_path, csv_path, sampled_file,nodata=nodata_value)
    if sample_code is not None:
        print('Error in sampling raster.')
    df_sampled_original = pd.read_csv(sampled_file,header=None,names=['lon','lat','height_icesat2','time','height_dem'],dtype={'lon':'float','lat':'float','height_icesat2':'float','time':'str','height_dem':'float'})
    df_sampled_filtered,raster_shifted = vertical_shift_raster(raster_path,df_sampled_original,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold)
    if no_writing_flag == False:
        lon_filtered = np.asarray(df_sampled_filtered.lon)
        lat_filtered = np.asarray(df_sampled_filtered.lat)
        height_icesat2_filtered = np.asarray(df_sampled_filtered.height_icesat2)
        time_filtered = np.asarray(df_sampled_filtered.time)
        output_csv = f'{os.path.splitext(csv_path)[0]}_Filtered_{mean_median_mode}_{n_sigma_filter}sigma_Threshold_{str(vertical_shift_iterative_threshold).replace(".","p")}m{os.path.splitext(csv_path)[1]}'
        np.savetxt(output_csv,np.c_[lon_filtered,lat_filtered,height_icesat2_filtered,time_filtered.astype(object)],fmt='%f,%f,%f,%s',delimiter=',')

    if resample_flag == True:
        resampled_file = f'{os.path.splitext(output_csv)[0]}_Sampled_Coregistered_{os.path.splitext(os.path.basename(raster_path))[0]}{os.path.splitext(output_csv)[1]}'
        resample_code = sample_raster(raster_shifted, output_csv, resampled_file,nodata=nodata_value)
        if resample_code is not None:
            print('Error in sampling co-registered raster.')
    if keep_original_sample_flag == False:
        os.remove(sampled_file)

if __name__ == '__main__':
    main()