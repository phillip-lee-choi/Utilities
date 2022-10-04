import subprocess
import argparse
import csv
import numpy as np
import pandas as pd
import datetime
import os
import sys
from osgeo import gdal,gdalconst,osr

def detect_header(csv_file):
    with open(csv_file,'r') as f:
        has_header = csv.Sniffer().has_header(f.read(1024))
    return has_header

def find_lonlat_headers(csv_file):
    lon_checklist = ['lon','longitude','long']
    lat_checklist = ['lat','latitude','latt']
    idx_lon,idx_lat = find_headers(csv_file,lon_checklist,lat_checklist)
    if np.logical_or(idx_lon is None,idx_lat is None):
        print('Could not find lon/lat headers.')
    return idx_lon,idx_lat

def find_xy_headers(csv_file):
    x_checklist = ['x','easting','east']
    y_checklist = ['y','northing','north']
    idx_x,idx_y = find_headers(csv_file,x_checklist,y_checklist)
    if np.logical_or(idx_x is None,idx_y is None):
        print('Could not find x/y headers.')
    return idx_x,idx_y

def find_headers(csv_file,lon_checklist,lat_checklist):
    csv_head = subprocess.check_output(f'head -n 1 {csv_file}', shell=True).decode('utf-8').strip().split('\n')
    headers = csv_head[0].split('\t')[0].split(',')
    headers = np.asarray([h.strip().lower().split(' ')[0] for h in headers])
    idx_lon = np.zeros(len(headers),dtype=bool)
    idx_lat = np.zeros(len(headers),dtype=bool)
    for lon_check in lon_checklist:
        idx_lon = np.any((idx_lon,headers==lon_check),axis=0)
    for lat_check in lat_checklist:
        idx_lat = np.any((idx_lat,headers==lat_check),axis=0)
    idx_lon = np.atleast_1d(np.argwhere(idx_lon).squeeze())
    idx_lat = np.atleast_1d(np.argwhere(idx_lat).squeeze())
    if len(idx_lon) == 0 or len(idx_lat) == 0:
        return None,None
    idx_lon = idx_lon[0]+1
    idx_lat = idx_lat[0]+1
    return idx_lon,idx_lat

def deg2utm(lon,lat):
    pi = np.math.pi
    n1 = np.asarray(lon).size
    n2 = np.asarray(lat).size
    if n1 != n2:
        print('Longitude and latitude vectors not equal in length.')
        print('Exiting')
        return
    lon_deg = lon
    lat_deg = lat
    lon_rad = lon*pi/180
    lat_rad = lat*pi/180
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    tan_lat = np.tan(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    tan_lon = np.tan(lon_rad)
    x = np.empty([n1,1],dtype=float)
    y = np.empty([n2,1],dtype=float)
    zone_letter = [None]*n1
    semi_major_axis = 6378137.0
    semi_minor_axis = 6356752.314245
    second_eccentricity = np.sqrt(semi_major_axis**2 - semi_minor_axis**2)/semi_minor_axis
    second_eccentricity_squared = second_eccentricity**2
    c = semi_major_axis**2 / semi_minor_axis
    utm_number = np.fix(lon_deg/6 + 31)
    S = utm_number*6 - 183
    delta_S = lon_rad - S*pi/180
    epsilon = 0.5*np.log((1+cos_lat * np.sin(delta_S))/(1-cos_lat * np.sin(delta_S)))
    nu = np.arctan(tan_lat / np.cos(delta_S)) - lat_rad
    v = 0.9996 * c / np.sqrt(1+second_eccentricity_squared * cos_lat**2)
    tau = 0.5*second_eccentricity_squared * epsilon**2 * cos_lat**2
    a1 = np.sin(2*lat_rad)
    a2 = a1 * cos_lat**2
    j2 = lat_rad + 0.5*a1
    j4 = 0.25*(3*j2 + a2)
    j6 = (5*j4 + a2*cos_lat**2)/3
    alpha = 0.75*second_eccentricity_squared
    beta = (5/3) * alpha**2
    gamma = (35/27) * alpha**3
    Bm = 0.9996 * c * (lat_rad - alpha*j2 + beta*j4 - gamma*j6)
    x = epsilon * v * (1+tau/3) + 500000
    y = nu * v * (1+tau) + Bm
    idx_y = y<0
    y[idx_y] = y[idx_y] + 9999999
    for i in range(n1):
        if lat_deg[i]<-72:
            zone_letter[i] = ' C'
        elif lat_deg[i] < -64:
            zone_letter[i] = ' D'
        elif lat_deg[i] < -56:
            zone_letter[i] = ' E'
        elif lat_deg[i] < -48:
            zone_letter[i] = ' F'
        elif lat_deg[i] < -40:
            zone_letter[i] = ' G'
        elif lat_deg[i] < -32:
            zone_letter[i] = ' H'
        elif lat_deg[i] < -24:
            zone_letter[i] = ' J'
        elif lat_deg[i] < -16:
            zone_letter[i] = ' K'
        elif lat_deg[i] < -8:
            zone_letter[i] = ' L'
        elif lat_deg[i] < 0:
            zone_letter[i] = ' M'
        elif lat_deg[i] < 8:
            zone_letter[i] = ' N'
        elif lat_deg[i] < 16:
            zone_letter[i] = ' P'
        elif lat_deg[i] < 24:
            zone_letter[i] = ' Q'
        elif lat_deg[i] < 32:
            zone_letter[i] = ' R'
        elif lat_deg[i] < 40:
            zone_letter[i] = ' S'
        elif lat_deg[i] < 48:
            zone_letter[i] = ' T'
        elif lat_deg[i] < 56:
            zone_letter[i] = ' U'
        elif lat_deg[i] < 64:
            zone_letter[i] = ' V'
        elif lat_deg[i] < 72:
            zone_letter[i] = ' W'
        else:
            zone_letter[i] = ' X'
    utm_int = np.char.mod('%02d',utm_number.astype(int))
    utm_int_list = utm_int.tolist()
    utmzone = [s1 + s2 for s1, s2 in zip(utm_int_list, zone_letter)]
    return x, y, utmzone

def get_epsg(input_file):
    src = gdal.Open(input_file)
    proj = osr.SpatialReference(wkt=src.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY',1)
    return epsg

def utm2epsg(utm_code,north_south_flag=False):
    utm_code = np.asarray([z.replace(' ','') for z in utm_code])
    lat_band_number = np.asarray([ord(u[2].upper()) for u in utm_code])
    if north_south_flag == True:
        hemisphere_ID = np.zeros(len(lat_band_number),dtype=int)
        hemisphere_ID[lat_band_number == 83] = 7 #south
        hemisphere_ID[lat_band_number == 78] = 6 #north
    else:
        hemisphere_ID = np.zeros(len(lat_band_number),dtype=int)
        hemisphere_ID[lat_band_number <= 77] = 7 #south
        hemisphere_ID[lat_band_number >= 78] = 6 #north
    epsg_code = np.asarray([f'32{a[1]}{a[0][0:2]}' for a in zip(utm_code,hemisphere_ID)])
    if len(epsg_code) == 1:
        epsg_code = epsg_code[0]
    return epsg_code

def find_column_12_21(csv,raster,nodata_value=-9999,geolocation='wgs84'):
    ''''
    Assumes that the csv has spatial coordinates in first two columns
    '''
    gdallocationinfo_input_12 = subprocess.check_output(f"cat {csv} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -{geolocation} {raster}",shell=True).decode('utf-8').split('\n')
    gdallocationinfo_input_12 = np.asarray(gdallocationinfo_input_12,dtype='<U18')
    gdallocationinfo_input_12[gdallocationinfo_input_12==''] = 'nan'
    gdallocationinfo_input_12 = gdallocationinfo_input_12.astype(float)
    percent_valid_12 = np.sum(gdallocationinfo_input_12 > nodata_value) / len(gdallocationinfo_input_12)
    gdallocationinfo_input_21 = subprocess.check_output(f"cat {csv} | cut -d, -f1-2 | sed 's/,/ /g' | awk '{{print $2 \" \" $1}}' | gdallocationinfo -valonly -{geolocation} {raster}",shell=True).decode('utf-8').split('\n')
    gdallocationinfo_input_21 = np.asarray(gdallocationinfo_input_21,dtype='<U18')
    gdallocationinfo_input_21[gdallocationinfo_input_21==''] = 'nan'
    gdallocationinfo_input_21 = gdallocationinfo_input_21.astype(float)
    percent_valid_21 = np.sum(gdallocationinfo_input_21 > nodata_value) / len(gdallocationinfo_input_21)
    return percent_valid_12,percent_valid_21

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster', help="Path to DEM file")
    parser.add_argument('--csv', help="Path to txt/csv file")
    parser.add_argument('--output_file', nargs='?')
    parser.add_argument('--standard',default=False,action='store_true')
    parser.add_argument('--filter',default=False,action='store_true')
    parser.add_argument('--lonlat',default=False,action='store_true')
    parser.add_argument('--latlon',default=False,action='store_true')
    parser.add_argument('--utm',default=False,action='store_true')

    args = parser.parse_args()
    raster_path = args.raster
    csv_path = args.csv
    output_file = args.output_file
    standard_flag = args.standard
    filter_flag = args.filter
    lonlat_flag = args.lonlat
    latlon_flag = args.latlon
    utm_flat = args.utm

    if np.sum((lonlat_flag,latlon_flag,utm_flat)) > 1:
        print('Please choose one coordinate system.')
        sys.exit()
    if output_file is None:
        output_file = f'{os.path.splitext(csv_path)[0]}_Sampled_{os.path.splitext(os.path.basename(raster_path))[0]}{os.path.splitext(csv_path)[1]}'
    
    if standard_flag == True:
        cat_command = f"cat {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -wgs84 {raster_path} > tmp.txt"
    else:
        csv_has_header = detect_header(csv_path)
        if csv_has_header == True:
            idx_lon,idx_lat = find_lonlat_headers(csv_path)
            cat_command = f"tail -n +2 {csv_path} | cut -d, -f1-{np.max((idx_lon,idx_lat))} | awk -F, '{{print ${idx_lon} \" \" ${idx_lat}}}' | gdallocationinfo -valonly -wgs84 {raster_path} > tmp.txt"
            if np.logical_or(idx_lon is None,idx_lat is None):
                idx_x,idx_y = find_xy_headers(csv_path)
                cat_command = f"tail -n +2 {csv_path} | cut -d, -f1-{np.max((idx_x,idx_y))} | awk -F, '{{print ${idx_x} \" \" ${idx_y}}}' | gdallocationinfo -valonly -geoloc {raster_path} > tmp.txt"
                if np.logical_or(idx_x is None,idx_y is None):
                    print("ERROR: No lon/lat or x/y headers found!")
                    sys.exit()
        else:
            N_subset = 1000
            nodata_value = -9999
            csv_base = os.path.splitext(csv_path)[0]
            csv_ext = os.path.splitext(csv_path)[1]
            csv_downsampled_path = f'{csv_base}_subset_n{N_subset}{csv_ext}'
            subprocess.run(f"awk 'NR % {N_subset} == 0' {csv_path} > {csv_downsampled_path}",shell=True)
            wc_downsampled = int(subprocess.check_output(f'wc -l {csv_downsampled_path}',shell=True).decode('utf-8').strip().split(' ')[0])
            if wc_downsampled < 100:
                csv_analysis = csv_path
            else:
                csv_analysis = csv_downsampled_path
            #assume lon/lat, if not, check lat/lon
            percent_valid_lonlat,percent_valid_latlon = find_column_12_21(csv_analysis,raster_path,nodata_value=nodata_value,geolocation='wgs84')
            percent_valid_xy,percent_valid_yx = find_column_12_21(csv_analysis,raster_path,nodata_value=nodata_value,geolocation='geoloc')
            os.remove(csv_downsampled_path)
            max_percent_valid = np.max((percent_valid_lonlat,percent_valid_latlon,percent_valid_xy,percent_valid_yx))
            if max_percent_valid == percent_valid_lonlat:
                cat_command = f"tail -n +2 {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -wgs84 {raster_path} > tmp.txt"
            elif max_percent_valid == percent_valid_latlon:
                cat_command = f"tail -n +2 {csv_path} | cut -d, -f1-2 | awk -F, '{{print $2 \" \" $1}}' | gdallocationinfo -valonly -wgs84 {raster_path} > tmp.txt"
            elif max_percent_valid == percent_valid_xy:
                cat_command = f"tail -n +2 {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -geoloc {raster_path} > tmp.txt"
            elif max_percent_valid == percent_valid_yx:
                cat_command = f"tail -n +2 {csv_path} | cut -d, -f1-2 | awk -F, '{{print $2 \" \" $1}}' | gdallocationinfo -valonly -geoloc {raster_path} > tmp.txt"

    subprocess.run(cat_command,shell=True)
    fill_nan_command = f"awk '!NF{{$0=\"NaN\"}}1' tmp.txt > tmp2.txt"
    subprocess.run(fill_nan_command,shell=True)
    if csv_has_header == True:
        header_command = f"sed -i '1s/^/Sampled Raster \\n/' tmp2.txt"
        subprocess.run(header_command,shell=True)
    paste_command = f"paste -d , {csv_path} tmp2.txt > {output_file}"
    subprocess.run(paste_command,shell=True)
    # os.remove('tmp.txt')
    # os.remove('tmp2.txt')
    if filter_flag == True:
        head_in = subprocess.check_output(f"head -n 1 {csv_path}",shell=True).decode('utf-8')
        n_column_sampled = len(head_in.split(',')) + 1
        subprocess.run(f"sed -i '/-9999/d' {output_file}",shell=True)
        subprocess.run(f"sed -i '/NaN/d' {output_file}",shell=True)
        subprocess.run(f"sed -i '/nan/d' {output_file}",shell=True)
        subprocess.run(f"awk -F, '${n_column_sampled}!=\"\"' {output_file} > tmp.txt",shell=True)
        subprocess.run(f'mv tmp.txt {output_file}',shell=True)



'''
cat /media/heijkoop/DATA/DEM/Accuracy_Assessment/Strip/Rural/US_Savannah_ATL03_Rural_Strip.txt | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -wgs84 /media/heijkoop/DATA/DEM/Accuracy_Assessment/Strip/Rural/WV01_20190126_1020010082E41B00_1020010083282500_2m_lsf_seg4_dem.tif > tmp_orig.txt



input_file = '/media/heijkoop/DATA/DEM/Locations/US_Savannah/ICESat-2/US_Savannah_Full_Mosaic_1_Sampled_ICESat2.txt'
df = pd.read_csv(input_file,header=None,names=['lon','lat','height_icesat2','time','height_dem'],dtype={'lon':'float','lat':'float','height_icesat2':'float','time':'str','height_dem':'float'})
lon = np.asarray(df.lon)
lat = np.asarray(df.lat)
height_icesat2 = np.asarray(df.height_icesat2)
time = np.asarray(df.time)
height_dem = np.asarray(df.height_dem)

idx_nan = np.isnan(height_dem)
idx_9999 = height_dem == -9999
idx_filter = ~np.any((idx_nan,idx_9999),axis=0)

lon_filtered = lon[idx_filter]
lat_filtered = lat[idx_filter]
height_icesat2_filtered = height_icesat2[idx_filter]
time_filtered = time[idx_filter]
height_dem_filtered = height_dem[idx_filter]

dh_filtered = height_dem_filtered - height_icesat2_filtered
rmse = np.sqrt(np.sum(dh_filtered**2)/len(dh_filtered))

time_filtered_datetime = np.asarray([datetime.datetime.strptime(t,'%Y-%m-%d %H:%M:%S.%f') for t in time_filtered])
date_filtered = np.asarray([t.date() for t in time_filtered_datetime])
unique_dates = np.unique(date_filtered)

rmse_date = np.zeros((len(unique_dates)))
for i in range(len(unique_dates)):
    idx_date = date_filtered == unique_dates[i]
    dh_tmp = dh_filtered[idx_date]
    rmse_date[i] = np.sqrt(np.sum(dh_tmp**2)/len(dh_tmp))
'''

if __name__ == '__main__':
    main()
