#!/usr/bin/env python

import subprocess
import argparse
import numpy as np
import json
from osgeo import gdal,gdalconst,osr
import pandas as pd
import datetime
import os,sys
import matplotlib

def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            # print(x,y)
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        y,x,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def get_raster_extents(raster,global_local_flag='global'):
    '''
    Get global or local extents of a raster
    '''
    src = gdal.Open(raster,gdalconst.GA_ReadOnly)
    gt = src.GetGeoTransform()
    cols = src.RasterXSize
    rows = src.RasterYSize
    local_ext = GetExtent(gt,cols,rows)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src.GetProjection())
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    global_ext = ReprojectCoords(local_ext,src_srs,tgt_srs)
    x_local = [item[0] for item in local_ext]
    y_local = [item[1] for item in local_ext]
    x_min_local = np.nanmin(x_local)
    x_max_local = np.nanmax(x_local)
    y_min_local = np.nanmin(y_local)
    y_max_local = np.nanmax(y_local)
    x_global = [item[0] for item in global_ext]
    y_global = [item[1] for item in global_ext]
    x_min_global = np.nanmin(x_global)
    x_max_global = np.nanmax(x_global)
    y_min_global = np.nanmin(y_global)
    y_max_global = np.nanmax(y_global)
    if global_local_flag.lower() == 'global':
        return x_min_global,x_max_global,y_min_global,y_max_global
    elif global_local_flag.lower() == 'local':
        return x_min_local,x_max_local,y_min_local,y_max_local
    else:
        return None


def utm2proj4(utm_code,hemisphere=None):
    if 'UTM' in utm_code:
        utm_code = utm_code.replace('UTM','')
    if 'utm' in utm_code:
        utm_code = utm_code.replace('utm','')
    north_south = ''
    zone = utm_code[0:2]
    if len(utm_code)==2 and hemisphere is not None:
        hemisphere = hemisphere.lower()
        if hemisphere == 'north' or hemisphere == 'south':
            north_south = hemisphere
        elif hemisphere == 'n':
            north_south = 'north'
        elif hemisphere == 's':
            north_south = 'south'
        else:
            print('Invalid hemisphere definition!')
            return
    else:
        lat_band_number = ord(utm_code[2])
        if lat_band_number >= 97 and lat_band_number <= 122:
            lat_band_number = lat_band_number - 96
        elif lat_band_number >= 65 and lat_band_number <= 90:
            lat_band_number = lat_band_number - 64

        if lat_band_number <= 13 and lat_band_number >= 3:
            north_south = 'south'
        elif lat_band_number <= 24 and lat_band_number >= 14:
            north_south = 'north'

    if len(north_south) == 0:
        print('Error! North/South not created!')
        return

    proj4 = '+proj=utm +zone='+zone+' +'+north_south+' +datum=WGS84 +units=m +no_defs'
    return proj4

def epsg2proj4(epsg_code):
    epsg_code = str(epsg_code) #forces string, if input is int for example
    zone = epsg_code[3:5]
    if epsg_code[2] == '6':
        north_south = 'north'
    elif epsg_code[2] == '7':
        north_south = 'south'
    proj4 = '+proj=utm +zone='+zone+' +'+north_south+' +datum=WGS84 +units=m +no_defs'
    return proj4

def utmcode2epsg(utm_code):
    #takes utm code in the form of "43Q" and converts to EPSG STRING (not int)
    #letter needs to be between C and X, so not N(orth)/S(outh)
    #EPSG is 32 + 6/7 (North/South) + longitude band
    #Q is in Northern hemisphere so it becomes 32643
    utm_number = utm_code[0:2]
    lat_band = utm_code[2]
    lat_band_number = ord(lat_band)-64

    if lat_band_number <= 13 and lat_band_number >= 3:
        ns_number = '7'
    elif lat_band_number <= 24 and lat_band_number >= 14:
        ns_number = '6'

    epsg = '32' + ns_number + utm_number
    return epsg

def epsg2utm(epsg):
    #EPSG code only gives you north/south information, not full zone
    epsg = str(epsg) #just in case it's delivered as an int
    if epsg[0:2] != '32' or len(epsg) != 5:
        print('Not a valid EPSG to transform to convert to UTM!')
        return
    if epsg[2] == '6':
        north_south = 'North'
    elif epsg[2] == '7':
        north_south = 'South'
    utm_code = 'UTM' + epsg[3:5]
    return utm_code, north_south

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

def main():
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster', help="Path to DEM file")
    parser.add_argument('--csv', help="Path to txt/csv file")
    parser.add_argument('--output_dir', nargs='?')
    parser.add_argument('--max_iters',default='20',nargs='?')
    args = parser.parse_args()

    

    dem_path = args.raster
    csv_path = args.csv
    output_dir = args.output_dir
    max_iter = int(args.max_iter)

    tmp_dir = '/BhaltosMount/Bhaltos/EDUARD/tmp/'

    if os.getcwd() == '/':
        print('Don\'t run this in "/" directory!')
        sys.exit()

    #if no output directory is given, output will be DEM directory
    if output_dir is None:
        output_dir = os.path.dirname(dem_path) + '/'
    elif output_dir is not None:
        if output_dir[len(output_dir)-1] != '/':
            output_dir = output_dir + '/' #force trailing slash
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    #if dem/csv are given as-is, with no folder structure, specify that they're in cwd
    if len(os.path.dirname(csv_path)) == 0:
        csv_path = os.getcwd() + '/' + csv_path
    if len(os.path.dirname(dem_path)) == 0:
        dem_path = os.getcwd() + '/' + dem_path

    epsg_code = osr.SpatialReference(wkt=gdal.Open(dem_path,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1)
    lon_min,lon_max,lat_min,lat_max = get_raster_extents(dem_path,'global')
    x_min,x_max,y_min,y_max = get_raster_extents(dem_path,'local')
    csv_downsampled = csv_path.split('.')[0] + '_n1000.' + csv_path.split('.')[1]

    subprocess.run('awk \'NR % 1000 == 0\' ' + csv_path + ' > ' + csv_downsampled,shell=True)
    df_downsampled = pd.read_csv(csv_downsampled,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})
    
    if np.sum(pd.isnull(df_downsampled.time)) > 0.9*len(df_downsampled):
        time_incl = False
    else:
        time_incl = True

    #lon may actually be x, but we will find out
    lon_downsampled = np.asarray(df_downsampled.lon)
    lat_downsampled = np.asarray(df_downsampled.lat)
    height_downsampled = np.asarray(df_downsampled.height)
    if time_incl:
        time_downsampled = np.asarray(df_downsampled.time)

    if len(lon_downsampled) < 10:
        print(f'Only {len(lon_downsampled)*1000} points for GCP, exiting!')
        sys.exit()

    cond_x_local = np.logical_and(lon_downsampled > x_min,lon_downsampled < x_max)
    cond_lon_global = np.logical_and(lon_downsampled > lon_min,lon_downsampled < lon_max)

    cond_y_local = np.logical_and(lat_downsampled > y_min,lat_downsampled < y_max)
    cond_lat_global = np.logical_and(lat_downsampled > lat_min,lat_downsampled < lat_max)

    cond_xy_local = np.logical_and(cond_x_local,cond_y_local)
    cond_lonlat_global = np.logical_and(cond_lon_global,cond_lat_global)

    xy_local_overlap = np.sum(cond_xy_local)/len(cond_xy_local)
    lonlat_global_overlap = np.sum(cond_lonlat_global)/len(cond_lonlat_global)
    max_overlap = np.maximum(xy_local_overlap,lonlat_global_overlap)

    print(' ')
    print("{:.1f}".format(100*xy_local_overlap) + '% of GCP points in the DEM\'s local projection.')
    print("{:.1f}".format(100*lonlat_global_overlap) + '% of GCP points in the DEM\'s WGS84 projection.')
    print(' ')

    if np.logical_and(xy_local_overlap == lonlat_global_overlap,xy_local_overlap==0):
        print('CSV does not overlap DEM.')
        sys.exit()
    elif max_overlap == xy_local_overlap:
        print('Assuming GCP points are in the same projection, not converting.')
        proj4_str = epsg2proj4(epsg_code)
        output_file = csv_path
        point2dem_output_file = f'{output_dir}{os.path.basename(os.path.splitext(output_file)[0])}_point2dem_results.txt'
        point2dem_command = f'point2dem {output_file} -o {output_dir + os.path.basename(os.path.splitext(output_file)[0])} --nodata-value -9999 --tr 2 --csv-format \"1:easting 2:northing 3:height_above_datum\" --csv-proj4 \"{proj4_str}\" > {point2dem_output_file}'
        subprocess.run(point2dem_command,shell=True)
        dem_align_output_file = f'{output_dir + os.path.basename(os.path.splitext(output_file)[0])}_dem_align_results.txt'
        dem_align_command = f'dem_align.py -outdir {output_dir} -max_iter {max_iter} {output_dir + os.path.basename(os.path.splitext(output_file)[0])}-DEM.tif {dem_path} > {dem_align_output_file}'
        subprocess.run(dem_align_command,shell=True)

    elif max_overlap == lonlat_global_overlap:
        print('Assuming GCP points are in WGS84, converting!')
        df_full = pd.read_csv(csv_path,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})
        lon_full = np.asarray(df_full.lon)
        lat_full = np.asarray(df_full.lat)
        height_full = np.asarray(df_full.height)
        if time_incl:
            time_full = np.asarray(df_full.time)
        x,y,zone = deg2utm(lon_full,lat_full)
        zone = np.asarray(zone)
        epsg_full = [utmcode2epsg(u.replace(' ','')) for u in zone]
        unique_epsg = np.unique(epsg_full)
        for ue in unique_epsg:
            if ue != epsg_code:
                continue
            idx_epsg = [e == ue for e in epsg_full]
            x_zone = x[idx_epsg]
            y_zone = y[idx_epsg]
            height_zone = height_full[idx_epsg]
            if time_incl:
                time_zone = time_full[idx_epsg]
            output_file = os.path.splitext(csv_path)[0] + '_' + ue + '.txt'
            if time_incl:
                np.savetxt(output_file,np.c_[x_zone,y_zone,height_zone,time_zone],fmt='%10.5f,%10.5f,%10.5f,%s')
            else:
                np.savetxt(output_file,np.c_[x_zone,y_zone,height_zone],fmt='%10.5f,%10.5f,%10.5f')
            proj4_str = epsg2proj4(ue)
            point2dem_output_file = f'{output_dir + os.path.basename(os.path.splitext(output_file)[0])}_point2dem_results.txt'
            point2dem_command = f'point2dem {output_file} -o {output_dir + os.path.basename(os.path.splitext(output_file)[0])} --nodata-value -9999 --tr 2 --csv-format \"1:easting 2:northing 3:height_above_datum\" --csv-proj4 \"{proj4_str}\" > {point2dem_output_file}'
            subprocess.run(point2dem_command,shell=True)
            dem_align_output_file = f'{output_dir + os.path.basename(os.path.splitext(output_file)[0])}_dem_align_results.txt'
            dem_align_command = f'dem_align.py -outdir {output_dir} -max_iter {max_iter} {output_dir + os.path.basename(os.path.splitext(output_file)[0])}-DEM.tif {dem_path} > {dem_align_output_file}'
            subprocess.run(dem_align_command,shell=True)
    
    print('Done at ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    main()
