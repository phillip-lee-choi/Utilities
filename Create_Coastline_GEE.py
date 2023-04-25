import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy import ndimage
from osgeo import gdal,gdalconst,osr
import argparse
import configparser
import warnings
import datetime
import glob
import sys
import os
import time
import io
import subprocess

import google.auth.transport.requests #Request
import google.oauth2.credentials #Credentials
import google_auth_oauthlib.flow #InstalledAppFlow
import googleapiclient.discovery #build
import googleapiclient.errors #HttpError
import googleapiclient.http #MediaIoBaseDownload

def get_strip_list(loc_dir,input_type,corrected_flag,dir_structure):
    '''
    Different input types:
        0: old and new methods
        1: old only (dem_browse.tif for 10 m and dem_smooth.tif for 2 m)
        2: new only (dem_10m.tif for 10 m and dem.tif for 2 m)
        3: input is from a list of strips
    '''
    if dir_structure == 'sealevel':
        if corrected_flag == True:
            strip_list_old = sorted(glob.glob(f'{loc_dir}*/strips/*dem_smooth_Shifted*.tif'))
            strip_list_old.extend(sorted(glob.glob(f'{loc_dir}UTM*/*/strips/*dem_smooth_Shifted*.tif')))
            strip_list_new = sorted(glob.glob(f'{loc_dir}*/strips/*dem_Shifted*.tif'))
            strip_list_new.extend(sorted(glob.glob(f'{loc_dir}UTM*/*/strips/*dem_Shifted*.tif')))
        else:
            strip_list_old = sorted(glob.glob(f'{loc_dir}*/strips/*dem_smooth.tif'))
            strip_list_old.extend(sorted(glob.glob(f'{loc_dir}UTM*/*/strips/*dem_smooth.tif')))
            strip_list_new = sorted(glob.glob(f'{loc_dir}*/strips/*dem.tif'))
            strip_list_new.extend(sorted(glob.glob(f'{loc_dir}UTM*/*/strips/*dem.tif')))
        if input_type == 0:
            strip_list = strip_list_old
            strip_list.extend(strip_list_new)
        elif input_type == 1:
            strip_list = strip_list_old
        elif input_type == 2:
            strip_list = strip_list_new
    elif dir_structure == 'simple':
        if corrected_flag == True:
            strip_list = sorted(glob.glob(f'{loc_dir}*dem_Shifted*.tif'))
        else:
            strip_list = sorted(glob.glob(f'{loc_dir}*dem.tif'))
    return np.asarray(strip_list)

def get_extent(gt,cols,rows):
    '''
    Return list of corner coordinates from a geotransform
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

def reproject_coords(coords,src_srs,tgt_srs):
    '''
    Reproject a list of x,y coordinates.
    x and y are in src coordinates, going to tgt
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        y,x,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def get_strip_extents(strip):
    '''
    Find extents of a given strip.
    Return will be lon/lat in EPSG:4326.
    '''
    src = gdal.Open(strip)
    gt = src.GetGeoTransform()
    cols = src.RasterXSize
    rows = src.RasterYSize
    ext = get_extent(gt,cols,rows)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src.GetProjection())
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    geo_ext = reproject_coords(ext,src_srs,tgt_srs)
    lon_strip = [item[0] for item in geo_ext]
    lat_strip = [item[1] for item in geo_ext]
    lon_min = np.nanmin(lon_strip)
    lon_max = np.nanmax(lon_strip)
    lat_min = np.nanmin(lat_strip)
    lat_max = np.nanmax(lat_strip)
    return lon_min,lon_max,lat_min,lat_max

def get_s2_image(polygon,s2,s2_cloud_probability,CLOUD_FILTER,
                 i_date=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),f_date=datetime.datetime.now().strftime("%Y-%m-%d")):
    s2_date_region = s2.filterDate(i_date,f_date).filterBounds(polygon)
    s2_cloud_probability_date_region = s2_cloud_probability.filterDate(i_date,f_date).filterBounds(polygon)
    s2_merged_date_region = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary':s2_date_region.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)),
        'secondary':s2_cloud_probability_date_region,
        'condition':ee.Filter.equals(**{
            'leftField':'system:index',
            'rightField':'system:index'
        })
    }))
    s2_median = s2_merged_date_region.median().clip(polygon)
    return s2_median

def get_ANDWI_threshold(s2_image,andwi_threshold):
    red = s2_image.select('B4')
    green = s2_image.select('B3')
    blue = s2_image.select('B2')
    nir = s2_image.select('B8')
    swir1 = s2_image.select('B11')
    swir2 = s2_image.select('B12')
    andwi = (red.add(green).add(blue).subtract(nir).subtract(swir1).subtract(swir2)).divide(red.add(green).add(blue).add(nir).add(swir1).add(swir2)).rename('ANDWI')
    return andwi.gt(andwi_threshold)

def export_to_drive(img,filename,geometry,loc_name,max_pixels=1e8):
    basename = os.path.splitext(filename)[0]
    export_task = ee.batch.Export.image.toDrive(image=img,
                                        description=basename,
                                        scale=10,
                                        region=geometry,
                                        fileNamePrefix=basename,
                                        crs='EPSG:4326',
                                        fileFormat='GeoTIFF',
                                        folder=f'GEE_{loc_name}',
                                        maxPixels=max_pixels)
    export_task.start()
    waiting = True
    while waiting:
        if export_task.status()['state'] == 'COMPLETED':
            waiting = False
        elif export_task.status()['state'] == 'FAILED':
            waiting = False
            return None
        else:
            time.sleep(5)
    return 0

def get_google_drive_credentials(token_json,credentials_json,SCOPES):
    creds = None
    if os.path.exists(token_json):
        creds = google.oauth2.credentials.Credentials.from_authorized_user_file(token_json, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                credentials_json, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_json, 'w') as token:
            token.write(creds.to_json())
    return creds

def get_google_drive_dir_id(gdrive_service,dir_name):
    page_token = None
    folders = []
    query = f"name = '{dir_name}' and mimeType = 'application/vnd.google-apps.folder'"
    try:
        while True:
            response = gdrive_service.files().list(q=query,
                                                   spaces='drive',
                                                   fields='nextPageToken, '
                                                   'files(id, name)',
                                                   pageToken=page_token).execute()
            folders.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
    except googleapiclient.errors.HttpError as error:
        print(f'An error occurred: {error}')
        folders = None
    return folders

def get_google_drive_file_id(gdrive_service,dir_id,file_name):
    '''
    query = f"name = '{file_base}' and mimeType = 'image/tiff' and '{dir_id}' in parents"
    '''
    file_base = os.path.splitext(file_name)[0]
    page_token = None
    files = []
    query = f"mimeType = 'image/tiff' and '{dir_id}' in parents"
    try:
        while True:
            response = gdrive_service.files().list(q=query,
                                                   spaces='drive',
                                                   fields='nextPageToken, '
                                                   'files(id, name)',
                                                   pageToken=page_token).execute()
            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
    except googleapiclient.errors.HttpError as error:
        print(f'An error occurred: {error}')
        files = None
    return files

def download_google_drive_id(gdrive_service,file_id):
    try:
        request = gdrive_service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = googleapiclient.http.MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
    except googleapiclient.errors.HttpError as error:
        print(F'An error occurred: {error}')
        file = None
        return 0
    return file.getvalue()

def download_img_google_drive(filename,output_folder,output_dir,token_json,credentials_json,SCOPES):
    creds = get_google_drive_credentials(token_json,credentials_json,[SCOPES])
    service = googleapiclient.discovery.build('drive', 'v3', credentials=creds)
    folder_list = get_google_drive_dir_id(service,output_folder)
    for folder in folder_list:
        folder_id = folder['id']
        folder_name = folder['name']
        if folder_name != output_folder:
            continue
        file_list = get_google_drive_file_id(service,folder_id,filename)
        idx_select = np.argwhere([f['name'] == filename for f in file_list])
        if len(idx_select) == 0:
            continue
        idx_select = np.atleast_1d(idx_select.squeeze())[0]
        file_id = file_list[idx_select]['id']
        download_code = download_google_drive_id(service,file_id)
        if download_code == 0:
            continue
        else:
            f = open(f'{output_dir}{filename}','wb')
            f.write(download_code)
            f.close()
            return 0
    return None

def connected_components(arr,threshold=0.01):
    area = arr.shape[0]*arr.shape[1]
    label, num_label = ndimage.label(arr == 1)
    size = np.bincount(label.ravel())
    label_IDs_sorted = np.argsort(size)[::-1] #sort the labels by size (descending, so biggest first), then remove label=0, because that one is land
    label_IDs_sorted = label_IDs_sorted[label_IDs_sorted != 0]
    clump = np.zeros(arr.shape,dtype=int)
    for label_id in label_IDs_sorted:
        if size[label_id]/area < threshold:
            break
        clump = clump + np.asarray(label==label_id,dtype=int)
    return clump

def write_new_array_geotiff(src,arr,filename,dtype):
    '''
    Based on an array read from src, write a new geotiff with the array
    '''
    wide = src.RasterXSize
    high = src.RasterYSize
    src_andwi_proj = src.GetProjection()
    src_andwi_geotrans = src.GetGeoTransform()
    dst = gdal.GetDriverByName('GTiff').Create(filename, wide, high, 1 , dtype)
    outBand = dst.GetRasterBand(1)
    outBand.WriteArray(arr,0,0)
    outBand.FlushCache()
    outBand.SetNoDataValue(0)
    dst.SetProjection(src_andwi_proj)
    dst.SetGeoTransform(src_andwi_geotrans)
    del dst
    return None

def polygonize_tif(img):
    img_nodata = img.replace('.tif','_nodata_0.tif')
    shp = img.replace('.tif','.shp')
    nodata_command = f'gdal_translate -q -a_nodata 0 {img} {img_nodata}'
    polygonize_command = f'gdal_polygonize.py -q {img_nodata} -f "ESRI Shapefile" {shp}'
    subprocess.run(nodata_command,shell=True)
    subprocess.run(polygonize_command,shell=True)
    subprocess.run(f'rm {img_nodata}',shell=True)
    return shp

def error_handling(input_location,lonlat_extents,loc_name):
    if input_location is None:
        if lonlat_extents is None and loc_name is None:
            print('Please provide either an input location or lonlat extents or a location name.')
            return 1
        elif lonlat_extents is None and loc_name is not None:
            print('Please provide lon/lat extents.')
            return 1
        elif lonlat_extents is not None and loc_name is None:
            print('Please provide a location name.')
            return 1
        else:
            return 0
    else:
        if lonlat_extents is None:
            if loc_name is None:
                return 0
            else:
                print('Using input location name.')
                return 0
        else:
            print('Won\'t override location extents.')
            return 1

def get_lonlat_bounds_gdf(gdf):
    '''
    Returns the lon/lat boundarys of an entire GeoDataFrame.
    '''
    lon_min = np.min(gdf.bounds.minx)
    lon_max = np.max(gdf.bounds.maxx)
    lat_min = np.min(gdf.bounds.miny)
    lat_max = np.max(gdf.bounds.maxy)
    return lon_min,lon_max,lat_min,lat_max

def lonlat2epsg(lon,lat):
    '''
    Finds the EPSG code for a given lon/lat coordinate.
    '''
    if lat >= 0:
        NS_code = '6'
    elif lat < 0:
        NS_code = '7'
    EW_code = f'{int(np.floor(lon/6.0))+31:02d}'
    epsg_code = f'32{NS_code}{EW_code}'
    return epsg_code

def simplify_polygon(geom,threshold):
    '''
    Simplifies the polygon
    '''
    new_geom = geom.simplify(threshold)
    return new_geom


def main():
    t_start = datetime.datetime.now()
    ee.Initialize()
    s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    s2_cloud_probability = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    #select (in order): Blue, Green, Red, NIR, Cloud Probability Map, Cloud mask
    print('Loaded Sentinel 2.')
    warnings.simplefilter(action='ignore')
    config_file = 'utils_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input location with DSMs.',default=None)
    parser.add_argument('--lonlat', help='Input lonlat extents (lon_min,lon_max,lat_min,lat_max).',nargs=4,default=None)
    parser.add_argument('--loc_name', help='Location name.',default=None)
    parser.add_argument('--t_start',help='Start date in YYYY-MM-DD format.',default=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"))
    parser.add_argument('--t_end',help='End date in YYYY-MM-DD format.',default=datetime.datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--simplify_radius',help='Radius with which to simplify coastline.',default=None)
    args = parser.parse_args()
    input_location = args.input
    lonlat_extents = args.lonlat
    loc_name = args.loc_name
    t_start = args.t_start
    t_end = args.t_end
    simplify_radius = int(args.simplify_radius)
    error_code = error_handling(input_location,lonlat_extents,loc_name)
    if error_code == 1:
        sys.exit()
    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')

    if input_location is None:
        input_location = tmp_dir
    if input_location[-1] != '/':
        input_location += '/'
    if loc_name is None:
        loc_name = input_location.split('/')[-2]
    if os.path.isdir(f'{input_location}Coast') == False:
        os.mkdir(f'{input_location}Coast')
    coast_dir = f'{input_location}Coast/'

    output_folder_gdrive = f'GEE_{loc_name}'
    
    SCOPES = config.get('GENERAL_CONSTANTS','SCOPES')
    CLOUD_FILTER = config.getint('GEE_CONSTANTS','CLOUD_FILTER')
    NDWI_THRESHOLD = config.getfloat('GEE_CONSTANTS','NDWI_THRESHOLD')
    token_json = config.get('GDRIVE_PATHS','token_json')
    credentials_json = config.get('GDRIVE_PATHS','credentials_json')

    input_type = 0
    corrected_flag = False
    dir_structure = 'sealevel'

    t_start_process1 = datetime.datetime.now()
    if lonlat_extents is None:
        strip_list = get_strip_list(input_location,input_type,corrected_flag,dir_structure)
        if len(strip_list) == 0:
            print('No strips found.')
            sys.exit()
        lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips = 180,-180,90,-90
        for strip in strip_list:
            lon_min_single_strip,lon_max_single_strip,lat_min_single_strip,lat_max_single_strip = get_strip_extents(strip)
            lon_min_strips = np.min((lon_min_strips,lon_min_single_strip))
            lon_max_strips = np.max((lon_max_strips,lon_max_single_strip))
            lat_min_strips = np.min((lat_min_strips,lat_min_single_strip))
            lat_max_strips = np.max((lat_max_strips,lat_max_single_strip))
        if lon_min_strips == 180 or lon_max_strips == -180 or lat_min_strips == 90 or lat_max_strips == -90:
            print('No valid strips found.')
            sys.exit()
        lonlat_extents = [lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips]
    else:
        lonlat_extents = [float(item) for item in lonlat_extents]

    gdf_extents = gpd.GeoDataFrame({'geometry': [shapely.geometry.Polygon([(lonlat_extents[0],lonlat_extents[2]),(lonlat_extents[1],lonlat_extents[2]),(lonlat_extents[1],lonlat_extents[3]),(lonlat_extents[0],lonlat_extents[3]),(lonlat_extents[0],lonlat_extents[2])])]},crs='EPSG:4326')
    geometry_xy = [[x,y] for x,y in zip(gdf_extents.exterior[0].xy[0],gdf_extents.exterior[0].xy[1])]
    extents_polygon = ee.Geometry.Polygon(geometry_xy)

    s2_median = get_s2_image(extents_polygon,s2,s2_cloud_probability,CLOUD_FILTER,t_start,t_end)
    andwi_threshold = get_ANDWI_threshold(s2_median,NDWI_THRESHOLD)

    andwi_threshold_filename = f'{loc_name}_S2_ANDWI_threshold.tif'
    export_code = export_to_drive(andwi_threshold,andwi_threshold_filename,extents_polygon,loc_name)
    if export_code is None:
        print('Google Drive export failed.')
        print('Trying with higher number of max pixels.')
        export_code = export_to_drive(andwi_threshold,andwi_threshold_filename,extents_polygon,loc_name,max_pixels=1e9)
        if export_code is None:
            print('Google Drive export failed.')
            sys.exit()
    t_end_process1 = datetime.datetime.now()
    dt_process1 = t_end_process1 - t_start_process1
    print(f'Processing Sentinel-2 took {dt_process1.seconds + dt_process1.microseconds/1e6:.1f} s.')
    t_start_process2 = datetime.datetime.now()
    download_code = download_img_google_drive(andwi_threshold_filename,output_folder_gdrive,coast_dir,token_json,credentials_json,SCOPES)
    if download_code is None:
        print('Could not download image from Google Drive.')
        sys.exit()
    andwi_threshold_local_file = f'{coast_dir}{andwi_threshold_filename}'
    andwi_coastline_tif_file = f'{coast_dir}{loc_name}_S2_ANDWI_Coastline.tif'
    andwi_surface_water_tif_file = f'{coast_dir}{loc_name}_S2_ANDWI_Surface_Water.tif'
    src_andwi = gdal.Open(andwi_threshold_local_file)
    andwi_data = np.asarray(src_andwi.GetRasterBand(1).ReadAsArray())
    surface_water_data = connected_components(andwi_data)
    coastline_data = surface_water_data*-1 + 1
    write_code = write_new_array_geotiff(src_andwi,coastline_data,andwi_coastline_tif_file,gdalconst.GDT_Byte)
    andwi_coastline_shp_file = polygonize_tif(andwi_coastline_tif_file)
    write_code = write_new_array_geotiff(src_andwi,surface_water_data,andwi_surface_water_tif_file,gdalconst.GDT_Byte)
    andwi_surface_water_shp_file = polygonize_tif(andwi_surface_water_tif_file)
    '''
    Add code to simplify to ~10 m or 20 m
    '''
    if simplify_radius is not None:
        simplified_coastline_shp_file = f'{coast_dir}{loc_name}_S2_ANDWI_Coastline_Simplified.shp'
        simplified_surface_water_shp_file = f'{coast_dir}{loc_name}_S2_ANDWI_Surface_Water_Simplified.shp'
        gdf_coastline = gpd.read_file(andwi_coastline_shp_file)
        gdf_surface_water = gpd.read_file(andwi_surface_water_shp_file)
        lon_min,lon_max,lat_min,lat_max = get_lonlat_bounds_gdf(gdf_coastline)
        lon_center = np.mean((lon_min,lon_max))
        lat_center = np.mean((lat_min,lat_max))
        epsg_code = lonlat2epsg(lon_center,lat_center)
        gdf_coastline = gdf_coastline.to_crs(f'EPSG:{epsg_code}')
        gdf_coastline_simplified = gdf_coastline.copy()
        gdf_coastline_simplified['geometry'] = gdf_coastline_simplified.apply(lambda x : simplify_polygon(x.geometry,simplify_radius),axis=1)
        gdf_coastline_simplified = gdf_coastline_simplified.to_crs('EPSG:4326')
        gdf_coastline_simplified.to_file(simplified_coastline_shp_file)
        gdf_surface_water = gdf_surface_water.to_crs(f'EPSG:{epsg_code}')
        gdf_surface_water_simplified = gdf_surface_water.copy()
        gdf_surface_water_simplified['geometry'] = gdf_surface_water_simplified.apply(lambda x : simplify_polygon(x.geometry,simplify_radius),axis=1)
        gdf_surface_water_simplified = gdf_surface_water_simplified.to_crs('EPSG:4326')
        gdf_surface_water_simplified.to_file(simplified_surface_water_shp_file)




    t_end_process2 = datetime.datetime.now()
    dt_process2 = t_end_process2 - t_start_process2
    print(f'Processing image took {dt_process2.seconds + dt_process2.microseconds/1e6:.1f} s.')

if __name__ == '__main__':
    main()