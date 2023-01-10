import ee
import numpy as np
import geopandas as gpd
import pandas as pd
import os,sys
import subprocess
import datetime
import shapely
import configparser
import argparse
import warnings
import time
import ctypes as c
import glob
import io

import google.auth.transport.requests #Request
import google.oauth2.credentials #Credentials
import google_auth_oauthlib.flow #InstalledAppFlow
import googleapiclient.discovery #build
import googleapiclient.errors #HttpError
import googleapiclient.http #MediaIoBaseDownload

def get_lonlat_geometry(geom):
    '''
    Returns lon/lat of all exteriors and interiors of a Shapely geomery:
        -Polygon
        -MultiPolygon
        -GeometryCollection
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    if geom.geom_type == 'Polygon':
        lon_geom,lat_geom = get_lonlat_polygon(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'MultiPolygon':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'GeometryCollection':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    return lon,lat

def get_lonlat_polygon(polygon):
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    exterior_xy = np.asarray(polygon.exterior.xy)
    lon = np.append(lon,exterior_xy[0,:])
    lon = np.append(lon,np.nan)
    lat = np.append(lat,exterior_xy[1,:])
    lat = np.append(lat,np.nan)
    for interior in polygon.interiors:
        interior_xy = np.asarray(interior.coords.xy)
        lon = np.append(lon,interior_xy[0,:])
        lon = np.append(lon,np.nan)
        lat = np.append(lat,interior_xy[1,:])
        lat = np.append(lat,np.nan)
    return lon,lat

def get_lonlat_gdf(gdf):
    '''
    Returns lon/lat of all exteriors and interiors of a GeoDataFrame.
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    for geom in gdf.geometry:
        lon_geom,lat_geom = get_lonlat_geometry(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    return lon,lat

def landmask_csv(lon,lat,lon_coast,lat_coast,landmask_c_file,inside_flag):
    '''
    Given lon/lat of points, and lon/lat of coast (or any other boundary),
    finds points inside the polygon. Boundary must be in the form of separate lon and lat arrays,
    with polygons separated by NaNs
    '''
    c_float_p = c.POINTER(c.c_float)
    landmask_so_file = landmask_c_file.replace('.c','.so') #the .so file is created
    subprocess.run('cc -fPIC -shared -o ' + landmask_so_file + ' ' + landmask_c_file,shell=True)
    landmask_lib = c.cdll.LoadLibrary(landmask_so_file)
    arrx = (c.c_float * len(lon_coast))(*lon_coast)
    arry = (c.c_float * len(lat_coast))(*lat_coast)
    arrx_input = (c.c_float * len(lon))(*lon)
    arry_input = (c.c_float * len(lat))(*lat)
    landmask = np.zeros(len(lon),dtype=c.c_int)
    landmask_lib.pnpoly(c.c_int(len(lon_coast)),c.c_int(len(lon)),arrx,arry,arrx_input,arry_input,c.c_void_p(landmask.ctypes.data))
    landmask = landmask == inside_flag #just to be consistent and return Boolean array
    return landmask

def get_NDVI(s2_image):
    ndvi = s2_image.normalizedDifference(['B8','B4']).rename('NDVI')
    return ndvi

def get_NDWI(s2_image):
    ndwi = s2_image.normalizedDifference(['B3','B8']).rename('NDWI')
    return ndwi

def getANDWI(s2_image):
    red = s2_image.select('B4')
    green = s2_image.select('B3')
    blue = s2_image.select('B2')
    nir = s2_image.select('B8')
    swir1 = s2_image.select('B11')
    swir2 = s2_image.select('B12')
    andwi = (red.add(green).add(blue).subtract(nir).subtract(swir1).subtract(swir2)).divide(red.add(green).add(blue).add(nir).add(swir1).add(swir2)).rename('ANDWI')
    return andwi

def get_NDSI(s2_image):
    ndsi = s2_image.normalizedDifference(['B11','B3']).rename('NDSI')
    return ndsi

def get_Optical(s2_image):
    optical = s2_image.select('B2','B3','B4').rename('Optical')
    return optical

def get_FalseColor(s2_image):
    false_color = s2_image.select('B3','B4','B8').rename('False_Color')
    return false_color

def get_NDVI_threshold(s2_image,NDVI_THRESHOLD):
    return s2_image.normalizedDifference(['B8','B4']).lt(NDVI_THRESHOLD)

def get_NDWI_threshold(s2_image,NDWI_THRESHOLD):
    return s2_image.normalizedDifference(['B3','B8']).gt(NDWI_THRESHOLD)

def get_NDSI_threshold(s2_image,NDSI_THRESHOLD):
    return s2_image.normalizedDifference(['B11','B3']).gt(NDSI_THRESHOLD)

def get_NDVI_NDWI_threshold(s2_image,NDVI_THRESHOLD,NDWI_THRESHOLD):
    return s2_image.normalizedDifference(['B8','B4']).lt(NDVI_THRESHOLD).multiply(s2_image.normalizedDifference(['B3','B8']).lt(NDWI_THRESHOLD))

def get_NDVI_ANDWI_threshold(s2_image,NDVI_THRESHOLD,NDWI_THRESHOLD):
    red = s2_image.select('B4')
    green = s2_image.select('B3')
    blue = s2_image.select('B2')
    nir = s2_image.select('B8')
    swir1 = s2_image.select('B11')
    swir2 = s2_image.select('B12')
    andwi = (red.add(green).add(blue).subtract(nir).subtract(swir1).subtract(swir2)).divide(red.add(green).add(blue).add(nir).add(swir1).add(swir2)).rename('ANDWI')
    return s2_image.normalizedDifference(['B8','B4']).lt(NDVI_THRESHOLD).multiply(andwi.lt(NDWI_THRESHOLD))

def clip_to_geometry(s2_image,geometry):
    return s2_image.clip(geometry)

def count_pixels(image,geometry):
    reduced = image.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13)
    return reduced

def add_cloud_bands(s2_image,CLD_PRB_THRESH):
    cloud_probability = ee.Image(s2_image.get('s2cloudless')).select('probability')
    is_cloud = cloud_probability.gt(CLD_PRB_THRESH).rename('clouds')
    return s2_image.addBands(ee.Image([cloud_probability, is_cloud]))

def add_shadow_bands(s2_image,NIR_DRK_THRESH,SR_BAND_SCALE,CLD_PRJ_DIST):
    not_water = s2_image.select('SCL').neq(6) 
    dark_pixels = s2_image.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    shadow_azimuth = ee.Number(90).subtract(ee.Number(s2_image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    cld_proj = (s2_image.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': s2_image.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return s2_image.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cloud_shadow_mask(s2_image,BUFFER,CLD_PRB_THRESH,NIR_DRK_THRESH,SR_BAND_SCALE,CLD_PRJ_DIST):
    img_cloud = add_cloud_bands(s2_image,CLD_PRB_THRESH)
    img_cloud_shadow = add_shadow_bands(img_cloud,NIR_DRK_THRESH,SR_BAND_SCALE,CLD_PRJ_DIST)
    is_cloud_shadow = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
    is_cloud_shadow = (is_cloud_shadow.focal_min(2).focal_max(BUFFER*2/20)
        .reproject(**{'crs': s2_image.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))
    #return s2_image.addBands(is_cloud_shadow)
    return img_cloud_shadow.addBands(is_cloud_shadow)

def apply_cloud_shadow_mask(s2_image):
    not_cloud_shadow = s2_image.select('cloudmask').Not()
    return s2_image.select('B.*').updateMask(not_cloud_shadow)

def csv_to_convex_hull_shp(df,csv_file,writing=True):
    lon = np.asarray(df.lon)
    lat = np.asarray(df.lat)
    df['day'] = df['time'].str[:10]
    day_list = df.day.unique().tolist()
    day_list_cleaned = [x for x in day_list if str(x) != 'nan']
    idx = [list(df.day).index(x) for x in set(list(df.day))]
    idx_sorted = np.sort(idx)
    idx_sorted = np.append(idx_sorted,len(df))
    gdf = gpd.GeoDataFrame()
    for i in range(len(day_list_cleaned)):
        day_str = day_list_cleaned[i]
        lonlat = np.column_stack((lon[idx_sorted[i]:idx_sorted[i+1]],lat[idx_sorted[i]:idx_sorted[i+1]]))
        if len(lonlat) == 1:
            icesat2_day_polygon = shapely.geometry.Point(lonlat)
        elif len(lonlat) == 2:
            icesat2_day_polygon = shapely.geometry.LineString(lonlat)
        else:
            icesat2_day_polygon = shapely.geometry.Polygon(lonlat)
        conv_hull = icesat2_day_polygon.convex_hull
        conv_hull_buffered = conv_hull.buffer(5E-4)
        df_tmp = pd.DataFrame({'day':[day_str]})
        gdf_tmp = gpd.GeoDataFrame(df_tmp,geometry=[conv_hull_buffered],crs='EPSG:4326')
        gdf = gpd.GeoDataFrame(pd.concat([gdf,gdf_tmp],ignore_index=True))
    gdf = gdf.set_crs('EPSG:4326')
    if writing == True:
        output_file = f'{os.path.splitext(csv_file)[0]}.shp'
        gdf.to_file(output_file)
    return gdf

def find_s2_image(gdf,s2,s2_cloud_probability,DT_SEARCH,CLOUD_FILTER,BUFFER,CLD_PRB_THRESH,NIR_DRK_THRESH,SR_BAND_SCALE,CLD_PRJ_DIST):
    csv_day_ee = ee.Date.parse('YYYY-MM-dd',gdf.day[0])
    csv_geometry = gdf.geometry[0]
    csv_geometry_bounds = csv_geometry.bounds
    csv_geometry_xy = [[x,y] for x,y in zip(csv_geometry.exterior.xy[0],csv_geometry.exterior.xy[1])]
    polygon = ee.Geometry.Polygon(csv_geometry_xy)
    i_date = datetime.datetime.strptime(gdf.day[0],'%Y-%m-%d') - datetime.timedelta(days=DT_SEARCH)
    i_date = i_date.strftime('%Y-%m-%d')
    f_date = datetime.datetime.strptime(gdf.day[0],'%Y-%m-%d') + datetime.timedelta(days=DT_SEARCH+1) #because f_date is exclusive
    f_date = f_date.strftime('%Y-%m-%d')
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
    ymd_ee = (s2_merged_date_region
        .map(lambda image : image.set('date', image.date().format("YYYYMMdd")))
        .distinct('date')
        .aggregate_array('date'))
    ymd_dates = ymd_ee.map(lambda s : ee.Date.parse('YYYYMMdd',s))
    ymd_length = ymd_dates.length().getInfo()
    if ymd_length == 0:
        return None,None,None
    dt_s2_images = ee.Array(ymd_dates.map(lambda s : ee.Date(s).difference(csv_day_ee,'day')))
    s2_subset = (ymd_ee.map(lambda date : s2_merged_date_region.filterMetadata('system:index','contains', date)))
    overlap_ratio = ee.Array(s2_subset.map(lambda img : ee.ImageCollection(img).geometry().intersection(polygon).area().divide(polygon.area())))
    filtered_clouds_single_day = s2_subset.map(lambda img : ee.ImageCollection(img).map(lambda img2 : img2.clip(polygon)))
    filtered_clouds_single_day = filtered_clouds_single_day.map(lambda img : ee.ImageCollection(img).map(lambda img2 : add_cloud_shadow_mask(img2,BUFFER,CLD_PRB_THRESH,NIR_DRK_THRESH,SR_BAND_SCALE,CLD_PRJ_DIST)))
    cloudmask_single_day = filtered_clouds_single_day.map(lambda img : ee.ImageCollection(img).mosaic().select('cloudmask').selfMask())
    notcloudmask_single_day = filtered_clouds_single_day.map(lambda img : ee.ImageCollection(img).mosaic().select('cloudmask').neq(1).selfMask())
    n_clouds = ee.Array(cloudmask_single_day.map(lambda img : count_pixels(ee.Image(img),polygon).get('cloudmask')))
    n_not_clouds = ee.Array(notcloudmask_single_day.map(lambda img : count_pixels(ee.Image(img),polygon).get('cloudmask')))
    cloud_percentage = n_clouds.divide(n_clouds.add(n_not_clouds))
    f1_score = (cloud_percentage.multiply(ee.Number(-1)).add(ee.Number(1))).multiply(overlap_ratio).divide((cloud_percentage.multiply(ee.Number(-1)).add(ee.Number(1))).add(overlap_ratio)).multiply(ee.Number(2))
    f1_modified = f1_score.subtract(dt_s2_images.abs().divide(ee.Number(100))).subtract(dt_s2_images.gt(ee.Number(0)).divide(ee.Number(200)))
    idx_select = f1_modified.argmax().get(0)
    ymd_select = ymd_ee.get(idx_select)
    ymd_select_info = ymd_select.getInfo()
    s2_select = ee.ImageCollection(filtered_clouds_single_day.get(idx_select))
    s2_select_clouds_removed = s2_select.map(lambda img : apply_cloud_shadow_mask(img))
    s2_select_clouds_removed_mosaic = s2_select_clouds_removed.mosaic()
    return s2_select_clouds_removed_mosaic,ymd_select_info,polygon


def export_to_drive(img,filename,geometry,loc_name):
    basename = os.path.splitext(filename)[0]
    export_task = ee.batch.Export.image.toDrive(image=img,
                                        description=basename,
                                        scale=10,
                                        region=geometry,
                                        fileNamePrefix=basename,
                                        crs='EPSG:4326',
                                        fileFormat='GeoTIFF',
                                        folder=f'GEE_{loc_name}')
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

    


def download_img_google_drive(filename,output_folder,tmp_dir,token_json,credentials_json,SCOPES):
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
            f = open(f'{tmp_dir}{filename}','wb')
            f.write(download_code)
            f.close()
            return 0
    return None

def get_idx_subset_day(t_full,t_select):
    t_day = np.asarray([t[:10] for t in t_full])
    idx = t_day == t_select
    return idx

def polygonize_tif(img):
    img_nodata = img.replace('.tif','_nodata_0.tif')
    shp = img.replace('.tif','.shp')
    nodata_command = f'gdal_translate -q -a_nodata 0 {img} {img_nodata}'
    polygonize_command = f'gdal_polygonize.py -q {img_nodata} -f "ESRI Shapefile" {shp}'
    subprocess.run(nodata_command,shell=True)
    subprocess.run(polygonize_command,shell=True)
    return shp

def main():
    ee.Initialize()
    s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    s2_cloud_probability = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    #select (in order): Blue, Green, Red, NIR, Cloud Probability Map, Cloud mask
    # s2 = s2.select('B2','B3','B4','B8','B11','MSK_CLDPRB','QA60')
    print('Loaded Sentinel 2.')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    warnings.simplefilter(action='ignore')
    config_file = 'utils_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Path to csv to filter')
    args = parser.parse_args()
    input_file = args.input_file

    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file')
    
    SCOPES = config.get('GENERAL_CONSTANTS','SCOPES')

    DT_SEARCH = config.getint('GEE_CONSTANTS','DT_SEARCH')
    CLOUD_FILTER = config.getint('GEE_CONSTANTS','CLOUD_FILTER')
    CLD_PRB_THRESH = config.getint('GEE_CONSTANTS','CLD_PRB_THRESH')
    NIR_DRK_THRESH = config.getfloat('GEE_CONSTANTS','NIR_DRK_THRESH')
    CLD_PRJ_DIST = config.getint('GEE_CONSTANTS','CLD_PRJ_DIST')
    BUFFER = config.getint('GEE_CONSTANTS','BUFFER')
    OVERLAP_MINIMUM = config.getfloat('GEE_CONSTANTS','OVERLAP_MINIMUM')
    SR_BAND_SCALE = config.getfloat('GEE_CONSTANTS','SR_BAND_SCALE')
    NDVI_THRESHOLD = config.getfloat('GEE_CONSTANTS','NDVI_THRESHOLD')
    NDWI_THRESHOLD = config.getfloat('GEE_CONSTANTS','NDWI_THRESHOLD')

    token_json = config.get('GDRIVE_PATHS','token_json')
    credentials_json = config.get('GDRIVE_PATHS','credentials_json')

    df = pd.read_csv(input_file,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})
    lon_i2 = np.asarray(df.lon)
    lat_i2 = np.asarray(df.lat)
    height_i2 = np.asarray(df.height)
    time_i2 = np.asarray(df.time)
    gdf_conv_hull = csv_to_convex_hull_shp(df,input_file)

    loc_name = input_file.split('/')[-1].split('_ATL03')[0]
    output_folder_gdrive = f'GEE_{loc_name}'

    t_start_full = datetime.datetime.now()
    for i in range(len(gdf_conv_hull)):
        print(f'{i+1}/{len(gdf_conv_hull)}')
        t_start = datetime.datetime.now()
        i2_ymd = gdf_conv_hull.day[i].replace('-','')
        s2_image,s2_ymd,s2_geometry = find_s2_image(gdf_conv_hull.iloc[[i]].reset_index(drop=True),s2,s2_cloud_probability,DT_SEARCH,CLOUD_FILTER,BUFFER,CLD_PRB_THRESH,NIR_DRK_THRESH,SR_BAND_SCALE,CLD_PRJ_DIST)
        if s2_image is None:
            print('No suitable Sentinel-2 data.')
            continue
        ndvi_ndwi_threshold = get_NDVI_ANDWI_threshold(s2_image,NDVI_THRESHOLD,NDWI_THRESHOLD)
        ndvi_ndwi_threshold_filename = f'{loc_name}_ATL03_{i2_ymd}_S2_{s2_ymd}_NDVI_NDWI_threshold.tif'
        export_code = export_to_drive(ndvi_ndwi_threshold,ndvi_ndwi_threshold_filename,s2_geometry,loc_name)
        if export_code is None:
            print('Google Drive export failed.')
            continue
        t_end = datetime.datetime.now()
        dt = t_end - t_start
        print(f'Processing Sentinel-2 took {dt.seconds + dt.microseconds/1e6:.1f} s.')

        '''
        Access that particular file from Google Drive
            Do credentials at very beginning of script
        Download it
        gdal_translate it with -a_nodata 0
        gdal_polygonize it 
        load up shapefile as numpy array
        landmask the particular segment of the ICESat-2 csv with that lon/lat array
        write out filtered csv
        Concatenate all filtered csvs into one csv
        '''
        t_start = datetime.datetime.now()
        download_code = download_img_google_drive(ndvi_ndwi_threshold_filename,output_folder_gdrive,tmp_dir,token_json,credentials_json,SCOPES)
        if download_code is None:
            print('Could not download image from Google Drive.')
            continue
        ndvi_ndwi_threshold_local_file = f'{tmp_dir}{ndvi_ndwi_threshold_filename}'
        ndvi_ndwi_threshold_shp = polygonize_tif(ndvi_ndwi_threshold_local_file)
        gdf_ndvi_ndwi_threshold = gpd.read_file(ndvi_ndwi_threshold_shp)
        lon_ndvi_ndwi,lat_ndvi_ndwi = get_lonlat_gdf(gdf_ndvi_ndwi_threshold)
        idx_subset_day = get_idx_subset_day(time_i2,gdf_conv_hull.day[i])
        lon_subset_day = lon_i2[idx_subset_day]
        lat_subset_day = lat_i2[idx_subset_day]
        height_subset_day = height_i2[idx_subset_day]
        time_subset_day = time_i2[idx_subset_day]

        landmask = landmask_csv(lon_subset_day,lat_subset_day,lon_ndvi_ndwi,lat_ndvi_ndwi,landmask_c_file,1)
        lon_masked = lon_subset_day[landmask]
        lat_masked = lat_subset_day[landmask]
        height_masked = height_subset_day[landmask]
        time_masked = time_subset_day[landmask]

        output_file = f'{tmp_dir}{loc_name}_{i2_ymd}_Filtered_NDVI_NDWI.txt'
        np.savetxt(output_file,np.c_[lon_masked,lat_masked,height_masked,time_masked.astype(object)],fmt='%f,%f,%f,%s',delimiter=',')
        t_end = datetime.datetime.now()
        dt = t_end - t_start
        print(f'Applying filter took {dt.seconds + dt.microseconds/1e6:.1f} s.')
    
    file_list = sorted(glob.glob(f'{tmp_dir}{loc_name}_*_Filtered_NDVI_NDWI.txt'))
    output_full_file = input_file.replace('.txt','_Filtered_NDVI_NDWI.txt')
    cat_command = f'cat {" ".join(file_list)} > {output_full_file}'
    subprocess.run(cat_command,shell=True)
    t_end_full = datetime.datetime.now()
    dt_full = t_end_full - t_start_full
    if dt_full.seconds > 3600:
        print(f'{loc_name} took {np.floor(dt_full.seconds/3600).astype(int)} hour(s), {np.floor(dt_full.seconds/60).astype(int)} minute(s), {np.mod(dt_full.seconds,60) + dt_full.microseconds/1e6:.1f} s.')
    else:
        print(f'{loc_name} took {np.floor(dt_full.seconds/60).astype(int)} minute(s), {np.mod(dt_full.seconds,60) + dt_full.microseconds/1e6:.1f} s.')

if __name__ == '__main__':
    main()