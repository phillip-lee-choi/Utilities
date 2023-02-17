import numpy as np
from osgeo import gdal, osr
import glob
import warnings
import configparser


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



def main():
    config_file = 'utils_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    warnings.simplefilter(action='ignore')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--loc_name',default=None,help='name of location')
    # args = parser.parse_args()

    NASA_SEALEVEL_dir = config.get('GENERAL_PATHS','NASA_SEALEVEL_dir')
    input_type = 0
    corrected_flag = False
    dir_structure = 'sealevel'

    continents_list = ['Africa','Asia','Europe','Middle_East','Oceania','South_America','North_America']
    for continent in continents_list:
        continent_dir = f'{NASA_SEALEVEL_dir}{continent}/'
        output_file = f'{NASA_SEALEVEL_dir}{continent}_DSM_extents.txt'
        f = open(output_file,'w')
        if continent == 'North_America':
            continent_dir = continent_dir.replace('/BhaltosMount/Bhaltos/','/home/eheijkoop/Bhaltos2/')
        loc_list = sorted(glob.glob(f'{continent_dir}*/'))
        for loc_dir in loc_list:
            loc_name = loc_dir.replace(continent_dir,'').replace('/','')
            print(loc_name)
            strip_list = get_strip_list(loc_dir,input_type,corrected_flag,dir_structure)
            lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips = 180,-180,90,-90
            for strip in strip_list:
                lon_min_single_strip,lon_max_single_strip,lat_min_single_strip,lat_max_single_strip = get_strip_extents(strip)
                lon_min_strips = np.min((lon_min_strips,lon_min_single_strip))
                lon_max_strips = np.max((lon_max_strips,lon_max_single_strip))
                lat_min_strips = np.min((lat_min_strips,lat_min_single_strip))
                lat_max_strips = np.max((lat_max_strips,lat_max_single_strip))
            f.write(f'{loc_name},{lon_min_strips:.2f},{lon_max_strips:.2f},{lat_min_strips:.2f},{lat_max_strips:.2f}')
        f.close()

if __name__ == '__main__':
    main()