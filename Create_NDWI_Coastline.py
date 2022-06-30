import numpy as np
import scipy.ndimage
import geopandas as gpd
from osgeo import gdal,gdalconst,osr
import argparse
import subprocess
from shapely.geometry import Polygon, MultiPolygon

def load_image(img_path):
    '''
    Loads a geotiff file as a numpy array.
    '''
    img = gdal.Open(img_path,gdalconst.GA_ReadOnly)
    img_array = np.array(img.GetRasterBand(1).ReadAsArray())
    return img_array

def flood_fill(img_binary,erosion_struc=np.ones((3,3))):
    '''
    Performs a flood-fill operation on a binary image.
    '''
    img_binary_eroded = scipy.ndimage.binary_dilation(img_binary)
    img_binary_floodfill = scipy.ndimage.binary_erosion(img_binary_eroded,structure=erosion_struc)
    return img_binary_floodfill

def write_file(raster,file_name,src):
    '''
    Given a numpy array and a file name, writes the array to a geotiff file, using the information in src.
    '''
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(file_name,raster.shape[1],raster.shape[0],1,gdal.GDT_Byte)
    out_raster.SetGeoTransform(src.GetGeoTransform())
    out_raster.SetProjection(src.GetProjection())
    out_raster.GetRasterBand(1).WriteArray(raster)
    out_raster.FlushCache()
    out_raster = None

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

def remove_interior_holes_polygon(geom,threshold):
    '''
    Removes interior holes smaller than a given threshold from the given geometry.
    '''
    if geom.geom_type == 'Polygon':
        interior_list = []
        for interior in geom.interiors:
            p = Polygon(interior)
            if p.area > threshold:
                interior_list.append(interior)
        new_geom = Polygon(geom.exterior.coords,holes=interior_list)
    elif geom.geom_type == 'MultiPolygon':
        polygon_list = []
        for polygon in geom.geoms:
            interior_list = []
            for interior in geom.interiors:
                p = Polygon(interior)
                if p.area > threshold:
                    interior_list.append(interior)
            polygon_list.append(Polygon(polygon.exterior.coords, holes=interior_list))
        new_geom = MultiPolygon(geom.exterior.coords,holes=interior_list)
    else:
        new_geom = geom
    return new_geom

def main():
    '''
    Write by Eduard Heijkoop, University of Colorado
    Update June 2022: Created script

    This script creates a coastlines shapefile from an NDWI image (e.g. from Sentinel-2 or Landsat-8/-9).
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Path to input NDWI file.')
    args = parser.parse_args()
    input_file = args.input_file

    NDWI_THRESHOLD = 0.0
    SHP_POLYGON_THRESHOLD = 0.005
    INTERIOR_THRESHOLD = 50000

    output_file_floodfill = input_file.replace('.tif','_floodfill.tif')
    output_file_floodfill_nodata = output_file_floodfill.replace('.tif','_nodata_0.tif')
    output_file_floodfill_shp = output_file_floodfill.replace('.tif','.shp')
    output_file_shp = input_file.replace('.tif','.shp')

    src = gdal.Open(input_file,gdalconst.GA_ReadOnly)
    ndwi = np.array(src.GetRasterBand(1).ReadAsArray())
    ndwi_lt_0p0 = ndwi < NDWI_THRESHOLD
    ndwi_lt_0p0 = ndwi_lt_0p0.astype(int)

    ndwi_lt_0p0_floodfill = flood_fill(ndwi_lt_0p0)
    write_file(ndwi_lt_0p0_floodfill,output_file_floodfill,src)
    nodata_command = f'gdal_translate -a_nodata 0 -of GTiff -co "COMPRESS=LZW" {output_file_floodfill} {output_file_floodfill_nodata}'
    polygonize_command = f'gdal_polygonize.py {output_file_floodfill_nodata} -f "ESRI Shapefile" {output_file_floodfill_shp}'
    subprocess.run(nodata_command,shell=True)
    subprocess.run(polygonize_command,shell=True)
    subprocess.run(f'rm {output_file_floodfill}',shell=True)
    subprocess.run(f'rm {output_file_floodfill_nodata}',shell=True)

    gdf = gpd.read_file(output_file_floodfill_shp)

    lon_min,lon_max,lat_min,lat_max = get_lonlat_bounds_gdf(gdf)
    lon_center = np.mean((lon_min,lon_max))
    lat_center = np.mean((lat_min,lat_max))
    epsg_code = lonlat2epsg(lon_center,lat_center)
    gdf = gdf.to_crs(f'EPSG:{epsg_code}')
    idx_area = np.argwhere(np.asarray(gdf.area) > SHP_POLYGON_THRESHOLD * np.sum(gdf.area)).squeeze()
    gdf = gdf.iloc[idx_area].reset_index(drop=True)

    gdf['geometry'] = gdf.apply(lambda x : remove_interior_holes_polygon(x.geometry,INTERIOR_THRESHOLD),axis=1)
    gdf = gdf.to_crs('EPSG:4326')
    gdf.to_file(output_file_shp)
    subprocess.run(f'rm {output_file_floodfill_shp.replace(".shp",".*")}',shell=True)

if __name__ == '__main__':
    main()