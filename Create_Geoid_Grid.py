import pyshtools as pysh
import matplotlib.pyplot as plt
import argparse
import datetime
import xarray as xr
from osgeo import gdal,gdalconst,osr
import numpy as np
import os


def main():
    '''
    input_file = '/media/heijkoop/DATA/GEOID/EGM2008_to2190_TideFree'
    output_file = '/media/heijkoop/DATA/GEOID/EGM2008_lmax_899.tif'
    resolution = 0.1
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Path to input spherical harmonics file.')
    parser.add_argument('--output_file',help='Path to output geoid grid file.')
    parser.add_argument('--resolution',help='Resolution of the output grid.')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    resolution = float(args.resolution)
    lmax = int(90/resolution - 1)
    if output_file is None:
        output_file = f'{os.path.splitext(input_file)[0]}_lmax_{lmax}.tif'

    t_start = datetime.datetime.now()

    if input_file.split('/')[-1].split('_')[0] == 'EGM2008':
        clm = pysh.SHGravCoeffs.from_file(input_file,error=True,header=None)
        ellipsoid = 'WGS84'
        clm.gm = pysh.constants.Earth.egm2008.gm.value
        clm.r0 = 6378136.3
        clm.omega = pysh.constants.Earth.egm2008.omega.value
        clm.error_kind = 'formal'
    else:
        clm = pysh.SHGravCoeffs.from_file(input_file,error=True)

    a = pysh.constants.Earth.wgs84.a.value
    b = pysh.constants.Earth.wgs84.b.value
    f = pysh.constants.Earth.wgs84.f.value
    u0 = pysh.constants.Earth.wgs84.u0.value

    geoid_ellipsoid = clm.geoid(u0,a=a,f=f,lmax=lmax)
    geoid_xarray = geoid_ellipsoid.to_xarray()
    geoid_array = geoid_xarray.data
    lon = geoid_xarray.longitude.data
    lat = geoid_xarray.latitude.data

    idx_180 = np.atleast_1d(np.argwhere(lon==180).squeeze())[0]
    geoid_array_180 = np.hstack([geoid_array[:,idx_180:-1],geoid_array[:,0:idx_180+1]])
    lon_180 = np.hstack((lon[idx_180:-1]-360.,lon[0:idx_180+1]))

    nrows,ncols = geoid_array.shape
    xres = resolution
    yres = resolution
    xmin,ymin,xmax,ymax = [np.min(lon_180)-xres/2,np.min(lat)-yres/2,np.max(lon_180)+xres/2,np.max(lat)+xres/2]
    geotransform = (xmin,xres,0,ymax,0, -yres)

    output_raster = gdal.GetDriverByName('GTiff').Create(output_file,ncols, nrows, 1 ,gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    output_raster.SetProjection( srs.ExportToWkt() )
    output_raster.GetRasterBand(1).WriteArray(geoid_array_180)
    output_raster.FlushCache()
    output_raster = None
    
    t_end = datetime.datetime.now()
    print('Time taken: ', t_end - t_start)


if __name__ == '__main__':
    main()