import subprocess
import argparse
from osgeo import gdal,gdalconst,osr
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster',help='Path to raster to clip.',default=None)
    parser.add_argument('--shp',help='Path to shapefile to clip with.',default=None)
    parser.add_argument('--crop',help='Crop to cutline.',default=False,action='store_true')
    parser.add_argument('--output_file',help='Output file.',default=None)
    args = parser.parse_args()
    
    raster = args.raster
    shp = args.shp
    output_file = args.output_file
    crop_flag = args.crop

    epsg_code = osr.SpatialReference(wkt=gdal.Open(raster,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1)
    shp_base = os.path.splitext(os.path.basename(shp))[0]
    if output_file is None:
        output_file = f'{os.path.splitext(raster)[0]}_Clipped{os.path.splitext(raster)[1]}'
    if crop_flag == True:
        crop_str = '-crop_to_cutline '
    else:
        crop_str = ''
    
    cmd = (
            f'gdalwarp '
            f'-s_srs EPSG:{epsg_code} '
            f'-t_srs EPSG:{epsg_code} '
            f'-of GTiff '
            f'-cutline {shp} '
            f'-cl {shp_base} '
            f'{crop_str}'
            f'-co COMPRESS=LZW -co BIGTIFF=YES '
            f'{raster} '
            f'{output_file}'
            )
    
    subprocess.run(cmd,shell=True)


if __name__ == '__main__':
    main()