import argparse
import numpy as np
import sys
import geopandas as gpd
import shapely
import warnings

def coords_to_xy(coords,reverse_coords_flag=False):
    coords = np.asarray(coords)
    if not np.isnan(coords[-1]):
        coords = np.append(coords,np.nan)
    if reverse_coords_flag == False:
        x = coords[:-1][0::2]
        y = coords[:-1][1::2]
    else:
        x = coords[:-1][1::2]
        y = coords[:-1][0::2]
    if x[0] != x[-1] or y[0] != y[-1]:
        print('First and last coordinates must be the same.')
        if x[0] != x[-1]:
            x = np.append(x,x[0])
        if y[0] != y[-1]:
            y = np.append(y,y[0])
    if len(x) != len(y):
        print('x and y must have the same length.')
        return None,None
    return x,y


def filter_xy(coords,reverse_coords_flag=False):
    coords = np.asarray([c.lower().replace('none','NaN').replace('break','NaN').replace('pause','NaN') for c in coords])
    coords = coords.astype(float)
    if not np.isnan(coords[-1]):
        coords = np.append(coords,np.nan)
    idx_nan = np.atleast_1d(np.argwhere(np.isnan(coords)).squeeze()) + 1
    if idx_nan.size > 0:
        idx_nan = np.append(0,idx_nan)
        polygon_list = []
        for i in range(len(idx_nan)-1):
            coords_subset = coords[idx_nan[i]:idx_nan[i+1]][:-1]
            x,y = coords_to_xy(coords_subset,reverse_coords_flag=reverse_coords_flag)
            p = shapely.geometry.Polygon(zip(x,y))
            polygon_list.append(p)
    else:
        x,y = coords_to_xy(coords_subset,reverse_coords_flag=reverse_coords_flag)
        p = shapely.geometry.Polygon(zip(x,y))
        polygon_list = [p]
    return polygon_list
    

def main():
    warnings.simplefilter(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--coords',nargs='*',help='Coordinates (lon,lat)',default=None)
    parser.add_argument('--output_file',help='Output file',default='output.shp')
    parser.add_argument('--epsg',help='EPSG code',default='4326')
    parser.add_argument('--reverse_coords',help='Reverse order of coordinates',default=False,action='store_true')
    args = parser.parse_args()
    output_file = args.output_file
    epsg_code = args.epsg
    reverse_coords_flag = args.reverse_coords
    if args.coords is None:
        print('No coordinates provided.')
        sys.exit()
    else:
        coords = args.coords
    
    polygon_list = filter_xy(coords,reverse_coords_flag=reverse_coords_flag)    
    gdf = gpd.GeoDataFrame(geometry=polygon_list,crs=f'EPSG:{epsg_code}')
    gdf.to_file(output_file)

if __name__ == '__main__':
    main()