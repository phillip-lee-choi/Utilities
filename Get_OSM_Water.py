import overpy
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import argparse
import sys


def get_osm_water(lon_min,lon_max,lat_min,lat_max):
    '''
    Given a lon/lat extent (order for OSM is lat/lon),
    downloads all buildings in that region
    Returns result, which is an overpy structure
    '''
    api = overpy.Overpass()
    bbox = str(lat_min)+','+str(lon_min)+','+str(lat_max)+','+str(lon_max)
    result = api.query("""
    [out:json][timeout:2000][maxsize:1073741824];
    (
    way["natural"="water"]("""+bbox+""");
    relation["natural"="water"]("""+bbox+""");
    );
    out body;
    >;
    out skel qt;
    """)
    return result

def find_polygon_members(lon_start,lat_start,lon_end,lat_end,idx_relations):
    '''
    Given a set of lon/lat start & end, find the best polygon outline 
    	-By sorting in the right order, and
        -By flipping the correct members
    '''
    idx_start = 0
    lon_start_full = lon_start[idx_start]
    lat_start_full = lat_start[idx_start]
    lon_end_full = lon_end[idx_start]
    lat_end_full = lat_end[idx_start]
    idx_sorted = np.atleast_1d(idx_relations[idx_start])
    idx_flipped = np.atleast_1d(False)
    lon_start = np.delete(lon_start,idx_start)
    lat_start = np.delete(lat_start,idx_start)
    lon_end = np.delete(lon_end,idx_start)
    lat_end = np.delete(lat_end,idx_start)
    idx_relations = np.delete(idx_relations,idx_start)
    while len(lon_start) > 0:
        idx_next = np.argwhere(np.logical_and(lon_start == lon_end_full,lat_start == lat_end_full))
        idx_next_flipped = np.argwhere(np.logical_and(lon_end == lon_end_full,lat_end == lat_end_full))
        #build condition to select the right one
        if len(idx_next) == 1:
            flip = False
        elif len(idx_next) == 0 and len(idx_next_flipped) == 1:
            flip = True
            idx_next = idx_next_flipped
        else:
            return None,None
        idx_next = np.atleast_1d(idx_next.squeeze())[0]
        idx_sorted = np.append(idx_sorted,idx_relations[idx_next])
        idx_flipped = np.append(idx_flipped,flip)
        #update
        if flip == True:
            lon_end_full = lon_start[idx_next]
            lat_end_full = lat_start[idx_next]
        else:
            lon_end_full = lon_end[idx_next]
            lat_end_full = lat_end[idx_next]
        lon_start = np.delete(lon_start,idx_next)
        lat_start = np.delete(lat_start,idx_next)
        lon_end = np.delete(lon_end,idx_next)
        lat_end = np.delete(lat_end,idx_next)
        idx_relations = np.delete(idx_relations,idx_next)
    return idx_sorted,idx_flipped

def overpy_to_gdf(overpy_struc,unary_flag=True):
    '''
    Given an overpy structure, subsets into numpy arrays of lon/lat
    Turns these into a Shapely polygon which can be used in a GeoDataFrame
    OpenStreetMap Way ID is included for added information (e.g. reverse lookup)
    Lines are of length 2, can't turn those into a polygon, so they are skipped
    '''
    poly_list = []
    idx_pop = []
    for way in overpy_struc.ways:
        lon = np.asarray([float(node.lon) for node in way.nodes])
        lat = np.asarray([float(node.lat) for node in way.nodes])
        if len(lon) < 3:
            continue
        if lon[0] != lon[-1]:
            continue
        lonlat = np.stack((lon,lat),axis=-1)
        poly = shapely.geometry.Polygon(lonlat)
        poly_list.append(poly)
    for relation in overpy_struc.relations:
        lon_start = np.asarray([float(member.resolve().nodes[0].lon) for member in relation.members if member.role == 'outer'])
        lat_start = np.asarray([float(member.resolve().nodes[0].lat) for member in relation.members if member.role == 'outer'])
        lon_end = np.asarray([float(member.resolve().nodes[-1].lon) for member in relation.members if member.role == 'outer'])
        lat_end = np.asarray([float(member.resolve().nodes[-1].lat) for member in relation.members if member.role == 'outer'])
        idx_relations = np.asarray([m.ref for m in relation.members if m.role == 'outer'])
        idx_sorted,idx_flipped = find_polygon_members(lon_start,lat_start,lon_end,lat_end,idx_relations)
        idx_searchsorted = np.argsort(idx_sorted)[np.searchsorted(idx_sorted[np.argsort(idx_sorted)],idx_relations)]
        lon_outer = np.empty([0,1],dtype=float)
        lat_outer = np.empty([0,1],dtype=float)
        for i,idx, in enumerate(idx_searchsorted):
            lon = np.asarray([float(n.lon) for n in relation.members[idx].resolve().nodes])
            lat = np.asarray([float(n.lat) for n in relation.members[idx].resolve().nodes])
            if idx_flipped[i] == True:
                lon = lon[::-1]
                lat = lat[::-1]
            if i != len(idx_searchsorted)-1:
                lon = lon[:-1]
                lat = lat[:-1]
            lon_outer = np.append(lon_outer,lon)
            lat_outer = np.append(lat_outer,lat)
        lonlat = np.stack((lon_outer,lat_outer),axis=-1)
        shell = shapely.geometry.LineString(lonlat)
        holes = []
        for member in relation.members:
            if member.role == 'inner':
                lon = np.asarray([float(n.lon) for n in member.resolve().nodes])
                lat = np.asarray([float(n.lat) for n in member.resolve().nodes])
                lonlat = np.stack((lon,lat),axis=-1)
                hole = shapely.geometry.LineString(lonlat)
                hole_polygon = shapely.geometry.Polygon(hole)
                intersection_area_hole = np.asarray([p.intersection(hole_polygon).area/p.area for p in poly_list if p.area > 0])
                if np.max(intersection_area_hole) > 0.999:
                    idx_pop.append(np.argmax(intersection_area_hole))
                holes.append(hole)
        poly = shapely.geometry.Polygon(shell=shell,holes=holes)
        poly_list.append(poly)
    gdf = gpd.GeoDataFrame(geometry=poly_list,crs='EPSG:4326')
    gdf = gdf.drop(idx_pop).reset_index(drop=True)
    if unary_flag == True:
        gdf = gpd.GeoDataFrame(geometry=[p for p in gdf.buffer(0).unary_union.geoms],crs='EPSG:4326')
    return gdf


def main():
    parser = argparse.ArgumentParser(description='Download OSM water polygons')
    parser.add_argument('--extents',type=float,nargs=4,help='lon_min lon_max lat_min lat_max')
    parser.add_argument('--out_file',type=str,help='Output file name')
    parser.add_argument('--min_size',type=float,default=10000.0,help='Minimum polygon size (m^2)')
    parser.add_argument('--unary_union',type='store_true',help='Perform unary union',default=False)
    parser.add_argument('--inverse',type='store_true',help='Inverse selection',default=False)
    args = parser.parse_args()

    lon_min,lon_max,lat_min,lat_max = args.extents
    output_file = args.out_file
    unary_flag = args.unary_union
    min_size = args.min_size
    inverse_flag = args.inverse

    water_result = get_osm_water(lon_min,lon_max,lat_min,lat_max)
    gdf = overpy_to_gdf(water_result,unary_flag=unary_flag)
    if min_size > 0:
        #requires DEM package for deg2tum and utm2epsg
        #need to convert to a non-degree projection to properly compute area
        sys.path.insert(0,'../DEM')
        from dem_utils import deg2utm,utm2epsg
        lon_mean = (gdf.bounds.minx.min() + gdf.bounds.maxx.max())*0.5
        lat_mean = (gdf.bounds.miny.min() + gdf.bounds.maxy.max())*0.5
        x,y,zone = deg2utm(lon_mean,lat_mean)
        epsg_code = utm2epsg(zone)
        gdf = gdf[gdf.to_crs(f'EPSG:{epsg_code}').area > min_size].reset_index(drop=True)

    if inverse_flag == True:
        buffer_val = 0.1
        bbox = shapely.geometry.box(lon_min-buffer_val,lat_min+buffer_val,lon_max-buffer_val,lat_max+buffer_val)
        inverse_geom = bbox
        for i in range(len(gdf)):
            inverse_geom = inverse_geom.symmetric_difference(gdf.geometry[i])
        gdf = gpd.GeoDataFrame(geometry=[g for g in inverse_geom.geoms],crs='EPSG:4326')
        gdf = gdf.clip(bbox).reset_index(drop=True)
        gdf = gdf[gdf.geom_type != 'LineString'].reset_index(drop=True)


    gdf.to_file(output_file)
if __name__ == '__main__':
    main()