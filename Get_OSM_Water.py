import overpy
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import argparse
import sys

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

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


def relation_to_geometry(relation):
    shell = []
    holes = []
    for m in relation.members:
        w = m.resolve()
        lon = np.asarray([float(n.lon) for n in w.nodes])
        lat = np.asarray([float(n.lat) for n in w.nodes])
        if m.role == 'inner':
            holes.append(shapely.geometry.LineString(np.stack((lon,lat),axis=-1)))
        elif m.role == 'outer':
            shell.append(shapely.geometry.LineString(np.stack((lon,lat),axis=-1)))
    if len(shell) > 1:
        shell_merged = shapely.ops.linemerge(shell)
    elif len(shell) == 1:
        shell_merged = shell[0]
    else:
        return None
    if len(holes) > 1:
        holes_merged = shapely.ops.linemerge(holes)
        if holes_merged.geom_type == 'MultiLineString':
            holes_merged = [g for g in holes_merged.geoms]
        else:
            holes_merged = [holes_merged]
    elif len(holes) == 1:
        holes_merged = holes
    else:
        holes_merged = None
    if shell_merged.geom_type == 'MultiLineString':
        geom = []
        for g in shell_merged.geoms:
            shell_holes = []
            if holes_merged is None:
                geom.append(shapely.geometry.Polygon(shell=np.asarray(g.xy).T,holes=[]))
            else:
                for h in holes_merged:
                    if g.contains(h):
                        shell_holes.append(h)
                geom.append(shapely.geometry.Polygon(shell=np.asarray(g.xy).T,holes=shell_holes))
    elif shell_merged.geom_type == 'LineString':
        geom = shapely.geometry.Polygon(shell=np.asarray(shell_merged.xy).T,holes=holes_merged)
    else:
        geom = None
    return geom

def osm_ways_to_poly(osm_result,relation_way_ids):
    '''
    
    '''
    lss = [] #convert ways to linstrings
    for ii_w,way in enumerate(osm_result.ways):
        if way in relation_way_ids:
            continue
        ls_coords = []
        for node in way.nodes:
            ls_coords.append((node.lon,node.lat)) # create a list of node coordinates
        lss.append(shapely.geometry.LineString(ls_coords)) # create a LineString from coords 
    merged = shapely.ops.linemerge([*lss]) # merge LineStrings
    borders = shapely.ops.unary_union(merged) # linestrings to a MultiLineString
    polygons = list(shapely.ops.polygonize(borders))
    return polygons


# def ways_to_geom(way,relation_way_ids):
#     '''
    
#     '''
#     if way.id not in relation_way_ids:
#         return None
#     lon = np.asarray([float(n.lon) for n in way.nodes])
#     lat = np.asarray([float(n.lat) for n in way.nodes])
#     if len(lon) < 4 or len(lat) < 4:
#         return None
#     geom = shapely.geometry.Polygon(np.stack((lon,lat),axis=-1))
#     return geom


def main():
    parser = argparse.ArgumentParser(description='Download OSM water polygons')
    parser.add_argument('--extents',type=float,nargs=4,help='lon_min lon_max lat_min lat_max')
    parser.add_argument('--out_file',type=str,help='Output file name')
    # parser.add_argument('--min_size',type=float,default=10000.0,help='Minimum polygon size (m^2)')
    # parser.add_argument('--unary_union',action='store_true',help='Perform unary union',default=False)
    parser.add_argument('--inverse',action='store_true',help='Inverse selection',default=False)
    # parser.add_argument('--clip',action='store_true',help='Clip to OSM coastline?',default=False)
    args = parser.parse_args()

    lon_min,lon_max,lat_min,lat_max = args.extents
    output_file = args.out_file
    # unary_flag = args.unary_union
    # min_size = args.min_size
    inverse_flag = args.inverse
    # clip_flag = args.clip

    water_result = get_osm_water(lon_min,lon_max,lat_min,lat_max)

    relation_way_ids = []
    relation_ways_outer = []
    relation_ways_inner = []
    for r in water_result.relations:
        for m in r.members:
            relation_way_ids.append(m.ref)
            if m.role == 'inner':
                relation_ways_inner.append(m.ref)
            elif m.role == 'outer':
                relation_ways_outer.append(m.ref)
    way_ids = water_result.way_ids
    
    poly_list = [relation_to_geometry(r) for r in water_result.relations]
    poly_list = flatten_list(poly_list)
    poly_list = [p for p in poly_list if p is not None]

    poly_list_ways = osm_ways_to_poly(water_result,relation_way_ids)

    gdf_relations = gpd.GeoDataFrame(geometry=poly_list,crs='EPSG:4326')
    gdf_ways = gpd.GeoDataFrame(geometry=poly_list_ways,crs='EPSG:4326')

    idx_valid_relations = np.asarray([g.is_valid for g in gdf_relations.geometry])
    idx_valid_ways = np.asarray([g.is_valid for g in gdf_ways.geometry])
    gdf_relations = gdf_relations[idx_valid_relations].reset_index(drop=True)
    gdf_ways = gdf_ways[idx_valid_ways].reset_index(drop=True)

    #Need to find a way to isolate incorrect ways
    idx_contains = np.zeros(len(gdf_ways),dtype=bool)
    for i in range(len(gdf_ways)):
        for j in range(len(gdf_relations)):
            if gdf_relations.geometry[j].intersection(gdf_ways.geometry[i]).area/gdf_ways.geometry[i].area > 0.8:
                idx_contains[i] = True
                break
            if gdf_relations.geometry[j].contains(gdf_ways.geometry[i]):
                idx_contains[i] = True
                break
    gdf_ways = gdf_ways[~idx_contains].reset_index(drop=True)
    gdf_total = pd.concat([gdf_relations,gdf_ways],ignore_index=True)

    mp = shapely.ops.unary_union(gdf_total.geometry)
    if mp.geom_type == 'MultiPolygon':
        poly_list_full = [p for p in mp.geoms]
    elif mp.geom_type == 'Polygon':
        poly_list_full = [mp]
    else:
        poly_list_full = []
    gdf = gpd.GeoDataFrame(geometry=poly_list_full,crs='EPSG:4326')
    # poly_list.extend(poly_list_ways)
    # gdf = gpd.GeoDataFrame(geometry=poly_list,crs='EPSG:4326')

    # if min_size > 0:
    #     #requires DEM package for deg2tum and utm2epsg
    #     #need to convert to a non-degree projection to properly compute area
    #     sys.path.insert(0,'../DEM')
    #     from dem_utils import deg2utm,utm2epsg
    #     lon_mean = (gdf.bounds.minx.min() + gdf.bounds.maxx.max())*0.5
    #     lat_mean = (gdf.bounds.miny.min() + gdf.bounds.maxy.max())*0.5
    #     x,y,zone = deg2utm(lon_mean,lat_mean)
    #     epsg_code = utm2epsg(zone)
    #     gdf = gdf[gdf.to_crs(f'EPSG:{epsg_code}').area > min_size].reset_index(drop=True)

    if inverse_flag == True:
        buffer_val = 0.1
        bbox = shapely.geometry.box(lon_min-buffer_val,lat_min-buffer_val,lon_max+buffer_val,lat_max+buffer_val)
        inverse_geom = bbox
        for i in range(len(gdf)):
            inverse_geom = inverse_geom.symmetric_difference(gdf.geometry[i])
        if inverse_geom.geom_type == 'Polygon':
            gdf = gpd.GeoDataFrame(geometry=[inverse_geom],crs='EPSG:4326')
        else:
            gdf = gpd.GeoDataFrame(geometry=[g for g in inverse_geom.geoms],crs='EPSG:4326')
        gdf = gdf.clip(bbox).reset_index(drop=True)
        gdf = gdf[gdf.geom_type != 'LineString'].reset_index(drop=True)
        gdf = gdf[gdf.geom_type != 'MultiLineString'].reset_index(drop=True)


    gdf.to_file(output_file)
if __name__ == '__main__':
    main()
