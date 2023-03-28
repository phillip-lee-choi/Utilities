import numpy as np
import datetime
import argparse
import configparser
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
import subprocess
import ctypes as c
import shapely
import itertools
import multiprocessing

import pyTMD.time
import pyTMD.model
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.read_FES_model import extract_FES_constants
from pyTMD.load_nodal_corrections import load_nodal_corrections
from pyTMD.calc_astrol_longitudes import calc_astrol_longitudes

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

def landmask_pts(lon,lat,lon_coast,lat_coast,landmask_c_file,inside_flag):
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


def compute_zmin_fes(z,n):
    '''
    z is not the same as zmajor!
    '''
    # mu2 = [0.069439968323, 0.351535557706, -0.046278307672]
    # nu2 = [-0.006104695053, 0.156878802427, 0.006755704028]
    # l2 = [0.077137765667, -0.051653455134, 0.027869916824]
    # t2 = [0.180480173707, -0.020101177502, 0.008331518844]
    # lda2 = [0.016503557465, -0.013307812292, 0.007753383202]
    zmin = np.zeros((n,20),dtype=np.complex64)
    zmin[:,0] = 0.263*z[:,0] - 0.0252*z[:,1]#-- 2Q1
    zmin[:,1] = 0.297*z[:,0] - 0.0264*z[:,1]#-- sigma1
    zmin[:,2] = 0.164*z[:,0] + 0.0048*z[:,1]#-- rho1
    zmin[:,3] = 0.0140*z[:,1] + 0.0101*z[:,3]#-- M12
    zmin[:,4] = 0.0389*z[:,1] + 0.0282*z[:,3]#-- M11
    zmin[:,5] = 0.0064*z[:,1] + 0.0060*z[:,3]#-- chi1
    zmin[:,6] = 0.0030*z[:,1] + 0.0171*z[:,3]#-- pi1
    zmin[:,7] = -0.0015*z[:,1] + 0.0152*z[:,3]#-- phi1
    zmin[:,8] = -0.0065*z[:,1] + 0.0155*z[:,3]#-- theta1
    # zmin[:,9] = -0.0389*z[:,1] + 0.0836*z[:,3]#-- J1
    zmin[:,10] = -0.0431*z[:,1] + 0.0613*z[:,3]#-- OO1
    # zmin[:,11] = 0.264*z[:,4] - 0.0253*z[:,5]#-- 2N2
    # zmin[:,12] = 0.298*z[:,4] - 0.0264*z[:,5]#-- mu2
    # zmin[:,13] = 0.165*z[:,4] + 0.00487*z[:,5]#-- nu2
    # zmin[:,14] = 0.0040*z[:,5] + 0.0074*z[:,6]#-- lambda2
    # zmin[:,15] = 0.0131*z[:,5] + 0.0326*z[:,6]#-- L2
    # zmin[:,16] = 0.0033*z[:,5] + 0.0082*z[:,6]#-- L2
    # zmin[:,17] = 0.0585*z[:,6]#-- t2
    # zmin[:,12] = mu2[0]*z[:,7] + mu2[1]*z[:,4] + mu2[2]*z[:,5]#-- mu2
    # zmin[:,13] = nu2[0]*z[:,7] + nu2[1]*z[:,4] + nu2[2]*z[:,5]#-- nu2
    # zmin[:,14] = lda2[0]*z[:,7] + lda2[1]*z[:,4] + lda2[2]*z[:,5]#-- lambda2
    # zmin[:,16] = l2[0]*z[:,7] + l2[1]*z[:,4] + l2[2]*z[:,5]#-- L2
    # zmin[:,17] = t2[0]*z[:,7] + t2[1]*z[:,4] + t2[2]*z[:,5]#-- t2
    # zmin[:,18] = 0.53285*z[:,8] - 0.03304*z[:,4]#-- eps2
    zmin[:,19] = -0.0034925*z[:,5] + 0.0831707*z[:,7]#-- eta2
    return zmin

def compute_arg_fes(n,t1,t2,S,H,P,pp):
    '''
    Independent of location!
    '''
    arg = np.zeros((n,20))
    arg[:,0] = t1 - 4.0*S + H + 2.0*P - 90.0#-- 2Q1
    arg[:,1] = t1 - 4.0*S + 3.0*H - 90.0#-- sigma1
    arg[:,2] = t1 - 3.0*S + 3.0*H - P - 90.0#-- rho1
    arg[:,3] = t1 - S + H - P + 90.0#-- M12
    arg[:,4] = t1 - S + H + P + 90.0#-- M11
    arg[:,5] = t1 - S + 3.0*H - P + 90.0#-- chi1
    arg[:,6] = t1 - 2.0*H + pp - 90.0#-- pi1
    arg[:,7] = t1 + 3.0*H + 90.0#-- phi1
    arg[:,8] = t1 + S - H + P + 90.0#-- theta1
    arg[:,9] = t1 + S + H - P + 90.0#-- J1
    arg[:,10] = t1 + 2.0*S + H + 90.0#-- OO1
    arg[:,11] = t2 - 4.0*S + 2.0*H + 2.0*P#-- 2N2
    arg[:,12] = t2 - 4.0*S + 4.0*H#-- mu2
    arg[:,13] = t2 - 3.0*S + 4.0*H - P#-- nu2
    arg[:,14] = t2 - S + P + 180.0#-- lambda2
    arg[:,15] = t2 - S + 2.0*H - P + 180.0#-- L2
    arg[:,16] = t2 - S + 2.0*H + P#-- L2
    arg[:,17] = t2 - H + pp#-- t2
    arg[:,18] = t2 - 5.0*S + 4.0*H + P #-- eps2
    arg[:,19] = t2 + S + 2.0*H - pp #-- eta2
    return arg

def compute_f_fes(n,sinn,cosn,II,Ra1):
    '''
    Independent of location!
    '''
    f = np.ones((n,20))
    # f[:,0] = np.sqrt((1.0 + 0.189*cosn - 0.0058*cos2n)**2 + (0.189*sinn - 0.0058*sin2n)**2)#-- 2Q1
    # f[:,1] = f[:,0]#-- sigma1
    # f[:,2] = f[:,0]#-- rho1
    # f[:,3] = np.sqrt((1.0 + 0.185*cosn)**2 + (0.185*sinn)**2)#-- M12
    # f[:,4] = np.sqrt((1.0 + 0.201*cosn)**2 + (0.201*sinn)**2)#-- M11
    # f[:,5] = np.sqrt((1.0 + 0.221*cosn)**2 + (0.221*sinn)**2)#-- chi1
    # f[:,9] = np.sqrt((1.0 + 0.198*cosn)**2 + (0.198*sinn)**2)#-- J1
    # f[:,10] = np.sqrt((1.0 + 0.640*cosn + 0.134*cos2n)**2 + (0.640*sinn + 0.134*sin2n)**2)#-- OO1
    # f[:,11] = np.sqrt((1.0 - 0.0373*cosn)**2 + (0.0373*sinn)**2)#-- 2N2
    # f[:,12] = f[:,11]#-- mu2
    # f[:,13] = f[:,11]#-- nu2
    # f[:,15] = f[:,11]#-- L2
    f[:,16] = np.sqrt((1.0 + 0.441*cosn)**2 + (0.441*sinn)**2)#-- L2
    f[:,0] = np.sin(II)*(np.cos(II/2.0)**2)/0.38 #-- 2Q1
    f[:,1] = f[:,0] #-- sigma1
    f[:,2] = f[:,0] #-- rho1
    f[:,3] = f[:,0] #-- M12
    f[:,4] = np.sin(2.0*II)/0.7214 #-- M11
    f[:,5] = f[:,4] #-- chi1
    f[:,9] = f[:,4] #-- J1
    f[:,10] = np.sin(II)*np.power(np.sin(II/2.0),2.0)/0.01640 #-- OO1
    f[:,11] = np.power(np.cos(II/2.0),4.0)/0.9154 #-- 2N2
    f[:,12] = f[:,11] #-- mu2
    f[:,13] = f[:,11] #-- nu2
    f[:,14] = f[:,11] #-- lambda2
    f[:,15] = f[:,11]*Ra1 #-- L2
    f[:,18] = f[:,11] #-- eps2
    f[:,19] = np.power(np.sin(II),2.0)/0.1565 #-- eta2
    return f

def compute_u_fes(n,sinn,cosn,dtr,xi,nu,R):
    '''
    Independent of location!
    '''
    u = np.zeros((n,20))
    # u[:,0] = np.arctan2(0.189*sinn - 0.0058*sin2n,1.0 + 0.189*cosn - 0.0058*sin2n)/dtr#-- 2Q1
    # u[:,1] = u[:,0]#-- sigma1
    # u[:,2] = u[:,0]#-- rho1
    # u[:,3] = np.arctan2( 0.185*sinn, 1.0 + 0.185*cosn)/dtr#-- M12
    # u[:,4] = np.arctan2(-0.201*sinn, 1.0 + 0.201*cosn)/dtr#-- M11
    # u[:,5] = np.arctan2(-0.221*sinn, 1.0 + 0.221*cosn)/dtr#-- chi1
    # u[:,9] = np.arctan2(-0.198*sinn, 1.0 + 0.198*cosn)/dtr#-- J1
    # u[:,10] = np.arctan2(-0.640*sinn - 0.134*sin2n,1.0 + 0.640*cosn + 0.134*cos2n)/dtr#-- OO1
    # u[:,11] = np.arctan2(-0.0373*sinn, 1.0 - 0.0373*cosn)/dtr#-- 2N2
    # u[:,12] = u[:,11]#-- mu2
    # u[:,13] = u[:,11]#-- nu2
    # u[:,15] = u[:,11]#-- L2
    u[:,16] = np.arctan2(-0.441*sinn, 1.0 + 0.441*cosn)/dtr#-- L2
    u[:,0] = (2.0*xi - nu)/dtr #-- 2Q1
    u[:,1] = u[:,0] #-- sigma1
    u[:,2] = u[:,0] #-- rho1
    u[:,3] = u[:,0] #-- M12
    u[:,4] = -nu/dtr #-- M11
    u[:,5] = u[:,4] #-- chi1
    u[:,9] = u[:,4] #-- J1
    u[:,10] = (-2.0*xi - nu)/dtr #-- OO1
    u[:,11] = (2.0*xi - 2.0*nu)/dtr #-- 2N2
    u[:,12] = u[:,11] #-- mu2
    u[:,13] = u[:,11] #-- nu2
    u[:,14] = (2.0*xi - 2.0*nu)/dtr #-- lambda2
    u[:,15] = (2.0*xi - 2.0*nu - R)/dtr#-- L2
    u[:,18] = u[:,12] #-- eps2
    u[:,19] = -2.0*nu/dtr #-- eta2
    return u

def prep_minor_tide_inference(t,DELTAT):
    dtr = np.pi/180.0
    # npts,nc = z_shape
    n = len(np.atleast_1d(t))
    # n = nt if ((npts == 1) & (nt > 1)) else npts
    MJD = 48622.0 + t
    # cindex = ['q1','o1','p1','k1','n2','m2','s2','k2','2n2']
    # z = np.ma.zeros((n,9),dtype=np.complex64)
    # nz = 9
    # minor = ['2q1','sigma1','rho1','m12','m11','chi1','pi1','phi1','theta1',
        # 'j1','oo1','2n2','mu2','nu2','lambda2','l2','l2','t2','eps2','eta2']
    # minor_indices = [i for i,m in enumerate(minor) if m not in constituents]
    hour = (t % 1)*24.0
    t1 = 15.0*hour
    t2 = 30.0*hour
    ASTRO5 = True
    S,H,P,omega,pp = calc_astrol_longitudes(MJD+DELTAT, ASTRO5=ASTRO5)
    sinn = np.sin(omega*dtr)
    cosn = np.cos(omega*dtr)
    # sin2n = np.sin(2.0*omega*dtr)
    # cos2n = np.cos(2.0*omega*dtr)
    II = np.arccos(0.913694997 - 0.035692561*np.cos(omega*dtr))
    at1 = np.arctan(1.01883*np.tan(omega*dtr/2.0))
    at2 = np.arctan(0.64412*np.tan(omega*dtr/2.0))
    xi = -at1 - at2 + omega*dtr
    xi[xi > np.pi] -= 2.0*np.pi
    nu = at1 - at2
    I2 = np.tan(II/2.0)
    Ra1 = np.sqrt(1.0 - 12.0*(I2**2)*np.cos(2.0*(P - xi)) + 36.0*(I2**4))
    P2 = np.sin(2.0*(P - xi))
    Q2 = 1.0/(6.0*(I2**2)) - np.cos(2.0*(P - xi))
    R = np.arctan(P2/Q2)
    arg = compute_arg_fes(n,t1,t2,S,H,P,pp)
    f = compute_f_fes(n,sinn,cosn,II,Ra1)
    u = compute_u_fes(n,sinn,cosn,dtr,xi,nu,R)
    th = (arg + u)*dtr
    return th,f

def get_z_matrix():
    z_matrix = np.array([
        [0.263,-0.0252,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.297,-0.0264,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.164,0.0048,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0140,0.0,0.0101,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0389,0.0,0.0282,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0064,0.0,0.0060,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0030,0.0,0.0171,0.0,0.0,0.0,0.0,0.0],
        [0.0,-0.0015,0.0,0.0152,0.0,0.0,0.0,0.0,0.0],
        [0.0,-0.0065,0.0,0.0155,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,-0.0431,0.0,0.0613,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,-0.0034925,0.0,0.0831707,0.0],
    ])
    return z_matrix

def compute_tides(lon,lat,utc_time,model_dir,N_cpus):
    '''
    Assumes utc_time is in datetime format
    lon/lat as flattened meshgrid arrays
    '''
    print('Pre-processing...')
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])
    model = pyTMD.model(model_dir,format='netcdf',compressed=False).elevation('FES2014')
    constituents = model.constituents
    amp,ph = extract_FES_constants(np.atleast_1d(lon),
            np.atleast_1d(lat), model.model_file, TYPE=model.type,
            VERSION=model.version, METHOD='spline', EXTRAPOLATE=False,
            SCALE=model.scale, GZIP=model.compressed)
    idx_no_data = np.unique(np.where(amp.mask == True)[0])
    amp = np.delete(amp,idx_no_data,axis=0)
    ph = np.delete(ph,idx_no_data,axis=0)
    lon = np.delete(lon,idx_no_data)
    lat = np.delete(lat,idx_no_data)
    ymd = np.asarray([t.date() for t in utc_time])
    delta_days = np.asarray([y.days for y in ymd-ymd[0]]) #working with integers is much faster than datetime objects
    unique_delta_days,idx_delta_days = np.unique(delta_days,return_index=True)
    seconds = np.asarray([t.hour*3600 + t.minute*60 + t.second + t.microsecond/1000000 for t in utc_time])
    tide_time = np.asarray([pyTMD.time.convert_calendar_dates(y.year,y.month,y.day,second=s) for y,s in zip(ymd,seconds)])
    deltat = calc_delta_time(delta_file, tide_time)
    cph = -1j*ph*np.pi/180.0
    hc = amp*np.exp(cph)
    pu,pf,G = load_nodal_corrections(np.atleast_1d(tide_time) + 48622.0, constituents,DELTAT=deltat, CORRECTIONS=model.format)
    th = G*np.pi/180 + pu
    pf_costh = (pf*np.cos(th)).transpose()
    pf_sinth = (pf*np.sin(th)).transpose()
    hc_real = hc.data.real
    hc_imag = hc.data.imag
    tide_min = np.zeros(len(lon))
    tide_max = np.zeros(len(lon))
    mhhw = np.zeros(len(lon))
    mllw = np.zeros(len(lon))
    th_minor,f_minor = prep_minor_tide_inference(tide_time,deltat)
    Z_matrix = get_z_matrix()
    f_costh = f_minor * np.cos(th_minor)
    f_sinth = f_minor * np.sin(th_minor)
    minor = ['2q1','sigma1','rho1','m12','m11','chi1','pi1','phi1','theta1',
        'j1','oo1','2n2','mu2','nu2','lambda2','l2','l2','t2','eps2','eta2']
    minor_indices = [i for i,m in enumerate(minor) if m not in constituents]
    f_costh = f_costh[:,minor_indices]
    f_sinth = f_sinth[:,minor_indices]
    f_costh = f_costh.transpose()
    f_sinth = f_sinth.transpose()
    constituent_reorder = np.asarray([26,24,25,3,21,7,29,4,0])
    z = hc[:,constituent_reorder]
    zmin = np.matmul(z.data,Z_matrix.transpose())
    zmin = zmin[:,minor_indices]
    zmin_real = zmin.real
    zmin_imag = zmin.imag
    ir = itertools.repeat
    idx = np.arange(len(lon))
    print('Finished pre-processing.')
    p = multiprocessing.Pool(N_cpus)
    tides_tuple = p.starmap(parallel_tides,zip(idx,
                                               ir(hc_real),ir(hc_imag),ir(pf_costh),ir(pf_sinth),
                                               ir(zmin_real),ir(zmin_imag),ir(f_costh),ir(f_sinth),
                                               ir(idx_delta_days)))
    p.close()
    tides_array = np.asarray(tides_tuple)
    tide_min = tides_array[:,0]
    tide_max = tides_array[:,1]
    mllw = tides_array[:,2]
    mhhw = tides_array[:,3]
    return lon,lat,tide_min,tide_max,mhhw,mllw

def parallel_tides(i,hc_real,hc_imag,pf_costh,pf_sinth,zmin_real,zmin_imag,f_costh,f_sinth,idx_delta_days):
    tmp_tide = np.dot(hc_real[i,:],pf_costh) - np.dot(hc_imag[i,:],pf_sinth)
    tmp_minor = np.dot(zmin_real[i,:],f_costh) - np.dot(zmin_imag[i,:],f_sinth)
    tmp_tide += tmp_minor
    tide_min = np.min(tmp_tide)
    tide_max = np.max(tmp_tide)
    tide_split = np.split(tmp_tide,idx_delta_days[1:])
    llw = np.min(tide_split,axis=1)
    hhw = np.max(tide_split,axis=1)
    mllw = np.mean(llw)
    mhhw = np.mean(hhw)
    return tide_min,tide_max,mllw,mhhw

def main():
    config_file = 'utils_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--t_resolution',help='Temporal resolution (mins).',default=5)
    parser.add_argument('--model_dir',help='Model directory',default=config.get('FES2014','model_dir'))
    parser.add_argument('--buffer',help='Flag if only coastal points should be used',action='store_true',default=False)
    parser.add_argument('--output_file',help='Output file')
    parser.add_argument('--N_cpus',help='Number of CPUs to use',default='1')
    args = parser.parse_args()
    t_start = '2010-01-01 00:00:00'
    t_end = '2019-12-31 23:59:59'
    t_resolution = float(args.t_resolution)
    model_dir = args.model_dir
    output_file = args.output_file
    buffer_flag = args.buffer
    N_cpus = int(args.N_cpus)

    date_range = pd.date_range(t_start,t_end,freq=f'{t_resolution}min')
    date_range_datetime = np.asarray([datetime.datetime.strptime(str(t),'%Y-%m-%d %H:%M:%S') for t in date_range])

    m2_file = f'{model_dir}fes2014/ocean_tide/m2.nc'
    m2_data = nc.Dataset(m2_file)
    lon = np.asarray(m2_data['lon'][:])
    lat = np.asarray(m2_data['lat'][:])
    lon = np.concatenate(([-180.0],lon[2881:]-360,lon[:2881]))
    phase_mask = np.asarray(m2_data['phase'][:].mask).astype(int)
    phase_mask = np.concatenate((np.atleast_2d(phase_mask[:,2880]).T,phase_mask[:,2881:],phase_mask[:,:2881]),axis=1)
    lon_mesh,lat_mesh = np.meshgrid(lon,lat)
    lon_array = lon_mesh.flatten()
    lat_array = lat_mesh.flatten()
    phase_mask = phase_mask.flatten()
    lon_array = lon_array[phase_mask==0]
    lat_array = lat_array[phase_mask==0]
    
    if buffer_flag == True:
        print('Buffering coast...')
        res = 'h'
        area_threshold = 5e6
        buffer_dist = 500e3
        x_max_3857 = 20037508.342789244
        antimeridian = shapely.geometry.LineString([[180.0,90.0],[180.0,-90.0]])
        meridian = shapely.geometry.LineString([[0.0,90.0],[0.0,-90.0]])
        coast_dir = config.get('COAST','coast_dir')
        coast_shp = f'{coast_dir}{res}/GSHHS_{res}_L1.shp'
        landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file')
        gdf_coast = gpd.read_file(coast_shp)
        gdf_coast_orig = gdf_coast.copy()
        gdf_coast = gdf_coast.to_crs('EPSG:3857')
        idx_area_threshold = gdf_coast.area > area_threshold
        gdf_coast = gdf_coast[idx_area_threshold]
        gdf_coast = gdf_coast.buffer(buffer_dist)
        gdf_coast = gpd.GeoDataFrame(geometry=[gdf_coast.unary_union],crs='EPSG:3857')
        tmp_gdf = gpd.GeoDataFrame()
        for i in range(len(gdf_coast)):
            tmp_gdf = gpd.GeoDataFrame(pd.concat([tmp_gdf,gpd.GeoDataFrame(geometry=[p for p in gdf_coast.geometry[i].geoms],crs='EPSG:3857')],ignore_index=True),crs='EPSG:3857')
        gdf_coast = tmp_gdf.copy()
        gdf_buffered_filtered = gpd.GeoDataFrame()
        for i in range(len(gdf_coast)):
            x_poly,y_poly = get_lonlat_polygon(gdf_coast.geometry[i])
            if np.nanmax(np.abs(x_poly)) < x_max_3857:
                geom = gdf_coast.geometry[i]
                gdf_buffered_filtered = gpd.GeoDataFrame(pd.concat([gdf_buffered_filtered,gpd.GeoDataFrame(geometry=[geom],crs='EPSG:3857')],ignore_index=True),crs='EPSG:3857')
            else:
                x_inside = x_poly.copy()
                y_inside = y_poly.copy()
                x_inside[x_inside > x_max_3857] = x_max_3857
                x_inside[x_inside < -x_max_3857] = -x_max_3857
                idx_nan = np.atleast_1d(np.argwhere(np.isnan(x_inside)).squeeze())
                if len(idx_nan) == 0:
                    idx_nan = np.array([0,len(x_inside)])
                else:
                    idx_nan = np.append(0,idx_nan+1,len(x_inside))
                holes_list = []
                for j in range(len(idx_nan)-1):
                    if j ==0:
                        x_tmp_exterior = x_inside[idx_nan[j]:idx_nan[j+1]]
                        y_tmp_exterior = y_inside[idx_nan[j]:idx_nan[j+1]]
                        geom_inside_exterior = shapely.geometry.Polygon(np.stack((x_tmp_exterior[~np.isnan(x_tmp_exterior)],y_tmp_exterior[~np.isnan(y_tmp_exterior)]),axis=1))
                        geom_inside_exterior = geom_inside_exterior.buffer(0)
                    else:
                        x_tmp_hole = x_inside[idx_nan[j]:idx_nan[j+1]]
                        y_tmp_hole = y_inside[idx_nan[j]:idx_nan[j+1]]
                        geom_inside_hole = shapely.geometry.Polygon(np.stack((x_tmp_hole[~np.isnan(x_tmp_hole)],y_tmp_hole[~np.isnan(y_tmp_hole)]),axis=1))
                        geom_inside_hole = geom_inside_hole.buffer(0)
                        holes_list.append(geom_inside_hole)
                geom_inside = shapely.geometry.Polygon(geom_inside_exterior.exterior.coords, [inner.exterior.coords for inner in holes_list])
                gdf_buffered_filtered = gpd.GeoDataFrame(pd.concat([gdf_buffered_filtered,gpd.GeoDataFrame(geometry=[geom_inside],crs='EPSG:3857')],ignore_index=True),crs='EPSG:3857')
                idx_outside = np.argwhere(np.abs(x_poly) > x_max_3857).squeeze()
                x_outside = x_poly[idx_outside]
                y_outside = y_poly[idx_outside]
                x_outside[x_outside > x_max_3857] = x_outside[x_outside > x_max_3857] - 2*x_max_3857
                x_outside[x_outside < -x_max_3857] = x_outside[x_outside < -x_max_3857] + 2*x_max_3857
                dist_outside = np.sqrt((x_outside[1:]-x_outside[:-1])**2 + (y_outside[1:]-y_outside[:-1])**2)
                idx_buffer_dist = np.argwhere(np.abs(x_outside) == x_max_3857 - buffer_dist).squeeze()
                idx_dist = np.atleast_1d(np.argwhere(dist_outside > 1e5).squeeze())
                if len(idx_dist) == 0:
                    idx_dist = np.array([0,len(x_outside)])
                else:
                    idx_dist = idx_dist[np.asarray([idx not in idx_buffer_dist for idx in idx_dist])]
                    idx_dist = np.concatenate(([0],idx_dist+1,[len(x_outside)]))
                for j in range(len(idx_dist)-1):
                    x_segment = x_outside[idx_dist[j]:idx_dist[j+1]]
                    y_segment = y_outside[idx_dist[j]:idx_dist[j+1]]
                    if x_segment[0] > 0:
                        x_segment = np.concatenate(([x_max_3857],x_segment[~np.isnan(x_segment)],[x_max_3857],[x_max_3857]))
                        y_segment = np.concatenate(([y_segment[0]],y_segment[~np.isnan(y_segment)],[y_segment[-1]],[y_segment[0]]))
                    elif x_segment[0] < 0:
                        x_segment = np.concatenate(([-x_max_3857],x_segment[~np.isnan(x_segment)],[-x_max_3857],[-x_max_3857]))
                        y_segment = np.concatenate(([y_segment[0]],y_segment[~np.isnan(y_segment)],[y_segment[-1]],[y_segment[0]]))
                    geom_segment = shapely.geometry.Polygon(np.stack((x_segment,y_segment),axis=1))
                    gdf_buffered_filtered = gpd.GeoDataFrame(pd.concat([gdf_buffered_filtered,gpd.GeoDataFrame(geometry=[geom_segment],crs='EPSG:3857')],ignore_index=True),crs='EPSG:3857')
        gdf_buffered_filtered = gpd.GeoDataFrame(geometry=[gdf_buffered_filtered.unary_union],crs='EPSG:3857')
        tmp_gdf = gpd.GeoDataFrame()
        for i in range(len(gdf_buffered_filtered)):
            tmp_gdf = gpd.GeoDataFrame(pd.concat([tmp_gdf,gpd.GeoDataFrame(geometry=[p for p in gdf_buffered_filtered.geometry[i].geoms],crs='EPSG:3857')],ignore_index=True),crs='EPSG:3857')
        gdf_buffered_filtered = tmp_gdf.copy()
        gdf_buffered_filtered = gdf_buffered_filtered.to_crs('EPSG:4326')
        print('Buffer complete. Running mask...')
        lon_coast,lat_coast = get_lonlat_gdf(gdf_buffered_filtered)
        landmask = landmask_pts(lon_array,lat_array,lon_coast,lat_coast,landmask_c_file,inside_flag=1)
        lon_array = lon_array[landmask == 1]
        lat_array = lat_array[landmask == 1]
        print('Mask complete.')
    print('Running tides...')
    lon_tide,lat_tide,tide_min,tide_max,mhhw,mllw = compute_tides(lon_array,lat_array,date_range_datetime,model_dir,N_cpus)
    np.savetxt(output_file,np.c_[lon_tide,lat_tide,tide_min,tide_max,mhhw,mllw],fmt='%.4f,%.4f,%.4f,%.4f,%.4f,%.4f',delimiter=',',comments='',header='lon,lat,tide_min,tide_max,MHHW,MLLW')
    print('Tides complete.')

if __name__ == '__main__':
    main()