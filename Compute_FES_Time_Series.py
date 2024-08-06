import os
os.environ["OMP_NUM_THREADS"] = "8" # Need this so numpy.dot doesn't use all threads/cores
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import datetime
import argparse
import configparser
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
import ctypes as c
import itertools
import multiprocessing
import numpy as np
import glob
import subprocess
import scipy.interpolate

import pyTMD


'''
Written by Eduard Heijkoop, University of Colorado Boulder, May 2023.
July 2024: Updated to pyTMD v2.1.2 to add option to compute FES2022 time series.

Purpose is to compute FES2014 time series for given lon/lat and time.
pyTMD package (v1.0.5) is used to compute tidal heights from the netCDF files.
The preprocessing optimizes for multiple locations at the same time, removing duplicate computations.

TO DO:
- Add option to interpolate between grid points
'''

def great_circle_distance(lon1,lat1,lon2,lat2,R=6378137.0):
    lon1 = deg2rad(lon1)
    lat1 = deg2rad(lat1)
    lon2 = deg2rad(lon2)
    lat2 = deg2rad(lat2)
    DL = np.abs(lon2 - lon1)
    DP = np.abs(lat2 - lat1)
    tmp = np.sqrt( np.sin(0.5*DP)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(0.5*DL)**2)
    tmp[tmp>1] = 1
    dsigma = 2*np.arcsin(tmp)
    distance = R*dsigma
    return distance

def deg2rad(deg):
    rad = deg*np.pi/180
    return rad

def compute_zmin_fes(z,n):
    '''
    z is not the same as zmajor!
    '''
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
    zmin[:,10] = -0.0431*z[:,1] + 0.0613*z[:,3]#-- OO1
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
    f[:,16] = np.sqrt((1.0 + 0.441*cosn)**2 + (0.441*sinn)**2)#-- L2
    f[:,18] = f[:,11] #-- eps2
    f[:,19] = np.power(np.sin(II),2.0)/0.1565 #-- eta2
    return f

def compute_u_fes(n,sinn,cosn,dtr,xi,nu,R):
    '''
    Independent of location!
    '''
    u = np.zeros((n,20))
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
    n = len(np.atleast_1d(t))
    MJD = 48622.0 + t
    hour = (t % 1)*24.0
    t1 = 15.0*hour
    t2 = 30.0*hour
    ASTRO5 = True
    # S,H,P,omega,pp = calc_astrol_longitudes(MJD+DELTAT, ASTRO5=ASTRO5)
    S,H,P,omega,pp = pyTMD.astro.mean_longitudes(MJD+DELTAT, ASTRO5=ASTRO5)
    sinn = np.sin(omega*dtr)
    cosn = np.cos(omega*dtr)
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

def decimal_to_datetime(decimal_year):
    year = int(decimal_year)
    rem = decimal_year - year
    base = datetime.datetime(year,1,1)
    result = base + datetime.timedelta(seconds=(base.replace(year=base.year+1)-base).total_seconds()*rem)
    return result

def calc_delta_time(utc_time):
    '''
    Downloads data from https://maia.usno.navy.mil/ser7/deltat.data (and maybe https://maia.usno.navy.mil/ser7/deltat.preds)
    and interpolates
    '''
    subprocess.run(f'wget -q https://maia.usno.navy.mil/ser7/deltat.data',shell=True)
    # df_deltat_data = pd.read_csv('deltat.data',delim_whitespace=True,header=None,names=['year','month','day','deltat'])
    df_deltat_data = pd.read_csv('deltat.data',sep='\\s+',header=None,names=['year','month','day','deltat'])
    t_datetime = np.asarray([datetime.datetime.strptime(f'{y}-{m}-{d}','%Y-%m-%d') for y,m,d in zip(df_deltat_data.year,df_deltat_data.month,df_deltat_data.day)])
    deltat = np.asarray(df_deltat_data.deltat)
    if np.max(utc_time) > np.max(t_datetime):
        subprocess.run(f'wget -q https://maia.usno.navy.mil/ser7/deltat.preds',shell=True)
        df_deltat_preds = pd.read_csv('deltat.preds',sep='\t')
        idx_year_name = np.atleast_1d(np.argwhere(['Year' in a for a in np.asarray(df_deltat_preds.keys())]).squeeze())[0]
        year_name = df_deltat_preds.keys()[idx_year_name]
        df_deltat_preds.rename({year_name:'year'},axis=1,inplace=True)
        t_datetime_preds = np.asarray([decimal_to_datetime(y) for y in df_deltat_preds['year']])
        deltat_preds = np.asarray(df_deltat_preds['TT-UT'])
        idx_preds = np.argwhere(t_datetime_preds > np.max(t_datetime)).squeeze()
        t_datetime = np.concatenate((t_datetime,t_datetime_preds[idx_preds]))
        deltat = np.concatenate((deltat,deltat_preds[idx_preds]))
        subprocess.run(f'rm deltat.preds',shell=True)
    subprocess.run(f'rm deltat.data',shell=True)
    dt_deltat = np.asarray([t.total_seconds() for t in t_datetime - datetime.datetime(1992,1,1)])
    dt_input = np.asarray([t.total_seconds() for t in utc_time - datetime.datetime(1992,1,1)])
    interp_func = scipy.interpolate.UnivariateSpline(dt_deltat,deltat)
    delta_time = interp_func(dt_input)/86400
    tide_time = np.asarray([t.days + (t.seconds+t.microseconds/1e6)/86400 for t in utc_time - datetime.datetime(1992,1,1)])
    return delta_time,tide_time

def get_fes_mask(fes_m2_data):
    '''
    Returns lon/lat of valid points for FES2014/2022 model
    N_lon is different due to different resolution (1/16 for FES2014 and 1/30 for FES2022)
    '''
    N_lon = fes_m2_data.variables['amplitude'].shape[0]
    lon = np.asarray(fes_m2_data['lon'][:])
    lat = np.asarray(fes_m2_data['lat'][:])
    lon = np.concatenate(([-180.0],lon[N_lon:]-360,lon[:N_lon]))
    amp_mask = np.asarray(fes_m2_data['amplitude'][:].mask).astype(int)
    amp_data = np.asarray(fes_m2_data['amplitude'][:])
    #Extrapolated data will have M2 values of 0 (rather than NoData) over lakes, which we want to exclude
    amp_data_mask = 1-(amp_data>0).astype(int)
    amp_mask += amp_data_mask
    amp_mask = np.concatenate((np.atleast_2d(amp_mask[:,N_lon-1]).T,amp_mask[:,N_lon:],amp_mask[:,:N_lon]),axis=1)
    lon_mesh,lat_mesh = np.meshgrid(lon,lat)
    lon_array = lon_mesh.flatten()
    lat_array = lat_mesh.flatten()
    amp_mask = amp_mask.flatten()
    lon_array = lon_array[amp_mask==0]
    lat_array = lat_array[amp_mask==0]
    return lon_array,lat_array

def get_fes_lonlat(model_dir,extrapolate_flag=False):
    if model_dir.split('/')[-2] == 'FES2014':
        lon,lat = get_fes2014_lonlat(model_dir,extrapolate_flag)
    elif model_dir.split('/')[-2] == 'FES2022':
        lon,lat = get_fes2022_lonlat(model_dir,extrapolate_flag)
    return lon,lat

def get_fes2014_lonlat(model_dir,extrapolate_flag=False):
    if extrapolate_flag == True:
        full_model_dir = f'{model_dir}fes2014/ocean_tide_extrapolated/'
    else:
        full_model_dir = f'{model_dir}fes2014/ocean_tide/'
    m2_file = f'{full_model_dir}/m2.nc'
    m2_data = nc.Dataset(m2_file)
    lon_fes2014,lat_fes2014 = get_fes_mask(m2_data)
    return lon_fes2014,lat_fes2014

def get_fes2022_lonlat(model_dir,extrapolate_flag=False):
    if extrapolate_flag == True:
        full_model_dir = f'{model_dir}fes2022b/ocean_tide_extrapolated/'
    else:
        full_model_dir = f'{model_dir}fes2022b/ocean_tide/'
    m2_file = f'{full_model_dir}/m2_fes2022.nc'
    m2_data = nc.Dataset(m2_file)
    lon_fes2022,lat_fes2022 = get_fes_mask(m2_data)
    return lon_fes2022,lat_fes2022

def get_model_files(model_dir,extrapolate_flag=False):
    if extrapolate_flag == True:
        model_files = sorted(glob.glob(f'{model_dir}fes*/ocean_tide_extrapolated/*.nc'))
    else:
        model_files = sorted(glob.glob(f'{model_dir}fes*/ocean_tide/*.nc'))
    return model_files

def nearest_neighbor(lon,lat,arr,ilon,ilat,dist_threshold=2000):
    arr_return = np.zeros(len(ilon),dtype=np.complex128)
    lon_grid,lat_grid = np.meshgrid(lon,lat)
    lon_grid[arr.mask] = np.nan
    lat_grid[arr.mask] = np.nan
    lon_arr = lon_grid.flatten()
    lat_arr = lat_grid.flatten()
    lon_arr = lon_arr[~np.isnan(lon_arr)]
    lat_arr = lat_arr[~np.isnan(lat_arr)]
    for i in range(len(ilon)):
        dist = great_circle_distance(lon_arr,lat_arr,ilon[i],ilat[i])
        idx = np.argmin(dist)
        min_dist = dist[idx]
        if min_dist >= dist_threshold:
            arr_return[i] = np.nan
            continue
        lon_min_dist = lon_arr[idx]
        lat_min_dist = lat_arr[idx]
        idx_lon = np.atleast_1d(np.argwhere(lon == lon_min_dist).squeeze())[0]
        idx_lat = np.atleast_1d(np.argwhere(lat == lat_min_dist).squeeze())[0]
        arr_return[i] = arr[idx_lat,idx_lon]
    arr_return = np.ma.array(arr_return,mask=np.isnan(arr_return),fill_value=np.complex128(np.nan))
    return arr_return

def read_netcdf_file(model_file):
    dataset = nc.Dataset(model_file)
    lon = np.asarray(dataset['lon'])
    lat = np.asarray(dataset['lat'])
    amp = dataset['amplitude'][:]
    ph = dataset['phase'][:]
    mask = np.logical_or(amp.data == amp.fill_value,ph.data == ph.fill_value)
    hc = np.ma.array(amp*np.exp(-1j*ph*np.pi/180.0),mask=mask,fill_value=np.ma.default_fill_value(np.dtype(complex)))
    return (hc,lon,lat)

def extract_constants(ilon,ilat,model_files,interpolate_method='nearest'):
    ilon = np.atleast_1d(np.copy(ilon))
    ilat = np.atleast_1d(np.copy(ilat))
    model_files = np.asarray(np.copy(model_files))
    ilon[ilon<0.0] += 360.0
    npts = len(ilon)
    nconst = len(model_files)
    amplitude = np.ma.zeros((npts,nconst))
    amplitude.mask = np.zeros((npts,nconst),dtype=bool)
    ph = np.ma.zeros((npts,nconst))
    ph.mask = np.zeros((npts,nconst),dtype=bool)
    # read and interpolate each constituent
    for i, model_file in enumerate(model_files):
        if not os.path.isfile(model_file):
            raise FileNotFoundError(str(model_file))
        hc, lon, lat = read_netcdf_file(model_file)
        # grid step size of tide model
        dlon = lon[1] - lon[0]
        # replace original values with extend arrays/matrices
        # if np.isclose(lon[-1] - lon[0], 360.0 - dlon):
        lon = pyTMD.io.FES._extend_array(lon, dlon)
        hc = pyTMD.io.FES._extend_matrix(hc)
        # determine if any input points are outside of the model bounds
        invalid = (ilon < lon.min()) | (ilon > lon.max()) | \
                  (ilat < lat.min()) | (ilat > lat.max())
        # interpolate amplitude and phase of the constituent
        if interpolate_method == 'nearest':
            hci = nearest_neighbor(lon,lat,hc,ilon,ilat)
        elif interpolate_method == 'bilinear':
            hc.data[hc.mask] = np.nan
            hci = pyTMD.interpolate.bilinear(lon, lat, hc, ilon, ilat,
                dtype=hc.dtype)
            hci.mask[:] |= np.isnan(hci.data)
            hci.data[hci.mask] = hci.fill_value
        elif interpolate_method == 'spline':
            hci = pyTMD.interpolate.spline(lon, lat, hc, ilon, ilat,
                dtype=hc.dtype,
                reducer=np.ceil,
                kx=1, ky=1)
            # replace invalid values with fill_value
            hci.data[hci.mask] = hci.fill_value
        # convert amplitude from centimeters to meters
        amplitude.data[:,i] = np.abs(hci.data)*0.01
        amplitude.mask[:,i] = np.copy(hci.mask)
        # phase of the constituent in radians
        ph.data[:,i] = np.arctan2(-np.imag(hci.data),np.real(hci.data))
        ph.mask[:,i] = np.copy(hci.mask)
        # update mask to invalidate points outside model domain
        amplitude.mask[:,i] |= invalid
        ph.mask[:,i] |= invalid
    # convert phase to degrees
    phase = ph*180.0/np.pi
    phase.data[phase.data < 0] += 360.0
    # replace data for invalid mask values
    amplitude.data[amplitude.mask] = amplitude.fill_value
    phase.data[phase.mask] = phase.fill_value
    # return the interpolated values
    return (amplitude, phase)

def compute_tides(lon,lat,utc_time,loc_names,model_dir,interpolate_method='nearest',extrapolate_flag=False,N_cpus=1):
    '''
    Assumes utc_time is in datetime format
    lon/lat as flattened meshgrid arrays
    '''
    print('Pre-processing...')
    # lon_array,lat_array = get_fes_lonlat(model_dir,extrapolate_flag)
    constituents = ['2n2', 'eps2', 'j1', 'k1', 'k2', 'l2', 'la2', 'm2',
                        'm3', 'm4', 'm6', 'm8', 'mf', 'mks2', 'mm', 'mn4', 'ms4',
                        'msf', 'msqm', 'mtm', 'mu2', 'n2', 'n4', 'nu2', 'o1',
                            'p1', 'q1', 'r2', 's1', 's2', 's4', 'sa', 'ssa', 't2']
    #FES uses la2 instead of lambda2
    constituents_lambda2 = [cst.replace('la2','lambda2') if cst == 'la2' else cst for cst in constituents]
    model_files = get_model_files(model_dir,extrapolate_flag=extrapolate_flag)
    amp,ph = extract_constants(lon,lat,model_files,interpolate_method=interpolate_method)
    idx_no_data = np.unique(np.where(amp.mask == True)[0])
    if len(idx_no_data) > 0:
        print(f'No data for {len(idx_no_data)} location(s).')
        amp = np.delete(amp,idx_no_data,axis=0)
        ph = np.delete(ph,idx_no_data,axis=0)
        lon = np.delete(lon,idx_no_data)
        lat = np.delete(lat,idx_no_data)
        loc_names = np.delete(loc_names,idx_no_data)
    deltat,tide_time = calc_delta_time(utc_time)
    # tide time is time since 1992-01-01 00:00:00 in decimal days, e.g. 11901.75332123
    # deltat is the TT-UT difference, obtained from https://maia.usno.navy.mil/ser7/deltat.data
    # (or https://maia.usno.navy.mil/ser7/deltat.preds if time is after latest time in aforementioned dataset)
    cph = -1j*ph*np.pi/180.0
    hc = amp*np.exp(cph)
    # the  48622.0 is the MJD of 1992-01-01 00:00:00
    pu,pf,G = pyTMD.arguments.arguments(np.atleast_1d(tide_time) + 48622.0,constituents_lambda2,corrections='FES')
    th = G*np.pi/180 + pu
    pf_costh = (pf*np.cos(th)).transpose()
    pf_sinth = (pf*np.sin(th)).transpose()
    hc_real = hc.data.real
    hc_imag = hc.data.imag
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
                                            ir(zmin_real),ir(zmin_imag),ir(f_costh),ir(f_sinth)))
    p.close()
    tides_array = np.asarray(tides_tuple)
    return loc_names,tides_array

def parallel_tides(i,hc_real,hc_imag,pf_costh,pf_sinth,zmin_real,zmin_imag,f_costh,f_sinth):
    tmp_tide = np.dot(hc_real[i,:],pf_costh) - np.dot(hc_imag[i,:],pf_sinth)
    tmp_minor = np.dot(zmin_real[i,:],f_costh) - np.dot(zmin_imag[i,:],f_sinth)
    tmp_tide += tmp_minor
    return tmp_tide

def main():
    config_file = 'utils_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--t_start')
    parser.add_argument('--t_end')
    parser.add_argument('--t_resolution',help='Temporal resolution (mins).',default=5.0,type=float)
    parser.add_argument('--coords',nargs='*',type=float,help='Coordinates (lon,lat)')
    parser.add_argument('--names',help='Names of locations',nargs='*')
    parser.add_argument('--model',help='FES model',default='fes2014',choices=['fes2014','fes2022'])
    parser.add_argument('--output_file',help='Output file')
    parser.add_argument('--machine',help='Machine name',default='t')
    parser.add_argument('--N_cpus',help='Number of CPUs to use',default=1,type=int)
    parser.add_argument('--interpolate',help='Interpolate between FES grid points.',default='nearest',choices=['nearest','bilinear','spline'])
    parser.add_argument('--extrapolate',help='Extrapolate beyond FES grid points.',action='store_true',default=False)
    args = parser.parse_args()
    t_start = args.t_start
    t_end = args.t_end
    t_resolution = args.t_resolution
    coords = args.coords
    loc_names = args.names
    fes_model = args.model
    output_file = args.output_file
    machine_name = args.machine
    N_cpus = args.N_cpus
    interpolate_method = args.interpolate
    extrapolate_flag = args.extrapolate

    if np.mod(len(coords),2) != 0:
        raise ValueError('Please provide an even number of coordinates.')
    
    lon_input = np.asarray(coords[0::2])
    lon_input[lon_input>180] -= 360
    lat_input = np.asarray(coords[1::2])
    N_coords = len(lon_input)

    if N_coords != len(loc_names):
        raise ValueError('Number of coordinates and names do not match.')

    model_dir = config['FES'][f'{fes_model}_model_dir']
    if machine_name == 'b':
        model_dir = model_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    elif machine_name == 'local':
        model_dir = model_dir.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
    date_range = pd.date_range(t_start,t_end,freq=f'{t_resolution}min')
    date_range_datetime = np.asarray([datetime.datetime.strptime(str(t),'%Y-%m-%d %H:%M:%S') for t in date_range])

    new_loc_names,tide_array = compute_tides(lon_input,lat_input,date_range_datetime,loc_names,model_dir,interpolate_method,extrapolate_flag,N_cpus)

    deleted_locs = list(set(loc_names) - set(new_loc_names))
    if len(deleted_locs) > 0:
        print('Warning. The following locations were omitted:')
        for i in range(len(deleted_locs)):
            print(deleted_locs[i])


    df = pd.DataFrame()
    df['time'] = date_range_datetime
    for i in range(len(new_loc_names)):
        # col_name = f'tide_lon_{lon_input_nearest[i]:.4f}_lat_{lat_input_nearest[i]:.4f}'.replace('.','p').replace('-','neg')
        col_name = loc_names[i]
        df[col_name] = tide_array[i,:]
    df.to_csv(output_file,index=False,float_format='%.3f')
    

if __name__ == '__main__':
    main()