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

import pyTMD.time
import pyTMD.model
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.read_FES_model import extract_FES_constants
from pyTMD.load_nodal_corrections import load_nodal_corrections
from pyTMD.calc_astrol_longitudes import calc_astrol_longitudes


'''
Written by Eduard Heijkoop, University of Colorado Boulder, May 2023.

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
    dsigma = 2*np.arcsin( np.sqrt( np.sin(0.5*DP)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(0.5*DL)**2))
    distance = R*dsigma
    return distance

def deg2rad(deg):
    rad = deg*np.math.pi/180
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
    S,H,P,omega,pp = calc_astrol_longitudes(MJD+DELTAT, ASTRO5=ASTRO5)
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
    return tides_array

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
    parser.add_argument('--model_dir',help='Model directory',default=config.get('FES2014','model_dir'))
    parser.add_argument('--output_file',help='Output file')
    parser.add_argument('--machine',help='Machine name',default='t')
    parser.add_argument('--N_cpus',help='Number of CPUs to use',default=1,type=int)
    parser.add_argument('--interpolate',help='Interpolate between FES2014 grid points.',action='store_true',default=False)
    args = parser.parse_args()
    t_start = args.t_start
    t_end = args.t_end
    t_resolution = args.t_resolution
    coords = args.coords
    model_dir = args.model_dir
    output_file = args.output_file
    machine_name = args.machine
    N_cpus = args.N_cpus
    interpolate_flag = args.interpolate

    if np.mod(len(coords),2) != 0:
        raise ValueError('Please provide an even number of coordinates.')
    
    lon_input = np.asarray(coords[0::2])
    lon_input[lon_input>180] -= 360
    lat_input = np.asarray(coords[1::2])
    N_coords = len(lon_input)

    if machine_name == 'b':
        model_dir = model_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    elif machine_name == 'local':
        model_dir = model_dir.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
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

    if interpolate_flag:
        # Interpolate between grid points
        print('Not yet available.')
    else:
        lon_input_nearest = np.zeros(N_coords)
        lat_input_nearest = np.zeros(N_coords)
        for i in range(N_coords):
            dist = great_circle_distance(lon_input[i],lat_input[i],lon_array,lat_array)
            idx = np.argmin(dist)
            min_dist = dist[idx]
            lon_input_nearest[i] = lon_array[idx]
            lat_input_nearest[i] = lat_array[idx]
            print(f'Moving from ({lon_input[i]:.2f},{lat_input[i]:.2f}) to ({lon_input_nearest[i]:.2f},{lat_input_nearest[i]:.2f}), with distance {min_dist/1000:.2f} km.')
        tide_array = compute_tides(lon_input_nearest.copy(),lat_input_nearest.copy(),date_range_datetime,model_dir,N_cpus)

    df = pd.DataFrame()
    df['time'] = date_range_datetime
    for i in range(N_coords):
        col_name = f'tide_lon_{lon_input_nearest[i]:.4f}_lat_{lat_input_nearest[i]:.4f}'.replace('.','p').replace('-','neg')
        df[col_name] = tide_array[i,:]
    df.to_csv(output_file,index=False,float_format='%.3f')
    

if __name__ == '__main__':
    main()