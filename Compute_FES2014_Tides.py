import numpy as np
import datetime
import argparse
import configparser
import matplotlib.pyplot as plt
import pandas as pd
import sys

import pyTMD.time
import pyTMD.model
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.infer_minor_corrections import infer_minor_corrections
from pyTMD.predict_tidal_ts import predict_tidal_ts
from pyTMD.read_FES_model import extract_FES_constants

def compute_tides(lon,lat,utc_time,model_dir):
    '''
    Assumes utc_time is in datetime format
    '''
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])
    model = pyTMD.model(model_dir,format='netcdf',compressed=False).elevation('FES2014')
    constituents = model.constituents
    amp,ph = extract_FES_constants(np.atleast_1d(lon),
            np.atleast_1d(lat), model.model_file, TYPE=model.type,
            VERSION=model.version, METHOD='spline', EXTRAPOLATE=False,
            SCALE=model.scale, GZIP=model.compressed)
    if np.any(amp.mask) == True:
        return None
    YMD = np.asarray([t.date() for t in utc_time])
    seconds = np.asarray([t.hour*3600 + t.minute*60 + t.second + t.microsecond/1000000 for t in utc_time])
    tide_time = np.asarray([pyTMD.time.convert_calendar_dates(y.year,y.month,y.day,second=s) for y,s in zip(YMD,seconds)])
    DELTAT = calc_delta_time(delta_file, tide_time)
    cph = -1j*ph*np.pi/180.0
    hc = amp*np.exp(cph)
    TIDE = predict_tidal_ts(np.atleast_1d(tide_time),hc,constituents,DELTAT=DELTAT,CORRECTIONS=model.format)
    MINOR = infer_minor_corrections(np.atleast_1d(tide_time),hc,constituents,DELTAT=DELTAT,CORRECTIONS=model.format)
    TIDE.data[:] += MINOR.data[:]
    return TIDE.data





def main():
    config_file = 'utils_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--lon',help='Longitude')
    parser.add_argument('--lat',help='Latitude')
    parser.add_argument('--t_start',help='Start time',default=datetime.datetime(datetime.datetime.now().year,1,1,0,0,0).strftime('%Y-%m-%d %H:%M:%S'))
    parser.add_argument('--t_end',help='End time',default=datetime.datetime(datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day,0,0,0).strftime('%Y-%m-%d %H:%M:%S'))
    parser.add_argument('--resolution',help='Temporal resolution (mins).',default=5)
    parser.add_argument('--model_dir',help='Model directory',default=config['FES2014']['model_dir'])
    parser.add_argument('--output_file',help='Output file')
    parser.add_argument('--plot',help='Plot',action='store_true')
    args = parser.parse_args()
    lon = float(args.lon)
    lat = float(args.lat)
    t_start = datetime.datetime.strptime(args.t_start,'%Y-%m-%d %H:%M:%S')
    t_end = datetime.datetime.strptime(args.t_end,'%Y-%m-%d %H:%M:%S')
    t_resolution = args.resolution
    model_dir = args.model_dir
    output_file = args.output_file
    plot_flag = args.plot

    date_range = pd.date_range(t_start,t_end,freq=f'{t_resolution}min')
    date_range_datetime = np.asarray([datetime.datetime.strptime(str(t),'%Y-%m-%d %H:%M:%S') for t in date_range])
    fes2014_heights = compute_tides(lon,lat,date_range_datetime,model_dir)
    if fes2014_heights is None:
        print('No FES2014 data found for this location')
        sys.exit()
    
    if output_file is not None:
        np.savetxt(output_file,np.c_[date_range.astype(object),fes2014_heights],fmt='%s,%f',delimiter=',',header=f'FES2014 Tidal heights at lon: {lon:.3f}, lat: {lat:.3f}')
    
    if plot_flag == True:
        plt.plot(date_range_datetime,fes2014_heights)
        plt.xlabel('Time')
        plt.ylabel('Height (m)')
        plt.title(f'FES2014 Tidal heights at lon: {lon:.3f}, lat: {lat:.3f}')
        plt.show()



if __name__ == '__main__':
    main()


