
import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
from tools.timeseries import convert_time_to_datetime
from tools.arrays import get_closest_index
from tools import log
import numpy as np
from netCDF4 import Dataset
from dataclasses import dataclass

@dataclass
class WindData:
    time: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    u: np.ndarray
    v: np.ndarray
    vel: np.ndarray
    dir: np.ndarray

def convert_u_v_to_meteo_vel_dir(u:np.ndarray, v:np.ndarray) -> tuple:
    vel = np.sqrt(u**2+v**2)

    dir = np.mod(180+180/np.pi*np.arctan2(u, v), 360)

    return vel, dir

def get_wind_dir_and_text() -> tuple:
    dir = [0, 45, 90, 135, 180, 225, 270, 315]
    text = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    return dir, text

def read_era5_wind_data(input_path:str) -> WindData:
    log.info(f'Reading wind data from {input_path}')
    nc = Dataset(input_path)

    time_org = nc['time'][:].filled(fill_value=np.nan)
    time_units = nc['time'].units
    time = convert_time_to_datetime(time_org, time_units)

    lon = nc['lon'][:].filled(fill_value=np.nan)
    lat = nc['lat'][:].filled(fill_value=np.nan)
    
    u = nc['Uwind'][:].filled(fill_value=np.nan)
    v = nc['Vwind'][:].filled(fill_value=np.nan)

    nc.close()

    vel, dir = convert_u_v_to_meteo_vel_dir(u, v)

    return WindData(time, lon, lat, u, v, vel, dir)

def get_wind_data_in_point(wind_data:WindData, lon_p:float, lat_p:float) -> WindData:
    i = get_closest_index(wind_data.lon[0, :], lon_p)
    j = get_closest_index(wind_data.lat[:, 0], lat_p)

    time = wind_data.time
    lon = wind_data.lon[j, i]
    lat = wind_data.lat[j, i]
    u = wind_data.u[:, j, i]
    v = wind_data.v[:, j, i]
    vel = wind_data.vel[:, j, i]
    dir = wind_data.dir[:, j, i]

    return WindData(time, lon, lat, u, v, vel, dir)

def get_wind_vel_and_dir_in_point(wind_data:WindData, lon_p:float, lat_p:float) -> tuple:
    i = get_closest_index(wind_data.lon[0, :], lon_p)
    j = get_closest_index(wind_data.lat[:, 0], lat_p)

    vel = wind_data.vel[:, j, i]
    dir = wind_data.dir[:, j, i]

    return vel, dir
