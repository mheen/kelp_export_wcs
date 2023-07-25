
import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
from tools.timeseries import convert_time_to_datetime, get_daily_means, get_l_time_range
from tools.arrays import get_closest_index
from tools import log
import numpy as np
from netCDF4 import Dataset
from dataclasses import dataclass
from datetime import datetime

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

def get_lon_lat_range_indices(lon:np.ndarray, lat:np.ndarray,
                              lon_range:list, lat_range:list) -> tuple[int, int, int, int]:
    i0 = get_closest_index(lon[0, :], lon_range[0])
    i1 = get_closest_index(lon[0, :], lon_range[1])
    j0 = get_closest_index(lat[:, 0], lat_range[0])
    j1 = get_closest_index(lat[:, 0], lat_range[1])
    return i0, i1, j0, j1

def read_era5_wind_data_from_netcdf(input_dir:str,
                                     start_date:datetime,
                                     end_date:datetime,
                                     lon_range=None,
                                     lat_range=None,
                                     path_format='era5_roms_forcing_') -> WindData:
    input_path = f'{input_dir}{path_format}{start_date.strftime("%Y")}0101.nc'
    log.info(f'Reading wind data from {input_path}')
    nc = Dataset(input_path)

    time_org = nc['time'][:].filled(fill_value=np.nan)
    time_units = nc['time'].units
    time_all = convert_time_to_datetime(time_org, time_units)
    l_time = get_l_time_range(time_all, start_date, end_date)
    time = time_all[l_time]

    lon_all = nc['lon'][:].filled(fill_value=np.nan)
    lat_all = nc['lat'][:].filled(fill_value=np.nan)

    if lon_range is not None and lat_range is not None:
        i0, i1, j0, j1 = get_lon_lat_range_indices(lon_all, lat_all, lon_range, lat_range)
    else:
        i0 = None
        i1 = None
        j0 = None
        j1 = None
    
    lon = lon_all[j0:j1, i0:i1]
    lat = lat_all[j0:j1, i0:i1]
    
    u = nc['Uwind'][l_time, j0:j1, i0:i1].filled(fill_value=np.nan)
    v = nc['Vwind'][l_time, j0:j1, i0:i1].filled(fill_value=np.nan)

    nc.close()

    vel, dir = convert_u_v_to_meteo_vel_dir(u, v)

    return WindData(time, lon, lat, u, v, vel, dir)

def get_wind_data_in_lon_lat_range(wind_data:WindData, lon_range:list, lat_range:list) -> WindData:
    i0, i1, j0, j1 = get_lon_lat_range_indices(wind_data.lon, wind_data.lat, lon_range, lat_range)
    lon = wind_data.lon[j0:j1, i0:i1]
    lat = wind_data.lat[j0:j1, i0:i1]
    u = wind_data.u[j0:j1, i0:i1]
    v = wind_data.v[j0:j1, i0:i1]
    vel = wind_data.vel[j0:j1, i0:i1]
    dir = wind_data.dir[j0:j1, i0:i1]
    return WindData(wind_data.time, lon, lat, u, v, vel, dir)

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

def get_daily_mean_wind_data(wind_data:WindData) -> WindData:
    dm_time, dm_u = get_daily_means(wind_data.time, wind_data.u)
    _, dm_v = get_daily_means(wind_data.time, wind_data.v)

    dm_vel, dm_dir = convert_u_v_to_meteo_vel_dir(dm_u, dm_v)

    return WindData(dm_time, wind_data.lon, wind_data.lat, dm_u, dm_v, dm_vel, dm_dir)
