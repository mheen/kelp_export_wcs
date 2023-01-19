import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.timeseries import convert_time_to_datetime, convert_datetime_to_time
from tools.files import get_dir_from_json
from tools import log
from dataclasses import dataclass
from netCDF4 import Dataset
import numpy as np
from datetime import datetime

@dataclass
class SatelliteSST:
    time: np.ndarray # [time]
    lon: np.ndarray # [lon]
    lat: np.ndarray # [lat]
    sst: np.ndarray # [time, lat, lon]

def read_satellite_sst_from_netcdf(input_path:str) -> SatelliteSST:
    nc = Dataset(input_path)

    time_org = nc['time'][:].filled(fill_value=np.nan).astype(float)
    time_units = nc['time'].units
    time = convert_time_to_datetime(time_org, time_units)

    lon = nc['lon'][:].filled(fill_value=np.nan)
    lat = nc['lat'][:].filled(fill_value=np.nan)

    sst = nc['sea_surface_temperature'][:].filled(fill_value=np.nan)

    if nc['sea_surface_temperature'].units == 'kelvin':
        sst = sst-273.15

    nc.close()

    return SatelliteSST(time, lon, lat, sst)

def get_monthly_mean_sst(sst_data:SatelliteSST, month:int) -> SatelliteSST:
    months = np.array([t.month for t in sst_data.time])
    l_month = months==month

    mean_sst = np.nanmean(sst_data.sst[l_month, :, :], axis=0)

    return SatelliteSST(np.array([datetime(1800, month, 1)]), sst_data.lon, sst_data.lat, mean_sst)

def write_sst_data_to_netcdf(sst_data:SatelliteSST, output_path:str):

    log.info(f'Writing satellite SST data to netcdf: {output_path}')

    nc = Dataset(output_path, 'w', format='NETCDF4')

    # define dimensions
    nc.createDimension('time', len(sst_data.time))
    nc.createDimension('lat', len(sst_data.lat))
    nc.createDimension('lon', len(sst_data.lon))

    # define variables
    nc_time = nc.createVariable('time', float, 'time', zlib=True)
    nc_lon = nc.createVariable('lon', float, 'lon', zlib=True)
    nc_lat = nc.createVariable('lat', float, 'lat', zlib=True)
    if len(sst_data.sst.shape) == 3:
        nc_sst = nc.createVariable('sea_surface_temperature', float, ('time', 'lat', 'lon'), zlib=True)
    elif len(sst_data.sst.shape) == 2:
        nc_sst = nc.createVariable('sea_surface_temperature', float, ('lat', 'lon'), zlib=True)
    else:
        raise ValueError('Unknown dimensions of SST data')

    # write variables
    time, time_units = convert_datetime_to_time(sst_data.time)
    nc_time[:] = time
    nc_time.units = time_units
    nc_lon[:] = sst_data.lon
    nc_lat[:] = sst_data.lat
    nc_sst[:] = sst_data.sst
    nc_sst.units = 'celsius'
    nc.close()

if __name__ == '__main__':
    input_path = f'{get_dir_from_json("satellite_sst")}IMOS_aggregation_gsr_monthly_mean_2012-2022.nc'
    sst_data = read_satellite_sst_from_netcdf(input_path)

    month = 6
    sst_month = get_monthly_mean_sst(sst_data, month)
    output_path = f'{get_dir_from_json("satellite_sst")}gsr_monthly_mean_{sst_month.time[0].strftime("%B")}.nc'
    write_sst_data_to_netcdf(sst_month, output_path)