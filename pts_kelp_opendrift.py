from opendrift.readers import reader_ROMS_native
from models.opendrift_bottomdrifters import BottomDrifters
from bathymetry_data import BathymetryData
from kelp_map import get_kelp_coordinates
from datetime import datetime, timedelta
import os
import numpy as np

import sys
sys.path.append('..')
from py_tools.files import get_dir_from_json
from py_tools import log

# TEMP: old release locations
# lon_release = [115.71, 115.68]
# lat_release = [-31.87, -31.77]
# release_description = '3milereef-line'
# lon_release = 115.42
# lat_release = -31.93
# release_description = 'rottnest_litterbag'
# lon_release = 115.68
# lat_release = -31.78
# release_description = 'marmion_litterbag'

def get_lon_lat_release_roms_depth(min_depth:float, max_depth:float) -> tuple:
    bathymetry = BathymetryData.read_from_csv()

    l_depth = np.logical_and(min_depth<bathymetry.h, bathymetry.h<max_depth)
    lon0 = bathymetry.lon[l_depth]
    lat0 = bathymetry.lat[l_depth]
    return lon0, lat0

def get_lon_lat_release_kelp_locations(probability_threshold=0.8, i_thin=10):
    lon, lat = get_kelp_coordinates()
    lon0 = lon[::i_thin]
    lat0 = lat[::i_thin]
    return lon0, lat0

def get_n_hourly_release_times(year:int, month:int, n_months=1, n_hours=3) -> np.ndarray:
    start_date = datetime(year, month, 1)
    n_days = (datetime(year, month+n_months, 1)-start_date).days
    n_hours = int(n_days*24/n_hours-n_hours)

    release_times = []
    for i in range(n_hours):
        release_times.append(start_date+timedelta(hours=i))

    return np.array(release_times)

def run(release_times:np.ndarray,
        lon_release:np.ndarray,
        lat_release:np.ndarray,
        file_description:str,
        run_duration=60,
        dt=300, dt_out=3600):
    
    run_duration = timedelta(days=run_duration)

    input_dir = get_dir_from_json('input/dirs.json', 'roms_data')
    input_files = f'{input_dir}{year}/perth_his_*.nc'
    output_dir = get_dir_from_json('input/dirs.json', 'opendrift')
    output_file = f'{output_dir}perth_{file_description}.nc'
    log.info(f'Simulation output will be saved to: {output_file}')

    roms_reader = reader_ROMS_native.Reader(filename=input_files)
    
    o = BottomDrifters(loglevel=20)
    o.add_reader(roms_reader)

    for release_time in release_times:
        o.seed_elements(lon=lon_release, lat=lat_release, time=release_time, z='seafloor')

    o.set_config('drift:advection_scheme', 'runge-kutta4')
    o.set_config('drift:vertical_advection', False) # turn on when considering particle properties
    o.set_config('drift:vertical_mixing', False) # consider adding: could be relevant
    o.set_config('drift:horizontal_diffusivity', 1) # [m2/s]
    # o.set_config('general:use_auto_landmask', False) # (uses landmask from ROMS) -> turned off: not working (?)
    o.set_config('general:coastline_action', 'stranding') # consider changing

    o.run(duration=run_duration, time_step=dt, time_step_output=dt_out,
          export_variables=['z'], outfile=output_file)

    log.info(f'Simulated done, saved to: {output_file}')

if __name__ == '__main__':
    years = [2022]
    
    lon0, lat0 = get_lon_lat_release_kelp_locations()

    for year in years:
        file_description = f'{year}'

        times0 = get_n_hourly_release_times(year, 4, n_months=4, n_hours=24)

        run(times0, lon0, lat0, file_description, dt=60*10)
