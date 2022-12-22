from readers import reader_ROMS
from models.opendrift_bottomdrifters import BottomDrifters
from bathymetry_data import BathymetryData
from kelp_map import generate_random_releases_based_on_probability
from datetime import datetime, timedelta
import os
import numpy as np

import sys
sys.path.append('..')
from py_tools.files import get_dir_from_json
from py_tools.timeseries import add_month_to_time
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

def get_n_hourly_release_times(year:int, month:int, day:int, n_months=1, n_hours=3) -> np.ndarray:
    start_date = datetime(year, month, day)
    end_date = add_month_to_time(start_date, n_months)
    n_days = (end_date-start_date).days
    n_hours = int(n_days*24/n_hours-n_hours)

    release_times = []
    for i in range(n_hours):
        release_times.append(start_date+timedelta(hours=i))

    return np.array(release_times)

def get_n_daily_release_times(year:int, month:int, day:int, n_months=4, n_days=1) -> np.ndarray:
    start_date = datetime(year, month, day)
    end_date = add_month_to_time(start_date, n_months)
    n_days = (end_date-start_date).days

    release_times = []
    for i in range(n_days):
        release_times.append(start_date+timedelta(days=i))

    return np.array(release_times)

def run(release_times:np.ndarray,
        lon_releases:np.ndarray,
        lat_releases:np.ndarray,
        file_description:str,
        run_duration=60,
        dt=300, dt_out=3600):
    
    export_variables = ['z', 'age_seconds', 'origin_marker',
                        'terminal_velocity', 'x_sea_water_velocity', 'y_sea_water_velocity',
                        'sea_floor_depth_below_sea_level', 'sea_water_temperature', 'sea_water_salinity']

    run_duration = timedelta(days=run_duration)

    input_dir = get_dir_from_json('input/dirs.json', 'roms_data')
    input_files = f'{input_dir}{release_times[0].year}/cwa_his_*.nc'
    output_dir = get_dir_from_json('input/dirs.json', 'opendrift')
    output_file = f'{output_dir}cwa-perth_{file_description}.nc'
    log.info(f'Simulation output will be saved to: {output_file}')

    roms_reader = reader_ROMS.Reader(filename=input_files)
    print(roms_reader)
    sys.stdout.flush()
    
    o = BottomDrifters(loglevel=20)
    o.add_reader(roms_reader)

    if len(lon_releases[0].shape) == 1:
        for t, release_time in enumerate(release_times):
            o.seed_elements(lon=lon_releases[t], lat=lat_releases[t], time=release_time, z='seafloor')
    else:
        for release_time in release_times:
            o.seed_elements(lon=lon_releases, lat=lat_releases, time=release_time, z='seafloor')

    o.set_config('drift:advection_scheme', 'runge-kutta4')
    o.set_config('drift:vertical_advection', False) # turn on when considering particle properties
    o.set_config('drift:vertical_mixing', False) # consider adding: could be relevant
    o.set_config('drift:horizontal_diffusivity', 1) # [m2/s]
    o.set_config('general:use_auto_landmask', False) # (uses landmask from ROMS) -> turned off: not working (?)
    o.set_config('general:coastline_action', 'stranding') # consider changing

    o.run(duration=run_duration, time_step=dt, time_step_output=dt_out,
          export_variables=export_variables, outfile=output_file)

    log.info(f'Simulation done, saved to: {output_file}')

    return o

def run_multiple_releases(year:int, start_month:int, start_day:int,
                          run_months:int, release_months:int, n_thin=10,
                          dt=60*10):

    rng = np.random.default_rng(42) # fix random seed to create random releases based on kelp probability
    times0 = get_n_daily_release_times(year, start_month, start_day, n_months=release_months)
    lons0 = []
    lats0 = []
    for t in times0:
        lon0, lat0 = generate_random_releases_based_on_probability(rng, n_thin=n_thin)
        lons0.append(lon0)
        lats0.append(lat0)

    print(f'times0={times0}')
    sys.stdout.flush()

    start_date = datetime(year, start_month, start_day)
    end_date = add_month_to_time(start_date, run_months)
    run_duration = (end_date-start_date).days

    file_description = f'{year}-{start_date.strftime("%b")}-{end_date.strftime("%b")}'

    log.info(f'Running simulation for: {year} {start_date.strftime("%b")} to {end_date.strftime("%b")}...')
    _ = run(times0, lons0, lats0, file_description, run_duration=run_duration, dt=dt)

def run_single_event(start_date:datetime, end_date:datetime,
                     n_thin=10, dt=60*10):
    rng = np.random.default_rng(42) # fix random seed to create random releases based on kelp probability
    lon0, lat0 = generate_random_releases_based_on_probability(rng, n_thin=n_thin)
    print(f'lon0={lon0}\nlat0={lat0}')
    sys.stdout.flush()

    times0 = [start_date]
    print(f'times0={times0}')
    sys.stdout.flush()

    run_duration = (end_date-start_date).days

    file_description = f'event-{start_date.strftime("%Y-%m-%d")}-{end_date.strftime("%Y-%m-%d")}'

    log.info(f'Running simulation for event: {start_date.strftime("%Y-%m-%d")}')
    _ = run(times0, lon0, lat0, file_description, run_duration=run_duration, dt=dt)

if __name__ == '__main__':
    
    run_multiple_releases(2017, 4, 1, 5, 4)

    # run_single_event(datetime(2022, 6, 28), datetime(2022, 7, 5))
