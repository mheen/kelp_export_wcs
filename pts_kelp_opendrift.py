from tools.files import get_dir_from_json
from tools.timeseries import add_month_to_time
from tools import log
from pts_tools import opendrift_reader_ROMS as reader_ROMS
from pts_tools.opendrift_bottomdrifters import BottomDrifters
from pts_tools.releases import get_n_daily_release_times
from data.kelp_data import generate_random_releases_based_on_probability
from datetime import datetime, timedelta
import numpy as np
import sys

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

def run(release_times:np.ndarray,
        lon_releases:np.ndarray,
        lat_releases:np.ndarray,
        file_description:str,
        run_duration=60,
        dt=300, dt_out=3600):
    
    export_variables = ['z', 'age_seconds',
                        'sea_water_temperature', 'sea_water_salinity']

    run_duration = timedelta(days=run_duration)

    input_dir = get_dir_from_json('roms_data')
    input_files = f'{input_dir}{release_times[0].year}/cwa_his_*.nc'
    output_dir = get_dir_from_json('opendrift')
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
                          run_months:int, release_months:int, n_thin=100,
                          dt=60*10, dt_out=3*60*60):

    rng = np.random.default_rng(42) # fix random seed to create random releases based on kelp probability
    times0 = get_n_daily_release_times(year, start_month, start_day, n_months=release_months)
    lons0 = []
    lats0 = []
    n_particles = 0
    for t in times0:
        lon0, lat0 = generate_random_releases_based_on_probability(rng, 'input/perth_kelp_probability.tif', n_thin=n_thin)
        lons0.append(lon0)
        lats0.append(lat0)
        n_particles += len(lon0)

    log.info(f'Releasing {n_particles} in total spread over times:')
    print(f'times0={times0}')
    sys.stdout.flush()

    start_date = datetime(year, start_month, start_day)
    end_date = add_month_to_time(start_date, run_months)
    run_duration = (end_date-start_date).days

    file_description = f'{year}-{start_date.strftime("%b")}-{end_date.strftime("%b")}'

    log.info(f'Running simulation for: {year} {start_date.strftime("%b")} to {end_date.strftime("%b")}...')
    _ = run(times0, lons0, lats0, file_description, run_duration=run_duration, dt=dt, dt_out=dt_out)

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
    
    run_multiple_releases(2017, 3, 1, 5, 4)

    # run_single_event(datetime(2022, 6, 28), datetime(2022, 7, 5))
