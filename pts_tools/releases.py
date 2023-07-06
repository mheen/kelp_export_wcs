import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from config import PtsConfig
from data.kelp_data import generate_random_releases_based_on_probability
from tools.timeseries import add_month_to_time
from tools import log
from data.bathymetry_data import BathymetryData
from datetime import datetime, timedelta
import numpy as np

def get_releases(config:PtsConfig):
    rng = np.random.default_rng(42) # fix random seed to create random releases based on kelp probability
    release_file = f'input/{config.release_region}_kelp_probability.tif'
    if not os.path.exists(release_file):
        raise ValueError(f'''Kelp probability file for specified release region {config.release_region}
                             does not exist: {release_file}. Create this first.''')

    times0 = get_n_daily_release_times(config.start_date, config.end_date_releases)
    lons0 = []
    lats0 = []
    n_particles = 0
    for t in times0:
        lon0, lat0 = generate_random_releases_based_on_probability(rng, release_file, n_thin=config.n_thin_initial, log_info=False)
        lons0.append(lon0)
        lats0.append(lat0)
        n_particles += len(lon0)
    log.info(f'Created {n_particles} initial particle releases.')
    return times0, lons0, lats0

def get_lon_lat_releases_based_on_depth(min_depth:float, max_depth:float) -> tuple:
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

def get_n_daily_release_times(start_date:datetime, end_date:datetime) -> np.ndarray:
    n_days = (end_date-start_date).days+1

    release_times = []
    for i in range(n_days):
        release_times.append(start_date+timedelta(days=i))

    return np.array(release_times)