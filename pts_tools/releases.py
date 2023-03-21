import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.timeseries import add_month_to_time
from data.bathymetry_data import BathymetryData
from datetime import datetime, timedelta
import numpy as np

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
    n_days = (end_date-start_date).days

    release_times = []
    for i in range(n_days):
        release_times.append(start_date+timedelta(days=i))

    return np.array(release_times)