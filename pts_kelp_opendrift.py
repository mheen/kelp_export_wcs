from opendrift.readers import reader_ROMS_native
from models.opendrift_bottomdrifters import BottomDrifters
from bathymetry_data import BathymetryData
from datetime import datetime, timedelta
import os
import numpy as np

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

def get_hourly_release_times(year:int, month:int, n_months=1) -> np.ndarray:
    start_date = datetime(year, month, 1)
    n_days = (datetime(year, month+n_months, 1)-start_date).days
    n_hours = n_days*24-1

    release_times = []
    for i in range(n_hours):
        release_times.append(start_date+timedelta(hours=i))

    return np.array(release_times)

def run(release_times:np.ndarray,
        lon_release:np.ndarray,
        lat_release:np.ndarray,
        file_description:str,
        run_duration=90, dt=300):
    
    run_duration = timedelta(days=run_duration)

    input_files = f'/mnt/qnap/OPERATIONAL/ROMS/CWA/archive/{year}/perth_his_*.nc'
    output_file = f'opendrift_output/perth_{year}-{month}_{file_description}.nc'

    roms_reader = reader_ROMS_native.Reader(filename=input_files)
    
    o = BottomDrifters(loglevel=20)
    o.add_reader(roms_reader)

    for release_time in release_times:
        o.seed_elements(lon=lon_release, lat=lat_release, time=release_time, z='seafloor')

    o.set_config('drift:advection_scheme', 'runge-kutta4')
    o.set_config('drift:vertical_mixing', False) # consider adding: could be relevant
    o.set_config('drift:horizontal_diffusivity', 1) # [m2/s]
    o.set_config('general:coastline_action', 'stranding') # consider changing

    o.run(duration=run_duration, time_step=dt, outfile=output_file)

if __name__ == '__main__':
    years = [2022]
    months = [4, 5, 6, 7]
    min_depths = [5, 10, 20, 30, 40]
    max_depths = [10, 20, 30, 40, 50]

    for d in range(len(min_depths)):
        lon0, lat0 = get_lon_lat_release_roms_depth(min_depths[d], max_depths[d])
        file_description = f'{max_depths[d]}m'

        for year in years:
            for month in months:
                times0 = get_hourly_release_times(year, month)
                print(f'Running simulation for depth {max_depths[d]}, {year}-{month}')
                run(times0, lon0, lat0, file_description)
