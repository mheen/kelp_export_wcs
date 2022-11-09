from opendrift.readers import reader_ROMS_native
from models.opendrift_bottomdrifters import BottomDrifters
from datetime import datetime, timedelta
import os

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

def run(year:int, month:int, depth_release:int, n_particles=3*10**5, run_duration=90, dt=300):
    
    start_time0 = datetime(year, month, 1)
    start_time1 = datetime(year, month+1, 1)-timedelta(days=1)
    run_duration = timedelta(days=run_duration)

    input_files = f'/mnt/qnap/OPERATIONAL/ROMS/CWA/archive/{year}/perth_his_*.nc'
    release_shapefile = f'input/perth_{depth_release}m_polygon.shp'
    if not os.path.exists(release_shapefile):
        raise ValueError(f'Release shapefile for depth {depth_release} does not exist: {release_shapefile}')
    output_file = f'opendrift_output/perth_{year}-{month}_{depth_release}m.nc'

    roms_reader = reader_ROMS_native.Reader(filename=input_files)
    
    o = BottomDrifters(loglevel=20)
    o.add_reader(roms_reader)
    o.seed_from_shapefile(release_shapefile, number=n_particles, time=[start_time0, start_time1], z='seafloor')

    o.set_config('drift:advection_scheme', 'runge-kutta4')
    o.set_config('drift:vertical_mixing', False) # consider adding: could be relevant
    o.set_config('drift:horizontal_diffusivity', 1) # [m2/s]
    o.set_config('general:coastline_action', 'stranding') # consider changing

    o.run(duration=run_duration, time_step=dt, outfile=output_file)

if __name__ == '__main__':
    years = [2022]
    months = [4, 5, 6, 7]
    depth_releases = [10, 20, 30, 40, 50]

    for year in years:
        for month in months:
            for depth_release in depth_releases:
                run(year, month, depth_release)
