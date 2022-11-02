from tools import get_files_in_dir, get_dir_from_json
from opendrift.readers import reader_ROMS_native
from models.opendrift_bottomdrifters import BottomDrifters
from datetime import datetime, timedelta

data_files = get_files_in_dir(f'{get_dir_from_json("roms_data")}', 'nc')

start_time = datetime(2022, 6, 16)
run_duration = timedelta(days=14)
dt = 1800 # [s]

# lon_release = [115.71, 115.68]
# lat_release = [-31.87, -31.77]
# release_description = '3milereef-line'
# lon_release = 115.42
# lat_release = -31.93
# release_description = 'rottnest_litterbag'
lon_release = 115.68
lat_release = -31.78
release_description = 'marmion_litterbag'

filename = f'{get_dir_from_json("opendrift")}{start_time.strftime("%Y%m%d")}_{release_description}_v{datetime.today().strftime("%Y%m%d")}'
pts_file = f'{filename}.nc'

roms_reader = reader_ROMS_native.Reader(filename=data_files)

o = BottomDrifters()
o.add_reader(roms_reader)
# o.seed_cone(lon=lon_release, lat=lat_release, number=1000, time=[start_time, start_time+run_duration], z='seafloor')
o.seed_elements(lon=lon_release, lat=lat_release, number=1000, radius=100, time=[start_time, start_time+timedelta(days=1)], z='seafloor')

o.set_config('drift:advection_scheme', 'runge-kutta4')
o.set_config('drift:vertical_mixing', False)
o.set_config('drift:horizontal_diffusivity', 1)  # [m2/s]
o.set_config('general:coastline_action', 'stranding')


o.run(duration=run_duration, time_step=dt, outfile=pts_file)

o.plot(background=['x_sea_water_velocity', 'y_sea_water_velocity'], filename=f'{filename}_uv.jpg')
o.plot(background='sea_water_temperature', filename=f'{filename}_temp.jpg')
o.plot(background='sea_water_salinity', filename=f'{filename}_salt.jpg')
