from config import PtsConfig, get_pts_config
from tools.files import get_dir_from_json, create_dir_if_does_not_exist
from tools import log
from pts_tools import opendrift_reader_ROMS as reader_ROMS
from pts_tools import opendrift_reader_ROMS_separate_gridfile as reader_ROMS_separate_grid
from pts_tools.opendrift_bottomdrifters import BottomDrifters
from pts_tools.opendrift_bottomthresholddrifters import BottomThresholdDrifters
from pts_tools.releases import get_n_daily_release_times
from data.kelp_data import generate_random_releases_based_on_probability
from datetime import datetime, timedelta
import numpy as np
import sys
import os

input_dir = f'{get_dir_from_json("roms_data")}'
grid_file = True
run_region = 'cwa'
region_name = 'litterbags'
start_date = datetime(2017, 1, 1)
end_date_releases = datetime(2017, 12, 31)
end_date_run = datetime(2018, 2, 1)
dt = 600
dt_out = 10800
export_variables = ['z', 'age_seconds', 'sea_water_temperature', 'sea_water_salinity']
n_particles = 10 # per day per location

output_dir = f'{get_dir_from_json("opendrift_output")}litterbags/'

# release locations
lons0 = np.repeat(np.array([115.4344167, 115.4117, 115.4425667,
         115.70884, 115.67305, 115.6718833,
         115.7079, 115.6790167, 115.6761833]), n_particles)
lats0 = np.repeat(np.array([-31.9188333, -31.9344167, -31.9116333,
         -31.86408, -31.7905667, -31.7791,
         -31.86505, -31.7893667, -31.7779833]), n_particles)
radius = 5000
times0 = get_n_daily_release_times(start_date, end_date_releases)

# set-up files for simulation
if grid_file is True:
    grid_file = f'{input_dir}grid.nc'
    if not os.path.exists(grid_file):
        raise ValueError(f'''Expected separate ROMS grid file here: {grid_file}.
                             If no separate file, set grid_file=False in config,
                             otherwise place grid file in correct location.''')
input_files = f'{input_dir}{start_date.year}/{run_region}_*.nc'
filename = f'{region_name}_{start_date.strftime("%b")}{end_date_run.strftime("%b%Y")}'
output_file = f'{output_dir}{filename}.nc'
create_dir_if_does_not_exist(os.path.dirname(output_file))
log.info(f'Simulation output will be saved to: {output_file}')

reader = reader_ROMS_separate_grid.Reader(filename=input_files, gridfile=grid_file)

# initialise elements
o = BottomDrifters(loglevel=20)
z_seed = 'seafloor'
o.add_reader(reader)

# seed elements
for time0 in times0:
    o.seed_elements(lon=lons0, lat=lats0, time=time0, z=z_seed)

# set simulation configuration
o.set_config('drift:advection_scheme', 'runge-kutta4')
o.set_config('drift:vertical_advection', False)
o.set_config('drift:vertical_mixing', False)
o.set_config('drift:horizontal_diffusivity', 10.0)
o.set_config('general:use_auto_landmask', False)
o.set_config('general:coastline_action', 'stranding')
o.set_config('seed:ocean_only', True)

# run simulation
run_duration = end_date_run-start_date
o.run(duration=run_duration, time_step=dt, time_step_output=dt_out,
      export_variables=export_variables, outfile=output_file)
