from config import PtsConfig, get_pts_config
from tools.files import get_dir_from_json, create_dir_if_does_not_exist
from tools import log
from pts_tools.releases import get_releases
from pts_tools import opendrift_reader_ROMS as reader_ROMS
from pts_tools import opendrift_reader_ROMS_separate_gridfile as reader_ROMS_separate_grid
from pts_tools.opendrift_bottomdrifters import BottomDrifters
from pts_tools.opendrift_bottomthresholddrifters import BottomThresholdDrifters
from datetime import datetime, timedelta
import numpy as np
import sys
import os

config_file = sys.argv[1]

config = get_pts_config(config_file)

times0, lons0, lats0 = get_releases(config)

# set-up files for simulation
if config.grid_file is True:
    grid_file = f'{config.input_dir}grid.nc'
    if not os.path.exists(grid_file):
        raise ValueError(f'''Expected separate ROMS grid file here: {grid_file}.
                             If no separate file, set grid_file=False in config,
                             otherwise place grid file in correct location.''')
input_files = f'{config.input_dir}{config.start_date.year}{config.input_dir_description}/{config.run_region}_*_his.nc'
filename = f'{config.region_name}_{config.start_date.strftime("%b")}{config.end_date_run.strftime("%b%Y")}_{config.extra_description}'
output_file = f'{config.output_dir}{config.sub_output_dir}{filename}.nc'
create_dir_if_does_not_exist(os.path.dirname(output_file))
log.info(f'Simulation output will be saved to: {output_file}')

if config.reader == 'roms_reader' and config.grid_file == False:
    reader = reader_ROMS.Reader(filename=input_files)
elif config.reader == 'roms_reader' and config.grid_file == True:
    reader = reader_ROMS_separate_grid.Reader(filename=input_files, gridfile=grid_file)
else:
    raise ValueError(f'Unknown reader {config.reader} requested.')

# initialise elements
if config.elements == 'bottom_drifters':
    o = BottomDrifters(loglevel=20)
    z_seed = 'seafloor'
elif config.elements == 'bottom_threshold_drifters':
    o = BottomThresholdDrifters(loglevel=20)
    z_seed = 'seafloor'
else:
    raise ValueError(f'Unknown elements {config.elements} requested.')
o.add_reader(reader)

# seed elements
list_types = [list, np.ndarray]
if type(lons0[0]) in list_types:
    for t, time0 in enumerate(times0):
        o.seed_elements(lon=lons0[t], lat=lats0[t], time=time0, z=z_seed)
else:
    for time0 in times0:
        o.seed_elements(lon=lons0, lat=lats0, time=time0, z=z_seed)

# set simulation configuration
o.set_config('drift:advection_scheme', config.advection_scheme)
o.set_config('drift:vertical_advection', config.vertical_advection)
o.set_config('drift:vertical_mixing', config.vertical_mixing)
o.set_config('drift:horizontal_diffusivity', config.horizontal_diffusivity)
o.set_config('general:use_auto_landmask', config.use_auto_landmask)
o.set_config('general:coastline_action', config.coastline_action)
o.set_config('seed:ocean_only', True)
if config.elements == 'bottom_threshold_drifters':
    o.set_config('transport:threshold_velocity', config.threshold_velocity)

# run simulation
run_duration = config.end_date_run-config.start_date
o.run(duration=run_duration, time_step=config.dt, time_step_output=config.dt_out,
      export_variables=config.export_variables, outfile=output_file)
