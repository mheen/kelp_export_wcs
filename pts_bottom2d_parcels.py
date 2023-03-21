from parcels import Field, FieldSet, ParticleSet, ParticleFile, ScipyParticle, JITParticle, AdvectionRK4, ErrorCode, Variable
from parcels import plotTrajectoriesFile
from pts_tools.pclasses_and_kernels import create_samplingparticle, SamplingParticle, KERNELS, delete_particle
from pts_tools.roms_c_grid_conversions import add_z_rho_psi_to_grid_file
from pts_tools.releases import get_n_daily_release_times
from data.kelp_data import generate_random_releases_based_on_probability
from tools.files import get_dir_from_json, get_daily_files_in_time_range, create_dir_if_does_not_exist
from tools import log
from datetime import datetime, timedelta
from netCDF4 import Dataset
import numpy as np
import os

# need to extract bottom layer from ROMS files first: ncea -d s_rho,1,1 -F input_file.nc output_file.nc

# ADD BROWNIAN MOTION, FIX "CORRECT CELL NOT FOUND" -> WHY NOT OUT OF BOUNDS?

start_date = datetime(2017, 3, 1, 0, 0)
end_date_release = datetime(2017, 7, 31)
end_date_run = datetime(2017, 8, 31)

time_str = f'{start_date.strftime("%b")}{end_date_run.strftime("%b")}{start_date.year}'
output_path = f'{get_dir_from_json("pts_parcels")}output/{time_str}_2d_bl'
create_dir_if_does_not_exist(os.path.dirname(output_path))

dt = timedelta(minutes=10)
outputdt = timedelta(hours=3)

grid_file = 'input/cwa_roms_grid.nc'
his_dir = f'{get_dir_from_json("roms_data")}{start_date.year}_bl/'
his_files = get_daily_files_in_time_range(his_dir, start_date-timedelta(days=1), end_date_run+timedelta(days=1),
                                          'nc', timeformat='cwa_%Y%m%d')

use_kernels = ['sample_age', 'sample_temp', 'sample_sal', 'sample_bathy']

n_thin_releases = 100

# create initial positions
rng = np.random.default_rng(42) # fix random seed to create random releases based on kelp probability
release_times = get_n_daily_release_times(start_date, end_date_release)
lons0 = np.array([])
lats0 = np.array([])
times0 = np.array([])
for t in release_times:
    lon0, lat0 = generate_random_releases_based_on_probability(rng, 'input/perth_kelp_probability.tif', n_thin=n_thin_releases)
    lons0 = np.concatenate([lons0, np.array(lon0)])
    lats0 = np.concatenate([lats0, np.array(lat0)])
    times0 = np.concatenate([times0, np.repeat(t, len(lon0))])

# set-up to read fieldset
filenames = {'U': {'lon': grid_file, 'lat': grid_file, 'data': his_files},
             'V': {'lon': grid_file, 'lat': grid_file, 'data': his_files},
             'temperature': {'lon': grid_file, 'lat': grid_file, 'data': his_files},
             'salinity': {'lon': grid_file, 'lat': grid_file, 'data': his_files},
             'bathymetry': {'lon': grid_file, 'lat': grid_file, 'data': grid_file}}

variables = {'U': 'u',
             'V': 'v',
             'temperature': 'temp',
             'salinity': 'salt',
             'bathymetry': 'h'}

dimensions = {'U': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'ocean_time'},
              'V': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'ocean_time'},
              'temperature': {'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'ocean_time'},
              'salinity': {'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'ocean_time'},
              'bathymetry': {'lon': 'lon_rho', 'lat': 'lat_rho'}}

# reshape u and v to be the same size as lon_psi, lat_psi to fit MITgcm convention
nc = Dataset(grid_file)
eta_size, xi_size = nc['lon_psi'].shape
nc.close()
indices = {'U': {'lat': range(0, eta_size)},
           'V': {'lon': range(0, xi_size)},
           'temperature': None,
           'salinity': None,
           'bathymetry': None}

# create fieldset
fieldset = FieldSet.from_c_grid_dataset(filenames, variables, dimensions,
                                        indices=indices, gridindexingtype='mitgcm')

# create particleset
# particle_class = create_samplingparticle(fieldset)
pset = ParticleSet(fieldset=fieldset, pclass=SamplingParticle,
                   lon=lons0, lat=lats0, time=times0,
                   lonlatdepth_dtype=np.float64) # for better precision with C-grids

pfile = ParticleFile(output_path, pset, outputdt=outputdt)

kernels = pset.Kernel(AdvectionRK4)
for k in use_kernels:
    kernels += KERNELS[k]

runtime = timedelta(days=(end_date_run-start_date).days)
pset.execute(kernels, runtime=runtime, dt=dt, output_file=pfile,
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})
