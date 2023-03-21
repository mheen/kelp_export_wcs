from parcels import Field, FieldSet, ParticleSet, ParticleFile, ScipyParticle, JITParticle, AdvectionRK4, ErrorCode, Variable
from pts_tools.pclasses_and_kernels import create_samplingparticle, AdvectionRK4DepthCorrector, KERNELS
from pts_tools.roms_c_grid_conversions import add_z_rho_psi_to_grid_file
from pts_tools.releases import get_n_daily_release_times
from data.kelp_data import generate_random_releases_based_on_probability
from tools.files import get_dir_from_json, get_daily_files_in_time_range
from tools import log
from datetime import datetime, timedelta
from netCDF4 import Dataset
import numpy as np

start_date = datetime(2017, 3, 1, 3, 0)
end_date_release = datetime(2017, 3, 1)
end_date_run = datetime(2017, 3, 3)

output_path = f'{get_dir_from_json("pts_parcels")}test_pts'

dt = timedelta(hours=0.5)
outputdt = timedelta(hours=1)

grid_file = 'input/cwa_roms_grid.nc'
his_dir = f'{get_dir_from_json("roms_data")}{start_date.year}/'
his_files = get_daily_files_in_time_range(his_dir, start_date, end_date_run, 'nc',
                                          timeformat='cwa_%Y%m%d')

# use_kernels = ['sample_age', 'sample_temp', 'sample_sal', 'sample_bathy']
use_kernels = []

lon0 = [115.3]
lat0 = [-31.6]
depth0 = [-20]
time0 = start_date

# release_times = get_n_daily_release_times(year, start_month, start_day, n_months=release_months)
# lons0 = []
# lats0 = []
# n_particles = 0
# for t in times0:
#     lon0, lat0 = generate_random_releases_based_on_probability(rng, 'input/perth_kelp_probability.tif', n_thin=n_thin)
#     lons0.append(lon0)
#     lats0.append(lat0)
#     n_particles += len(lon0)


# add z_psi to grid file if it doesn't already exist
add_z_rho_psi_to_grid_file(grid_file)

filenames = {'U': {'lon': grid_file, 'lat': grid_file, 'depth': grid_file, 'data': his_files},
            'V': {'lon': grid_file, 'lat': grid_file, 'depth': grid_file, 'data': his_files},
            'temperature': {'lon': grid_file, 'lat': grid_file, 'depth': grid_file, 'data': his_files},
            'salinity': {'lon': grid_file, 'lat': grid_file, 'depth': grid_file, 'data': his_files},
            'bathymetry': {'lon': grid_file, 'lat': grid_file, 'data': grid_file}}

variables = {'U': 'u',
            'V': 'v',
            'temperature': 'temp',
            'salinity': 'salt',
            'bathymetry': 'h'}

dimensions = {'U': {'lon': 'lon_psi', 'lat': 'lat_psi', 'depth': 'z_psi', 'time': 'ocean_time'},
              'V': {'lon': 'lon_psi', 'lat': 'lat_psi', 'depth': 'z_psi', 'time': 'ocean_time'},
              'temperature': {'lon': 'lon_rho', 'lat': 'lat_rho', 'depth': 'z_rho', 'time': 'ocean_time'},
              'salinity': {'lon': 'lon_rho', 'lat': 'lat_rho', 'depth': 'z_rho', 'time': 'ocean_time'},
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
maxdepth = Field('maxdepth', data=fieldset.U.grid.depth[0, :, :],
                 lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                 mesh='spherical', interp_method='cgrid_tracer', gridindexingtype='mitgcm')
fieldset.add_field(maxdepth)

# particle_class = create_samplingparticle(fieldset)
particle_class = ScipyParticle
pset = ParticleSet(fieldset=fieldset, pclass=particle_class,
                   lon=lon0, lat=lat0, depth=depth0, time=time0,
                   lonlatdepth_dtype=np.float64) # for better precision with C-grids

pfile = ParticleFile(output_path, pset, outputdt=outputdt)

kernels = pset.Kernel(AdvectionRK4DepthCorrector)
for k in use_kernels:
    kernels += KERNELS[k]

runtime = timedelta(days=(end_date_run-start_date).days)
pset.execute(kernels, runtime=runtime, dt=dt, output_file=pfile)
