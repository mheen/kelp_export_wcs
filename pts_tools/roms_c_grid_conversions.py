from data.roms_data import get_z as convert_s_rho_to_z_rho
from scipy.interpolate import RectBivariateSpline
from tools import log
from netCDF4 import Dataset
import numpy as np

def add_z_rho_psi_to_grid_file(grid_file:str):
    nc = Dataset(grid_file, mode='r+')
    if 'z_psi' in nc.variables:
        nc.close()
        return
    
    s_rho_dim = nc.dimensions['s_rho']
    eta_psi = nc.dimensions['eta_psi']
    xi_psi = nc.dimensions['xi_psi']
    eta_rho = nc.dimensions['eta_rho']
    xi_rho = nc.dimensions['xi_rho']

    lon_rho = nc.variables['lon_rho'][:].filled(fill_value=np.nan)
    lat_rho = nc.variables['lat_rho'][:].filled(fill_value=np.nan)
    lon_psi = nc.variables['lon_psi'][:].filled(fill_value=np.nan)
    lat_psi = nc.variables['lat_psi'][:].filled(fill_value=np.nan)

    s_rho = nc.variables['s_rho'][:].filled(fill_value=np.nan)
    h = nc.variables['h'][:].filled(fill_value=np.nan)
    cs_r = nc.variables['Cs_r'][:].filled(fill_value=np.nan)
    hc = nc.variables['hc'][:].filled(fill_value=np.nan)

    z_rho = convert_s_rho_to_z_rho(s_rho, h, cs_r, hc)

    z_psi = np.zeros((s_rho_dim.size, eta_psi.size, xi_psi.size))
    for k in range(s_rho_dim.size):
        z_interp = RectBivariateSpline(lat_rho[:, 0], lon_rho[0, :], z_rho[k, :, :]) # is this really correct for curvilinear grid?
        z_psi[k, :, :] = z_interp(lat_psi[:, 0], lon_psi[0, :])

    nc_z_rho = nc.createVariable('z_rho', float, (s_rho_dim, eta_rho, xi_rho), zlib=True)
    nc_z_rho[:] = z_rho
    nc_z_psi = nc.createVariable('z_psi', float, (s_rho_dim, eta_psi, xi_psi), zlib=True)
    nc_z_psi[:] = z_psi

    nc.close()
    log.info(f'Wrote z_psi variable to {grid_file}')
