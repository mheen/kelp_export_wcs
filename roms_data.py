from location_info import LocationInfo
from basic_maps import plot_basic_map
from dataclasses import dataclass
from matplotlib import path
import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from py_tools.files import get_dir_from_json
from py_tools.timeseries import convert_time_to_datetime, get_l_time_range
from py_tools import log

def bbox2ij(lon:np.ndarray, lat:np.ndarray, bbox:list) -> tuple:
    '''Return indices for i,j that will completely cover the specified bounding box.     
    i0, i1, j0, j1 = bbox2ij(lon, lat, bbox)
    lon, lat = 2D arrays that are the target of the subset
    bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]

    Example
    -------  
    >>> i0, i1, j0, j1 = bbox2ij(lon_rho, [-71, -63., 39., 46])
    >>> h_subset = nc.variables['h'][j0:j1, i0:i1]       
    '''
    bbox = np.array(bbox)
    mypath = np.array([bbox[[0,1,1,0]], bbox[[2,2,3,3]]]).T
    p = path.Path(mypath)
    points = np.vstack((lon.flatten(), lat.flatten())).T
    n,m = np.shape(lon)
    inside = p.contains_points(points).reshape((n,m))
    ii,jj = np.meshgrid(range(m), range(n))
    return min(ii[inside]), max(ii[inside]), min(jj[inside]), max(jj[inside])

def get_z(s:np.ndarray, h:np.ndarray, cs_r:np.ndarray, hc:np.ndarray) -> np.ndarray:
    '''Gets depth of ROMS sigma layers.
    
    Input parameters:
    s: sigma layers [s] (using "s_rho" in ROMS, but "s_w" is also an option)
    h: bottom depths [eta, xi] ("h" in ROMS)
    cs_r: s-level stretching curve [s] ("Cs_r" in ROMS)
    hc: critical depth ("hc" in ROMS)
    
    IMPORTANT: Assuming Vtransform = 2'''

    output_shape = (len(cs_r),) + h.shape

    n = hc*s[:, None] + np.outer(cs_r, h)
    d = (1.0 + hc/h)

    z = n.reshape(output_shape)/D

    return z

@dataclass
class RomsGrid:
    lon: np.ndarray # [eta, xi]
    lat: np.ndarray # [eta, xi]
    s: np.ndarray # [s]
    angle: np.ndarray # [eta, xi]
    h: np.ndarray # [eta, xi]
    z: np.ndarray # [s, eta, xi]

    def get_lon_lat_range(self) -> tuple:
        lon_range = [np.nanmin(self.lon), np.nanmax(self.lon)]
        lat_range = [np.nanmin(self.lat), np.nanmax(self.lat)]

        return lon_range, lat_range

    def get_eta_xi_of_lon_lat_point(self, lon_p:np.ndarray, lat_p:np.ndarray) -> tuple:
        etas = []
        xis = []
        for i in range(len(lon_p)):
            xi, _, eta, _ = bbox2ij(self.lon, self.lat, [lon_p[i], lon_p[i]+0.1, lat_p[i], lat_p[i]+0.1])
            etas.append(eta)
            xis.append(xi)
        return np.array(etas), np.array(xis)

def read_roms_grid_from_netcdf(input_path:str) -> RomsGrid:
    nc = Dataset(input_path)

    lon_rho = nc['lon_rho'][:].filled(fill_value=np.nan)
    lat_rho = nc['lat_rho'][:].filled(fill_value=np.nan)
    s_rho = nc['s_rho'][:].filled(fill_value=np.nan)
    angle = nc['angle'][:].filled(fill_value=np.nan)

    h = nc['h'][:].filled(fill_value=np.nan)
    hc = nc['hc'][:].filled(fill_value=np.nan)
    cs_r = nc['Cs_r'][:].filled(fill_value=np.nan)

    z = get_z(s_rho, h, cs_r, hc)

    nc.close()

    return RomsGrid(lon_rho, lat_rho, s_rho, angle, h, z)

def get_subgrid(grid:RomsGrid, lon_range:list, lat_range:list) -> RomsGrid:
    i0, i1, j0, j1 = bbox2ij(grid.lon, grid.lat, [lon_range[0], lon_range[1], lat_range[0], lat_range[1]])

    lon = grid.lon[j0:j1, i0:i1]
    lat = grid.lat[j0:j1, i0:i1]
    angle = grid.angle[j0:j1, i0:i1]
    h = grid.h[j0:j1, i0:i1]
    z = grid.z[j0:j1, i0:i1]

    return RomsGrid(lon, lat, grid.s, angle, h, z)
    
def convert_roms_u_v_to_u_east_v_north(u:np.ndarray, v:np.ndarray, angle:np.ndarray) -> tuple:
    '''Convert u and v from curvilinear ROMS output to u eastwards and v northwards.
    This is done by:
    1. Converting u and v so that they are on rho-coordinate point (cell center).
    2. Rotating u and v so they are directed eastwards and northwards respectively.'''

    def u2rho(var_u:np.ndarray) -> np.ndarray:
        '''Convert variable on u-coordinate to rho-coordinate.'''
        var_u_size = var_u.shape
        n_dimension = len(var_u_size)
        L = var_u_size[-1]
        if n_dimension == 4:
            var_rho = np.zeros((var_u_size[0], var_u_size[1], var_u_size[2], L+1))
            var_rho[:, :, :, 1:L] = 0.5*(var_u[:, :, :, 0:L-1]+var_u[:, :, :, 1:L])
            var_rho[:, :, :, 0] = var_rho[:, :, :, 1]
            var_rho[:, :, :, L] = var_rho[:, :, :, L-1]
        return var_rho

    def v2rho(var_v:np.ndarray) -> np.ndarray:
        '''Convert variable on v-coordinate to rho-coordinate.'''
        var_v_size = var_v.shape
        n_dimension = len(var_v_size)
        M = var_v_size[-2]      
        if n_dimension == 4:
            var_rho = np.zeros((var_v_size[0], var_v_size[1], M+1, var_v_size[3]))
            var_rho[:, :, 1:M, :] = 0.5*(var_v[:, :, 0:M-1, :]+var_v[:, :, 1:M, :])
            var_rho[:, :, 0, :] = var_rho[:, :, 1, :]
            var_rho[:, :, M, :] = var_rho[:, :, M-1, :]
        return var_rho

    def rotate_u_v(u:np.ndarray, v:np.ndarray, angle:np.ndarray) -> tuple:
        '''Rotate u and v velocities on curvilinear grid so that
        they are directed east- and northwards respectively.'''
        u_east = u*np.cos(angle)-v*np.sin(angle)
        v_north = v*np.cos(angle)+u*np.sin(angle)
        return u_east, v_north

    u_rho = u2rho(u)
    v_rho = v2rho(v)
    u_east, v_north = rotate_u_v(u_rho, v_rho, angle)

    return u_east, v_north

@dataclass
class RomsData:
    time: np.ndarray
    grid: RomsGrid
    u_east: np.ndarray
    v_north: np.ndarray
    temp: np.ndarray
    salt: np.ndarray

def read_roms_data_from_netcdf(input_path:list, lon_range=None, lat_range=None, time_range=None) -> RomsData:
    grid = read_roms_grid_from_netcdf(input_path)

    if lon_range and lat_range is not None:
        i0, i1, j0, j1 = bbox2ij(grid.lon, grid.lat, [lon_range[0], lon_range[1], lat_range[0], lat_range[1]])
        grid = get_subgrid(grid, lon_range, lat_range)
    else:
        i0 = 0
        i1 = -1
        j0 = 0
        j1 = -1

    nc = Dataset(input_path)

    time_org = nc['ocean_time'][:].filled(fill_value=np.nan)
    time_units = nc['ocean_time'].units
    time = convert_time_to_datetime(time_org, time_units)

    if time_range is not None:
        l_time = get_l_time_range(time, time_range[0], time_range[1])
        i_time = np.where(l_time)[0]
        t0 = i_time[0]
        t1 = i_time[-1]
    else:
        t0 = 0
        t1 = -1

    time = time[t0:t1]

    temp = nc['temp'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan) # [time, s, eta, xi]
    salt = nc['salt'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan) # [time, s, eta, xi]

    u = nc['u'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan) # [time, s, eta, xi]
    v = nc['v'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan) # [time, s, eta, xi]

    nc.close()

    u_east, v_north = convert_roms_u_v_to_u_east_v_north(u, v, grid.angle)

    return RomsData(time, grid, u_east, v_north, temp, salt)

def plot_map_roms_data(roms_data:RomsData, location_info:LocationInfo, parameter:str, t:int, s:int,
                       ax=None, show=True,
                       cmap='RdBu_r', clabel='', vmin=None, vmax=None) -> plt.axes:
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        if len(values.shape) == 4:
            values = values[t, s, :, :] # [time, s, eta, xi]
        else:
            raise ValueError(f'Map plotting currently only works for 4D variables')
    elif parameter == 'velocity':
        u = roms_data.u[t, s, :, :]
        v = roms_data.v[t, s, :, :]
        values = np.sqrt(u**2+v**2)
    else:
        raise ValueError(f'Unknown parameter {parameter} in RomsData')

    c = ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, values, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(c)
    cbar.set_label(clabel)

    if parameter == 'velocity':
        thin = 5
        i = np.arange(0, u.shape[0], thin)
        j = np.arange(0, u.shape[1], thin)
        u_q = u[i][:, j]
        v_q = v[i][:, j]
        lon_q = roms_data.grid.lon[i][:, j]
        lat_q = roms_data.grid.lat[i][:, j]
        ax.quiver(lon_q, lat_q, u_q, v_q, scale=10, transform=ccrs.PlateCarree())

    if show is True:
        plt.show()
    else:
        return ax
