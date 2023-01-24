import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_daily_files_in_time_range
from tools.timeseries import convert_time_to_datetime, get_l_time_range, get_closest_time_index
from tools.coordinates import get_distance_between_points, get_points_on_line_between_points
from tools.arrays import get_closest_index
from tools import log
from dataclasses import dataclass
from matplotlib import path
import numpy as np
from netCDF4 import Dataset
from datetime import datetime

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

    z = n.reshape(output_shape)/d

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
        if type(lon_p) is float:
            lon_p = [lon_p]
        if type(lat_p) is float:
            lat_p = [lat_p]
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

def get_subgrid(grid:RomsGrid, lon_range:list, lat_range:list, s_range:list) -> RomsGrid:
    i0, i1, j0, j1 = bbox2ij(grid.lon, grid.lat, [lon_range[0], lon_range[1], lat_range[0], lat_range[1]])

    s0 = None
    s1 = None
    if s_range is not None:
        s0 = s_range[0]
        s1 = s_range[1]

    lon = grid.lon[j0:j1, i0:i1]
    lat = grid.lat[j0:j1, i0:i1]
    s = grid.s[s0:s1]
    angle = grid.angle[j0:j1, i0:i1]
    h = grid.h[j0:j1, i0:i1]
    z = grid.z[s0:s1, j0:j1, i0:i1]

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

def _get_roms_data_from_netcdf(input_path:str, lon_range:list, lat_range:list, time_range:list) -> tuple:
    full_grid = read_roms_grid_from_netcdf(input_path)

    i0 = None
    i1 = None
    j0 = None
    j1 = None
    if lon_range is not None:
        i0 = get_closest_index(full_grid.lon[0, :], lon_range[0])
        if lon_range[1] == lon_range[0]:
            i1 = i0+1
        else:
            i1 = get_closest_index(full_grid.lon[0, :], lon_range[1])
    if lat_range is not None:
        j0 = get_closest_index(full_grid.lat[:, 0], lat_range[0])
        if lat_range[1] == lat_range[0]:
            j1 = j0+1
        else:
            j1 = get_closest_index(full_grid.lat[:, 0], lat_range[1])

    log.info(f'Reading ROMS data from {input_path}')
    nc = Dataset(input_path)

    time_org = nc['ocean_time'][:].filled(fill_value=np.nan)
    time_units = nc['ocean_time'].units
    time = convert_time_to_datetime(time_org, time_units)

    t0 = None
    t1 = None
    if time_range is not None:
        l_time = get_l_time_range(time, time_range[0], time_range[1])
        i_time = np.where(l_time)[0]
        if len(i_time) != 0:
            t0 = i_time[0]
            t1 = i_time[-1]+1

    time = time[t0:t1]

    temp = nc['temp'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan) # [time, s, eta, xi]
    salt = nc['salt'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan) # [time, s, eta, xi]

    # read in full u and v first, then select lon and lat range after conversion to u_east and v_north
    # ugly and slow way of doing this: figure out how to select correct ranges for u and v
    u = nc['u'][t0:t1, :, :, :].filled(fill_value=np.nan) # [time, s, eta, xi]
    v = nc['v'][t0:t1, :, :, :].filled(fill_value=np.nan) # [time, s, eta, xi]

    nc.close()

    u_east, v_north = convert_roms_u_v_to_u_east_v_north(u, v, full_grid.angle)
    u_east = u_east[:, :, j0:j1, i0:i1]
    v_north = v_north[:, :, j0:j1, i0:i1]

    lon = full_grid.lon[j0:j1, i0:i1]
    lat = full_grid.lat[j0:j1, i0:i1]
    s = full_grid.s[:]
    angle = full_grid.angle[j0:j1, i0:i1]
    h = full_grid.h[j0:j1, i0:i1]
    z = full_grid.z[:, j0:j1, i0:i1]
    grid = RomsGrid(lon, lat, s, angle, h, z)

    return time, grid, u_east, v_north, temp, salt

def read_roms_data_from_netcdf(input_path:list, lon_range=None, lat_range=None, time_range=None) -> RomsData:
    time, grid, u_east, v_north, temp, salt = _get_roms_data_from_netcdf(input_path,
                                                                         lon_range,
                                                                         lat_range,
                                                                         time_range)

    return RomsData(time, grid, u_east, v_north, temp, salt)

def read_roms_data_from_multiple_netcdfs(input_dir:str, start_time:datetime, end_time:datetime,
                                         lon_range=None, lat_range=None) -> RomsData:
    time_range = [start_time, end_time]

    nc_files = get_daily_files_in_time_range(input_dir, start_time, end_time, 'nc')

    time, grid, u_east, v_north, temp, salt = _get_roms_data_from_netcdf(nc_files[0], lon_range, lat_range, time_range)

    for i in range(1, len(nc_files)):
        t, _, u, v, tp, s = _get_roms_data_from_netcdf(nc_files[i], lon_range, lat_range, time_range)
        time = np.concatenate((time, t))
        u_east = np.concatenate((u_east, u))
        v_north = np.concatenate((v_north, v))
        temp = np.concatenate((temp, tp))
        salt = np.concatenate((salt, s))

    return RomsData(time, grid, u_east, v_north, temp, salt)

def get_eta_xi_along_transect(grid:RomsGrid, lon1:float, lat1:float, lon2:float, lat2:float, ds:float) -> tuple:
    lons, lats = get_points_on_line_between_points(lon1, lat1, lon2, lat2, ds)
    eta, xi = grid.get_eta_xi_of_lon_lat_point(lons, lats)
    return eta, xi

def get_distance_along_transect(lons:np.ndarray, lats:np.ndarray):
    distance = [0]
    
    for i in range(len(lons)-1):
        d = get_distance_between_points(lons[i], lats[i], lons[i+1], lats[i+1])
        distance.append(d)
    distance = np.array(distance)
    
    return np.cumsum(distance) # distance in meters

def get_gradient_along_transect(roms_data:RomsData, parameter:str, s_layer:int, time:datetime,
                                lon1:float, lat1:float, lon2:float, lat2:float, ds:float) -> tuple:
    
    eta, xi = get_eta_xi_along_transect(roms_data.grid, lon1, lat1, lon2, lat2, ds)
    lon = roms_data.grid.lon[eta, xi]
    lat = roms_data.grid.lat[eta, xi]
    distance = get_distance_along_transect(lon, lat)/1000 # distance in km

    t = get_closest_time_index(roms_data.time, time)

    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
    elif hasattr(roms_data.grid, parameter):
        values = getattr(roms_data.grid, parameter)
    else:
        raise ValueError(f'Unknown ROMS parameter {parameter} requested.')

    if len(values.shape) == 2: # [eta, xi]
        values = values[eta, xi]
    elif len(values.shape) == 3: # [time, eta, xi]
        values = values[t, eta, xi]
    elif len(values.shape) == 4: # [time, s, eta, xi]
        values = values[t, s_layer, eta, xi]

    dvalue = np.diff(values)
    dx = np.diff(distance)

    gradient = dvalue/dx

    return gradient, values, distance
