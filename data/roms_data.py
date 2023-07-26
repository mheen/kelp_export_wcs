import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_daily_files_in_time_range, get_files_in_dir, create_dir_if_does_not_exist
from tools.timeseries import convert_time_to_datetime, convert_datetime_to_time, get_l_time_range, get_closest_time_index
from tools.coordinates import get_distance_between_points, get_points_on_line_between_points, get_transect_lons_lats_ds_from_json
from tools.coordinates import get_bearing_between_points
from tools.velocity_shore_angles import get_cross_shelf_velocity
from tools import log
from tools.seawater_density import calculate_density
from dataclasses import dataclass
from matplotlib import path
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import distutils.spawn
import subprocess
from scipy import spatial
from scipy.ndimage import gaussian_filter
import warnings
import shutil
import matplotlib.pyplot as plt

def bbox2ij(lon:np.ndarray, lat:np.ndarray, bbox:list) -> tuple:
    '''Return indices for i,j that will completely cover the specified bounding box.     
    i0, i1, j0, j1 = bbox2ij(lon, lat, bbox)
    lon, lat = 2D arrays that are the target of the subset
    bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]

    Example
    -------  
    >>> i0, i1, j0, j1 = bbox2ij(lon_rho, [-71, -63., 39., 46])
    >>> h_subset = nc.variables['h'][j0:j1, i0:i1]

    Copied from Rich Signell
    https://gis.stackexchange.com/questions/71630/subsetting-a-curvilinear-netcdf-file-roms-model-output-using-a-lon-lat-boundin
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

class RomsGrid:
    def __init__(self, lon:np.ndarray,
                 lat:np.ndarray,
                 s:np.ndarray,
                 angle:np.ndarray,
                 h:np.ndarray,
                 z:np.ndarray,
                 s_w:np.ndarray,
                 z_w:np.ndarray,
                 pm:np.ndarray,
                 pn:np.ndarray):
        self.lon = lon # [eta, xi]
        self.lat = lat # [eta, xi]
        self.s = s # [s]
        self.angle = angle # [eta, xi]
        self.h = h # [eta, xi]
        self.z = z # [s, eta, xi]
        self.s_w = s_w # [s_w] (len(s)+1)
        self.z_w = z_w # [s_w, eta, xi]
        self.pm = pm # [eta, xi] curvilinear coordinate metric in xi (1/m)
        self.pn = pn # [eta, xi] curvilinear coordinate metric in eta (1/m)
    
        grid_coords_1d = list(zip(np.ravel(self.lon), np.ravel(self.lat)))
        self.kdtree = spatial.KDTree(grid_coords_1d)

    def get_grid_cell_areas(self) -> np.ndarray:
        area = 1/(self.pm*self.pn)
        return area

    def get_lon_lat_range(self) -> tuple:
        lon_range = [np.nanmin(self.lon), np.nanmax(self.lon)]
        lat_range = [np.nanmin(self.lat), np.nanmax(self.lat)]

        return lon_range, lat_range

    def get_eta_xi_of_lon_lat_point(self, lon_p:np.ndarray, lat_p:np.ndarray) -> tuple:
        float_types = [float, np.float64]
        if type(lon_p) in float_types and type(lat_p) in float_types:
            distance, index = self.kdtree.query([lon_p, lat_p])
            eta, xi = np.unravel_index(index, self.lon.shape)
            return xi, eta
        etas = []
        xis = []
        for i in range(len(lon_p)):
            distance, index = self.kdtree.query([lon_p[i], lat_p[i]])
            eta, xi = np.unravel_index(index, self.lon.shape)
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
    
    s_w = nc['s_w'][:].filled(fill_value=np.nan)
    cs_w = nc['Cs_w'][:].filled(fill_value=np.nan)
    z_w = get_z(s_w, h, cs_w, hc)

    pm = nc['pm'][:].filled(fill_value=np.nan)
    pn = nc['pn'][:].filled(fill_value=np.nan)

    nc.close()

    return RomsGrid(lon_rho, lat_rho, s_rho, angle, h, z, s_w, z_w, pm, pn)

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

    s_w = grid.s_w[s0:s1] # not strictly correct since s_range will be different for s_rho and s_w
    z_w = grid.z_w[s0:s1, j0:j1, i0:i1]

    pm = grid.pm[j0:j1, i0:i1]
    pn = grid.pn[j0:j1, i0:i1]

    return RomsGrid(lon, lat, s, angle, h, z, s_w, z_w, pm, pn)

@dataclass
class RomsData:
    time: np.ndarray
    grid: RomsGrid
    u_east: np.ndarray
    v_north: np.ndarray
    temp: np.ndarray
    salt: np.ndarray
    density: np.ndarray
    sigma_t: np.ndarray

def get_roms_data_from_netcdf(full_grid:RomsGrid, input_path:str, lon_range:list, lat_range:list, time_range:list) -> tuple:
    i0 = None
    i1 = None
    j0 = None
    j1 = None
    if lon_range is not None and lat_range is not None:
        i0, i1, j0, j1 = bbox2ij(full_grid.lon, full_grid.lat, [lon_range[0], lon_range[1], lat_range[0], lat_range[1]])
        if i0 == i1:
            i1 = i0+1
        if j0 == j1:
            j1 = j0+1
    if lon_range is not None and lat_range is None or lat_range is not None and lon_range is None:
        raise ValueError('You must specify both a lon and lat range, not just one.')

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

    # subset u and v (potentially needs padding to convert u- and v- to rho-points)
    # based on position of slice, add padding if needed
    len_eta = full_grid.lon.shape[0]
    len_xi = full_grid.lon.shape[1]
    if i0 is None and i1 is None or i0 == 0 and i1 == len_xi-1: # all (pad both sides)
        u_org = nc['u'][t0:t1, :, j0:j1, :].filled(fill_value=np.nan)
        u = np.empty((u_org.shape[0], u_org.shape[1], u_org.shape[2], u_org.shape[3]+2))
        u[:, :, :, 1:-1] = u_org
        u[:, :, :, 0] = u_org[:, :, :, 0]
        u[:, :, :, -1] = u_org[:, :, :, -1]
    elif i0 == 0 and i1 < len_xi-1: # left (pad left)
        u_org = nc['u'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan)
        u = np.empty((u_org.shape[0], u_org.shape[1], u_org.shape[2], u_org.shape[3]+1))
        u[:, :, :, 1:] = u_org
        u[:, :, :, 0] = u_org[:, :, :, 0]
    elif i0 > 0 and i1 < len_xi-1: # middle (no padding)
        u = nc['u'][t0:t1, :, j0:j1, i0-1:i1].filled(fill_value=np.nan)
    elif i0 > 0 and i1 == len_xi-1: # right (pad right)
        u_org = nc['u'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan)
        u = np.empty((u_org.shape[0], u_org.shape[1], u_org.shape[2], u_org.shape[3]+1))
        u[:, :, :, 0:-1] = u_org
        u[:, :, :, -1] = u_org[:, :, :, -1]

    if j0 is None and j1 is None or j0 == 0 and j1 == len_eta-1: # all (pad both sides)
        v_org = nc['v'][t0:t1, :, :, i0:i1].filled(fill_value=np.nan)
        v = np.empty((v_org.shape[0], v_org.shape[1], v_org.shape[2]+2, v_org.shape[3]))
        v[:, :, 1:-1, :] = v_org
        v[:, :, 0, :] = v_org[:, :, 0, :]
        v[:, :, -1, :] = v_org[:, :, -1, :]
    elif j0 == 0 and j1 < len_eta-1: # bottom (pad bottom)
        v_org = nc['v'][t0:t1, :, j0:j1, i0:i1].filled(fill_value=np.nan)
        v = np.empty((v_org.shape[0], v_org.shape[1], v_org.shape[2]+1, v_org.shape[3]))
        v[:, :, 1:, :] = v_org
        v[:, :, 0, :] = v_org[:, :, 0, :]
    elif j0 > 0 and j1 < len_eta-1: # middle (no padding)
        v = nc['v'][t0:t1, :, j0-1:j1, i0:i1].filled(fill_value=np.nan)
    elif j0 > 0 and j1 == len_eta-1: # top (pad top)
        v_org = nc['v'][t0:t1, :, j0-1:j1, i0:i1].filled(fill_value=np.nan)
        v = np.empty((v_org.shape[0], v_org.shape[1], v_org.shape[2]+1, v_org.shape[3]))
        v[:, :, 0:-1, :] = v_org
        v[:, :, -1, :] = v_org[:, :, -1, :]

    nc.close()

    def convert_roms_u_v_to_u_east_v_north(u:np.ndarray, v:np.ndarray, angle:np.ndarray) -> tuple:
        '''Convert u and v from curvilinear ROMS output to u eastwards and v northwards.
        This is done by:
        1. Converting u and v so that they are on rho-coordinate point (cell center).
        2. Rotating u and v so they are directed eastwards and northwards respectively.
        
        Example grid: "." = rho-point, "x" = u-point, "*" = v-point
            _________ _________ _________ _________
            |         |         |         |         |
            |    .    x    .    x    .    x    .    |
            |   0,2  0,2  1,2  1,2  2,2  2,2  3,2   |
            |         |         |         |         |
            |____*____|____*____|____*____|____*____|
            |   0,1   |   1,1   |   2,1   |   3,1   |
            |    .    x    .    x    .    x    .    |
            |   0,1  0,1  1,1  1,1  2,1  2,1  3,1   |
            |         |         |         |         |
            |____*____|____*____|____*____|____*____|
            |   0,0   |   1,0   |   2,0   |   3,0   |
            |    .    x    .    x    .    x    .    |
            |   0,0  0,0  1,0  1,0  2,0  2,0  3,0   |
        ^  |         |         |         |         |
        |  |_________|_________|_________|_________|
        eta  xi ->
        '''

        def u2rho(var_u:np.ndarray) -> np.ndarray:
            '''Convert variable on u-coordinate to rho-coordinate.'''
            var_u_size = var_u.shape
            n_dimension = len(var_u_size)
            L = var_u_size[-1]
            if n_dimension == 4:
                var_rho = 0.5*(var_u[:, :, :, 0:-1]+var_u[:, :, :, 1:])
            return var_rho

        def v2rho(var_v:np.ndarray) -> np.ndarray:
            '''Convert variable on v-coordinate to rho-coordinate.'''
            var_v_size = var_v.shape
            n_dimension = len(var_v_size)
            M = var_v_size[-2]      
            if n_dimension == 4:
                var_rho = 0.5*(var_v[:, :, 0:-1, :]+var_v[:, :, 1:, :])
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

    angle = full_grid.angle[j0:j1, i0:i1]
    u_east, v_north = convert_roms_u_v_to_u_east_v_north(u, v, angle)

    lon = full_grid.lon[j0:j1, i0:i1]
    lat = full_grid.lat[j0:j1, i0:i1]
    s = full_grid.s[:]
    
    h = full_grid.h[j0:j1, i0:i1]
    z = full_grid.z[:, j0:j1, i0:i1]
    s_w = full_grid.s[:]
    z_w = full_grid.z_w[:, j0:j1, i0:i1]

    pm = full_grid.pm[j0:j1, i0:i1]
    pn = full_grid.pn[j0:j1, i0:i1]

    grid = RomsGrid(lon, lat, s, angle, h, z, s_w, z_w, pm, pn)

    density = calculate_density(salt, temp, -z)
    sigma_t = density-1000

    return time, grid, u_east, v_north, temp, salt, density, sigma_t

def read_roms_data_from_netcdf(input_path:list, lon_range=None, lat_range=None, time_range=None) -> RomsData:
    try:
        full_grid = read_roms_grid_from_netcdf(input_path)
    except:
        grid_file = f'{os.path.abspath(os.path.join(os.path.dirname(input_path), os.pardir))}/grid.nc'
        warnings.warn(f'No grid information found in {input_path}. Trying separate grid file instead: {grid_file}')
        full_grid = read_roms_grid_from_netcdf(grid_file)
    
    time, grid, u_east, v_north, temp, salt, density, sigma_t = get_roms_data_from_netcdf(full_grid, input_path,
                                                                                 lon_range,
                                                                                 lat_range,
                                                                                 time_range)

    return RomsData(time, grid, u_east, v_north, temp, salt, density, sigma_t)

def read_roms_data_from_multiple_netcdfs(input_dir:str, start_time:datetime, end_time:datetime,
                                         lon_range=None, lat_range=None) -> RomsData:
    time_range = [start_time, end_time]

    nc_files = get_daily_files_in_time_range(input_dir, start_time, end_time, 'nc')

    try:
        full_grid = read_roms_grid_from_netcdf(nc_files[0])
    except:
        grid_file = f'{os.path.abspath(os.path.join(os.path.dirname(nc_files[0]), os.pardir))}/grid.nc'
        warnings.warn(f'No grid information found in {nc_files[0]}. Trying separate grid file instead: {grid_file}')
        full_grid = read_roms_grid_from_netcdf(grid_file)
    time, grid, u_east, v_north, temp, salt, density, sigma_t = get_roms_data_from_netcdf(full_grid, nc_files[0], lon_range, lat_range, time_range)

    for i in range(1, len(nc_files)):
        t, _, u, v, tp, s, d, st = get_roms_data_from_netcdf(full_grid, nc_files[i], lon_range, lat_range, time_range)
        time = np.concatenate((time, t))
        u_east = np.concatenate((u_east, u))
        v_north = np.concatenate((v_north, v))
        temp = np.concatenate((temp, tp))
        salt = np.concatenate((salt, s))
        density = np.concatenate((density, d))
        sigma_t = np.concatenate((sigma_t, st))

    return RomsData(time, grid, u_east, v_north, temp, salt, density, sigma_t)

def get_daily_means(time:np.ndarray, values:np.ndarray, time_axis=0) -> tuple:
    daily_time = []
    daily_values = []

    n_days = (time[-1]-time[0]).days+1

    for n in range(n_days):
        start_date = datetime(time[0].year, time[0].month, time[0].day, 0, 0)+timedelta(days=n)
        end_date = start_date+timedelta(days=1)
        l_time = get_l_time_range(time, start_date, end_date)
        daily_time.append(start_date)
        daily_values.append(np.nanmean(values[l_time], axis=time_axis))

    return np.array(daily_time), np.array(daily_values)

def get_daily_mean_roms_data(roms_data:RomsData) -> RomsData:
    time_d, u_east_d = get_daily_means(roms_data.time, roms_data.u_east)
    _, v_north_d = get_daily_means(roms_data.time, roms_data.v_north)
    _, temp_d = get_daily_means(roms_data.time, roms_data.temp)
    _, salt_d = get_daily_means(roms_data.time, roms_data.salt)
    _, density_d = get_daily_means(roms_data.time, roms_data.density)
    _, sigma_t_d = get_daily_means(roms_data.time, roms_data.sigma_t)
    
    return RomsData(time_d, roms_data.grid, u_east_d, v_north_d, temp_d, salt_d, density_d, sigma_t_d)

def get_eta_xi_along_transect(grid:RomsGrid, lon1:float, lat1:float,
                              lon2:float, lat2:float, ds:float) -> tuple:
    lons, lats = get_points_on_line_between_points(lon1, lat1, lon2, lat2, ds)
    eta, xi = grid.get_eta_xi_of_lon_lat_point(lons, lats)

    coords = list(zip(eta, xi))
    unique_coords = list(dict.fromkeys(coords))
    unique_coords_list = list(zip(*unique_coords))
    eta_unique = unique_coords_list[0]
    xi_unique = unique_coords_list[1]
    
    return eta_unique, xi_unique

def get_lon_lat_along_depth_contour(roms_grid:RomsGrid, h_level=100) -> tuple[np.ndarray, np.ndarray]:
    levels = [0, 10, 20, 50, 100, 200, 400, 600, 1000]

    try:
        i_level = levels.index(h_level)
    except:
        raise ValueError(f'Levels does not contain {h_level}. Please request a valid value: {levels}')

    ax = plt.axes()
    contour_set = ax.contour(roms_grid.lon, roms_grid.lat, roms_grid.h, levels=levels)

    contour_line = contour_set.collections[i_level].get_paths()[0]
    contour_line_coords = contour_line.vertices
    lons = contour_line_coords[:, 0]
    lats = contour_line_coords[:, 1]

    plt.close()

    return lons, lats

def get_eta_xi_along_depth_contour(roms_grid:RomsGrid, h_level=100) -> tuple[np.ndarray, np.ndarray]:
    lons, lats = get_lon_lat_along_depth_contour(roms_grid, h_level=h_level)
    eta, xi = roms_grid.get_eta_xi_of_lon_lat_point(lons, lats)

    return eta, xi

def get_distance_along_transect(lons:np.ndarray, lats:np.ndarray):
    distance = [0]
    
    for i in range(len(lons)-1):
        d = get_distance_between_points(lons[i], lats[i], lons[i+1], lats[i+1])
        distance.append(d)
    distance = np.array(distance)
    
    return np.cumsum(distance) # distance in meters

def get_down_transect_velocity_component(u:np.ndarray, v:np.ndarray,
                                         lon1:float, lat1:float, lon2:float, lat2:float) -> np.ndarray:
    alpha = get_bearing_between_points(lon1, lat1, lon2, lat2)
    alpha_rad = np.deg2rad(alpha)
    down_transect = u*np.cos(alpha_rad)+v*np.sin(alpha_rad)
    return down_transect

def get_cross_shelf_velocity_component(roms_data:RomsData) -> np.ndarray:
    h_smooth = gaussian_filter(roms_data.grid.h, 2) # smooth bathymetry before calculating cross-shelf velocity
    u_cross = get_cross_shelf_velocity(h_smooth, roms_data.u_east, roms_data.v_north)
    return u_cross

@dataclass
class TransectData:
    time: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    distance: np.ndarray
    z: np.ndarray
    h: np.ndarray
    u_down: np.ndarray
    temp: np.ndarray
    salt: np.ndarray
    density: np.ndarray
    sigma_t: np.ndarray

def get_transect_data(roms_data:RomsData, lon1:float, lat1:float,
                      lon2:float, lat2:float, ds:float) -> TransectData:
    
    eta, xi = get_eta_xi_along_transect(roms_data.grid, lon1, lat1, lon2, lat2, ds)
    lon = roms_data.grid.lon[eta, xi]
    lat = roms_data.grid.lat[eta, xi]
    distance = get_distance_along_transect(lon, lat)/1000 # km
    z = roms_data.grid.z[:, eta, xi]
    h = roms_data.grid.h[eta, xi]
    distance2d = np.repeat(distance[np.newaxis, :], z.shape[0], axis=0)

    # u = roms_data.u_east[:, :, eta, xi]
    # v = roms_data.v_north[:, :, eta, xi]
    # u_down = get_down_transect_velocity_component(u, v, lon1, lat1, lon2, lat2)
    u_cross = get_cross_shelf_velocity_component(roms_data)
    u_down = u_cross[:, :, eta, xi]

    temp = roms_data.temp[:, :, eta, xi]
    salt = roms_data.salt[:, :, eta, xi]
    density = roms_data.density[:, :, eta, xi]
    sigma_t = roms_data.sigma_t[:, :, eta, xi]

    return TransectData(roms_data.time, lon, lat, distance2d, z, h, u_down, temp, salt, density, sigma_t)

def get_roms_data_for_transect(input_dir:str, start_time:datetime, end_time:datetime,
                               lon1:float, lat1:float, lon2:float, lat2:float) -> RomsData:
    lon_range = [np.nanmin([lon1, lon2]), np.nanmax([lon1, lon2])]
    lat_range = [np.nanmin([lat1, lat2]), np.nanmax([lat1, lat2])]
    roms_data = read_roms_data_from_multiple_netcdfs(input_dir, start_time, end_time,
                                                     lon_range=lon_range, lat_range=lat_range)
    return roms_data

def get_depth_integrated_gradient_along_transect(roms_data:RomsData, parameter:str,
                                                 lon1:float, lat1:float,
                                                 lon2:float, lat2:float, ds:float) -> tuple:
    warnings.warn('''The calculation of the gradient along a transect assumes that
                  increasing distance is means moving away from the coast:
                  i.e. the point (lon1, lat1) is closer to the coast than the point (lon2, lat2).
                  If this is not the case, you will get negative gradients where you
                  expect positive ones and vice versa.''')


    eta, xi = get_eta_xi_along_transect(roms_data.grid, lon1, lat1, lon2, lat2, ds)
    lon = roms_data.grid.lon[eta, xi]
    lat = roms_data.grid.lat[eta, xi]
    distance = get_distance_along_transect(lon, lat)/1000 # distance in km
    z = roms_data.grid.z[:, eta, xi]

    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
    elif hasattr(roms_data.grid, parameter):
        values = getattr(roms_data.grid, parameter)
    else:
        raise ValueError(f'Unknown ROMS parameter {parameter} requested.')

    if len(values.shape) == 2: # [eta, xi]
        values = values[eta, xi]
        dvalues = np.diff(values)
    elif len(values.shape) == 3: # [time, eta, xi]
        values = values[:, eta, xi]
        dvalues = np.diff(values)
    elif len(values.shape) == 4: # [time, s, eta, xi]
        values = values[:, :, eta, xi]
        depth_average_values = np.nanmean(values, axis=1)
        dvalues = np.diff(depth_average_values)

    gradient = np.nanmean(dvalues/np.diff(distance), axis=1) # mean gradient varying in time

    return gradient, values, distance, z, roms_data.time

def read_depth_integrated_gradient_data_from_netcdf(input_path:str) -> tuple:
    log.info(f'Reading gradient data from: {input_path}')
    nc = Dataset(input_path)

    time_org = nc['time'][:].filled(fill_value=np.nan)
    time_units = nc['time'].units
    time = convert_time_to_datetime(time_org, time_units)

    gradient = nc['gradient'][:].filled(fill_value=np.nan)
    values = nc['values'][:].filled(fill_value=np.nan)

    distance = nc['distance'][:].filled(fill_value=np.nan)
    z = nc['z2d'][:].filled(fill_value=np.nan)

    nc.close()

    return gradient, values, distance, z, time

def write_depth_integrated_gradient_data_to_netcdf(output_path:str, time:np.ndarray,
                                                   gradient:np.ndarray,
                                                   values:np.ndarray, distance:np.ndarray,
                                                   z:np.ndarray):
    
    create_dir_if_does_not_exist(os.path.dirname(output_path))

    log.info(f'Saving gradient data to: {output_path}')
    nc = Dataset(output_path, 'w', format='NETCDF4')
    
    # create dimensions
    nc.createDimension('time', len(time))
    nc.createDimension('x', len(distance))
    nc.createDimension('z', z.shape[0])
    # create variables
    nc_time = nc.createVariable('time', float, 'time', zlib=True)
    nc_dist = nc.createVariable('distance', float, 'x', zlib=True)
    nc_z = nc.createVariable('z2d', float, ('z', 'x'), zlib=True)
    nc_grad = nc.createVariable('gradient', float, 'time', zlib=True)
    nc_values = nc.createVariable('values', float, ('time', 'z', 'x'), zlib=True)
    # fill variables
    time_int, time_units = convert_datetime_to_time(time)
    nc_time[:] = time_int
    nc_time.units = time_units
    nc_dist[:] = distance
    nc_z[:] = z
    nc_grad[:] = gradient
    nc_values[:] = values
    nc.close()

def write_transect_data_to_netcdf(input_dir:str, output_dir:str, lon1:float, lat1:float,
                                  lon2:float, lat2:float, ds:float,
                                  output_file_description='',
                                  grid_file='input/cwa_roms_grid.nc'):
    
    grid = read_roms_grid_from_netcdf(grid_file)
    eta, xi = get_eta_xi_along_transect(grid, lon1, lat1, lon2, lat2, ds)
    eta0 = np.nanmax([np.nanmin(eta)-2, 0]) # margin of 2, but can't be less than 0
    eta1 = np.nanmin([np.nanmax(eta)+2, grid.lon.shape[0]-1]) # margin of 2, but can't be more than length
    xi0 = np.nanmax([np.nanmin(xi)-2, 0]) # margin of 2, but can't be less than 0
    xi1 = np.nanmin([np.nanmax(xi)+2, grid.lon.shape[1]-1]) # margin of 2, but can't be more than length

    create_dir_if_does_not_exist(output_dir)

    ncfiles = get_files_in_dir(input_dir, 'nc', return_full_path=False)
    for ncfile in ncfiles:
        output_path = f'{output_dir}{output_file_description}{ncfile}'
        if os.path.exists(output_path):
            log.info(f'Transect file already exists, skipping: {output_path}')
        input_path = f'{input_dir}{ncfile}'

        ncks_slice_rho = ['-d', f'eta_rho,{eta0},{eta1},1' ,'-d', f'xi_rho,{xi0},{xi1},1']
        ncks_slice_v = ['-d', f'eta_v,{eta0},{eta1-1},1', '-d', f'xi_v,{xi0},{xi1},1']
        ncks_slice_u = ['-d', f'eta_u,{eta0},{eta1},1', '-d', f'xi_u,{xi0},{xi1-1},1']
        ncks_options = ncks_slice_rho + ncks_slice_v + ncks_slice_u
        command = [distutils.spawn.find_executable('ncks')] + ncks_options + [input_path, output_path]
        log.info(f'Extracting transect data, saving to: {output_path}')
        subprocess.run(command)

def get_vel_correction_factor_for_specific_height_above_sea_floor(z_req:float, grid_path='input/cwa_roms_grid.nc'):
    '''Calculates correction factor that needs to be applied to velocities assuming a logarithmic
    vertical velocity profile in the bottom layer.
    
    Calculation assumes:
    u(z) = u*/kappa*log(z/z0)
    u* = sqrt(tau_b)
    tau_b = kappa**2*u_sigma0**2/log**2(z_sigma0/z0) : bottom stress calculated from velocity in lowest layer

    Using u_sigma0 = 1, we calculate a spatially varying correction factor for the requested depth.
    '''
    kappa = 0.41 # von Karman constant
    z0 = 1.65*10**(-5) # m bottom roughness (can potentially vary in space but is kept constant in ROMS)
    roms_grid = read_roms_grid_from_netcdf(grid_path)
    z_sigma0 = (roms_grid.z[1, :, :]-roms_grid.z[0, :, :])/2 # height of bottom layer above sea floor (divided by 2 because u,v-velocities are in middle of layer)
    u_sigma0 = 1 # m/s -> using this instead of actual velocity to get a correction factor
    tau_b = kappa**2*u_sigma0**2/(np.log(z_sigma0/z0))**2
    u_star = np.sqrt(tau_b)

    u_corr = u_star/kappa*np.log(z_req/z0)
    # if z_req > z_sigma0 then the correction will be > 1. However, we assume that if this is the case,
    # currents are beyond the logarithmic layer and so the ROMS velocity and not the log-layer velocity
    # should hold. So, we set u_corr to be 1 in these cases.
    u_corr[u_corr>1.0] = 1.0

    return u_corr

def write_roms_velocities_at_specific_depth(input_dir:str, output_dir:str, z_req:float):
    '''Assuming a logarithmic velocity profile in the bottom layer, calculate what the
    velocity would be at a specific requested depth.
    '''

    create_dir_if_does_not_exist(output_dir)
    corr = get_vel_correction_factor_for_specific_height_above_sea_floor(z_req)

    ncfiles = get_files_in_dir(input_dir, 'nc')
    for ncfile in ncfiles:
        ncfile_new = f'{output_dir}{os.path.basename(ncfile)}'
        log.info(f'Copied file {ncfile} to {ncfile_new}')
        shutil.copyfile(ncfile, ncfile_new)

        nc = Dataset(ncfile_new, mode='r+')
        u = nc['u'][:].filled(fill_value=np.nan)
        v = nc['v'][:].filled(fill_value=np.nan)
        nc['u'][:, 0, :, :] = np.multiply(corr[:, :-1], u[:, 0, :, :])
        nc['v'][:, 0, :, :] = np.multiply(corr[:-1, :], v[:, 0, :, :])
        nc.close()
        log.info(f'Applied vertical correction to file: {ncfile_new}')
