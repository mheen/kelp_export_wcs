from matplotlib import path
import numpy as np
from netCDF4 import Dataset

import sys
sys.path.append('..')
from py_tools.timeseries import convert_time_to_datetime, get_l_time_range

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

class RomsGrid:
    def __init__(self,
                 lon_rho:np.ndarray,
                 lat_rho:np.ndarray,
                 s_rho:np.ndarray,
                 angle:np.ndarray):

        self.lon = lon_rho # [eta, xi]
        self.lat = lat_rho # [eta, xi]
        self.s = s_rho
        self.angle = angle # [eta, xi]

    def get_lon_lat_range(self) -> tuple:
        lon_range = [np.nanmin(self.lon), np.nanmax(self.lon)]
        lat_range = [np.nanmin(self.lat), np.nanmax(self.lat)]

        return lon_range, lat_range

    def get_eta_xi_of_lon_lat_point(self, lon_p:float, lat_p:float) -> tuple:
        xi, _, eta, _ = bbox2ij(self.lon, self.lat, [lon_p, lon_p+0.1, lat_p, lat_p+0.1])
        return eta, xi

    def get_subgrid(self, i0:int, i1:int, j0:int, j1:int):
        lon = self.lon[j0:j1, i0:i1]
        lat = self.lat[j0:j1, i0:i1]
        angle = self.angle[j0:j1, i0:i1]
        
        return RomsGrid(lon, lat, self.s, angle)

    @staticmethod
    def read_from_netcdf(input_path='input/perth_roms_grid.nc'):
        netcdf = Dataset(input_path)

        lon_rho = netcdf['lon_rho'][:].filled(fill_value=np.nan)
        lat_rho = netcdf['lat_rho'][:].filled(fill_value=np.nan)
        s_rho = netcdf['s_rho'][:].filled(fill_value=np.nan)
        angle = netcdf['angle'][:].filled(fill_value=np.nan)

        netcdf.close()

        return RomsGrid(lon_rho, lat_rho, s_rho, angle)

class RomsData:
    def __init__(self,
                 time:np.ndarray,
                 grid:RomsGrid,
                 u_east:np.ndarray,
                 v_north:np.ndarray,
                 temp:np.ndarray,
                 salt:np.ndarray,
                 h:np.ndarray):
    
        self.time = time
        self.grid = grid
        self.u = u_east # [time, s, eta, xi]
        self.v = v_north # [time, s, eta, xi]
        self.temp = temp # [time, s, eta, xi]
        self.salt = salt # [time, s, eta, xi]
        self.h = h # [eta, xi]

    @staticmethod
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


    @staticmethod
    def read_from_netcdf(input_path:str, lon_range=None, lat_range=None, time_range=None):
        
        grid = RomsGrid.read_from_netcdf()

        if lon_range and lat_range is not None:
            i0, i1, j0, j1 = bbox2ij(grid.lon, grid.lat, [lon_range[0], lon_range[1], lat_range[0], lat_range[1]])
        else:
            i0 = 0
            i1 = -1
            j0 = 0
            j1 = -1

        sub_grid = grid.get_subgrid(i0, i1, j0, j1)

        netcdf = Dataset(input_path)

        time_org = netcdf['ocean_time'][:].filled(fill_value=np.nan)
        time_units = netcdf['ocean_time'].units
        time = convert_time_to_datetime(time_org, time_units)

        if time_range is not None:
            l_time = get_l_time_range(time, time_range[0], time_range[1])
            i_time = np.where(l_time)[0]
            t0 = i_time[0]
            t1 = i_time[-1]
        else:
            t0 = 0
            t1 = -1

        h = netcdf['h'][j0:j1, i0:i1].filled(fill_value=np.nan)

        temp = netcdf['temp'][:, :, j0:j1, i0:i1].filled(fill_value=np.nan)
        salt = netcdf['salt'][:, :, j0:j1, i0:i1].filled(fill_value=np.nan)

        u = netcdf['u'][:, :, j0:j1, i0:i1].filled(fill_value=np.nan)
        v = netcdf['v'][:, :, j0:j1, i0:i1].filled(fill_value=np.nan)

        netcdf.close()

        u_east, v_north = RomsData.convert_roms_u_v_to_u_east_v_north(u, v, sub_grid.angle)

        return RomsData(time, sub_grid, u_east, v_north, temp, salt, h)

if __name__ == '__main__':
    grid = RomsGrid.read_from_netcdf()
