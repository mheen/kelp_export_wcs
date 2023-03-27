from tools.timeseries import convert_time_to_datetime, get_l_time_range
from tools import log
from dataclasses import dataclass
from netCDF4 import Dataset
import numpy as np
from warnings import warn
from datetime import datetime

class Particles:
    def __init__(self, time:np.ndarray,
                 status:np.ndarray,
                 lon:np.ndarray,
                 lat:np.ndarray,
                 z:np.ndarray,
                 salt:np.ndarray,
                 temp:np.ndarray,
                 age:np.ndarray,
                 moving:np.ndarray):
        self.time = time
        self.status = status # [trajectory, time] # 0: active, 1: stranded, -999: not active yet or out of domain
        self.lon = lon # [trajectory, time]
        self.lat = lat # [trajectory, time]
        self.z = z # [trajectory, time]

        self.salt = salt # [trajectory, time]
        self.temp = temp # [trajectory, time]
        
        self.age = age # [trajectory, time] (days)

        self.moving = moving # [trajectory, time] # 0: not moving, 1: moving

        self.add_initial_positions()

    def add_initial_positions(self):
        t_release = self.get_release_time_index()

        self.lon0 = []
        self.lat0 = []
        self.z0 = []
        for i in range(self.lon.shape[0]):
            self.lon0.append(self.lon[i, t_release[i]])
            self.lat0.append(self.lat[i, t_release[i]])
            self.z0.append(self.z[i, t_release[i]])
        self.lon0 = np.array(self.lon0)
        self.lat0 = np.array(self.lat0)
        self.z0 = np.array(self.z0)

    def get_particles_in_time_range(self, start_time:datetime, end_time:datetime):
        l_time = get_l_time_range(self.time, start_time, end_time)
        time = np.copy(self.time[l_time])
        # remove particles that are not released within time range
        l_released = ~np.all(np.isnan(self.lon[:, l_time]), axis=1)
        status = np.copy(self.status[l_released, :][:, l_time])
        lon = np.copy(self.lon[l_released, :][:, l_time])
        lat = np.copy(self.lat[l_released, :][:, l_time])
        z = np.copy(self.z[l_released, :][:, l_time])
        salt = np.copy(self.salt[l_released, :][:, l_time])
        temp = np.copy(self.temp[l_released, :][:, l_time])
        age = np.copy(self.age[l_released, :][:, l_time])
        if moving is not None:
            moving = np.copy(self.moving[l_released, :][:, l_time])
        else:
            moving = None
        return Particles(time, status, lon, lat, z, salt, temp, age, moving)

    def filter_based_on_release_depth(self, h_min:float, h_max:float):
        l_depth = np.logical_and(self.z0<-h_min, self.z0>=-h_max)

        status = self.status[l_depth, :]
        lon = self.lon[l_depth, :]
        lat = self.lat[l_depth, :]
        z = self.z[l_depth, :]

        salt = self.salt[l_depth, :]
        temp = self.temp[l_depth, :]

        age = self.age[l_depth, :]

        if self.moving is not None:
            moving = self.moving[l_depth, :]
        else:
            moving = None

        return Particles(self.time, status, lon, lat, z, salt, temp, age, moving)

    def filter_based_on_release_lon_lat_range(self, lon_range:list, lat_range:list):
        l_lon_range = np.logical_and(self.lon0>=lon_range[0], self.lon0<=lon_range[1])
        l_lat_range = np.logical_and(self.lat0>=lat_range[0], self.lat0<=lat_range[1])
        l_range = np.logical_and(l_lon_range, l_lat_range)

        status = self.status[l_range, :]
        lon = self.lon[l_range, :]
        lat = self.lat[l_range, :]
        z = self.z[l_range, :]

        salt = self.salt[l_range, :]
        temp = self.temp[l_range, :]

        age = self.age[l_range, :]

        if self.moving is not None:
            moving = self.moving[l_range, :]
        else:
            moving = None

        return Particles(self.time, status, lon, lat, z, salt, temp, age, moving)

    def get_l_deep_sea(self, h_deep_sea:float, remain=False) -> np.ndarray:
        
        if remain is True:
            i_ds, j_ds = np.where(self.z <= -h_deep_sea)
            l_deep_sea = np.zeros(self.z.shape)
            for k in range(len(i_ds)):
                # once particles in deep sea, they will remain there
                # (stops particles that move out of simulation domain
                # from being removed from deep sea count)
                l_deep_sea[i_ds[k], j_ds[k]:] = 1
            return l_deep_sea

        l_deep_sea = self.z <= -h_deep_sea # particles can move into deep sea and then out again

        return l_deep_sea

    def get_indices_arriving_in_deep_sea(self, h_deep_sea:float) -> tuple:
        l_deep_sea = self.get_l_deep_sea(h_deep_sea)
        p_ds, t_ds = np.where(l_deep_sea==1)
        p_arriving_ds, i_sort = np.unique(p_ds, return_index=True)
        t_arriving_ds = t_ds[i_sort]
        return p_arriving_ds, t_arriving_ds

    def get_n_deep_sea(self, h_deep_sea:float) -> np.ndarray:
        l_deep_sea = self.get_l_deep_sea(h_deep_sea, remain=True)
        n_deep_sea = np.sum(l_deep_sea, axis=0)
        return n_deep_sea

    def bin_age_arriving_in_deep_sea(self, h_deep_sea:float) -> tuple:
        p_arriving_ds, t_arriving_ds = self.get_indices_arriving_in_deep_sea(h_deep_sea)
        age_arriving_ds = self.age[p_arriving_ds, t_arriving_ds]
        age_bins = np.arange(0, np.ceil(np.nanmax(self.age))+2, 1)
        i_age_arriving_ds = np.digitize(age_arriving_ds, age_bins)
        return age_bins, i_age_arriving_ds

    def get_matrix_release_age_arriving_deep_sea(self, h_deep_sea:float) -> tuple:
        p_arriving_ds, _ = self.get_indices_arriving_in_deep_sea(h_deep_sea)
        t_release = self.get_release_time_index()
        unique_t_release = np.unique(t_release)
        i_unique_t_release = []
        for p in p_arriving_ds:
            i_unique_t_release.append(np.where(unique_t_release==t_release[p])[0][0])
        i_unique_t_release = np.array(i_unique_t_release)

        age_bins, i_age_arriving_ds = self.bin_age_arriving_in_deep_sea(h_deep_sea)

        matrix_arriving_ds = np.zeros((len(unique_t_release), len(age_bins)))
        if len(i_age_arriving_ds)!=0 and len(i_unique_t_release)!=0:
            np.add.at(matrix_arriving_ds, (i_unique_t_release, i_age_arriving_ds), 1)

        return self.time[unique_t_release], age_bins, matrix_arriving_ds

    def get_release_time_index(self) -> np.ndarray:
        p_no_nan, t_no_nan = np.where(~np.isnan(self.lon))
        _, i_sort = np.unique(p_no_nan, return_index=True)
        t_release = t_no_nan[i_sort]
        return t_release

    def get_n_particles_in_simulation(self) -> np.ndarray:
        t_release = self.get_release_time_index()

        n_particles_in_sim = np.zeros((len(self.time)))
        for t in range(len(self.time)):
            n_particles_in_sim[t:] += np.sum(t_release==t)

        return n_particles_in_sim

    @staticmethod
    def read_from_netcdf(input_path:str):
        log.info(f'Reading Particles from {input_path}')

        netcdf = Dataset(input_path)
        time_org = netcdf['time'][:].filled(fill_value=np.nan)
        time_units = netcdf['time'].units
        time = convert_time_to_datetime(time_org, time_units)

        status = netcdf['status'][:].filled(fill_value=-999)

        lon = netcdf['lon'][:].filled(fill_value=np.nan)
        lat = netcdf['lat'][:].filled(fill_value=np.nan)
        z = netcdf['z'][:].filled(fill_value=np.nan)

        if 'sea_water_salinity' in netcdf.variables:
            salt = netcdf['sea_water_salinity'][:].filled(fill_value=np.nan)
        else:
            salt = None
        if 'sea_water_temperature' in netcdf.variables:
            temp = netcdf['sea_water_temperature'][:].filled(fill_value=np.nan)
        else:
            temp = None

        age = netcdf['age_seconds'][:].filled(fill_value=np.nan)/(24*60*60) # convert age in seconds to age in days

        if 'moving' in netcdf.variables:
            moving = netcdf['moving'][:].filled(fill_value=-999)
        else:
            moving = None

        netcdf.close()

        return Particles(time, status, lon, lat, z, salt, temp, age, moving)

class DensityGrid:
    def __init__(self, lon_range:list, lat_range:list, dx:float):
        self.dx = dx
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.lon = np.arange(self.lon_range[0], self.lon_range[1]+self.dx, self.dx)
        self.lat = np.arange(self.lat_range[0], self.lat_range[1]+self.dx, self.dx)

    def get_index(self, lon_p:np.ndarray, lat_p:np.ndarray) -> tuple:
        lon_index = np.floor((lon_p-self.lon_range[0])*1/self.dx)
        l_index_lon_over = lon_index >= abs(self.lon_range[1]-self.lon_range[0])*1/self.dx
        l_index_lon_under = lon_index < 0

        lat_index = np.floor((lat_p-self.lat_range[0])*1/self.dx)
        l_index_lat_over = lat_index >= abs(self.lat_range[1]-self.lat_range[0])*1/self.dx
        l_index_lat_under = lat_index < 0

        l_invalid_index = l_index_lon_over+l_index_lon_under+l_index_lat_over+l_index_lat_under

        if np.sum(l_invalid_index) != 0:
            warn(f'''Locations out of range of DensityGrid, using NaNs for these indices.''')

        lon_index[l_invalid_index] = np.nan
        lat_index[l_invalid_index] = np.nan

        return lon_index, lat_index


def get_particle_density(grid:DensityGrid, lon_p:np.ndarray, lat_p:np.ndarray) -> np.ndarray:
    lon_index, lat_index = grid.get_index(lon_p, lat_p)
    
    density = np.zeros((len(grid.lat), len(grid.lon)))
    
    density_1d = density.flatten()
    x = lon_index[~np.isnan(lon_index)].astype('int')
    y = lat_index[~np.isnan(lat_index)].astype('int')
    index_1d = np.ravel_multi_index(np.array([y, x]), density.shape)
    np.add.at(density_1d, index_1d, 1)
    density = density_1d.reshape(density.shape)

    return density