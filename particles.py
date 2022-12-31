from tools.timeseries import convert_time_to_datetime
from tools import log
from netCDF4 import Dataset
import numpy as np

class Particles:
    def __init__(self, time:np.ndarray,
                 status:np.ndarray,
                 lon:np.ndarray,
                 lat:np.ndarray,
                 z:np.ndarray,
                 salt:np.ndarray,
                 temp:np.ndarray,
                 age:np.ndarray):
        self.time = time
        self.status = status # 0: active, 1: stranded, -999: not active yet or out of domain
        self.lon = lon # [trajectory, time]
        self.lat = lat # [trajectory, time]
        self.z = z # [trajectory, time]

        self.salt = salt
        self.temp = temp
        
        self.age = age # [days]

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

    def filter_based_on_release_depth(self, h_min:float, h_max:float):
        l_depth = np.logical_and(self.z0<-h_min, self.z0>=-h_max)

        status = self.status[l_depth, :]
        lon = self.lon[l_depth, :]
        lat = self.lat[l_depth, :]
        z = self.z[l_depth, :]

        salt = self.salt[l_depth, :]
        temp = self.temp[l_depth, :]

        age = self.age[l_depth, :]

        return Particles(self.time, status, lon, lat, z, salt, temp, age)

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

        return Particles(self.time, status, lon, lat, z, salt, temp, age)

    def get_l_deep_sea(self, h_deep_sea:float) -> np.ndarray:
        i_ds, j_ds = np.where(self.z <= -h_deep_sea)
        l_deep_sea = np.zeros(self.z.shape)
        for k in range(len(i_ds)):
            # once particles in deep sea, they will remain there
            # (stops particles that move out of simulation domain
            # from being removed from deep sea count)
            # IMPORTANT: this may need to change
            l_deep_sea[i_ds[k], j_ds[k]:] = 1

        return l_deep_sea

    def get_indices_arriving_in_deep_sea(self, h_deep_sea:float) -> tuple:
        l_deep_sea = self.get_l_deep_sea(h_deep_sea)
        p_ds, t_ds = np.where(l_deep_sea==1)
        p_arriving_ds, i_sort = np.unique(p_ds, return_index=True)
        t_arriving_ds = t_ds[i_sort]
        return p_arriving_ds, t_arriving_ds

    def get_n_deep_sea(self, h_deep_sea:float) -> np.ndarray:
        l_deep_sea = self.get_l_deep_sea(h_deep_sea)
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

        salt = netcdf['sea_water_salinity'][:].filled(fill_value=np.nan)
        temp = netcdf['sea_water_temperature'][:].filled(fill_value=np.nan)

        age = netcdf['age_seconds'][:].filled(fill_value=np.nan)/(24*60*60) # convert age in seconds to age in days

        netcdf.close()

        return Particles(time, status, lon, lat, z, salt, temp, age)
