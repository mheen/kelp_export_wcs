from basic_maps import perth_map
from bathymetry_data import BathymetryData
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import numpy as np
from warnings import warn

import sys
sys.path.append('..')
from py_tools.timeseries import convert_time_to_datetime
from py_tools.files import get_dir_from_json
from py_tools import log

class Particles:
    def __init__(self, time:np.ndarray,
                 status:np.ndarray,
                 lon:np.ndarray,
                 lat:np.ndarray,
                 z:np.ndarray,
                 h:np.ndarray,
                 u: np.ndarray,
                 v: np.ndarray,
                 salt:np.ndarray,
                 temp:np.ndarray,
                 age:np.ndarray,
                 origin_marker:np.ndarray):
        self.time = time
        self.status = status # 0: active, 1: stranded, -999: not active yet or out of domain
        self.lon = lon
        self.lat = lat
        self.z = z

        self.h = h
        self.u = u
        self.v = v
        self.salt = salt
        self.temp = temp
        
        self.age = age # [days]
        self.origin_marker = origin_marker

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

        h = self.h[l_depth, :]
        u = self.u[l_depth, :]
        v = self.v[l_depth, :]
        salt = self.salt[l_depth, :]
        temp = self.temp[l_depth, :]

        age = self.age[l_depth, :]
        origin_marker = self.origin_marker[l_depth, :]

        return Particles(self.time, status, lon, lat, z, h, u, v, salt, temp, age, origin_marker)

    def filter_based_on_release_lon_lat_range(self, lon_range:list, lat_range:list):
        l_lon_range = np.logical_and(self.lon0>=lon_range[0], self.lon0<=lon_range[1])
        l_lat_range = np.logical_and(self.lat0>=lat_range[0], self.lat0<=lat_range[1])
        l_range = np.logical_and(l_lon_range, l_lat_range)

        status = self.status[l_range, :]
        lon = self.lon[l_range, :]
        lat = self.lat[l_range, :]
        z = self.z[l_range, :]

        h = self.h[l_range, :]
        u = self.u[l_range, :]
        v = self.v[l_range, :]
        salt = self.salt[l_range, :]
        temp = self.temp[l_range, :]

        age = self.age[l_range, :]
        origin_marker = self.origin_marker[l_range, :]

        return Particles(self.time, status, lon, lat, z, h, u, v, salt, temp, age, origin_marker)

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

    def plot_matrix_arriving_in_deep_sea(self, h_deep_sea:float,
                                         ax=None, show=True, cmap='plasma',
                                         output_path=None) -> plt.axes:
        if ax is None:
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes()
        
        time_release, age_arriving_ds, matrix_arriving_ds = self.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
        n_dim_matrix = len(matrix_arriving_ds.flatten().shape)
        if n_dim_matrix == 1:
            warn(f'''There is only a single release available in this simulation;
                 matrix_arriving_ds is {n_dim_matrix}D, so this plot cannot be made. Skipping.''')
            return ax

        n_particles_in_simulation = self.get_n_particles_in_simulation()

        y, x = np.meshgrid(age_arriving_ds, time_release)
        z = matrix_arriving_ds/n_particles_in_simulation[0]
        z[z==0] = np.nan

        c = ax.pcolormesh(x, y, z, cmap=cmap)
        cbar = plt.colorbar(c)
        cbar.set_label('Fraction of released particles arriving in deep sea')

        def _create_ticklabels_at_interval(tick_labels:np.ndarray, n_int:int) -> list:
            tick_labels_int = []
            for i in range(len(tick_labels)):
                if i%n_int == 0: # (% is modulo)
                    tick_labels_int.append(str(tick_labels[i]))
                else:
                    tick_labels_int.append('')
            return tick_labels_int

        myFmt = mdates.DateFormatter('%Y-%m-%d')
        ax.set_xticks(time_release[::5])
        ax.set_xticklabels(time_release[::5], rotation=90)
        ax.xaxis.set_major_formatter(myFmt)
        ax.set_xlabel('Release date')
        ax.set_xlim([time_release[0], time_release[-1]])

        ax.set_yticks(age_arriving_ds)
        ax.set_yticklabels(_create_ticklabels_at_interval(age_arriving_ds.astype(int), 5))
        ax.set_ylabel('Particle age (days)')
        ax.set_ylim([0, np.nanmax(age_arriving_ds)])

        if output_path is not None:
            log.info(f'Saving figure to: {output_path}')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()
        else:
            return ax

    def plot_age_in_deep_sea(self, h_deep_sea:float, total_particles=None,
                             linestyle='-', color='k', label='',
                             ax=None, show=True, output_path=None) -> plt.axes:
        if ax is None:
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes()
            ax.set_xlabel('Particle age (days)')
            ax.set_ylabel('Fraction in deep sea')

        _, age_arriving_ds, matrix_arriving_ds = self.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
        n_dim_matrix = len(matrix_arriving_ds.shape)
        if n_dim_matrix == 2:
            n_deep_sea_per_age = np.cumsum(np.sum(matrix_arriving_ds, axis=0))
        elif n_dim_matrix == 1:
            n_deep_sea_per_age = np.cumsum(matrix_arriving_ds)
        else:
            raise ValueError(f'You should not be able to get here: matrix_arriving_ds is {n_dim_matrix}D')

        if total_particles is None:
            total_particles = self.lon.shape[0]
        f_deep_sea_per_age = n_deep_sea_per_age/total_particles # divided by total # particles

        ax.plot(age_arriving_ds, f_deep_sea_per_age, linestyle, color=color, label=label)

        if output_path is not None:
            log.info(f'Saving figure to: {output_path}')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()
        else:
            return ax

    def plot_timeseries_in_deep_sea(self, h_deep_sea:float, total_particles=None,
                                    ax=None, show=True, output_path=None) -> plt.axes:
        if ax is None:
            fig = plt.figure(figsize=(10, 5))
            fig.tight_layout()
            ax = plt.axes()
            ax.set_ylabel('# of particles')

        n_deep_sea = self.get_n_deep_sea(h_deep_sea)
        n_particles_sim = self.get_n_particles_in_simulation()

        ax.plot(self.time, n_deep_sea, '-k', label='# in deep sea')
        ax.plot(self.time, n_particles_sim, '--k', label='# in simulation')
        ax.legend(loc='upper left')

        color_ax2='g'
        ax2 = ax.twinx()
        ax2.set_ylabel('Fraction in deep sea', color=color_ax2)
        if total_particles is None:
            total_particles = self.lon.shape[0]
        ax2.plot(self.time, n_deep_sea/total_particles, '-', color=color_ax2)
        ax2.tick_params(axis='y', labelcolor=color_ax2)

        if output_path is not None:
            log.info(f'Saving figure to: {output_path}')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()
        else:
            return ax

    def plot_initial(self, ax=None, show=True, color='k') -> plt.axes:
        if ax is None:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax = perth_map(ax)

        ax.plot(self.lon0, self.lat0, '.', color=color)

        if show is True:
            plt.show()
        else:
            return ax

    def animate(self, color_p='k', color_i='g', show_bathymetry=True,
            show_initial=True, output_path=None,
            dpi=100, fps=25):
    
        writer = animation.PillowWriter(fps=fps)

        # plot map
        plt.rcParams.update({'font.size' : 15})
        plt.rcParams.update({'font.family': 'arial'})
        plt.rcParams.update({'figure.dpi': dpi})
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = perth_map(ax)

        if show_bathymetry is True:
            bathymetry = BathymetryData.read_from_netcdf()
            ax = bathymetry.plot_contours(ax, show=False, color='#757575')
        
        if show_initial is True:
            ax.plot(self.lon0, self.lat0, 'o', color=color_i, markersize=2, zorder=1, label='initial')
            ax.legend(loc='upper right')

        # animated points
        point = ax.plot([], [], 'o', color=color_p, markersize=2, zorder=2)[0]

        # animated text
        ttl = ax.text(0.5, 1.04,'', transform=ax.transAxes,
                    ha='center', va='top',
                    bbox=dict(facecolor='w', alpha=0.3, edgecolor='w', pad=2))
        ttl.set_animated(True)
        
        def init():
            point.set_data([],[])
            ttl.set_text('')
            return point, ttl

        def animate(i):        
            x, y = (self.lon[:,i], self.lat[:,i])
            point.set_data(x,y)
            title = self.time[i].strftime('%d-%m-%Y %H:%M')
            ttl.set_text(title)
            return point, ttl

        anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=len(self.time), blit=True)
        if output_path is not None:
            log.info(f'Saving animation to: {output_path}')
            anim.save(output_path, writer=writer)
        else:
            plt.show()

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

        h = netcdf['sea_floor_depth_below_sea_level'][:].filled(fill_value=np.nan)
        u =  netcdf['x_sea_water_velocity'][:].filled(fill_value=np.nan)
        v = netcdf['y_sea_water_velocity'][:].filled(fill_value=np.nan)
        salt = netcdf['sea_water_salinity'][:].filled(fill_value=np.nan)
        temp = netcdf['sea_water_temperature'][:].filled(fill_value=np.nan)

        age = netcdf['age_seconds'][:].filled(fill_value=np.nan)/(24*60*60) # convert age in seconds to age in days
        origin_marker = netcdf['origin_marker'][:].filled(fill_value=-999)

        netcdf.close()

        return Particles(time, status, lon, lat, z, h, u, v, salt, temp, age, origin_marker)

if __name__ == '__main__':
    h_deep_sea = 150 # m depth
    file_name = 'perth_2017'
    get_animation = False

    p = Particles.read_from_netcdf(f'{get_dir_from_json("input/dirs.json", "opendrift")}{file_name}.nc')

    if get_animation is True:
        animation_path = f'{get_dir_from_json("input/dirs.json", "plots")}{file_name}.gif'
        p.animate(output_path=animation_path)

    p.plot_timeseries_in_deep_sea(h_deep_sea, show=False, output_path=f'{get_dir_from_json("input/dirs.json", "plots")}{file_name}_timeseries_deep_sea.jpg')

    p.plot_matrix_arriving_in_deep_sea(h_deep_sea, show=False, output_path=f'{get_dir_from_json("input/dirs.json", "plots")}{file_name}_matrix_arriving_deep_sea.jpg')

    p.plot_age_in_deep_sea(h_deep_sea, show=False, output_path=f'{get_dir_from_json("input/dirs.json", "plots")}{file_name}_age_deep_sea.jpg')
