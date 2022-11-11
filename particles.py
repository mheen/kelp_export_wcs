from roms_data import RomsGrid
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cftr
import numpy as np

import sys
sys.path.append('..')
from py_tools.timeseries import convert_time_to_datetime
from py_tools.files import get_dir_from_json

class Particles:
    def __init__(self, time:np.ndarray,
                 status:np.ndarray,
                 lon:np.ndarray,
                 lat:np.ndarray,
                 z:np.ndarray):
        self.time = time
        self.status = status # 0: active, 1: stranded, -999: not active yet or out of domain
        self.lon = lon
        self.lat = lat
        self.z = z

        self.add_initial_positions()
        self.add_total_number_particles()

    def get_fraction_in_deep_sea(self, h_deep_sea:float) -> np.ndarray:
        l_deep_sea = self.z <= -h_deep_sea
        n_deep_sea = np.sum(l_deep_sea, axis=0)
        f_deep_sea = n_deep_sea/self.total_particles

        return f_deep_sea

    def add_total_number_particles(self):
        t_release = self.get_release_time_index()

        self.total_particles = np.zeros((len(self.time)))
        for t in range(len(self.time)):
            self.total_particles[t:] += np.sum(t_release==t)

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

    def get_release_time_index(self) -> np.ndarray:
        p_no_nan, t_no_nan = np.where(~np.isnan(self.lon))
        _, i_sort = np.unique(p_no_nan,return_index=True)
        t_release = t_no_nan[i_sort]
        return t_release

    def filter_based_on_release_depth(self, h_min:float, h_max:float):
        l_depth = np.logical_and(self.z0<-h_min, self.z0>=-h_max)

        status = self.status[l_depth, :]
        lon = self.lon[l_depth, :]
        lat = self.lat[l_depth, :]
        z = self.z[l_depth, :]

        return Particles(self.time, status, lon, lat, z)

    def filter_based_on_release_lon_lat_range(self, lon_range:list, lat_range:list):
        l_lon_range = np.logical_and(self.lon0>=lon_range[0], self.lon0<=lon_range[1])
        l_lat_range = np.logical_and(self.lat0>=lat_range[0], self.lat0<=lat_range[1])
        l_range = np.logical_and(l_lon_range, l_lat_range)

        status = self.status[l_range, :]
        lon = self.lon[l_range, :]
        lat = self.lat[l_range, :]
        z = self.z[l_range, :]

        return Particles(self.time, status, lon, lat, z)

    @staticmethod
    def read_from_netcdf(input_path:str):
        netcdf = Dataset(input_path)
        time_org = netcdf['time'][:].filled(fill_value=np.nan)
        time_units = netcdf['time'].units
        time = convert_time_to_datetime(time_org, time_units)

        status = netcdf['status'][:].filled(fill_value=-999)

        lon = netcdf['lon'][:].filled(fill_value=np.nan)
        lat = netcdf['lat'][:].filled(fill_value=np.nan)
        z = netcdf['z'][:].filled(fill_value=np.nan)

        return Particles(time, status, lon, lat, z)

def animate(p:Particles,
            lon_range=None, lat_range=None, output_path=None,
            dpi=100, fps=25):

    color_p = 'k'

    grid = RomsGrid.read_from_netcdf()
    if lon_range is None:
        lon_range, _ = grid.get_lon_lat_range()
    if lat_range is None:
        _, lat_range = grid.get_lon_lat_range()
    
    writer = animation.PillowWriter(fps=fps)

    # plot map
    plt.rcParams.update({'font.size' : 15})
    plt.rcParams.update({'font.family': 'arial'})
    plt.rcParams.update({'figure.dpi': dpi})
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cftr.LAND, color='#FAF1D7', zorder=4)
    ax.add_feature(cftr.OCEAN, color='#67C1E7', zorder=0)
    ax.add_feature(cftr.COASTLINE, color='#8A8A8A', zorder=5)    
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], ccrs.PlateCarree())

    # animated points
    point = ax.plot([], [], 'o', color=color_p, markersize=2, zorder=1)[0]

    # animated text
    ttl = ax.text(0.02,0.98,'',transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor='w', alpha=0.3, edgecolor='w',pad=2))
    ttl.set_animated(True)
    
    def init():
        point.set_data([],[])
        ttl.set_text('')
        return point, ttl

    def animate(i):        
        x, y = (p.lon[:,i],p.lat[:,i])
        point.set_data(x,y)
        title = p.time[i].strftime('%d-%m-%Y %H:%M')
        ttl.set_text(title)
        return point, ttl

    anim = animation.FuncAnimation(plt.gcf(),animate,init_func=init,frames=len(time),blit=True)
    if output_path is not None:
        anim.save(output_path, writer=writer)
    else:
        plt.show()

def plot_percentage_in_deepsea(p:Particles, h_deep_sea=150,
                               depth_intervals=[5, 10, 20, 30, 40, 50],
                               show=True, output_path=None):

    f_ds = p.get_fraction_in_deep_sea(h_deep_sea)

    fig = plt.figure(figsize=(4, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(p.time, f_ds, '-k')
    ax1.set_ylabel(f'Fraction particles in deep sea (<{h_deep_sea} m)')

    ax2 = plt.subplot(2, 1, 2)
    for d in range(len(depth_intervals)-1):
        p_depth = p.filter_based_on_release_depth(depth_intervals[d], depth_intervals[d+1])
        f_ds_depth = p_depth.get_fraction_in_deep_sea(h_deep_sea)
        ax2.plot(p_depth.time, f_ds_depth, label=f'{depth_intervals[d+1]} m')
    ax2.set_ylabel(f'Fraction of particles in deep sea (<{h_deep_sea} m)')
    ax2.legend(title='Release depth')

    if show is True:
        plt.show()
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)  

if __name__ == '__main__':
    p = Particles.read_from_netcdf(f'{get_dir_from_json("input/dirs.json", "opendrift")}perth_2022.nc')

    # animation_path = f'{get_dir_from_json("input/dirs.json", "plots")}perth_2022.gif'
    # animate(p, output_path=animation_path)

    plot_percentage_in_deepsea(p)

