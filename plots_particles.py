from tools.timeseries import get_closest_time_index
from tools.files import get_dir_from_json
from tools import log
from particles import Particles
from bathymetry_data import BathymetryData
from kelp_map import KelpProbability
from location_info import LocationInfo, get_location_info
from basic_maps import plot_basic_map
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.units as munits
import cartopy.crs as ccrs
from warnings import warn
import numpy as np
from datetime import date, datetime, timedelta

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats = ['%y',  # ticks are mostly years
                    '%b',       # ticks are mostly months
                    '%d',       # ticks are mostly days
                    '%H:%M',    # hrs
                    '%H:%M',    # min
                    '%S.%f', ]  # secs
# these are mostly just the level above...
formatter.zero_formats = [''] + formatter.formats[:-1]
# ...except for ticks that are mostly hours, then it is nice to have
# month-day:
formatter.zero_formats[3] = '%d-%b'

formatter.offset_formats = ['',
                            '%Y',
                            '%b %Y',
                            '%d %b %Y',
                            '%d %b %Y',
                            '%d %b %Y %H:%M', ]

def animate_particles(particles:Particles, location_info:LocationInfo,
                      show_bathymetry=True, show_kelp_map=True,
                      output_path=None, color_p='k',
                      dpi=100, fps=25):

    writer = animation.PillowWriter(fps=fps)

    # plot map
    plt.rcParams.update({'font.size' : 15})
    plt.rcParams.update({'font.family': 'arial'})
    plt.rcParams.update({'figure.dpi': dpi})
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_basic_map(ax, location_info)

    if show_bathymetry is True:
        bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')
        ax = bathymetry.plot_contours(location_info, ax=ax, show=False, color='#757575')
    
    if show_kelp_map is True:
        kelp_prob = KelpProbability.read_from_tiff('input/perth_kelp_probability.tif')
        ax = kelp_prob.plot(location_info, ax=ax, show=False)

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
        x, y = (particles.lon[:,i], particles.lat[:,i])
        point.set_data(x,y)
        title = particles.time[i].strftime('%d-%m-%Y %H:%M')
        ttl.set_text(title)
        return point, ttl

    anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=len(particles.time), blit=True)
    if output_path is not None:
        log.info(f'Saving animation to: {output_path}')
        anim.save(output_path, writer=writer)
    else:
        plt.show()

def plot_matrix_arriving_in_deep_sea(particles:Particles, h_deep_sea:float,
                                     ax=None, show=True, cmap='plasma',
                                     output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()
    
    time_release, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
    n_dim_matrix = len(matrix_arriving_ds.shape)
    if n_dim_matrix == 1:
        warn(f'''There is only a single release available in this simulation;
                matrix_arriving_ds is {n_dim_matrix}D, so this plot cannot be made. Skipping.''')
        return ax

    n_particles_in_simulation = particles.get_n_particles_in_simulation()

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

    # myFmt = mdates.DateFormatter('%d-%m-%Y')
    # ax.set_xticks(time_release[::5])
    # ax.set_xticklabels(time_release[::5], rotation=90)
    # ax.xaxis.set_major_formatter(myFmt)
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

def plot_histogram_arriving_in_deep_sea(particles:Particles, h_deep_sea:float,
                                        ax=None, show=True, output_path=None,
                                        color='#1b7931', edgecolor='none'):

    _, t_arriving_ds = particles.get_indices_arriving_in_deep_sea(h_deep_sea)
    
    n_days = (particles.time[-1]-particles.time[0]).days
    time_days = np.array([particles.time[0]+timedelta(days=n) for n in range(n_days)])

    n_arriving, _ = np.histogram(particles.time[t_arriving_ds], bins=time_days)
    
    n_particles_in_simulation = particles.get_n_particles_in_simulation()
    n_particles_in_simulation_per_day = []
    for i in range(len(time_days)):
        i_time = np.where(particles.time==time_days[i])[0][0]
        n_particles_in_simulation_per_day.append(n_particles_in_simulation[i_time])
    n_particles_in_simulation_per_day = np.array(n_particles_in_simulation_per_day)

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()

    ax.bar(time_days[:-1], n_arriving/n_particles_in_simulation_per_day[:-1]*100, color=color, edgecolor=edgecolor)

    ax.set_xlim([time_days[0], time_days[-1]])
    ax.set_ylabel('Particles moving past shelf break (%)')

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_age_in_deep_sea(particles:Particles, h_deep_sea:float, total_particles=None,
                         linestyle='-', color='k', label='', age_lim=None,
                         ax=None, show=True, output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()
        ax.set_xlabel('Particle age (days)')
        ax.set_ylabel('Fraction in deep sea')
        if age_lim is not None:
            ax.set_xlim([0, age_lim])

    _, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
    n_dim_matrix = len(matrix_arriving_ds.shape)
    if n_dim_matrix == 2:
        n_deep_sea_per_age = np.cumsum(np.sum(matrix_arriving_ds, axis=0))
    elif n_dim_matrix == 1:
        n_deep_sea_per_age = np.cumsum(matrix_arriving_ds)
    else:
        raise ValueError(f'You should not be able to get here: matrix_arriving_ds is {n_dim_matrix}D')

    if total_particles is None:
        total_particles = particles.lon.shape[0]
    f_deep_sea_per_age = n_deep_sea_per_age/total_particles # divided by total # particles

    ax.plot(age_arriving_ds, f_deep_sea_per_age, linestyle, color=color, label=label)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_timeseries_in_deep_sea(particles:Particles, h_deep_sea:float, total_particles=None,
                                ax=None, show=True, output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        fig.tight_layout()
        ax = plt.axes()
        ax.set_ylabel('# of particles')

    n_deep_sea = particles.get_n_deep_sea(h_deep_sea)
    n_particles_sim = particles.get_n_particles_in_simulation()

    ax.plot(particles.time, n_deep_sea, '-k', label='# in deep sea')
    ax.plot(particles.time, n_particles_sim, '--k', label='# in simulation')
    ax.legend(loc='upper left')

    color_ax2='g'
    ax2 = ax.twinx()
    ax2.set_ylabel('Fraction in deep sea', color=color_ax2)
    if total_particles is None:
        total_particles = particles.lon.shape[0]
    ax2.plot(particles.time, n_deep_sea/total_particles, '-', color=color_ax2)
    ax2.tick_params(axis='y', labelcolor=color_ax2)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_particle_locations(particles:Particles, location_info:LocationInfo,
                            time=None, t=None,
                            ax=None, show=True, color='k') -> plt.axes:
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    if t is None and time is None:
        log.info('''Plotting initial particles locations. Specify either requested
                 time or time index t if you want to plot a different time.''')
        ax.plot(particles.lon0, particles.lat0, '.', color=color)
    elif t == 0:
        ax.plot(particles.lon0, particles.lat0, '.', color=color)
    elif t != 0:
        ax.plot(particles.lon[:, t], particles.lat[:, t], '.', color=color)
    elif t is None and time is not None:
        t = get_closest_time_index(particles.time, time)
        ax.plot(particles.lon[:, t], particles.lat[:, t], '.', color=color)

    if show is True:
        plt.show()
    else:
        return ax

if __name__ == '__main__':
    location_info = get_location_info('cwa_perth')
    h_deep_sea = 500 # m depth: edge of continental shelf

    input_path = f'{get_dir_from_json("opendrift")}cwa-perth_2017-Mar-Aug.nc'
    particles = Particles.read_from_netcdf(input_path)
    
    # animation_path = f'{get_dir_from_json("plots")}animation_cwa-perth_2017-Mar-Aug.gif'
    # animate_particles(particles, location_info, output_path=animation_path)

    # plot_particle_locations(particles, location_info, t=0)

    plot_histogram_arriving_in_deep_sea(particles, h_deep_sea)

